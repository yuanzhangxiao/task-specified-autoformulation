#!/usr/bin/env python3
"""Audit exact reuse and freeze a low-cost Phase-B candidate-generation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from autoformalism.data import BenchmarkRegistry

try:
    from scripts.replay_phase_b_frozen_candidates import (
        _development_context,
        _load_pool,
        _prepare_candidates,
        _write_json,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from replay_phase_b_frozen_candidates import (  # type: ignore[no-redef]
        _development_context,
        _load_pool,
        _prepare_candidates,
        _write_json,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-reusable-candidates", type=int, default=100)
    parser.add_argument("--local-proposer-call-budget", type=int, default=8)
    parser.add_argument("--hosted-rescue-call-cap", type=int, default=4)
    args = parser.parse_args()
    if (
        args.maximum_reusable_candidates < 1
        or args.local_proposer_call_budget < 1
        or args.hosted_rescue_call_cap < 0
    ):
        raise SystemExit("planning budgets must be positive")

    pool = _load_pool(args.candidate_pool)
    cells = []
    for spec in BenchmarkRegistry().specs():
        if not spec.benchmark_id.startswith("phase_b_"):
            continue
        tier = next(iter(spec.tier_roles))
        source = _legacy_source(spec.benchmark_id)
        helper_args = SimpleNamespace(
            public_data_root=args.public_data_root,
            destination_benchmark=spec.benchmark_id,
            source_benchmark=source,
            tier=tier,
            maximum_candidates=args.maximum_reusable_candidates,
            output_root=args.output.parent / "preflight" / spec.benchmark_id,
            seed=0,
        )
        _, context = _development_context(helper_args)
        source_artifacts, eligibility, eligible, selected = _prepare_candidates(
            helper_args, pool, context
        )
        exact_count = len(selected)
        cells.append(
            {
                "benchmark_id": spec.benchmark_id,
                "tier": tier,
                "legacy_source_benchmark": source,
                "source_artifacts": len(source_artifacts),
                "exact_contract_artifacts": sum(
                    bool(item["eligible"]) for item in eligibility
                ),
                "unique_reusable_structures": len(eligible),
                "replay_candidate_cap": exact_count,
                "generation_lane": (
                    "reuse_plus_local_diversification"
                    if exact_count
                    else "local_generation_required"
                ),
                "planned_local_proposer_calls": args.local_proposer_call_budget,
                "planned_generation_judge_calls": 0,
                "conditional_hosted_rescue_call_cap": args.hosted_rescue_call_cap,
                "hosted_rescue_trigger": (
                    "no deterministically valid local candidate or frozen "
                    "development threshold failure"
                ),
            }
        )

    reusable_cells = sum(bool(item["replay_candidate_cap"]) for item in cells)
    payload: dict[str, Any] = {
        "schema_version": "phase_b_candidate_generation_plan_v1",
        "stage": "development_only_preflight",
        "uses_llm_calls": False,
        "uses_test_data": False,
        "cell_count": len(cells),
        "cells_with_exact_reuse": reusable_cells,
        "cells_requiring_local_generation": len(cells) - reusable_cells,
        "planned_local_proposer_calls": sum(
            item["planned_local_proposer_calls"] for item in cells
        ),
        "planned_generation_judge_calls": 0,
        "maximum_conditional_hosted_rescue_calls": sum(
            item["conditional_hosted_rescue_call_cap"] for item in cells
        ),
        "policy": {
            "local_generation_first": True,
            "judge_during_pool_generation": False,
            "hosted_calls_are_conditional": True,
            "selection_split": "validation",
            "test_split_opened": False,
        },
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, payload)
    _write_markdown(args.output.with_suffix(".md"), payload)
    summary = {key: payload[key] for key in payload if key != "cells"}
    print(json.dumps(summary, indent=2))


def _legacy_source(benchmark_id: str) -> str:
    if benchmark_id.startswith("phase_b_dalla_man_"):
        return "perturbed_b1" if "_perturbed_" in benchmark_id else "original_b1"
    if benchmark_id.startswith(
        tuple(f"phase_b_anonymous_system_t{index}_" for index in range(1, 5))
    ):
        return (
            "obfuscated_perturbed_case01"
            if "_perturbed_" in benchmark_id
            else "obfuscated_original_case01"
        )
    if benchmark_id.startswith("phase_b_cstr_") or "_obfuscated_" in benchmark_id:
        return "benchmark5"
    if benchmark_id.startswith("phase_b_alien_device_") or "_opaque_" in benchmark_id:
        return "benchmark6"
    raise ValueError(f"cannot map Phase-B cell to a legacy source: {benchmark_id}")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase-B candidate-generation preflight",
        "",
        (
            "This development-only audit performs no LLM calls and does not "
            "open test data."
        ),
        "",
        f"- cells: {payload['cell_count']}",
        f"- cells with exact reusable structures: {payload['cells_with_exact_reuse']}",
        (
            "- cells requiring local generation: "
            f"{payload['cells_requiring_local_generation']}"
        ),
        f"- planned local proposer calls: {payload['planned_local_proposer_calls']}",
        "- planned generation-stage judge calls: 0",
        (
            "- maximum conditional hosted rescue calls: "
            f"{payload['maximum_conditional_hosted_rescue_calls']}"
        ),
        "",
        "| Cell | Exact reusable structures | Generation lane |",
        "|---|---:|---|",
    ]
    for item in payload["cells"]:
        lines.append(
            f"| {item['benchmark_id']} | {item['replay_candidate_cap']} | "
            f"{item['generation_lane']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
