#!/usr/bin/env python3
"""Render all Phase-B prompt pairs into a compact human-review bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks import (
    load_suite_spec,
    phase_b_public_spec,
    render_phase_b_prompts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("configs/benchmarks/phase_b_suite_v1.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    suite = load_suite_spec(args.suite)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for family in suite.families:
        for task in family.tasks:
            task_argument = task if family.family == "dalla_man" else None
            for condition in family.dynamics_conditions:
                dynamics = "canonical" if condition == "not_applicable" else condition
                for tier in family.tiers:
                    for variant in family.semantic_variants:
                        spec = phase_b_public_spec(
                            family.family,
                            tier.name,
                            variant,
                            task=task_argument,
                            dynamics=dynamics,
                            data_root=args.data_root,
                        )
                        proposer, judge = render_phase_b_prompts(spec)
                        cell = args.output_root / spec.benchmark_id
                        cell.mkdir(parents=True, exist_ok=True)
                        proposer_path = cell / "proposer_prompt.txt"
                        judge_path = cell / "judge_prompt.txt"
                        proposer_path.write_text(proposer, encoding="utf-8")
                        judge_path.write_text(judge, encoding="utf-8")
                        records.append(
                            {
                                "benchmark_id": spec.benchmark_id,
                                "family": family.family,
                                "task": task,
                                "dynamics": dynamics,
                                "tier": tier.name,
                                "semantic_variant": variant,
                                "proposer_prompt": str(
                                    proposer_path.relative_to(args.output_root)
                                ),
                                "judge_prompt": str(
                                    judge_path.relative_to(args.output_root)
                                ),
                            }
                        )

    (args.output_root / "index.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Phase-B prompt review index",
        "",
        "Generated from the frozen typed public specification. No trajectory or "
        "private-reference data are included.",
        "",
        "| Family | Task | Dynamics | Tier | Variant | Proposer | Judge |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in records:
        lines.append(
            "| {family} | {task} | {dynamics} | {tier} | {semantic_variant} | "
            "[{proposer_prompt}]({proposer_prompt}) | "
            "[{judge_prompt}]({judge_prompt}) |".format(**item)
        )
    (args.output_root / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Rendered {len(records)} prompt pairs to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
