#!/usr/bin/env python3
"""Freeze, run, and summarize a Phase-B LLM-ODE campaign.

The search itself comes from the pinned upstream checkout, which `run` binds
through AF_LLM_ODE_ROOT. `prepare` and `report` never contact a provider.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal.llm_ode_campaign import prepare, report, run
from autoformalism.rebuttal.llm_ode_driver import build_searcher
from autoformalism.rebuttal.prefit_replay import sealed_read


def _searcher(root: Path):
    """Bind the pinned checkout and the job-local endpoint for one task."""
    checkout = os.environ.get("AF_LLM_ODE_ROOT")
    if not checkout:
        raise ValueError("AF_LLM_ODE_ROOT must point at the pinned LLM-ODE checkout")
    base_url = os.environ.get("AF_VLLM_BASE_URL")
    if not base_url:
        raise ValueError("AF_VLLM_BASE_URL must point at the served endpoint")
    sealed = sealed_read(root / "plan.json")
    budget = sealed["plan"]["budget"]
    if budget["unit"] != "iterations":
        raise ValueError(f"unexpected budget unit {budget['unit']!r}")
    return build_searcher(
        upstream_root=Path(checkout),
        base_url=base_url.rstrip("/") + "/v1",
        iterations=int(budget["declared"]),
        # From the sealed plan, never the environment: the island count
        # changes the search and the call volume, so it is part of what the
        # frozen plan means.
        islands=int(sealed["search_config"]["n_islands"]),
    )


def main() -> None:
    """Discovery is explicit; prepare and report make no provider call."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("prepare")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--public-root", type=Path, required=True)
    freeze.add_argument("--root", type=Path, required=True)
    worker = sub.add_parser("run")
    worker.add_argument("--root", type=Path, required=True)
    worker.add_argument("--index", type=int, required=True)
    summary = sub.add_parser("report")
    summary.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.config, args.public_root, args.root)
        value = {
            "plan_sha256": result["artifact_sha256"],
            "expected": len(result["rows"]),
            "maximum_logical_calls": result["maximum_logical_calls"],
            "reporting_qualifications": result["reporting_qualifications"],
        }
    elif args.command == "run":
        result = run(args.root, args.index, search=_searcher(args.root))
        value = {key: item for key, item in result.items() if key != "rows"}
    else:
        result = report(args.root)
        value = {key: item for key, item in result.items() if key != "rows"}
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
