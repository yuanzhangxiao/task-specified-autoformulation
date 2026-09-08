#!/usr/bin/env python3
"""Freeze, audit, or run the public-only pre-fitting feedback experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.staged_prefit_feedback_campaign import (
    audit_from_plan,
    freeze_prefit_campaign,
    run_prefit_campaign,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--source-hybrid-plan", type=Path, required=True)
    freeze.add_argument("--source-results", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--plan", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--wall-seconds", type=float)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_prefit_campaign(
            args.config,
            args.source_hybrid_plan,
            args.source_results,
            args.output,
        )
        print(
            json.dumps(
                {"plan_sha256": result["plan_sha256"], "tasks": len(result["tasks"])}
            )
        )
    elif args.command == "audit":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        result = audit_from_plan(plan)
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        print(
            json.dumps(
                run_prefit_campaign(
                    args.plan,
                    args.output,
                    args.base_url,
                    wall_seconds=args.wall_seconds,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
