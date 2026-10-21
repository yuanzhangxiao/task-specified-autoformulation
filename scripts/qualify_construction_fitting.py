"""Freeze, run, and inspect the opt-in returned-construction fitting experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autoformalism.research import construction_fit_qualification as campaign

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    """Keep public preparation and each bounded CPU task independently runnable."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "verify", "run", "report", "inspect", "smoke"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--root", "--campaign-root", type=Path, required=True
        )
        if command == "prepare":
            command_parser.add_argument("--source", type=Path, required=True)
            command_parser.add_argument("--public-cell", type=Path, required=True)
            command_parser.add_argument(
                "--config",
                "--policy",
                type=Path,
                default=REPO / "configs/construction_fit_qualification_v1.json",
            )
        if command == "run":
            command_parser.add_argument(
                "--index", "--task-index", type=int, required=True
            )
    args = parser.parse_args()
    if args.command == "prepare":
        from autoformalism.research.construction_fit_inputs import load_inputs

        policy = campaign.Policy.model_validate_json(args.config.read_text())
        result = campaign.prepare(
            args.root, load_inputs(args.source, args.public_cell), policy
        )
    elif args.command == "verify":
        plan, _ = campaign.verify(args.root)
        result = {"verified": True, "tasks": len(plan["tasks"])}
    elif args.command == "run":
        result = campaign.run_task(args.root, args.index)
    else:
        result = getattr(campaign, args.command)(args.root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    main()
