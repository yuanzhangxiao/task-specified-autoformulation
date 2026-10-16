#!/usr/bin/env python3
"""Prepare, verify and complete M39 profile diagnostics without refitting."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import practical_uncertainty_supplement as campaign

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    """Keep the separately budgeted supplement runnable without notebooks."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify", "run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--source", type=Path, required=True)
            command.add_argument(
                "--config",
                type=Path,
                default=REPO
                / "configs/phase_c_practical_uncertainty_supplement_v1.json",
            )
        if name == "run":
            command.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = campaign.prepare(
            args.root,
            args.source,
            campaign.Policy.model_validate_json(args.config.read_text()),
        )
    elif args.command == "verify":
        plan, _ = campaign.verify(args.root)
        result = {
            "verified": True,
            "plan_sha256": campaign.public.content_sha256(plan),
            "tasks": len(plan["tasks"]),
            "parameter_refitting_performed": False,
        }
    elif args.command == "run":
        result = campaign.run_task(args.root, args.index)
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
