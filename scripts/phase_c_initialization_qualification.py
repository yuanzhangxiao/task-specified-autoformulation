#!/usr/bin/env python3
"""Run the isolated M43 initialization and strict-prefix forecast qualification."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import initialization_qualification as campaign

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    """Keep preparation, verification, execution and reporting notebook independent."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify", "run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--inputs", type=Path)
            command.add_argument(
                "--config",
                type=Path,
                default=REPO / "configs/phase_c_initialization_qualification_v1.json",
            )
        elif name == "run":
            command.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    print(f"M43: {args.command} root={args.root}", flush=True)
    if args.command == "prepare":
        result = campaign.prepare(
            args.root,
            args.inputs,
            campaign.Policy.model_validate_json(args.config.read_text()),
        )
    elif args.command == "verify":
        plan, _ = campaign.verify(args.root)
        result = {"verified": True, "tasks": len(plan["tasks"])}
    elif args.command == "run":
        result = campaign.run_task(args.root, args.index)
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
