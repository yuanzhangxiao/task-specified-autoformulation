#!/usr/bin/env python3
"""Export, prepare, replay or inspect M31 training-only fitting feedback."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import feedback_campaign as campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "verify", "run", "report"):
        child = commands.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            child.add_argument("--inputs", type=Path, required=True)
            child.add_argument("--config", type=Path, required=True)
        if name == "run":
            child.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "export":
        result = campaign.export_inputs(args.source, args.output)
    elif args.command == "prepare":
        result = campaign.prepare(
            args.root,
            args.inputs,
            campaign.Policy.model_validate_json(args.config.read_text()),
        )
    elif args.command == "verify":
        plan, _ = campaign.verify(args.root)
        result = {"status": "verified", "tasks": len(plan["tasks"])}
    elif args.command == "run":
        result = campaign.run_task(args.root, args.index)
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
