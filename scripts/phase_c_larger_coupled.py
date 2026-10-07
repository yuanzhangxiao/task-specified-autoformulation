#!/usr/bin/env python3
"""Prepare, run and report the M18 coupled linear rollout comparison."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import larger_coupled_campaign as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "qualify", "run", "report"):
        p = commands.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            p.add_argument("--inputs", type=Path, required=True)
            p.add_argument("--config", type=Path, required=True)
        if name == "run":
            p.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "export":
        result = campaign.controls.export_inputs(args.output)
    elif args.command == "prepare":
        result = campaign.prepare(
            args.root,
            args.inputs,
            campaign.PolishingPolicy.model_validate_json(args.config.read_text()),
        )
    elif args.command == "qualify":
        result = campaign.qualify(args.root)
    elif args.command == "run":
        result = campaign.run_task(args.root, args.index)
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
