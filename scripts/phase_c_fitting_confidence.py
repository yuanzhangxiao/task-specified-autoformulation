#!/usr/bin/env python3
"""Export M18 endpoints, run M19 training-only checks, and report recovery."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import confidence_campaign as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("export")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "run", "report"):
        p = commands.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            p.add_argument("--inputs", type=Path, required=True)
            p.add_argument("--config", type=Path, required=True)
        if name == "run":
            p.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "export":
        result = campaign.export(args.source, args.output)
    elif args.command == "prepare":
        result = campaign.prepare(
            args.root,
            args.inputs,
            campaign.ConfidencePolicy.model_validate_json(args.config.read_text()),
        )
    elif args.command == "run":
        result = campaign.run_task(args.root, args.index)
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
