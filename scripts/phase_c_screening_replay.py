#!/usr/bin/env python3
"""Export, freeze, calibrate, run and inspect the CPU-only M10 diagnostic."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import screening_replay as campaign
from autoformalism.fitting.screening_replay_export import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("export")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", default="alien_hard")
    p = commands.add_parser("prepare")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    for name in ("calibrate", "run", "report"):
        p = commands.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        if name != "report":
            p.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "export":
        result = export(args.source, args.output, case_name=args.case)
    elif args.command == "prepare":
        result = campaign.prepare(
            args.root,
            args.inputs,
            campaign.ReplayPolicy.model_validate_json(args.config.read_text()),
        )
    elif args.command == "calibrate":
        result = campaign.calibrate(args.root, args.index)
    elif args.command == "run":
        result = campaign.run_task(args.root, args.index)
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
