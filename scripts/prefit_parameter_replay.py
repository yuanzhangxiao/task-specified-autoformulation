#!/usr/bin/env python3
"""Replay a saved numerical proposal, then optionally fit its corrected child."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal import prefit_parameter_replay as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("replay", "fit", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "replay":
            command.add_argument("--source", type=Path, required=True)
            command.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    if args.command == "replay":
        result = campaign.prepare(args.source, args.root, attempt=args.attempt)
    else:
        result = getattr(campaign, args.command)(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
