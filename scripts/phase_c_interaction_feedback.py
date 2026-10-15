#!/usr/bin/env python3
"""Export, verify or locally qualify the M37 interaction-feedback handoff."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import interaction_feedback as feedback
from autoformalism.fitting import interaction_feedback_qualification as qualification


def main() -> None:
    """Keep packet export and four-control qualification independently runnable."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "verify"):
        child = commands.add_parser(name)
        child.add_argument("--diagnostics", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "qualify", "report"):
        child = commands.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        if name == "qualify":
            child.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command in {"export", "verify"}:
        operation = (
            feedback.export_packet if args.command == "export" else feedback.completed
        )
        result = operation(args.diagnostics, args.output).model_dump(mode="json")
    elif args.command == "prepare":
        result = qualification.prepare(args.root)
    elif args.command == "qualify":
        result = qualification.qualify(args.root, args.index)
    else:
        result = qualification.report(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
