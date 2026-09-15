#!/usr/bin/env python3
"""Prepare, run or report a separately bounded convergence diagnostic."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting.fit_convergence import (
    execute_convergence,
    prepare_convergence,
    report_convergence,
)
from autoformalism.schemas.fit_convergence import ConvergenceSelection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "inspect", "run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--parent", type=Path, required=True)
            command.add_argument("--continuation", type=Path, required=True)
            command.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_convergence(
            args.parent,
            args.continuation,
            ConvergenceSelection.model_validate_json(args.selection.read_text()),
            args.output,
        )
    elif args.command == "run":
        result = execute_convergence(args.output)
    else:
        result = report_convergence(args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
