#!/usr/bin/env python3
"""Prepare, inspect or execute an explicit public CPU fitting handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.fitting.public_fitting import execute_fit, inspect_fit, prepare_fit
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--request", type=Path, required=True)
    prepare.add_argument("--training", type=Path, required=True)
    prepare.add_argument("--validation", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("inspect", "run"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_fit(
            PublicFitRequest.model_validate_json(args.request.read_text()),
            PublicSplit.model_validate_json(args.training.read_text()),
            PublicSplit.model_validate_json(args.validation.read_text()),
            args.output,
        )
    elif args.command == "inspect":
        result = inspect_fit(args.output)
    else:
        result = execute_fit(args.output).model_dump(mode="json")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
