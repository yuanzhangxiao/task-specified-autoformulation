#!/usr/bin/env python3
"""Prepare/inspect/run/report one explicitly selected public fit continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.fitting.fit_continuation import (
    execute_continuation,
    inspect_continuation,
    prepare_continuation,
)
from autoformalism.schemas.fit_continuation import ContinuationSelection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "inspect", "run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--parent", type=Path, required=True)
            command.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_continuation(
            args.parent,
            ContinuationSelection.model_validate_json(args.selection.read_text()),
            args.output,
        )
    elif args.command == "inspect":
        result = inspect_continuation(args.output)
    elif args.command == "run":
        result = execute_continuation(args.output).model_dump(mode="json")
    else:
        inspection = inspect_continuation(args.output)
        path = args.output / "result.json"
        if not path.exists():
            result = {**inspection, "status": "pending_or_interrupted"}
        else:
            # A sealed terminal result is read/verified without new numerical work.
            terminal = execute_continuation(args.output).model_dump(mode="json")
            frozen = json.loads((args.output / "freeze.json").read_text())
            parent = frozen["parent"]["result"]["result"]
            result = {
                **inspection,
                "result": terminal,
                "parent_training": parent["training"],
                "parent_validation": parent["validation"],
            }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
