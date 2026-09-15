#!/usr/bin/env python3
"""Export, run, or inspect one preselected public repaired-candidate fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.prefit_fit_handoff import (
    HandoffSelection,
    prepare_handoff,
    run_handoff,
    summarize_handoff,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--construction-root", type=Path, required=True)
    prepare.add_argument("--public-root", type=Path, required=True)
    prepare.add_argument("--selection", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("run", "summary"):
        commands.add_parser(name).add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        selection = HandoffSelection.model_validate_json(args.selection.read_text())
        report = prepare_handoff(
            args.source,
            args.construction_root,
            args.public_root,
            selection,
            args.output,
        )
    elif args.command == "run":
        report = run_handoff(args.output)
    else:
        report = summarize_handoff(args.output)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
