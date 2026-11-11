#!/usr/bin/env python3
"""Freeze a Phase C classical plan on a Jetstream2 VM, then run its tasks.

`prepare` freezes the plan against the release and prints the task count and
the CPUs each task declares. `run` runs one task the way Delta's worker would,
unless that task has already recorded an outcome. Neither contacts a model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from autoformalism.rebuttal import phase_c_classical_vm as vm


def main() -> int:
    """Dispatch one subcommand and return its exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--release", type=Path, required=True)
    prepare.add_argument("--root", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--release", type=Path, required=True)
    run.add_argument("--julia-depot", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        summary = vm.prepare(args.config, args.release, args.root)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    return vm.run(
        args.root,
        args.index,
        release=args.release,
        julia_depot=args.julia_depot,
        base=os.environ,
    )


if __name__ == "__main__":
    sys.exit(main())
