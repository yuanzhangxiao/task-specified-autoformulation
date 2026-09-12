#!/usr/bin/env python3
"""Freeze, resume, or summarize the matched pre-fit judge comparison."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.repair_comparison import freeze, run_pass, summarize, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--source-rescue-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--judge-revision", required=True)
    p = sub.add_parser("run")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--role", choices=("proposer", "judge"), required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--wall-seconds", type=float, default=14400)
    for command in ("summary", "verify"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.source_rescue_root, args.output, args.judge_revision)
        result = {"plan_sha256": result["plan_sha256"], "tasks": len(result["tasks"])}
    elif args.command == "run":
        result = run_pass(args.root, args.role, args.base_url, args.wall_seconds)
    elif args.command == "verify":
        result = {"verified": True, "plan_sha256": verify(args.root)["plan_sha256"]}
    else:
        result = summarize(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
