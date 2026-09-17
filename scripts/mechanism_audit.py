#!/usr/bin/env python3
"""Snapshot saved models, compute graph metrics, and review public equations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.mechanism_audit_campaign import prepare, report, run
from autoformalism.rebuttal.mechanism_audit_sources import export_bundle


def main() -> None:
    """Only the explicit judge command may contact a local model endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--baseline-root", type=Path)
    export.add_argument("--d3-root", type=Path)
    export.add_argument("--review-root", type=Path)
    export.add_argument("--round", type=int)
    export.add_argument(
        "--arms",
        nargs="+",
        default=["full"],
        choices=["full", "brief_only", "refit_only", "no_spec", "no_latent"],
    )
    export.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("prepare")
    freeze.add_argument("--bundle", type=Path, action="append", required=True)
    freeze.add_argument("--public-root", type=Path, required=True)
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--model", default="openai/gpt-oss-120b")
    freeze.add_argument("--model-revision", required=True)
    judge = sub.add_parser("judge")
    judge.add_argument("--root", type=Path, required=True)
    judge.add_argument("--base-url", required=True)
    judge.add_argument("--wall-seconds", type=float, default=18000)
    summary = sub.add_parser("report")
    summary.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        result = export_bundle(
            args.output,
            baseline_root=args.baseline_root,
            d3_root=args.d3_root,
            review_root=args.review_root,
            round_index=args.round,
            arms=tuple(args.arms),
        )
    elif args.command == "prepare":
        result = prepare(
            args.bundle,
            args.public_root,
            args.root,
            model_revision=args.model_revision,
            model=args.model,
        )
    elif args.command == "judge":
        result = run(args.root, args.base_url, wall_seconds=args.wall_seconds)
    else:
        result = report(args.root)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"rows", "review_prompt", "settings"}
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
