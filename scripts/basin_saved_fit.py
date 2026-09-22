#!/usr/bin/env python3
"""Freeze, fit or report eligible historical basin repairs without LLM calls."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal import basin_saved_fit as io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "verify", "fit", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument(
        "--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.source is None or args.gate is None:
            parser.error("freeze requires --source and --gate")
        value = io.freeze(args.source, args.gate, args.root)
    elif args.stage == "fit":
        value = io.fit_task(args.root, args.index)
    elif args.stage == "report":
        value = io.report(args.root)
    else:
        value = io.verify(args.root)
    print(
        json.dumps(
            {
                k: v
                for k, v in value.items()
                if k
                in {
                    "protocol",
                    "artifact_sha256",
                    "status",
                    "status_counts",
                    "selection_counts",
                    "maximum_new_fits",
                    "new_fit_results",
                    "classification_counts",
                    "llm_calls",
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
