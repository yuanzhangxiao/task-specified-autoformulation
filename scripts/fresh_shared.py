#!/usr/bin/env python3
"""Run final pruning or inspect the fresh shared/multi-output campaign."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal import fresh_shared as campaign
from autoformalism.rebuttal import review_deadline_io as io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prune-task", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    )
    args = parser.parse_args()
    if args.command == "prune-task":
        with io.execution_lease(args.root):
            value = campaign.run_one(args.root, args.task_index)
        value = {k: value[k] for k in ("status", "task", "identity")}
    else:
        value = campaign.report(args.root)
        value = {k: v for k, v in value.items() if k != "rows"}
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
