#!/usr/bin/env python3
"""Summarize the public-only incremental-construction pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.incremental_construction_experiment import (
    summarize_incremental_construction_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_incremental_construction_experiment(
        args.plan,
        args.task_plan,
        args.result_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
