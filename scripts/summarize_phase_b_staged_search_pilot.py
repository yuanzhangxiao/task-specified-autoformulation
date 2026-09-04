#!/usr/bin/env python3
"""Summarize the public-only feedback-routed staged-search pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.staged_search_pilot import (
    summarize_staged_search_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_staged_search_pilot(
        args.plan,
        args.task_plan,
        args.search_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
