"""Tabulate the nine-case baselines' validation NMSE.

    python scripts/phase_b_nine_case_table.py \
        --run "D3 as published=RUN/summary.json" \
        --run "HDTwinGen=RUN/summary.json" \
        --run "Spec-informed SINDy=RUN/summary.json" \
        --literature LITERATURE/summary.json --json OUT.json

Prints the Markdown table; see phase_b_nine_case_table for the rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.phase_b_nine_case_table import load, markdown, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--literature", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        name, _, path = item.partition("=")
        runs[name] = load(Path(path))
    result = table(runs, load(args.literature) if args.literature else None)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, allow_nan=True))
    print(markdown(result), end="")


if __name__ == "__main__":
    main()
