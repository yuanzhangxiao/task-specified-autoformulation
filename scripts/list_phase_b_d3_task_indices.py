#!/usr/bin/env python3
"""Print D3 campaign task indices, optionally restricted to one tier.

`prepare()` builds rows as `for cell in cells: for repetition in repetitions`,
so an index is derivable from the frozen config alone and a batch can be
selected before the plan exists.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def task_indices(
    config: dict, tier: str, match: str | None = None
) -> tuple[int, ...]:
    """Return campaign indices in plan order, filtered by tier and name.

    `match` is a regular expression against benchmark_id, so a campaign can be
    run in priority order -- the cells a paper leads with first -- instead of
    all or nothing. Selecting a subset changes coverage, not the plan: the
    indices are the same ones the full roster uses, so a later run fills in
    the rest without recomputing anything.
    """
    pattern = re.compile(match) if match else None
    repetitions = list(config["repetitions"])
    selected: list[int] = []
    for position, cell in enumerate(config["cells"]):
        if tier not in {"all", str(cell["tier"])}:
            continue
        if pattern is not None and not pattern.search(str(cell["benchmark_id"])):
            continue
        base = position * len(repetitions)
        selected.extend(base + offset for offset in range(len(repetitions)))
    return tuple(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tier", default="all", choices=("all", "easy", "hard"))
    parser.add_argument(
        "--match",
        help="regular expression on benchmark_id, to run a subset in priority order",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(
        ",".join(str(item) for item in task_indices(config, args.tier, args.match))
    )


if __name__ == "__main__":
    main()
