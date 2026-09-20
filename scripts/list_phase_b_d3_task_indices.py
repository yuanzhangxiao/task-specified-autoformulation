#!/usr/bin/env python3
"""Print D3 campaign task indices, optionally restricted to one tier.

`prepare()` builds rows as `for cell in cells: for repetition in repetitions`,
so an index is derivable from the frozen config alone and a batch can be
selected before the plan exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def task_indices(config: dict, tier: str) -> tuple[int, ...]:
    """Return campaign indices in plan order, filtered by tier."""
    repetitions = list(config["repetitions"])
    selected: list[int] = []
    for position, cell in enumerate(config["cells"]):
        if tier not in {"all", str(cell["tier"])}:
            continue
        base = position * len(repetitions)
        selected.extend(base + offset for offset in range(len(repetitions)))
    return tuple(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tier", default="all", choices=("all", "easy", "hard"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(",".join(str(item) for item in task_indices(config, args.tier)))


if __name__ == "__main__":
    main()
