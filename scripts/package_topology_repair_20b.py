"""Package one verified topology case and the committed ACES worker."""

import argparse
import json
from pathlib import Path

from autoformalism.research import topology_repair_trial as trial
from scripts.package_t2_breadth import package_committed
from scripts.topology_repair_20b import ENTRYPOINTS


def package(destination: Path, case: Path, harmony: Path) -> Path:
    trial.verify_case(json.loads(case.read_text()))
    return package_committed(
        destination,
        "phase-c-topology-repair-20b-v1",
        ["src", "pyproject.toml", *ENTRYPOINTS],
        "inputs/case.json",
        case,
        harmony,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "out", "harmony"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    print(package(args.out, args.case, args.harmony))
