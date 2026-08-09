"""Validate and summarize a versioned benchmark-suite design."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoformalism.benchmarks import load_suite_spec


def main() -> None:
    """Validate the suite contract and print a compact planning summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/benchmarks/phase_b_suite_v1.json"),
    )
    args = parser.parse_args()
    suite = load_suite_spec(args.spec)
    print(f"suite={suite.suite_version} status={suite.status}")
    print(f"full_factorial_cells={suite.number_of_cells}")
    for family in suite.families:
        print(
            f"{family.family}: cells={family.number_of_cells} "
            f"tasks={len(family.tasks)} dynamics={len(family.dynamics_conditions)} "
            f"semantics={len(family.semantic_variants)} tiers={len(family.tiers)}"
        )


if __name__ == "__main__":
    main()
