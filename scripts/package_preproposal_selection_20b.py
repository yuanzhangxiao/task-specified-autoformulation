"""Package the committed screened-selection worker and verified train-only cases."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import preproposal_graph_portfolio as portfolio
from scripts.package_t2_breadth import package_committed
from scripts.preproposal_selection_20b import ENTRYPOINTS, input_payload
from scripts.t2_breadth import REPO


def package(
    destination: Path,
    case: Path,
    harmony: Path,
    *,
    matched: bool = False,
    complete_graphs: bool = False,
) -> Path:
    """Archive the committed experiment and its verified public inputs."""
    payload = input_payload(case)
    tracked = ["src", "pyproject.toml", *ENTRYPOINTS]
    if complete_graphs:
        payload["variable_exits"] = portfolio.read_variable_exits(
            REPO, payload["frozen_cases"]
        )
        tracked.extend([portfolio.LAUNCHER, "scripts/preproposal_graph_exports.py"])
    if matched or complete_graphs:
        tracked.extend(
            [
                "scripts/hpc/start_preproposal_matched_aces.sh",
                "docs/PHASE_C_PREPROPOSAL_PILOT.md",
            ]
        )
    with TemporaryDirectory() as folder:
        source = Path(folder) / "preproposal.json"
        source.write_text(json.dumps(payload, ensure_ascii=False))
        return package_committed(
            destination,
            portfolio.NAME
            if complete_graphs
            else "phase-c-preproposal-matched-20b-v2"
            if matched
            else "phase-c-preproposal-selection-20b-v1",
            tracked,
            "inputs/preproposal.json",
            source,
            harmony,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matched", action="store_true", help="Build the matched selection experiment."
    )
    parser.add_argument("--complete-graphs", action="store_true")
    for name in ("case", "out", "harmony"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    print(
        package(
            args.out,
            args.case,
            args.harmony,
            matched=args.matched,
            complete_graphs=args.complete_graphs,
        )
    )
