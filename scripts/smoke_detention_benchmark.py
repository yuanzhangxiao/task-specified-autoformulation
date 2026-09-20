#!/usr/bin/env python3
"""One real frozen-fitter recovery and immutable resume of the basin benchmark."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting.public_fitting import _jsonable
from autoformalism.rebuttal.detention_benchmark import REPO, prepare, report, run_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = prepare(args.output, REPO / "configs/detention_benchmark_v1.json")
    assert plan["generation_gate_passed"]
    result = run_task(args.output, 0)
    assert result["fit"]["status"] == "complete", result["fit"]["message"]
    assert result["diagnostic"]["strict_output_recovery"], result["diagnostic"]
    path = args.output / "results/coupled_noise0_start0/assessment.json"
    before = path.read_bytes()
    assert run_task(args.output, 0) == result
    assert path.read_bytes() == before
    report(args.output)
    print(
        json.dumps(
            _jsonable(
                {
                    "status": "passed",
                    "resume_unchanged": True,
                    "diagnostic": result["diagnostic"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
