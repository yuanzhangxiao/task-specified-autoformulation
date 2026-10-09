#!/usr/bin/env python3
"""Exercise both M22 strategies on an independent small nonlinear control."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import nonlinear_comparison as fitting
from autoformalism.fitting.confidence_checks import TrainingProblem
from tests.profiled_fixture import TRUTH, make_base


def run(root: Path):
    base, _ = make_base()
    problem = TrainingProblem.model_validate(
        {k: v for k, v in base.items() if k != "nodes"} | {"incumbent": base["start"]}
    )
    policy = fitting.ComparisonPolicy(
        warm_seconds=30,
        warm_calls=150,
        portfolio_size=1,
        trial_seconds=15,
        trial_calls=75,
        continuation_seconds=30,
        continuation_calls=150,
        check_seconds=15,
        sensitivity_seconds=15,
        target_nmse=1e-10,
        trajectory_nmse=1e-9,
        node_targets=(150, 300),
        penalties=(100, 10000),
        minimum_intervals=4,
        observation_anchors=2,
        cycles=2,
        block_iterations=8,
    )
    rows = []
    for method in fitting.METHODS:
        result = fitting.run(problem, policy, method, root / method)
        assert result == fitting.run(problem, policy, method, root / method)
        assert result["routing"]["selected"] == "terminal_output_profile"
        assert result["selected"]["training_nmse"] < 1e-8
        assert result["selected"]["maximum_trajectory_nmse"] < 1e-7
        # Truth is used only for this post-fit smoke assertion.
        error = max(abs(result["retained_parameters"][k] - v) for k, v in TRUTH.items())
        assert error < 0.01
        rows.append(
            {
                "method": method,
                "training_nmse": result["selected"]["training_nmse"],
                "maximum_parameter_error": error,
                "wall_seconds": result["additional_wall_seconds"],
                "resume_identical": True,
            }
        )
    return {"status": "passed", "rows": rows, "benchmark_fitting_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().root), indent=2))
