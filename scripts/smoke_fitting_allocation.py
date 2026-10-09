#!/usr/bin/env python3
"""M23 small nonlinear numerical smoke, including a real conditional restart."""

import argparse
import json
from pathlib import Path
from time import monotonic

from autoformalism.fitting import nonlinear_comparison as fitting
from autoformalism.fitting import nonlinear_rollout
from autoformalism.fitting.fitting_allocation import MeasuredConditionalEngine
from tests.profiled_fixture import TRUTH
from tests.test_trajectory_profile import problem as fixture


def run(root: Path) -> dict:
    problem = fixture()
    policy = fitting.ComparisonPolicy(
        allocation="measured",
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
        minimum_node_seconds=0.1,
        node_targets=(150, 300),
        minimum_intervals=4,
        observation_anchors=2,
        cycles=2,
        block_iterations=8,
    )
    rows = []
    for method in fitting.METHODS:
        result = fitting.run(problem, policy, method, root / method)
        assert result == fitting.run(problem, policy, method, root / method)
        assert not any(
            o["operation"] == "warm/conditional" for o in result["operations"]
        )
        assert result["routing"]["selected"] == "terminal_output_profile"
        loss = result["selected"]["training_nmse"]
        assert loss < 1e-8
        error = max(abs(result["retained_parameters"][k] - v) for k, v in TRUTH.items())
        assert error < 0.01
        rows.append(
            {
                "method": method,
                "training_nmse": loss,
                "maximum_parameter_error": error,
                "resume_identical": True,
            }
        )
    oracle, _, _ = nonlinear_rollout.build(problem)
    control = nonlinear_rollout.verify(
        oracle, problem.start, monotonic() + 15, lambda _: None, timing=True
    )
    engine = MeasuredConditionalEngine(oracle, problem, policy)
    engine.screen_cost = control["integrator_seconds"]["DOP853"]
    restart = engine.propose(
        problem.start, monotonic() + 15, lambda _: None, root / "nodes"
    )
    assert restart["selected"] and restart["calls"] >= 1
    assert not restart["control_rollout_repeated"]
    assert not restart["incumbent_replacement_authorized"]
    return {
        "status": "passed",
        "rows": rows,
        "conditional_screened": len(restart["candidates"]),
        "benchmark_fitting_performed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().root), indent=2))
