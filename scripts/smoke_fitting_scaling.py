#!/usr/bin/env python3
"""Check all M26 scaling arms on a tiny ODE, without benchmark fitting."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import nonlinear_rollout as rollout
from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting import scaling_recovery as fitting
from tests.profiled_fixture import TRUTH
from tests.test_trajectory_profile import problem as fixture


def run(root: Path):
    problem = fixture()
    oracle, _, _ = rollout.build(problem)
    starts = [problem.start, *numerical.diverse_starts(oracle, problem, 3, 261)]
    policy = fitting.Policy(
        block_seconds=30,
        block_calls=60,
        check_seconds=15,
        sensitivity_seconds=15,
        target_nmse=1e-10,
        trajectory_nmse=1e-9,
    )
    rows = []
    for method in fitting.METHODS:
        result = fitting.run(problem, starts, policy, method, root / method)
        assert result == fitting.run(problem, starts, policy, method, root / method)
        assert result["accuracy_verified"]
        assert result["cost_complete"]
        assert (
            result["search_calls_started"] <= policy.block_calls * policy.maximum_blocks
        )
        rows.append(
            {
                "scaling": method,
                "training_nmse": result["selected"]["training_nmse"],
                "maximum_parameter_error": max(
                    abs(result["retained_parameters"][k] - v) for k, v in TRUTH.items()
                ),
                "blocks": result["blocks_used"],
                "calls": result["search_calls_completed"],
                "resume_identical": True,
            }
        )
    return {"status": "passed", "rows": rows, "benchmark_fitting_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().root), indent=2))
