#!/usr/bin/env python3
"""Small real nonlinear rollout/profile/verification test; no benchmark fitting."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import incumbent_campaign as campaign
from autoformalism.fitting import nonlinear_comparison as fitting
from tests.profiled_fixture import TRUTH
from tests.test_trajectory_profile import problem as fixture


def run(root: Path):
    problem = fixture()
    policy = campaign.Policy(
        continuation_seconds=30,
        continuation_calls=150,
        trial_seconds=10,
        trial_calls=50,
        check_seconds=15,
        sensitivity_seconds=15,
        target_nmse=1e-10,
        trajectory_nmse=1e-9,
    )
    rows = []
    for method in campaign.METHODS:
        arguments = {"shared_warm": problem.incumbent, "continuation_order": method}
        result = fitting.run(
            problem, policy, "best_rollout", root / method, **arguments
        )
        assert result == fitting.run(
            problem, policy, "best_rollout", root / method, **arguments
        )
        assert result["routing"]["selected"] == "terminal_output_profile"
        assert result["after_warm"]["parameters"] == problem.incumbent
        assert not any(o["operation"] == "warm/rollout" for o in result["operations"])
        assert (
            result["selected"]["training_nmse"] <= result["after_warm"]["training_nmse"]
        )
        if method == "incumbent_first":
            assert result["selected"]["training_nmse"] < 1e-8
            assert result["trial_count"] == 0
        rows.append(
            {
                "method": method,
                "training_nmse": result["selected"]["training_nmse"],
                "maximum_parameter_error": max(
                    abs(result["retained_parameters"][k] - v) for k, v in TRUTH.items()
                ),
                "calls_started": result["search_calls_started"],
                "calls_completed": result["search_calls_completed"],
                "resume_identical": True,
            }
        )
    return {"status": "passed", "rows": rows, "benchmark_fitting_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().root), indent=2))
