#!/usr/bin/env python3
"""Exercise profiled and joint M27 kernels on small standalone synthetic controls."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import nonlinear_rollout, recovery_numerics, scaling_recovery
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.coordinates import training_coordinates
from autoformalism.schemas.public_fitting import PublicSplit
from scripts.smoke_fitting_scaling import run as profiled_smoke


def run(root: Path) -> dict:
    profiled = profiled_smoke(root / "profiled")
    case = controls.make_inputs()["cases"]["nonlinear"]
    request = controls.request("nonlinear", 0)
    model, start, _ = public._lower(request)
    problem = checks.TrainingProblem(
        request=request,
        training=case["training"],
        start=start,
        incumbent=start,
        coordinates=training_coordinates(
            model,
            public.unpack_split(PublicSplit.model_validate(case["training"])),
            start,
        ),
    )
    oracle, profile, route = nonlinear_rollout.build(problem)
    assert profile is None and route["selected"] == "joint_rollout"
    starts = [start, *recovery_numerics.diverse_starts(oracle, problem, 3, 270)]
    policy = scaling_recovery.Policy(
        block_seconds=30,
        block_calls=60,
        check_seconds=15,
        sensitivity_seconds=15,
        target_nmse=1e-10,
        trajectory_nmse=1e-9,
    )
    rows = []
    for method in scaling_recovery.METHODS:
        folder = root / "joint" / method
        result = scaling_recovery.run(problem, starts, policy, method, folder)
        assert result == scaling_recovery.run(problem, starts, policy, method, folder)
        assert result["accuracy_verified"] and result["cost_complete"]
        rows.append(
            {
                "method": method,
                "route": result["routing"]["selected"],
                "training_nmse": result["selected"]["training_nmse"],
                "calls": result["search_calls_completed"],
                "resume_identical": True,
            }
        )
    return {
        "status": "passed",
        "profiled": profiled,
        "joint": rows,
        "benchmark_fitting_performed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().root), indent=2))
