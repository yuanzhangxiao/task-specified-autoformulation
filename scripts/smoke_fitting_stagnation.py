#!/usr/bin/env python3
"""Exercise M25 on a tiny nonlinear ODE, never fit the research benchmark locally."""

import argparse
import json
from pathlib import Path
from time import monotonic

from autoformalism.fitting import nonlinear_comparison as fitting
from autoformalism.fitting import nonlinear_rollout as rollout
from autoformalism.fitting import stagnation_campaign as campaign
from autoformalism.fitting.stagnation_audit import AuditPolicy, audit
from tests.profiled_fixture import TRUTH
from tests.test_trajectory_profile import problem as fixture


def run(root: Path):
    problem = fixture()
    oracle, profile, _ = rollout.build(problem)
    diagnostic = audit(
        oracle,
        profile,
        problem.incumbent,
        AuditPolicy(),
        monotonic() + 30,
        lambda _: None,
    )
    assert diagnostic["status"] == "passed"
    policy = campaign.Policy(
        continuation_seconds=30,
        continuation_calls=120,
        check_seconds=15,
        sensitivity_seconds=15,
        target_nmse=1e-10,
        trajectory_nmse=1e-9,
    )
    rows = []
    for method in campaign.METHODS:
        kw = {
            "shared_warm": problem.incumbent,
            "continuation_order": "incumbent_first",
            "optimizer_scaling": method,
        }
        result = fitting.run(problem, policy, "best_rollout", root / method, **kw)
        assert result == fitting.run(
            problem, policy, "best_rollout", root / method, **kw
        )
        assert result["selected"]["training_nmse"] < 1e-8
        op = next(
            o
            for o in result["operations"]
            if o["operation"] == "incumbent-continuation/rollout"
        )
        assert op["value"]["accepted_steps"]
        rows.append(
            {
                "scaling": method,
                "training_nmse": result["selected"]["training_nmse"],
                "max_parameter_error": max(
                    abs(result["retained_parameters"][k] - v) for k, v in TRUTH.items()
                ),
                "calls": result["search_calls_completed"],
                "accepted_steps": len(op["value"]["accepted_steps"]),
                "resume_identical": True,
            }
        )
    return {
        "status": "passed",
        "audit_status": diagnostic["status"],
        "audit_calls": diagnostic["calls"],
        "rows": rows,
        "benchmark_fitting_performed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().root), indent=2))
