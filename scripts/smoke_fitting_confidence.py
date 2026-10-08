#!/usr/bin/env python3
"""Bounded real M19 smoke: numerical checks, nuisance fitting and exact resume."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import confidence_campaign as campaign
from autoformalism.fitting import confidence_checks, confidence_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    data = read_seal(args.inputs)
    endpoint = next(
        e
        for e in data["endpoints"]
        if e["task_id"] == "linear3_s0_coupled_profiled_rollout"
    )
    problem = campaign.training_problem(data, endpoint)
    policy = confidence_checks.ConfidencePolicy(
        restart_seconds=15,
        restart_calls=12,
        profile_seconds=6,
        profile_calls=6,
        check_seconds=15,
    )
    first = confidence_fit.run(problem, policy, args.root)
    assert first["status"] == "complete" and first["selected"]["training_nmse"] < 1e-8
    assert first["before"]["sensitivity"]["parameter_count"] == 6
    assert len(first["profiles"]) == 24
    assert confidence_fit.run(problem, policy, args.root) == first
    print(
        json.dumps(
            {
                "execution_passed": True,
                "confidence": first["confidence"],
                "additional_wall_seconds": first["additional_wall_seconds"],
                "limitation": "Smoke does not qualify recovery or confidence.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
