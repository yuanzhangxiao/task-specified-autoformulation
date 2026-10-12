#!/usr/bin/env python3
"""Small analytic multi-output/initializer check of the opt-in public bridge."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting import public_rollout as bridge
from autoformalism.fitting import verified_replay
from scripts.smoke_public_fitting import control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request, train, val = control("general-rollout-v1", multiple_targets=True)
    inputs = bridge.Inputs(request=request, training=train.model_dump(mode="json"))
    policy = bridge.Policy()
    result = bridge.run(inputs, policy, args.output / "fit")
    assert result.fit and result.fit.verification.target_reached, result
    assert result.fit.parameters is not None
    assert bridge.run(inputs, policy, args.output / "fit") == result
    evaluation = verified_replay.run(
        request,
        result.fit.parameters,
        train,
        val,
        verified_replay.ReplayPolicy(initial_seconds=30, retry_seconds=30),
        args.output / "evaluation",
    )
    assert evaluation["solver_agreement_passed"], evaluation
    assert max(evaluation["metrics"].values()) < 1e-10
    print(
        json.dumps(
            {
                "status": "passed",
                "resume_unchanged": True,
                "targets": 2,
                "shared_initializer": "init_z_scale",
                "metrics": evaluation["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
