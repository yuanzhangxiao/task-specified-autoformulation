#!/usr/bin/env python3
"""Analytical CPU handoff control, not a new benchmark recovery experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import (
    content_sha256,
    execute_fit,
    inspect_fit,
    prepare_fit,
)
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


def control(profile: str, *, multiple_targets: bool = False):
    """Two coupled states; latent initial value follows the initial output."""
    targets = ("v01", "v02") if multiple_targets else ("v01",)
    candidate = {
        "candidate_id": "public_fit_smoke",
        "parent_candidate_id": None,
        "states": [{"name": name, "kind": "latent"} for name in ("x", "z")],
        "state_equations": [
            {"state": "x", "rhs": "-a*x + z"},
            {"state": "z", "rhs": "-b*z"},
        ],
        "observation_mappings": [{"channel": "v01", "expression": "x"}],
        "parameters": [
            {"name": name, "scope": "global", "role": "rate"} for name in ("a", "b")
        ],
        "initial_conditions": [
            {"state": name, "scope": "global", "fixed_value": 0} for name in ("x", "z")
        ],
    }
    if multiple_targets:
        candidate["observation_mappings"].append(
            {"channel": "v02", "expression": "2*x"}
        )
    request = PublicFitRequest.model_validate(
        {
            "base_candidate": candidate,
            "context": {"targets": targets},
            "initialization_plan": {
                "rules": {
                    "z": {
                        "initial": {
                            "mode": "map",
                            "expression": "scale*v01",
                            "parameters": [
                                {"name": "scale", "role": "coefficient", "guess": 0.5}
                            ],
                        }
                    }
                }
            },
            "parameter_guesses": {"a": 0.7, "b": 1.2},
            "profile": profile,
            "source": {
                "stage": "synthetic_control",
                "task_id": "analytical-handoff",
                "artifact_sha256": content_sha256(candidate),
            },
        }
    )
    time = np.linspace(0, 2, 21)

    def split(name, initials):
        rows = []
        for index, initial in enumerate(initials):
            x = (
                initial * np.exp(-0.7 * time)
                + 0.5 * initial * (np.exp(-0.7 * time) - np.exp(-1.2 * time)) / 0.5
            )
            rows.append(
                {
                    "trajectory_id": f"{name}_{index}",
                    "time": time.tolist(),
                    "targets": {
                        key: (x if key == "v01" else 2 * x).tolist() for key in targets
                    },
                }
            )
        return PublicSplit.model_validate(
            {"name": name, "fingerprint": name, "rows": rows}
        )

    return request, split("train", (1.0, 1.7)), split("val", (0.6,))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("general-rollout-v1", "collocation-feasible-v1"),
        default="general-rollout-v1",
    )
    args = parser.parse_args()
    request, train, val = control(args.profile)
    prepare_fit(request, train, val, args.output)
    inspection = inspect_fit(args.output)
    if not inspection["capability_supported"]:
        raise RuntimeError(inspection["capability_message"])
    result = execute_fit(args.output)
    before = (args.output / "result.json").read_bytes()
    assert execute_fit(args.output) == result
    assert (args.output / "result.json").read_bytes() == before
    assert result.status == "complete", result.message
    assert result.training.normalized_mse < 1e-8
    assert result.validation.normalized_mse < 1e-8
    print(
        json.dumps(
            {
                "status": "passed",
                "identity": result.identity,
                "profile": result.profile,
                "resume_unchanged": True,
                "training_nmse": result.training.normalized_mse,
                "validation_nmse": result.validation.normalized_mse,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
