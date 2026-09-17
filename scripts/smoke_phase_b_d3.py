#!/usr/bin/env python3
"""Synthetic native increment parity smoke; no API calls or benchmark data."""

from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np

from autoformalism.baselines.d3_native import fit_native_d3
from autoformalism.baselines.d3_rollout import evaluate_validation
from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.schemas import CandidateModel


def main() -> None:
    """Exercise the actual Torch fitter when installed, before live discovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    model = CandidateModel.model_validate(
        {
            "candidate_id": "native_smoke",
            "parent_candidate_id": None,
            "states": [{"name": "x", "kind": "observed"}],
            "state_equations": [{"state": "x", "rhs": "k * x"}],
            "observation_mappings": [{"channel": "x", "expression": "x"}],
            "parameters": [
                {
                    "name": "k",
                    "scope": "global",
                    "bounds": {"lower": -1, "upper": 0},
                    "initialization_range": {"lower": -0.6, "upper": -0.4},
                }
            ],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "x"}
            ],
        }
    )

    def trajectory(initial: float) -> Trajectory:
        time = np.arange(6) * 0.1  # a dt multiplier would fail this smoke
        return Trajectory(
            trajectory_id=str(initial),
            time=time,
            targets={"x": initial * 0.5 ** np.arange(6)},
            auxiliaries={},
            external_inputs={},
            fixed_covariates={},
            derivatives={},
        )

    train = DatasetSplit(SplitName.TRAIN, (trajectory(1),), "training")
    val = DatasetSplit(SplitName.VALIDATION, (trajectory(2),), "validation")
    available = importlib.util.find_spec("torch") is not None
    if args.require_torch and not available:
        raise RuntimeError("Torch is required before any live D3 calls")
    if available:
        fitted = fit_native_d3(model, train, val, targets=("x",), seed=0, epochs=20)
        parameters, saved = fitted.parameters, fitted.validation_mse
    else:
        parameters, saved = {"k": -0.5}, 0.0
    result = evaluate_validation(
        model, parameters, train, val, saved_one_step_nmse=saved
    )
    assert result["saved_one_step_matches"]
    assert result["phase_b_rollout"]["normalized_mse"] < 1e-12
    print(
        json.dumps(
            {
                "status": "pass",
                "native_torch_fit_checked": available,
                "one_step_matches": result["saved_one_step_matches"],
                "recursive_validation_nmse": result["phase_b_rollout"][
                    "normalized_mse"
                ],
                "live_llm_calls": 0,
                "test_data_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
