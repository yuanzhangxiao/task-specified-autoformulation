"""Failed trials and fitter handoffs must retain useful physical progress."""

from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import least_squares

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import sensitivity_probe
from autoformalism.fitting.collocation_sensitivity import CollocationSensitivityConfig
from autoformalism.fitting.feasibility import (
    EvaluationBudget,
    GuardedOracle,
    SensitivityUnavailable,
    handoff_starts,
)
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.schemas import CandidateModel


def test_handoff_uses_augmented_best_before_older_screens():
    oracles = [
        SimpleNamespace(
            directory=Path("primal"), best={"parameters": {"a": 1}, "cost": 10}
        ),
        SimpleNamespace(
            directory=Path("augmented"), best={"parameters": {"a": 2}, "cost": 3}
        ),
    ]
    screened = [{"source": "ordinary", "parameters": {"a": 1}, "cost": 10}]
    result = handoff_starts(oracles, screened, {"a": 0})
    assert [r["parameters"]["a"] for r in result] == [2, 1]
    assert result[0]["source"] == "augmented"


def test_trf_rejects_failed_trial_and_recovers_without_zero_jacobian(
    tmp_path, monkeypatch
):
    model = compile_candidate(
        CandidateModel.model_validate(
            {
                "candidate_id": "trial_control",
                "parent_candidate_id": None,
                "states": [{"name": "x", "kind": "latent"}],
                "state_equations": [{"state": "x", "rhs": "a"}],
                "observation_mappings": [{"channel": "v01", "expression": "x"}],
                "parameters": [{"name": "a", "scope": "global", "role": "rate"}],
                "initial_conditions": [
                    {"state": "x", "scope": "global", "fixed_value": 0}
                ],
            }
        ),
        ValidationContext(targets=("v01",)),
    )
    row = Trajectory(
        "train", np.array([0.0, 1.0]), {"v01": np.full(2, 0.25)}, {}, {}, {}, {}
    )
    training = DatasetSplit(SplitName.TRAIN, (row,), "trial")
    attempted = []

    def response(system, trajectory, theta, *args, **kwargs):
        a = theta[0]
        attempted.append(a)
        if a > 0.505:
            raise ValueError("unavailable trial rollout")
        return (
            np.full((2, 1), a * a),
            np.full((2, 1, 1), 2 * a),
            np.zeros((2, 1)),
            {"nfev": 1, "njev": 0, "nlu": 0, "segments": 1},
        )

    monkeypatch.setattr(sensitivity_probe, "symbolic_rollout", response)
    budget = EvaluationBudget(monotonic() + 10, 50)
    oracle = GuardedOracle(
        SymbolicODE(model),
        training,
        1.0,
        CollocationSensitivityConfig().fit_config(),
        tmp_path,
        budget.deadline,
        sensitivities=True,
        budget=budget,
        point_seconds=5,
        reject_invalid_trials=True,
    )
    result = least_squares(
        oracle, [0.4], jac=oracle.jacobian, bounds=(oracle.lower, oracle.upper)
    )
    assert result.x[0] == pytest.approx(0.5, abs=1e-7)
    assert max(attempted) > 0.505
    assert oracle.rejected_trials > 0
    assert oracle.best["cost"] < 1e-14
    # Even after prior valid points, requesting the failed point's Jacobian fails.
    with pytest.raises(SensitivityUnavailable):
        oracle.jacobian(np.array([0.6]))
    assert oracle.last_jac is None


def test_invalid_first_point_remains_explicit_even_with_rejection(
    tmp_path, monkeypatch
):
    from autoformalism.fitting.initialization import (
        LatentInitializationPlan,
        apply_initialization_plan,
    )
    from autoformalism.rebuttal.initialization_campaign import synthetic_problem
    from autoformalism.rebuttal.piecewise_campaign import unpack_split

    raw, _ = synthetic_problem("shared", 0.0, 0)
    model = compile_candidate(
        CandidateModel.model_validate(raw["candidate"]),
        ValidationContext.model_validate(raw["context"]),
    )
    model, guesses, _ = apply_initialization_plan(
        model, LatentInitializationPlan.model_validate(raw["initialization_plan"])
    )
    starts = {**raw["start"], **guesses}
    budget = EvaluationBudget(monotonic() + 10, 3)
    oracle = GuardedOracle(
        SymbolicODE(model),
        unpack_split(raw["splits"]["train"]),
        1.0,
        CollocationSensitivityConfig().fit_config(),
        tmp_path,
        budget.deadline,
        sensitivities=True,
        budget=budget,
        point_seconds=5,
        reject_invalid_trials=True,
    )

    def failed(*args, **kwargs):
        raise ValueError("invalid first point")

    monkeypatch.setattr(sensitivity_probe, "symbolic_rollout", failed)
    with pytest.raises(SensitivityUnavailable):
        oracle(oracle.vector(starts))
    assert oracle.rejected_trials == 0
