"""External baselines go through the study's own held-out test-panel scoring."""

from __future__ import annotations

import numpy as np

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate
from scripts import iclr_stress_evaluation as stress
from scripts import score_baseline_interventions as scoring
from scripts import score_baseline_test_panel as panel

CONTEXT = ValidationContext.model_validate(
    {
        "targets": ["T"],
        "auxiliaries": [],
        "external_inputs": ["Cf", "Tf", "Tjf"],
        "fixed_covariates": [],
    }
)
CELL = {"cell": "toy", "family": "cstr", "noise": 0, "stride": 1, "mask": []}


def _row(law: str, context: ValidationContext = CONTEXT) -> dict:
    candidate = equation_candidate(
        "sindy", dict.fromkeys(context.targets, law), context
    )
    return {
        "source_id": "sindy:toy:0",
        "method": "sindy",
        "repetition": 0,
        "status": "ready",
        "model": {
            "candidate": candidate.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "parameters": {},
            "initials": {},
            "semantics": "continuous_time",
            "grammar": "baseline",
        },
    }


def _record(jacket: np.ndarray, time: np.ndarray) -> dict:
    n = len(time)
    return {
        "trajectory_id": "toy",
        "time": time.tolist(),
        "targets": {"T": [350.0] * n},
        "auxiliaries": {},
        "external_inputs": {
            "Cf": [1.0] * n,
            "Tf": [350.0] * n,
            "Tjf": jacket.tolist(),
        },
    }


def _pair(truth: dict) -> dict:
    """A jacket step and its quiet control, with the truth's own rollouts."""
    time = np.round(np.arange(0.0, 3.0001, 0.1), 10)
    step = np.where((time >= 1.0) & (time < 2.0), 340.0, 330.0)
    pair = {"case": "toy_step"}
    for key, jacket in (("perturbed", step), ("control", np.full_like(time, 330.0))):
        record = _record(jacket, time)
        curve = scoring.predict(truth, stress.public_input(record, CELL), "Radau")
        record["targets"]["T"] = list(curve["T"])
        pair[key] = record
    return pair


def test_the_true_model_scores_zero_and_a_wrong_one_does_not():
    truth = _row("-0.8*(T - 350) + 0.5*(Tjf - 330)")
    pair = _pair(truth)
    exact = stress.score(
        truth, CELL, [pair], {"T": 1.0}, panel.baseline_predictor(truth)
    )
    assert exact["evaluation_status"] == "complete"
    assert exact["test_nmse"] < 1e-12
    channel = exact["trajectories"][0]["channels"]["T"]
    assert channel["signed_peak_direction_correct"] is True
    wrong = _row("-0.8*(T - 350) + 0.25*(Tjf - 330)")
    off = stress.score(wrong, CELL, [pair], {"T": 1.0}, panel.baseline_predictor(wrong))
    assert off["test_nmse"] > 1e-3
    assert 0.1 < off["trajectories"][0]["channels"]["T"]["response_rse"] < 1.0


def test_a_failed_rollout_leaves_the_test_nmse_unavailable():
    pair = _pair(_row("-0.8*(T - 350) + 0.5*(Tjf - 330)"))
    other = ValidationContext.model_validate(
        {**CONTEXT.model_dump(mode="json"), "external_inputs": ["Cf", "Tjf"]}
    )
    broken = _row("-0.8*(T - 350)", other)
    result = stress.score(
        broken, CELL, [pair], {"T": 1.0}, panel.baseline_predictor(broken)
    )
    assert result["evaluation_status"] == "unresolved"
    assert result["test_nmse"] is None
