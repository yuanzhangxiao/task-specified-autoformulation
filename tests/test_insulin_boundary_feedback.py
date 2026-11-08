"""Witnesses and conditional clock diagnostics for the saved insulin balance."""

import json

import pytest

from autoformalism.research import insulin_boundary_revision as boundary
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from scripts import describe_insulin_boundary as diagnostics
from tests.test_construction_revision_edits import problem
from tests.test_insulin_boundary_revision import with_rhs


def point():
    return {
        "trajectory": "train_003",
        "time_min": 2,
        "inputs": {boundary.MEAL: 0, boundary.INFUSION: 0},
        "state_at_boundary": {"I": 0, "Gp": 1.5, "mem": 2},
        "reference_rate_pmol_L_min": -2,
    }


def test_depletion_feedback_requires_the_actual_public_training_witness():
    witness = point()
    inputs = {
        "problem": with_rhs("-q").model_dump(mode="json"),
        "revision_protocol": boundary.PROTOCOL,
        "evidence": {
            "split": "train",
            "trajectories": [{"trajectory_id": "train_003"}],
        },
        "boundary_witness": witness,
        "user": json.dumps(
            {"training_evidence": {"depletion": witness, "ingestion_windows": []}}
        ),
    }
    out = diagnostics.feedback(inputs)
    assert "-2.000000" in out["fitted_feedback"][0]
    assert diagnostics.feedback(inputs) == out
    with pytest.raises(ValueError, match="does not reproduce"):
        diagnostics.feedback(
            {**inputs, "problem": with_rhs("-2*q").model_dump(mode="json")}
        )
    with pytest.raises(ValueError, match="training"):
        diagnostics.feedback({**inputs, "evidence": {"split": "validation"}})
    with pytest.raises(ValueError, match="training"):
        diagnostics.feedback(
            {**inputs, "evidence": {"split": "train", "trajectories": []}}
        )


def test_new_state_boundary_and_meal_amount_are_reported_only_when_identified():
    def relation(rhs):
        return diagnostics._missing_state_term(with_rhs(rhs), point(), "dep")

    assert "dI/dt = dep." in relation("max(-q,0) + dep")
    assert "dI/dt = -dep +1.5" in relation("Gp - dep")
    assert relation("max(dep,0)") is None
    assert relation("dep * I") is None
    assert diagnostics._meal_coefficient(with_rhs(f"a*{boundary.MEAL}+max(-q,0)")) == 1
    assert diagnostics._meal_coefficient(with_rhs(f"I*{boundary.MEAL}")) is None
    assert diagnostics._meal_coefficient(with_rhs(f"max({boundary.MEAL},0)")) is None


def test_clock_checks_expand_processes_and_respect_parameter_scope_and_bounds():
    def findings(rhs, role="nonnegative_coefficient", scope="global", bounds=None):
        data = problem().model_dump(mode="json")
        model = data["base_candidate"]
        model["parameters"] = [
            {"name": "a", "role": role, "scope": scope, "bounds": bounds}
        ]
        model["processes"][0]["expression"] = rhs
        next(r for r in model["state_equations"] if r["state"] == "dep")["rhs"] = "q"
        return diagnostics._clock_feedback(
            PrefixFitProblem.model_validate(data), {"dep"}
        )

    assert len(findings("sigmoid(t/a)")) == 2
    assert len(findings("sigmoid(t/a)", role="time_constant")) == 1
    assert len(findings("sigmoid(t/a)", bounds={"lower": 1, "upper": 2})) == 1
    assert not findings(
        "sigmoid(t/a)", role="time_constant", scope="trajectory_specific"
    )
    assert not findings("sigmoid(t/a)*u", role="time_constant")
