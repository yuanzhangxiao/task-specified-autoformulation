"""Observation contracts and term attribution for frozen CSTR panel replay."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/iclr_cstr_observation_panel.py"
SPEC = importlib.util.spec_from_file_location("cstr_observation_adapter", SOURCE)
panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)


def context(auxiliaries=()):
    """Declare the public sensor and forcing roles."""
    return {
        "targets": ["T"],
        "auxiliaries": list(auxiliaries),
        "external_inputs": ["Cf", "Tf", "Tjf"],
        "fixed_covariates": [],
    }


def reference():
    """Include hidden channels to check that the adapter never exposes them."""
    return {
        "time": [0.0, 1.0, 2.0],
        "T": [350.0, 355.0, 360.0],
        "C": [1.0, 2.0, 3.0],
        "Tj": [330.0, 335.0, 340.0],
        "Cf": [1.0, 1.0, 1.0],
        "Tf": [350.0, 350.0, 350.0],
        "Tjf": [330.0, 330.0, 330.0],
        "reaction_heat": [2.0, 4.0, 6.0],
    }


@pytest.mark.parametrize("auxiliaries", panel.MASKS.values())
def test_future_temperature_and_undeclared_sensors_are_never_exposed(auxiliaries):
    tr = panel.observation_trajectory(context(auxiliaries), reference())
    np.testing.assert_array_equal(tr.targets["T"], [350.0] * 3)
    assert set(tr.auxiliaries) == set(auxiliaries)
    assert set(tr.external_inputs) == {"Cf", "Tf", "Tjf"}
    assert tr.derivatives == {}


def test_unsupported_observation_contract_is_rejected():
    with pytest.raises(ValueError, match="observation contract"):
        panel.observation_trajectory(context(["reaction_heat"]), reference())


def test_paired_solver_disagreement_is_not_scored():
    good = {
        "status": "complete",
        "curves": {"T": [350, 355, 360]},
        "alternate_temperature": [350, 355, 370],
    }
    base = {
        "status": "complete",
        "curves": {"T": [350] * 3},
        "alternate_temperature": [350] * 3,
    }
    verdict = panel.target_score(
        None, good, base, reference(), reference() | {"T": [350] * 3}, {}, 1.0
    )
    assert verdict == {"status": "effect_solver_disagreement"}


def test_metric_failure_is_preserved():
    def failure(*_):
        raise ArithmeticError("non-finite effect")

    assert (
        panel.metric(SimpleNamespace(scores=failure))["status"] == "metric_unavailable"
    )


def test_affine_map_serialization_resumes_without_changing_scores(tmp_path):
    path = tmp_path / "scores.json"
    value = {"map": (2.0, -1.0), "effect_rse": 0.25}
    first = panel.write(path, value)
    assert first == panel.write(path, value)
    assert first == panel.read(path)
    with pytest.raises(ValueError, match="frozen artifact differs"):
        panel.write(path, value | {"effect_rse": 0.5})


@pytest.fixture
def declared():
    # Integration checks run with the independently maintained scoring checkout
    # on PYTHONPATH. Core contract checks above run in the main checkout too.
    return pytest.importorskip("scripts.score_declared_mechanisms")


def view(equation, auxiliaries=(), latent=None):
    from autoformalism.expressions import ValidationContext
    from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate

    ctx = context(auxiliaries)
    candidate = equation_candidate(
        "sindy", {"T": equation}, ValidationContext.model_validate(ctx)
    ).model_dump(mode="json")
    for name, rhs in (latent or {}).items():
        candidate["states"].append({"name": name, "kind": "latent"})
        candidate["state_equations"].append({"state": name, "rhs": rhs})
    return {
        "candidate": candidate,
        "context": ctx,
        "grammar": "baseline",
        "parameters": {},
        "initials": {},
        "semantics": "continuous_time",
    }


def test_original_temperature_only_rule_is_unchanged(declared):
    model = view("Tf - T + R + 0.3*(J-T)", latent={"R": "Cf-R", "J": "Tjf-J"})
    assert panel.attribute_observed(model) == declared.attribute(model)


def test_supplied_sensors_attribute_terms_without_claiming_latent_recovery(declared):
    model = view("0.5*(Tf-T) + 0.2*C - 0.3*(T-Tj)", ["C", "Tj"])
    assignment = panel.attribute_observed(model)
    assert [t["mechanism"] for t in assignment["terms"]] == list(declared.MECHANISMS)
    assert assignment["hidden"] == {"C": None, "Tj": None}
    assert assignment["state_kinds"] == {}
    tr = panel.observation_trajectory(model["context"], reference())
    curves = declared.curves(model, assignment, tr)
    np.testing.assert_allclose(curves["reaction_heat"], 0.2 * tr.auxiliaries["C"])
    np.testing.assert_allclose(
        curves["jacket_heat"], -0.3 * (np.asarray(curves["T"]) - tr.auxiliaries["Tj"])
    )
    assert not any(k.startswith("state.") for k in curves)


def test_measured_concentration_can_drive_a_latent_reaction_process(declared):
    model = view("Tf - T + Q + Tj - T", ["C", "Tj"], latent={"Q": "C-Q"})
    assignment = panel.attribute_observed(model)
    assert assignment["state_kinds"] == {"Q": "concentration"}
    assert assignment["represented"]["reaction_heat"]
    assert assignment["hidden"] == {"C": None, "Tj": None}
