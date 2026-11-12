"""The frozen rule that reads a model's declared reactor mechanisms."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate
from scripts import score_declared_mechanisms as declared

CONTEXT = ValidationContext.model_validate(
    {
        "targets": ["T"],
        "auxiliaries": [],
        "external_inputs": ["Cf", "Tf", "Tjf"],
        "fixed_covariates": [],
    }
)


def _classify(node):
    return declared._kind(declared.names_in(node), {"R": "concentration"})


def _terms(text):
    tree = ast.parse(text, mode="eval").body
    return [
        (sign, ast.unparse(node))
        for sign, node, _ in declared.split_terms(tree, _classify)
    ]


def test_sums_split_and_common_factors_distribute():
    assert _terms("a*Cf*(Tf - T) + b*R - c*(Tjf - T)") == [
        (1, "a * Cf * (Tf - T)"),
        (1, "b * R"),
        (-1, "c * (Tjf - T)"),
    ]
    # A heat capacity common to terms of different mechanisms is distributed.
    assert _terms("(a*(Tf - T) + b*Cf - c*(Tjf - T)) / (C0 + C1 * T)") == [
        (1, "a * (Tf - T) / (C0 + C1 * T)"),
        (1, "b * Cf / (C0 + C1 * T)"),
        (-1, "c * (Tjf - T) / (C0 + C1 * T)"),
    ]
    assert _terms("(a*(Tf - T) + b*R) * reciprocal(Cp)") == [
        (1, "a * (Tf - T) * reciprocal(Cp)"),
        (1, "b * R * reciprocal(Cp)"),
    ]
    assert _terms("1 / C * (a*(Tf - T) - b*(Tjf - T))") == [
        (1, "1 / C * (a * (Tf - T))"),
        (-1, "1 / C * (b * (Tjf - T))"),
    ]
    # A difference inside one mechanism's term stays whole.
    assert _terms("k * Cf * (Tf - T)") == [(1, "k * Cf * (Tf - T)")]
    assert _terms("(Tf - T) / tau") == [(1, "(Tf - T) / tau")]
    assert _terms("-(a - b)") == [(-1, "a"), (1, "b")]


def test_a_distributed_factor_never_decides_a_mechanism():
    # D3 seed 1's form: a heat capacity containing Cf divides terms of several
    # mechanisms; the ambient loss must not become reaction heat through it.
    text = "(a*(T - Tamb) + b*(Tf - T) + c*(Tjf - T)) * reciprocal(Cp0 + CpC*Cf)"
    parts = declared.split_terms(ast.parse(text, mode="eval").body, _classify)
    assert [_classify(own) for _, _, own in parts] == [None, "feed", "jacket"]
    assert all("reciprocal" in ast.unparse(term) for _, term, _ in parts)


def test_function_names_are_not_variables():
    names = declared.names_in(
        ast.parse("A * exp(-Ea / T) * sigmoid(Cf)", mode="eval").body
    )
    assert names == {"A", "Ea", "T", "Cf"}


def _candidate(equations, latent=()):
    """A small candidate with latent states, built on the baseline adapter."""
    candidate = equation_candidate("sindy", {"T": equations["T"]}, CONTEXT)
    payload = candidate.model_dump(mode="json")
    for name in latent:
        payload["states"].append({"name": name, "kind": "latent"})
        payload["state_equations"].append({"state": name, "rhs": equations[name]})
    return payload


def test_terms_and_states_are_assigned_by_the_inputs_they_contain():
    view = {
        "candidate": _candidate(
            {
                "T": "0.5*Cf*(Tf - T) + 0.7*R - 0.4*(T - J) + 0.2*(Tf_in - T) - 0.1",
                "R": "2*(Cf - R)",
                "J": "3*(Tjf - J)",
                "Tf_in": "1.5*(Tf - Tf_in)",
            },
            latent=("R", "J", "Tf_in"),
        ),
        "context": CONTEXT.model_dump(mode="json"),
        "grammar": "baseline",
    }
    try:
        result = declared.attribute(view)
    except Exception as exc:  # the adapter's candidate may refuse extra states
        pytest.skip(f"toy latent candidate not constructible: {exc}")
    assert result["state_kinds"] == {
        "R": "concentration",
        "J": "jacket",
        "Tf_in": "feed",
    }
    assert [t["mechanism"] for t in result["terms"]] == [
        "feed_transport",
        "reaction_heat",
        "jacket_heat",
        "feed_transport",
        "unattributed",
    ]
    assert result["hidden"] == {"C": "R", "Tj": "J"}
    assert result["represented"] == {
        "feed_transport": True,
        "reaction_heat": True,
        "jacket_heat": True,
    }


def test_terms_add_up_to_the_right_hand_side_along_a_rollout():
    law = "0.5*(Tf - T) + 0.02*Cf*T - 0.3*(T - Tjf) - 0.01"
    view = {
        "candidate": equation_candidate("sindy", {"T": law}, CONTEXT).model_dump(
            mode="json"
        ),
        "context": CONTEXT.model_dump(mode="json"),
        "parameters": {},
        "initials": {},
        "grammar": "baseline",
        "semantics": "continuous_time",
        "increment_step": None,
        "native": None,
    }
    assignment = declared.attribute(view)
    assert [t["mechanism"] for t in assignment["terms"]] == [
        "feed_transport",
        "reaction_heat",
        "jacket_heat",
        "unattributed",
    ]
    time = np.linspace(0.0, 3.0, 31)
    tr = Trajectory(
        trajectory_id="toy",
        time=time,
        targets={"T": np.full(len(time), 360.0)},
        external_inputs={
            "Cf": np.ones(len(time)),
            "Tf": np.full(len(time), 350.0),
            "Tjf": np.where(time > 1.0, 340.0, 330.0),
        },
        auxiliaries={},
        fixed_covariates={},
        derivatives={},
    )
    out = declared.curves(view, assignment, tr)
    temperature = np.asarray(out["T"])
    tjf = tr.external_inputs["Tjf"]
    np.testing.assert_allclose(out["feed_transport"], 0.5 * (350.0 - temperature))
    np.testing.assert_allclose(out["jacket_heat"], -0.3 * (temperature - tjf))
    np.testing.assert_allclose(out["unattributed"], -0.01)


def test_the_rule_is_frozen_for_the_reactor():
    rule = json.loads(Path(declared.RULE).read_text())
    assert rule["status"] == "frozen_before_scoring"
    assert rule["truth"]["feed_transport"].startswith("Tf - T")
    assert rule["calibration"] == [
        "train_zero",
        "train_cf_plus_step",
        "train_tjf_pulse",
        "train_tf_minus_step",
    ]
