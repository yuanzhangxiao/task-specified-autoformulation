"""Regressions for the reachability, fixed-cohort and bounded-fit audits."""

from __future__ import annotations

import ast

import numpy as np
import pytest

from autoformalism.schemas import CandidateModel
from scripts import iclr_active_history as history
from scripts import iclr_reference_fits as reference
from scripts.iclr_transfer_sensitivity import conditioning, limits, parameter_rows


def endpoint():
    model = CandidateModel.model_validate(
        {
            "candidate_id": "trial",
            "parent_candidate_id": None,
            "states": [
                {"name": "y", "kind": "observed"},
                {"name": "x", "kind": "latent"},
                {"name": "z", "kind": "latent"},
            ],
            "state_equations": [
                {"state": "y", "rhs": "k*sink+u"},
                {"state": "x", "rhs": "-x+v"},
                {"state": "z", "rhs": "-z+w"},
            ],
            "processes": [{"name": "sink", "expression": "x+(z-z)"}],
            "observation_mappings": [{"channel": "y", "expression": "y"}],
            "initial_conditions": [
                {"state": "y", "expression": "y", "scope": "global"},
                {"state": "x", "fixed_value": 0, "scope": "global"},
                {"state": "z", "fixed_value": 0, "scope": "global"},
            ],
            "parameters": [{"name": "k", "scope": "global"}],
        }
    )
    return {
        "status": "ready",
        "candidate": model.model_dump(mode="json"),
        "parameters": {"k": 0.5},
        "context": {
            "targets": ["y"],
            "external_inputs": ["u", "v", "w"],
            "auxiliaries": [],
        },
    }


def test_fitted_process_reachability_and_ast_count_are_name_stable():
    r = endpoint()
    data = history.endpoint_complexity(r)
    assert data["state_count"] == 3 and data["target_reachable_state_count"] == 2
    assert data["target_reachable_latent_count"] == 1 and data["parameter_count"] == 1
    expected = sum(
        sum(1 for _ in ast.walk(ast.parse(e, mode="eval")))
        for e in ("k*sink+u", "-x+v", "-z+w", "x+(z-z)", "y", "y")
    )
    assert data["expression_nodes"] == expected
    r["parameters"]["k"] = 0.0
    g = history.reachable({**r, "lowered_context": r["context"]})
    assert g["inactive_states"] == ["x", "z"] and g["public_edges"] == ["u→y"]
    assert history.overlap([], []) is None


def test_terminal_unknown_or_bad_expression_cannot_claim_inactive_latents():
    row = endpoint()
    row["candidate"]["state_equations"][0]["rhs"] = "__import__('os').system('test')"
    c = history.endpoint_complexity(row)
    assert c["complexity_status"] == "model_or_expression_unavailable"
    assert c["target_reachable_latent_count"] is None
    assert (
        history.endpoint_complexity({"status": "no_retained_model"})["state_count"]
        is None
    )


def test_fixed_cohort_does_not_replace_failed_tasks_between_visits():
    r = {"arm": "full", "round": 0, "retained_validation_nmse": 0.5}
    rows = [
        dict(r, task_id=t, round=i, result_sha256=str(i))
        for t in ("one_c", "two_c")
        for i in (0, 1)
    ]
    scores = {("one_c", 0, "0"): 0, ("one_c", 1, "1"): 0.5, ("two_c", 1, "1"): 1}
    observed = history.fixed_curves(rows, scores, 2)
    assert [v["fixed_runs"] for v in observed] == [1, 1]
    assert [v["mean_confirmed"] for v in observed] == [0, 0.5]
    assert observed[0]["tasks"] == ["one_c"]


def test_wider_limits_contain_original_domain_and_preserve_gastric_ordering():
    c = reference.coefficients(reference.T1P)
    names = list(c)
    base_l, base_u = limits(reference.T1P, False)
    lo, hi = limits(reference.T1P, True)
    assert np.all(lo <= base_l) and np.all(hi >= base_u)
    assert c["kmin"] * np.exp(hi[names.index("kmin")]) == pytest.approx(
        c["kmax"] * np.exp(lo[names.index("kmax")])
    )
    assert c["b"] * np.exp(hi[names.index("b")]) == pytest.approx(0.98)
    result = parameter_rows(reference.T1P, lo, "wide_same_starts", 0)
    assert all(r["bound_hit"] == "lower" for r in result)
    assert conditioning(np.diag([1.0, 0]))["condition_number"] is None
