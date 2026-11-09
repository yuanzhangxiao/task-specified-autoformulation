"""Treatment isolation, fixed state semantics, paired receipts and split boundaries."""

from __future__ import annotations

from copy import deepcopy

import pytest

from autoformalism.rebuttal import iclr_controlled_search as campaign
from autoformalism.rebuttal import iclr_search_controls as controls
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write


def cell():
    return {
        "benchmark_id": "t2_hard",
        "tier": "hard",
        "limits": {},
        "meal_input": "meal",
        "context": {
            "targets": ["I", "Gp"],
            "auxiliaries": ["Uii"],
            "external_inputs": ["insulin_pmol_per_kg_min", "meal"],
        },
        "prompt": "A. Task specification\nConstruct a model of post-meal "
        "glucose regulation and insulin-dependent disposal.\n"
        "- a meal response.\n- a delayed insulin-action pathway regulating disposal.\n"
        "\nB. Available data\nGp and I are measured targets (mg/kg, pmol/L). "
        "Uii available.\n"
        "\nC. Modeling requirements\nUse shared parameters.\nF. Required response\n"
        "Provide equations.\n",
    }


def bundle(extra=False):
    states = ["Gp", "I"] + (["X"] if extra else [])
    c = {
        "candidate_id": "fixture",
        "parent_candidate_id": None,
        "states": [
            {"name": s, "kind": "latent" if s == "X" else "observed", "description": s}
            for s in states
        ],
        "state_equations": [
            {"state": "Gp", "rhs": "meal" + (" - X" if extra else "")},
            {"state": "I", "rhs": "insulin_pmol_per_kg_min"},
        ]
        + ([{"state": "X", "rhs": "I-X"}] if extra else []),
        "observation_mappings": [{"channel": t, "expression": t} for t in ("Gp", "I")],
        "initial_conditions": [
            {"state": s, "scope": "global", "expression": "0" if s == "X" else s}
            for s in states
        ],
    }
    return {
        "candidate": c,
        "initialization": {"context": cell()["context"]},
        "topology": {
            "inventory": [
                {"name": s, "definition": "differential", "scientific_role": s}
                for s in states
            ]
        },
    }


def test_fixed_gamma_allows_equation_changes_and_preserves_semantics_at_pruning():
    original = bundle(True)
    anchor = controls.gamma(original)
    active = controls.treatment(cell(), "fixed_gamma", anchor)
    task = {"scientific_verifier": True}
    legal = deepcopy(original)
    legal["candidate"]["state_equations"][2]["rhs"] = "I - 0.2 * X"
    assert controls.certificate(legal, active, task)[
        "eligible_for_development_selection"
    ]
    changed = deepcopy(legal)
    changed["topology"]["inventory"][2]["scientific_role"] = "absorbed meal"
    assert not controls.certificate(changed, active, task)[
        "eligible_for_development_selection"
    ]
    omitted = {k: v for k, v in legal.items() if k != "topology"}
    assert not controls.certificate(omitted, active, task)[
        "eligible_for_development_selection"
    ]
    assert controls.certificate(
        {**omitted, "lowered_pruning_request": True}, active, task
    )["eligible_for_development_selection"]
    omitted["candidate"]["states"][2]["description"] = "new meaning"
    assert not controls.certificate(
        {**omitted, "lowered_pruning_request": True}, active, task
    )["eligible_for_development_selection"]


def test_no_spec_removes_insulin_premises_but_preserves_roles_units_and_safety():
    a = controls.treatment(cell(), "no_spec")
    assert a["mechanism_spec"] is None and a["brief"]["requirements"] == []
    assert "insulin-dependent disposal" not in a["active_prompt"]
    assert "delayed insulin" not in a["active_prompt"]
    assert "pmol/L" in a["active_prompt"] and "post-meal glucose" in a["active_prompt"]
    assert controls.certificate(bundle(True), a, {"scientific_verifier": True})[
        "eligible_for_development_selection"
    ]


def test_no_latent_strict_and_prediction_capacity_are_separate():
    strict = controls.treatment(cell(), "no_latent")
    relaxed = controls.treatment(cell(), "observed_only")
    assert "only for directly observed" in strict["brief"]["scientific_context"]
    assert not controls.certificate(bundle(), strict, {"scientific_verifier": True})[
        "eligible_for_development_selection"
    ]
    assert controls.certificate(bundle(), relaxed, {"scientific_verifier": False})[
        "eligible_for_development_selection"
    ]
    assert not controls.certificate(
        bundle(True), relaxed, {"scientific_verifier": False}
    )["eligible_for_development_selection"]


def test_wrong_clause_is_consistent_across_critic_graph_and_proposer():
    wrong = controls.treatment(cell(), "incorrect")
    assert "I" not in wrong["brief"]["requirements"][1]["drivers"]
    assert "no causal pathway" in wrong["active_prompt"]
    assert not controls.certificate(bundle(True), wrong, {"scientific_verifier": True})[
        "controlled_checks"
    ]["no_insulin_to_balance_path"]
    under = controls.treatment(cell(), "underspecified")
    assert len(under["brief"]["requirements"]) == 1
    assert (
        "delayed insulin" not in under["active_prompt"]
        and "pmol/L" in under["active_prompt"]
    )
    assert "shared parameters" in under["active_prompt"]
    over = controls.treatment(cell(), "overrestrictive")
    assert not controls.certificate(bundle(), over, {"scientific_verifier": True})[
        "controlled_checks"
    ]["one_unobserved_state"]


def test_pair_shares_initial_failure_without_new_budget(tmp_path):
    parent = {"task_id": "full"}
    fixed = {"task_id": "fixed_gamma", "critic": False}
    original = sealed_write(
        io.round_path(tmp_path, parent, 0) / "result.json",
        {
            "task": parent,
            "round": 0,
            "selected": None,
            "trial": None,
            "status": "fit_failed",
            "cost": {"physical_requests": 5},
        },
    )
    result = campaign.share_initial(tmp_path, parent, fixed)
    assert (
        result["shared_source_sha256"] == original["artifact_sha256"]
        and not result["cost"]
    )
    assert sealed_read(io.round_path(tmp_path, fixed, 0) / "result.json") == result


def test_reject_invalid_roster_before_provider_or_files(tmp_path):
    with pytest.raises(ValueError, match="fixed_gamma requires full"):
        campaign.prepare(
            tmp_path / "run",
            tmp_path / "public",
            ["toy"],
            ["fixed_gamma"],
            [0],
            3,
            False,
            True,
        )
    with pytest.raises(ValueError, match="invalid public cell"):
        campaign.prepare(
            tmp_path / "run",
            tmp_path / "public",
            ["../test"],
            ["full"],
            [0],
            3,
            False,
            True,
        )


def test_parameter_cap_is_visible_and_enforced_on_lowered_coefficients():
    limited = controls.treatment({**cell(), "max_parameters": 1}, "full")
    assert "Fit at most 1 coefficients" in limited["active_prompt"]
    proposed = bundle(True)
    proposed["candidate"]["parameters"] = [
        {"name": n, "scope": "global"} for n in ("gain", "loss")
    ]
    proposed["candidate"]["state_equations"][-1]["rhs"] = "gain*I-loss*X"
    result = controls.certificate(proposed, limited, {"scientific_verifier": True})
    assert result["controlled_checks"]["fitted_parameter_budget"] is False
    assert not result["eligible_for_development_selection"]
    assert controls.certificate(
        proposed, controls.treatment(cell(), "full"), {"scientific_verifier": True}
    )["eligible_for_development_selection"]
