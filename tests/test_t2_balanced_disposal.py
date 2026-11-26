"""Matched whole-ODE construction and qualification of balanced disposal laws."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import run_t2_balanced_disposal as fit


def packet():
    graph = {
        "variable_types": {
            "Gp": "differential",
            "I": "differential",
            "I_act": "differential",
            "U": "algebraic",
        },
        "sources": {
            "Gp": ["meal_rate_g_per_min", "U"],
            "I": ["Gp"],
            "I_act": ["I"],
            "U": ["I_act", "Uii"],
        },
        "initializers": {
            "Gp": {"mode": "observed_initial", "channel": "Gp"},
            "I": {"mode": "observed_initial", "channel": "I"},
            "I_act": {"mode": "value", "scope": "shared_training"},
        },
    }
    formula = {
        "equations": {
            "Gp": "a*meal_rate_g_per_min+b*U",
            "I": "c*Gp",
            "I_act": "e*I",
            "U": "q+k*I_act+w*((Uii-(1))/1)",
        },
        "initializers": {"I_act": "z"},
        "parameters": {
            "a": 0.0,
            "b": 0.0,
            "c": 0.0,
            "e": 0.0,
            "q": 2.0,
            "k": 0.1,
            "w": 0.0,
            "z": 0.0,
        },
    }
    events = [
        {
            "time": [0.0, 1.0, 2.0],
            "trajectory_id": f"train_{i:03}",
            "targets": {
                "Gp": [3.0, 3.0, 3.0],
                "I": [2.0, 2.0, 2.0],
                "U": [2.0 + i] * 3,
            },
            "inputs": {"meal_rate_g_per_min": [0.0] * 3},
            "auxiliaries": {"Uii": [1.0] * 3, "Gt": [5.0 + i] * 3},
            "covariates": {},
        }
        for i in range(2)
    ]
    return {
        "graph": graph,
        "shape": {"candidates": [formula], "candidate_families": [{"U": "affine"}]},
        "train_rows": events,
    }


def screen(reject=""):
    return {
        "hidden_candidates": {
            "+".join(("I_act", *states)): {
                "eligible": reject != "+".join(("I_act", *states))
            }
            for states in fit.LAWS.values()
        }
    }


def test_paired_forms_and_frozen_native():
    p = packet()
    original = deepcopy(p)
    families = fit.candidates(p, screen("I_act+Gt"), "action")
    assert p == original and len(families) == 10
    assert families["free_action"]["formula"] == p["shape"]["candidates"][0]
    for law in fit.LAWS:
        pair = [families[f"{arm}_{law}"] for arm in ("free", "balanced")]
        assert pair[0]["graph"] == pair[1]["graph"]
        assert all(
            c["dependency_eligible"] == (law not in {"tissue", "interaction"})
            for c in pair
        )
        for c in pair:
            assert (
                c["formula"]["initializers"]
                == p["shape"]["candidates"][0]["initializers"]
            )
            assert (
                c["formula"]["equations"]["I_act"]
                == p["shape"]["candidates"][0]["equations"]["I_act"]
            )
    assert (
        "dJoint*I_act*" in families["balanced_interaction"]["formula"]["equations"]["U"]
    )
    p["graph"]["sources"]["I_act"] = []
    with pytest.raises(ValueError, match="required dependency"):
        fit.candidates(p, screen(), "action")


def test_balance_fixed_for_unseen_input_and_causal_initialization():
    p = packet()
    families = fit.candidates(p, screen(), "action")
    row = deepcopy(p["train_rows"][0])
    row["auxiliaries"]["Uii"] = [1.0, 2.0, 3.0]
    targets = ["Gp", "I", "U"]
    for arm in ("balanced", "free"):
        c = families[f"{arm}_interaction"]
        form = {**c["formula"], "parameters": c["starts"]["constant_observed"]}
        endpoint = {
            "parameters": form["parameters"],
            "scales": dict.fromkeys(targets, 1.0),
        }
        curve = fit.prior._forecast(c["graph"], form, endpoint, row, targets)
        assert curve["status"] == "scored"
        expected = np.array([2.5, 3.5, 4.5]) if arm == "balanced" else np.full(3, 2.5)
        np.testing.assert_allclose(curve["predicted"]["U"], expected)
        alternate = deepcopy(row)
        alternate["targets"]["U"] = [0.0, 7.0, 10.0]
        assert (
            fit.prior._forecast(c["graph"], form, endpoint, alternate, targets)[
                "predicted"
            ]
            == curve["predicted"]
        )


def test_endpoint_replays_actual_start_and_rejects_wrong_identity(tmp_path):
    family = "balanced_action"
    c = fit.candidates(packet(), screen(), "action")[family]
    plan = {"artifact_sha256": "fixed-plan"}
    panel = {"panel_id": "graph", "candidates": {family: c}}
    folder = tmp_path / "fits" / "graph" / family
    start = "constant_observed"
    original_form = {**c["formula"], "parameters": c["starts"][start]}
    fit_result = {
        "graph_sha256": fit.content_hash(c["graph"]),
        "formula_sha256": fit.content_hash(original_form),
        "status": "fitted",
    }
    receipt = fit.sealed_write(
        folder / f"{start}.json",
        {
            "plan": "fixed-plan",
            "start": start,
            "formula": fit.content_hash(original_form),
            "result": fit_result,
        },
    )
    chosen = fit.sealed_write(
        folder / "best.json",
        {
            "plan": "fixed-plan",
            "panel_id": "graph",
            "family": family,
            "best_start": start,
            "starts": {start: receipt["artifact_sha256"]},
        },
    )
    assert fit.load_endpoint(tmp_path, plan, panel, family) == (
        c["graph"],
        original_form,
        fit_result,
        chosen["artifact_sha256"],
    )
    with pytest.raises(ValueError, match="endpoint identity"):
        fit.load_endpoint(tmp_path, {"artifact_sha256": "other-plan"}, panel, family)
    c["starts"][start]["dOffset"] += 1.0
    with pytest.raises(ValueError, match="Starting vector"):
        fit.load_endpoint(tmp_path, plan, panel, family)
