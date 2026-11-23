"""Checks for nested free-rollout disposal comparisons and train-only selection."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import screen_t2_disposal_revisions as screen


def test_bounded_linear_response_reports_offset_and_handle_zero_columns():
    x = np.array([[1.0, 0], [0, 1], [0, 1]])
    y = np.array([-3.0, 5.0, 5.0])
    assert np.allclose(screen.solve(x, y, np.array([-np.inf, 0.0])), [3, 0])
    assert screen.solve(x[:, :0], y, np.empty(0)).size == 0
    with pytest.raises(ValueError, match="lengths"):
        screen.solve(x, y[:-1], np.array([-np.inf, 0.0]))


def test_menu_edits_only_disposal_and_fails_closed_on_unsupplied_predictor():
    parent = {
        "graph": {
            "variable_types": {
                "Gp": "differential",
                "I_act": "differential",
                "U": "algebraic",
                "I": "differential",
            },
            "sources": {
                "Gp": ["U"],
                "I_act": ["I"],
                "I": ["I"],
                "U": ["Gp", "I_act", "Uii"],
            },
            "initializers": {
                k: {"mode": "observed_initial", "channel": k}
                for k in ("Gp", "I_act", "I")
            },
        },
        "formula": {
            "equations": {
                "Gp": "-U",
                "I_act": "I",
                "I": "-I",
                "U": "afcoef11+afcoef12*Gp+afcoef13*I_act+Uii",
            },
            "parameters": {"afcoef11": 2, "afcoef12": 1, "afcoef13": 3},
            "initializers": {},
        },
        "training_formula": "old",
    }
    old = deepcopy(parent)
    for family in screen.menu(parent):
        new = screen.assemble(parent, family)
        if family.name not in ("incumbent", "refit_existing"):
            assert "training_formula" not in new
            assert set(new["graph"]["sources"]["U"]) <= {"Gp", "I_act", "Uii", "Gt"}
            assert "afcoef13" not in new["formula"]["parameters"]
        assert new["formula"]["equations"]["Gp"] == "-U"
        assert "I_act" in new["graph"]["sources"]["U"]
    assert parent == old
    with pytest.raises(ValueError, match="unsupplied"):
        screen.assemble(parent, screen.Family("bad", "future_G", (), (), ()))


def test_ineligible_candidate_and_wrong_fold_cannot_win():
    def result(loss, *, dep=0.5, action=True):
        scores = {
            "pooled_nmse": loss,
            "per_target_nmse": {"U": loss},
            "min_U_minus_Uii": dep,
            "minima": {"Gp": 1.0},
        }
        return {
            "training": dict(scores),
            "held_training": dict(scores),
            "active_action_gain": action,
        }

    good = result(0.8)
    wrong_fold = result(0.7)
    wrong_fold["held_training"]["per_target_nmse"] = {"U": 1.5}
    cases = {
        "incumbent": result(1.0),
        "refit_existing": result(0.5),
        "inactive": result(0.1, action=False),
        "wrong_fold": wrong_fold,
        "negative": result(0.1, dep=-0.4),
    }
    assert screen.select(cases)[0] == "incumbent"
    winner, reasons = screen.select(cases | {"good": good})
    assert winner == "good"
    assert reasons["wrong_fold"] == ["held_training_U"]
    assert "training_negative_contribution_or_target" in reasons["negative"]


def test_summaries_weight_by_sample_count_and_keep_event_disposal_sse():
    def row(name, n):
        return {
            "trajectory_id": name,
            "time": list(range(n)),
            "targets": {k: np.full(n, 10) for k in screen.TARGETS},
            "auxiliaries": {"Uii": np.ones(n)},
        }

    y = np.zeros((4, 3))
    y[-1, 2] = -4
    result = screen.summary(
        y.ravel(), [row("a", 3), row("b", 1)], {"Gp": 1, "I": 1, "U": 2}
    )
    assert result["event_U_sse"] == {"a": 0, "b": 64}
    assert result["per_target_nmse"]["U"] == 4
    assert result["min_U_minus_Uii"] == 1
