"""The one-round T2 edits use only permitted initials and matched free rollouts."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import revision_round_insulin as case


def toy():
    model = {
        "graph": {
            "variable_types": {
                "Gp": "differential",
                "I": "differential",
                "U": "algebraic",
            },
            "initializers": {
                k: {"mode": "observed_initial", "channel": k} for k in ("Gp", "I")
            },
            "sources": {"Gp": ["U"], "I": [], "U": ["Gt"]},
        },
        "formula": {
            "equations": {"Gp": "3-U", "I": "0", "U": "afcoef11+disp_tissue*Gt"},
            "initializers": {},
            "parameters": {"afcoef11": 2.0, "disp_tissue": 0.25},
        },
        "scales": dict.fromkeys(case.TARGETS, 1.0),
    }
    rows = []
    for i, gt in enumerate((1.0, 2.0, 3.0)):
        measured_u = 3.0 - 0.25 * gt
        rows.append(
            {
                "trajectory_id": f"train_{i:03}",
                "time": [0, 1, 2],
                "covariates": {},
                "inputs": {},
                "auxiliaries": {"Gt": [gt] * 3, "Uii": [1.0] * 3},
                "targets": {
                    "I": [2.0] * 3,
                    "U": [measured_u] * 3,
                    "Gp": [10 + j * (3.0 - measured_u) for j in range(3)],
                },
            }
        )
    return model, rows


@pytest.mark.parametrize(
    "edit",
    [
        {},
        {"initial_source": "Gt", "half_life_min": False},
        {"initial_source": "U", "half_life_min": 25},
        {"initial_source": "Gt", "half_life_min": 0.1},
        {"initial_source": "none", "half_life_min": 100},
        {"initial_source": "I", "half_life_min": float("nan")},
    ],
)
def test_invalid_edits_fail_before_data_access(edit, monkeypatch):
    monkeypatch.setattr(case, "_intake", lambda: pytest.fail("Training opened"))
    assert case.score(edit)["eligible"] is False


@pytest.mark.parametrize(("source", "half"), [("Gp", 0), ("Gt", 5), ("I", 30)])
def test_compiled_extra_state_uses_initial_value_only(source, half):
    parent, rows = toy()
    original = deepcopy(parent)
    e = {"initial_source": source, "half_life_min": half}
    request = case._assemble(e, parent, rows[0])
    form, graph = request["formula"], request["graph"]
    assert parent == original
    assert form["parameters"][case.GAIN] == 0.0
    assert "M_init" in graph["sources"]["U"]
    assert graph["sources"]["M_init"] == (["M_init"] if half else [])
    assert graph["initializers"]["M_init"]["state_initials"] == (
        [] if source == "Gt" else [source]
    )
    base = case.FitProblem(parent, rows)
    nested = case.FitProblem(request, rows)
    np.testing.assert_allclose(
        base.residual(base._vector(parent["formula"]["parameters"])),
        nested.residual(nested._vector(form["parameters"])),
        atol=1e-10,
    )


def test_training_refit_changes_slope_and_inverts_the_wrong_order():
    model, rows = toy()
    fits = case._restricted_fit(model, rows, case.COMMON)
    assert fits["updated_coefficients"] == pytest.approx(
        {"afcoef11": 3, "disp_tissue": -0.25}, abs=1e-5
    )


def test_reject_modified_model_before_opening_validation(monkeypatch):
    model, rows = toy()
    monkeypatch.setattr(case, "_intake", lambda: (model, rows))
    monkeypatch.setattr(
        case.replay, "load_validation", lambda *_: pytest.fail("Development opened")
    )
    bad = {
        "eligible": True,
        "edit": case.PARENT,
        "model_sha256": "bad",
        "fit": {"updated_coefficients": {"afcoef11": 2000}},
    }
    with pytest.raises(ValueError, match="differs"):
        case.validate(case.PARENT, bad)


def test_screen_collapses_proportional_initial_columns_without_using_observed_levels():
    _, rows = toy()
    for idx, row in enumerate(rows):
        row["targets"]["Gp"][0] = 12 - 2 * row["auxiliaries"]["Gt"][0]
        row["targets"]["I"][0] = float(idx == 1)
    unique, aliases = case._initial_contrasts(rows)
    assert set(unique) == {"Gt", "I"}
    assert aliases == {"Gp": "Gt"}
    for row in rows:
        row["targets"]["I"][0] = 4.5
    assert case._initial_contrasts(rows)[1] == {"Gp": "Gt", "I": "zero on training"}


def test_conditional_disposal_screen_reports_infeasible_order_and_best_affine_score():
    values = np.arange(4, dtype=float)
    prediction = 2.0 + values
    observed = 5.0 - values
    first = np.array([0, 2])
    assert (
        case._conditional_order_loss(np.ones((4, 1)), prediction, observed, first)
        is None
    )
    assert case._conditional_order_loss(
        np.column_stack((np.ones(4), values)), prediction, observed, first
    ) == pytest.approx(0.0, abs=1e-10)
