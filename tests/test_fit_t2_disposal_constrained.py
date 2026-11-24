"""Constraints, accounting and train-only access of the matched disposal solver."""

import numpy as np
import pytest

from scripts import fit_t2_disposal_constrained as trial
from tests.test_validate_positive_insulin_transfer import toy


def fixture(monkeypatch, name="train_000"):
    row, model = toy(name)
    # A reduced system whose unconstrained optimum would violate U >= Uii.
    row.auxiliaries["Uii"] = [1.0] * 4
    row.targets["U"] = [0.5] * 4
    f = model["formula"]
    f["equations"]["U"] = "gain+afcoef13*I+afcoef14*(Uii-1)"
    model["graph"]["sources"]["U"] = ["I", "Uii"]
    f["parameters"].update(gain=2.5, afcoef13=0.001, afcoef14=4)
    hash_ = trial.content_hash([model["graph"], f["equations"], f["initializers"]])
    monkeypatch.setattr(trial, "GRAPHS", {hash_})
    return model, [vars(row)]


def test_constrained_fit_protects_insulin_and_dependent_disposal(monkeypatch):
    model, rows = fixture(monkeypatch)
    response = trial.fit(model, rows, "warm")
    p, diagnostics = response["parameters"], response["diagnostics"]
    assert response["training_nmse"] < diagnostics["source_nmse"]
    fixed = set(model["formula"]["parameters"]) - {"gain"}
    assert all(p[n] == model["formula"]["parameters"][n] for n in fixed)
    assert diagnostics["retained_minimum_U_minus_Uii"] == pytest.approx(
        trial.DISPOSAL_BUFFER, abs=1e-7
    )
    assert diagnostics["solver_success"]
    assert diagnostics["objective_evaluations"] <= diagnostics["budget"]


def test_midpoint_witness_and_original_sample_objective(monkeypatch):
    model, rows = fixture(monkeypatch)
    data = trial._Training(model, rows)
    assert data.samples.tolist() == [0, 2, 4, 6]
    delta = np.zeros_like(data.observed)
    delta[1, 2] = -3
    margins, selected = data.margins(delta)
    assert selected[0, 1] == 1
    assert margins[1] == pytest.approx(-3.5 - trial.DISPOSAL_BUFFER)


def test_budget_preserves_source_and_unqualified_inputs_fail(monkeypatch):
    model, rows = fixture(monkeypatch)
    response = trial.fit(model, rows, "displaced", budget=1)
    assert response["parameters"] == model["formula"]["parameters"]
    assert response["diagnostics"]["objective_evaluations"] == 1
    assert not response["diagnostics"]["solver_success"]
    with pytest.raises(ValueError, match="train"):
        trial.fit(model, [rows[0] | {"trajectory_id": "val_000"}], "warm")
    model["formula"]["equations"]["Gp"] += "+1"
    with pytest.raises(ValueError, match="Unqualified graph"):
        trial.fit(model, rows, "warm")
