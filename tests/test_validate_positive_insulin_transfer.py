from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import validate_positive_insulin_transfer as trial
from tests.test_transfer_positive_insulin_law import selected


def toy(name="val_000"):
    knots, sm = selected()
    drug = trial.portable.production.prior.DRUG
    row = SimpleNamespace(
        trajectory_id=name,
        time=[0.0, 1.0, 2.0, 4.0],
        targets={n: [4.0, 6.0, 8.0, 3.0] for n in trial.TARGETS},
        inputs={drug: [2.0, 1.0, 3.0, 2.0]},
        auxiliaries={"Gt": [-2.0, 2.0, 5.0, 0.3], "EGP": [1.0, 3.0, 0.0, 2.0]},
        covariates={},
    )
    graph = {
        "variable_types": {"Gp": "differential", "I": "differential", "U": "algebraic"},
        "sources": {"I": ["I"], "Gp": ["Gp", "I"], "U": ["I"]},
        "initializers": {
            n: {"mode": "observed_initial", "channel": n} for n in ("Gp", "I")
        },
    }
    form = {
        "equations": {"I": "-I/5", "Gp": "I-Gp", "U": "I/2"},
        "initializers": {},
        "parameters": {},
    }
    model = trial.portable.export({"graph": graph, "formula": form}, knots, sm)
    return row, model | {"scales": dict.fromkeys(trial.TARGETS, 1.0)}


def test_qualified_validation_forecast_cannot_read_later_targets():
    row, m = toy()
    original = trial.forecast_inputs(row)
    changed = deepcopy(row)
    for y in changed.targets.values():
        y[1:] = [1e6] * 3
    p, log = trial.forecast(m, original)
    q, _ = trial.forecast(m, trial.forecast_inputs(changed))
    assert len(log) >= 2 and min(p[1]) >= 0
    np.testing.assert_array_equal(p, q)
    np.testing.assert_allclose(
        p[1],
        trial.scalar_solve(original, m["knots"], m["scalar_model"]),
        atol=1e-5,
        rtol=1e-7,
    )
    assert set(trial.IDS) == {"val_000", "val_001", "val_002", "val_003"}


def test_allows_only_registered_val_ids_and_supplied_roles():
    row, _ = toy("train_000")
    with pytest.raises(ValueError, match="ID"):
        trial.forecast_inputs(row)
    row, _ = toy()
    row.auxiliaries["I"] = row.targets["I"]
    with pytest.raises(ValueError, match="roles"):
        trial.forecast_inputs(row)


def test_scalar_constant_has_known_exact_solution():
    row, model = toy()
    event = trial.forecast_inputs(row)
    knots, sm = model["knots"], model["scalar_model"]
    sm["weights"] = [1.0] * len(sm["weights"])
    expected = 3 * sm["tau"] + (4 - 3 * sm["tau"]) * np.exp(-event.time / sm["tau"])
    np.testing.assert_allclose(
        trial.scalar_solve(event, knots, sm), expected, atol=1e-10
    )


def test_no_validation_access_without_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(
        trial, "read_validation", lambda *_: pytest.fail("Validation opened")
    )
    with pytest.raises(FileNotFoundError):
        trial.validate(tmp_path, tmp_path, tmp_path / "validation.json")


def test_source_receipt_hash_checked_before_validation_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(
        trial,
        "sealed_read",
        lambda _: {"artifact_sha256": "x", "training_intake_sha256": "y"},
    )
    monkeypatch.setattr(
        trial, "_rows", lambda *_: pytest.fail("Parsed unverified rows")
    )
    with pytest.raises(ValueError, match="intake"):
        trial.read_validation(
            tmp_path / "validation.json",
            {"validation_intake": "different", "training_intake": "y"},
        )
