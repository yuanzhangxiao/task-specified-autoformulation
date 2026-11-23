"""A serialized positive forcing law retains its scalar and complete-model meaning."""

import json
from copy import deepcopy

import numpy as np
import pytest

from autoformalism.expressions.compiler import _evaluate
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_function_fitting import validate_formula
from scripts import transfer_positive_insulin_law as p
from tests.test_transfer_insulin_order_law import parent


def selected():
    drug = p.production.prior.DRUG
    knots = {"Gt": [0.0, 1.0, 4.0], drug: [0.0], "EGP": [0.0, 1.0, 2.0]}
    weights = [0.0, 2.0, 8.0, 3.0, 10.0, 0.0, 2.0]
    model = {
        "coefficient_hats": p.production.coefficient_hats(knots),
        "weights": weights,
        "tau": 5.0,
    }
    return json.loads(json.dumps(knots, sort_keys=True)), model


def test_exported_cardinal_functions_saturate_and_preserve_coefficient_order(tmp_path):
    graph, form, coeff = parent()
    original = {"graph": graph, "formula": form | {"parameters": coeff}}
    before = deepcopy(original)
    knots, sm = selected()
    exported = p.export(original, knots, sm)
    path = tmp_path / "model.json"
    sealed_write(path, exported)
    model = sealed_read(path)
    value = validate_formula(model["graph"], model["formula"])
    assert value["initializers"] == form["initializers"]
    for k in ("U", "Z"):
        assert value["equations"][k] == form["equations"][k]
    for k in ("shared", "init"):
        assert value["parameters"][k] == coeff[k]
    assert "old_tau" not in value["parameters"]
    assert model["graph"]["sources"]["I"] == ["EGP", "Gt", "I"]
    assert model["constant_channels"] == {
        p.production.prior.DRUG: "single fitting knot"
    }
    compiled = RestrictedParser().parse(value["equations"]["I"], location="I")
    for x in (-3, 0, 0.5, 1.0, 1.4, 2.0, 3.5, 4.0, 5.0):
        for z in (-1, 0, 0.7, 1, 1.5, 2.0, 4.0):
            q = (
                np.interp(x, knots["Gt"], sm["weights"][:3])
                + 3
                + np.interp(z, knots["EGP"], sm["weights"][4:])
            )
            rhs = _evaluate(
                compiled, {**value["parameters"], "I": 20.0, "Gt": x, "EGP": z}
            )
            assert rhs == pytest.approx(q - 4.0, abs=1e-12)
            assert q >= 0
    assert original == before


def test_reject_unbound_order_negative_weights_and_collisions():
    g, f, v = parent()
    parent_model = {"graph": g, "formula": f | {"parameters": v}}
    knots, sm = selected()
    with pytest.raises(ValueError, match="identify"):
        p.export(
            parent_model, knots, sm | {"coefficient_hats": sm["coefficient_hats"][::-1]}
        )
    with pytest.raises(ValueError, match="nonnegative"):
        p.export(parent_model, knots, sm | {"weights": [-1] * 7})
    with pytest.raises(ValueError, match="timescale"):
        p.export(parent_model, knots, sm | {"tau": 0})
    v["pos_tau"] = 1.0
    with pytest.raises(ValueError, match="collides"):
        p.export(parent_model, knots, sm)


def test_identical_fitted_weights_collapse_without_adding_a_source():
    g, f, v = parent()
    knots, sm = selected()
    sm["weights"][:3] = [4, 4, 4]
    out = p.export({"graph": g, "formula": f | {"parameters": v}}, knots, sm)
    assert out["constant_channels"]["Gt"] == "identical fixed weights"
    assert "Gt" not in out["graph"]["sources"]["I"]
    assert out["formula"]["parameters"]["pos_Gt_flat"] == 4.0


def test_full_grid_reproduces_scalar_and_checks_archived_parent():
    drug = p.production.prior.DRUG
    knots, sm = selected()
    row = {
        "trajectory_id": "train_toy",
        "time": [0.0, 1.0, 2.0, 3.0, 4.0],
        "inputs": {drug: [0.0, 6.0, 3.0, 8.0, 4.0]},
        "auxiliaries": {
            "Gt": [-1.0, 0.4, 3.0, 4.7, 1.6],
            "EGP": [3.0, 0.0, 1.7, 1.0, 2.1],
        },
        "targets": {n: [3.0, 3.0, 4.0, 5.0, 6.0] for n in p.TARGETS},
        "covariates": {},
    }
    kinds = {"Gp": "differential", "I": "differential", "U": "algebraic"}
    g = {
        "variable_types": kinds,
        "sources": {"I": ["I", "Gt"], "Gp": ["I", "Gp"], "U": ["I"]},
        "initializers": {
            n: {"mode": "observed_initial", "channel": n} for n in ("Gp", "I")
        },
    }
    f = {
        "equations": {"I": "-I/5+Gt", "Gp": "(I-Gp)/2", "U": "I/3"},
        "initializers": {},
        "parameters": {},
    }
    original = {"formula": f, "graph": g}
    scales = {"I": 7.0, "Gp": 2.0, "U": 1.0}
    model = p.export(original, knots, sm) | {"scales": scales}
    old = p._metrics(row, p._predict(original, row, None, 4), scales)
    expected = p.production.score(row, knots, sm)
    base, child = p.compare_event(model, original, old, row, expected)
    assert base == old and child["status"] == "scored"
    assert np.isclose(child["sse"]["I"], expected["sse"], atol=2e-5, rtol=1e-6)
    assert (
        child["negative_samples"]["I"] == 0
        and child["levels"][-1]["substeps"] in p.LEVELS
    )
    with pytest.raises(ValueError, match="restoring"):
        p.compare_event(
            model, original, {"sse": dict.fromkeys(p.TARGETS, 100.0)}, row, expected
        )
    with pytest.raises(ValueError, match="forecast"):
        p.compare_event(
            model, original, old, row, expected | {"forecast_sha256": "other"}
        )
