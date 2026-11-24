"""Check total-mass fitting against its declared graph and state equations."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import t2_initial_memory_fit as plasma
from scripts import t2_total_glucose_fit as total
from scripts.positive_insulin_refit_kernel import FitProblem
from tests.test_t2_initial_memory_fit import _synthetic_fold


def _fold():
    rows, fold = _synthetic_fold()
    for r in rows:
        r["auxiliaries"]["E"] = [0.02, 0.03, 0.04, 0.05]
    return rows, fold


def test_independent_full_graph_replay_matches_specialized_fit():
    rows, fold = _fold()
    model = total.seed_model(rows, fold, "long")
    p = model["formula"]["parameters"]
    p.update(
        afcoef12=1.2,
        afcoef10=0.08,
        disp_tissue=-1.7,
        afcoef16=0.03,
        init_tissue_gain=5.7,
        meal_balance_gain=2.7,
    )
    engine = total._TotalTrajectories(model, rows)
    g, u = engine.predict([p[k] for k in engine.active])
    predicted = engine.residual(g, u).transpose(0, 2, 1)
    generic = FitProblem(model, rows, substeps=64)
    actual = generic.residual(generic._vector(p)).reshape(predicted.shape)
    np.testing.assert_allclose(actual, predicted, rtol=1e-8, atol=2e-6)
    assert model["formula"]["initializers"]["S"] == "Gp+Gt"
    assert model["graph"]["variable_types"]["Gp"] == "algebraic"
    assert set(generic.engine.forcings).isdisjoint({"Gp", "I", "U"})
    assert set(generic.engine.readings) == {"Gp", "Gt", "I"}
    assert not set(total.FIXED_SOURCE) & p.keys()


def test_mass_change_under_known_constant_source_and_tissue_change():
    rows, fold = _fold()
    m = total.seed_model(rows, fold, "short")
    for r in rows:
        r["inputs"]["meal_rate_g_per_min"] = [0] * 4
        r["auxiliaries"].update(EGP=[3.0] * 4, E=[0.25] * 4)
    p = m["formula"]["parameters"]
    p.update(
        afcoef9=0,
        afcoef10=0,
        disp_tissue=0,
        init_tissue_gain=0,
        afcoef12=0,
        afcoef16=0,
        afcoef11=2,
    )
    data = total._TotalTrajectories(m, rows)
    g, u = data.predict([p[k] for k in data.active])
    for i, r in enumerate(rows):
        t = np.asarray(r["time"])
        tissue = r["auxiliaries"]["Gt"]
        s0 = r["targets"]["Gp"][0] + tissue[0]
        np.testing.assert_allclose(g[i, :: plasma.SUBSTEPS] + tissue, s0 + 0.75 * t)
    np.testing.assert_allclose(u, 2.0)


def test_warm_mapping_is_locked_to_this_fold_and_insulin():
    rows, fold = _fold()
    m = plasma.seed_model(rows, fold, "fading", "short")
    m["formula"]["parameters"]["init_tissue_gain"] = -1.73
    warm = total.seed_model(rows, fold, "warm", m)
    assert warm["formula"]["parameters"]["init_tissue_gain"] == -1.73
    assert "meal_balance_offset" in m["formula"]["parameters"]
    with pytest.raises(ValueError, match="warm start"):
        total.seed_model(rows, fold, "short", m)
    f = deepcopy(m)
    f["fit_training_ids"][0] = "train_011"
    with pytest.raises(ValueError, match="fold"):
        total.seed_model(rows, fold, "warm", f)
    f = deepcopy(m)
    f["formula"]["parameters"]["pos_tau"] = 100
    with pytest.raises(ValueError, match="insulin"):
        total.seed_model(rows, fold, "warm", f)


def test_fit_preserves_order_domain_and_budget():
    rows, fold = _fold()
    attempt = total.fit(rows, fold, "long", budget=14)
    data = total._TotalTrajectories(attempt["model"], rows)
    p = attempt["model"]["formula"]["parameters"]
    g, u = data.predict([p[n] for n in data.active])
    assert attempt["forward_evaluations"] <= 14
    assert np.min(g) >= 0 and np.min(u) >= 1
    assert np.min(u[data.high, 0] - u[data.low, 0] - data.required) >= -1e-9
    assert attempt["training_nmse"] == pytest.approx(np.mean(data.residual(g, u) ** 2))
    with pytest.raises(ValueError, match="15 events"):
        total.fit(rows[:-1], fold, "short", budget=14)
