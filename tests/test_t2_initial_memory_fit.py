"""Train-only memory interventions and full-graph propagation on synthetic folds."""

import numpy as np
import pytest

from scripts import t2_initial_memory_fit as memory
from scripts.positive_insulin_refit_kernel import FitProblem


def _synthetic_fold():
    ids = [f"train_{i:03}" for i in range(16) if i != 11]
    rows = []
    for i, name in enumerate(ids):
        gt0 = 110 + i
        rows.append(
            {
                "trajectory_id": name,
                "time": [0.0, 1.0, 2.0, 3.0],
                "covariates": {},
                "inputs": {"meal_rate_g_per_min": [0.0, 1.0, 2.0, 0.0]},
                "targets": {
                    "Gp": [180.0, 181.0, 183.0, 185.0],
                    "I": [25.0, 23.0, 21.0, 20.0],
                    "U": [2 + 0.01 * i, 2.3, 2.4, 2.2],
                },
                "auxiliaries": {
                    "Gt": [gt0, gt0 + 2, gt0 + 4, gt0 + 6],
                    "EGP": [1.1, 1.1, 1.2, 1.2],
                    "Uii": [1.0, 1.0, 1.0, 1.0],
                },
            }
        )
    fold = {
        "centers": {"I": 22.0, "Gp": 180.0, "U": 2.22, "Gt_ref": 110.0},
        "scales": {"I": 5.0, "Gp": 20.0, "U": 0.5},
        "fit_ids": ids,
        "insulin": {
            "sources": ["I"],
            "knots": {"Gt": [110.0]},
            "scalar_model": {"weights": [2.0], "tau": 4.0},
            "formula": "-I/pos_tau+pos_Gt_flat",
            "parameters": {"pos_tau": 4.0, "pos_Gt_flat": 2.0},
        },
    }
    return rows, fold


@pytest.mark.parametrize("arm", ["persistent", "fading"])
def test_triangular_propagation_matches_free_rollout(arm):
    rows, fold = _synthetic_fold()
    model = memory.seed_model(rows, fold, arm, "long")
    p = model["formula"]["parameters"]
    p.update(
        {
            "afcoef12": 1.0,
            "afcoef16": 0.02,
            "afcoef10": 0.8,
            "afcoef9": 0.1,
            "init_tissue_gain": 1.5,
        }
    )
    engine = memory._Trajectories(model, rows)
    x = np.array([p[k] for k in engine.active])
    g, u = engine.predict(x)
    e = engine.residual(g, u).transpose(0, 2, 1)
    original = FitProblem(model, rows, substeps=32)
    expected = original.residual(original._vector(p)).reshape(e.shape)
    assert np.max(np.abs(e - expected)) < 2e-6


def test_only_initial_tissue_offset_fades_and_zero_decay_nests():
    rows, fold = _synthetic_fold()
    models = [
        memory.seed_model(rows, fold, a, "long") for a in ("persistent", "fading")
    ]
    for m in models:
        m["formula"]["parameters"].update(
            {
                "afcoef12": 0.0,
                "init_tissue_gain": 2.0,
                "afcoef10": 0.8,
                "afcoef16": 0.02,
            }
        )
    persistent, fade = [memory._Trajectories(m, rows) for m in models]
    v = np.array([models[0]["formula"]["parameters"][n] for n in persistent.active])
    gp, u = persistent.predict(v)
    gzero, uzero = fade.predict(np.r_[v, 0.0])
    np.testing.assert_allclose(gp, gzero, rtol=0, atol=0)
    np.testing.assert_allclose(u, uzero, rtol=0, atol=0)
    _, ufade = fade.predict(np.r_[v, 0.3])
    expected = 2 * fade.initial_tissue * (np.exp(-0.3 * fade.t[::2]) - 1)
    np.testing.assert_allclose(ufade - u, expected, atol=1e-12)


def test_fold_identity_guard_and_constrained_training():
    rows, fold = _synthetic_fold()
    with pytest.raises(ValueError, match="15 events"):
        memory.seed_model(rows[:-1], fold, "persistent", "long")
    with pytest.raises(ValueError, match="nested start"):
        memory.seed_model(rows, fold, "fading", "nested")
    fit = memory.fit(rows, fold, "persistent", "long", budget=10)
    assert fit["training_nmse"] <= fit["source_nmse"]
    model = fit["model"]
    data = memory._Trajectories(model, rows)
    _, u = data.predict([model["formula"]["parameters"][n] for n in data.active])
    assert np.min(u) > 1
    assert np.min(u[data.high, 0] - u[data.low, 0] - data.required) > -1e-9
    near = memory.seed_model(rows, fold, "fading", "nested", model)
    assert near["formula"]["parameters"]["memory_rate"] > 0
    assert model["formula"]["parameters"] == {
        k: v for k, v in near["formula"]["parameters"].items() if k != "memory_rate"
    }
