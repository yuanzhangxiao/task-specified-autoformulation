"""Latent-state SINDy: its library, its replay, its fitting and its handoff."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autoformalism.baselines import latent_ode as lo
from autoformalism.baselines import latent_sindy as ls
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal import noise_exchange as nx

REPO = Path(__file__).resolve().parents[1]


def test_the_library_is_constant_linear_then_quadratic():
    terms = ls.library_terms(3)
    assert terms[:4] == ((), (0,), (1,), (2,))
    assert terms[4:] == ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    z = np.array([2.0, 3.0, 5.0])
    np.testing.assert_allclose(
        ls.theta(z), [1, 2, 3, 5, 4, 6, 10, 9, 15, 25], rtol=0, atol=0
    )


def _decay_model() -> dict:
    """dy/dt = -y + u in unit scaling, one inert latent state: y = 1 - exp(-t)."""
    n_terms = len(ls.library_terms(3))  # y, h1, u
    coefficients = np.zeros((n_terms, 2))
    coefficients[1, 0] = -1.0  # y
    coefficients[3, 0] = 1.0  # u
    return {
        "targets": ["y"],
        "inputs": ["u"],
        "tau": 1.0,
        "standardization": {
            "target_mean": [0.0],
            "target_sd": [1.0],
            "input_mean": [0.0],
            "input_sd": [1.0],
        },
        "coefficients": coefficients.tolist(),
        "mask": np.ones_like(coefficients).tolist(),
        "initializer": [[0.0, 0.0, 0.0]],
    }


def test_the_replay_integrates_the_saved_equations():
    time = np.linspace(0.0, 4.0, 41)
    replay = ls.LatentSINDy(_decay_model())
    for method in ("Radau", "BDF"):
        y, h = replay.simulate(time, np.array([0.0]), np.ones((41, 1)), method=method)
        np.testing.assert_allclose(y[:, 0], 1 - np.exp(-time), rtol=1e-6, atol=1e-7)
        assert np.all(h == 0.0)
    equations = replay.equations()
    assert equations["y"] == "dy/dt = (-1*y +1*u) / 1"
    assert equations["h1"] == "dh1/dt = (0) / 1"


def test_inputs_are_scaled_by_their_largest_training_deviation():
    data = lo.Arrays(
        np.arange(3.0),
        np.zeros((2, 3, 1)),
        np.array([[[0.0], [10.0], [0.0]], [[0.0], [0.0], [2.0]]]),
    )
    scale = ls.scaling(data)
    mean = 2.0
    assert scale["input_mean"] == [pytest.approx(mean)]
    assert scale["input_sd"] == [pytest.approx(10.0 - mean)]


def _toy(batch: int, seed: int) -> lo.Arrays:
    rng = np.random.default_rng(seed)
    time = np.round(np.arange(0.0, 6.0001, 0.2), 10)
    inputs = np.zeros((batch, len(time), 1))
    targets = np.zeros((batch, len(time), 1))
    for b in range(batch):
        inputs[b, time >= rng.uniform(1.0, 3.0), 0] = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-0.5, 0.5)
        for k in range(len(time)):
            targets[b, k, 0] = y
            if k + 1 < len(time):
                for _ in range(40):
                    y += 0.2 / 40 * -(y - inputs[b, k, 0])
    return lo.Arrays(time, targets, inputs)


def test_a_short_fit_keeps_finite_sparse_checkpoints():
    pytest.importorskip("torch")
    config = ls.LatentSINDyConfig(
        latent_dimensions=(1,), iterations=30, refit_iterations=10, evaluate_every=5
    )
    out = ls.train(_toy(6, 0), _toy(3, 1), 1, 0, config)
    assert out["dense"]["validation_nmse_rk4"] < np.inf
    kept = [s["kept_terms"] for s in out["thresholds"]]
    assert kept == sorted(kept, reverse=True)
    for sparse in out["thresholds"]:
        snapshot = sparse["snapshot"]
        mask = np.asarray(snapshot["mask"])
        assert np.all(np.asarray(snapshot["coefficients"])[mask == 0] == 0)


def test_the_handoff_forecasts_a_latent_sindy_model():
    context = ValidationContext.model_validate(
        {
            "targets": ["Gp", "I"],
            "auxiliaries": ["Uii"],
            "external_inputs": ["meal_rate_g_per_min", "insulin_pmol_per_kg_min"],
            "fixed_covariates": [],
        }
    )
    targets, inputs = lo.channels(context)
    n_terms = len(ls.library_terms(len(targets) + 1 + len(inputs)))
    coefficients = np.zeros((n_terms, 3))
    coefficients[1, 0] = coefficients[2, 1] = -0.5  # each target decays
    model = {
        "kind": "latent_sindy",
        "context": context.model_dump(mode="json"),
        "targets": list(targets),
        "inputs": list(inputs),
        "tau": 30.0,
        "standardization": {
            "target_mean": [150.0, 30.0],
            "target_sd": [20.0, 10.0],
            "input_mean": [0.0, 0.0, 2.0],
            "input_sd": [5.0, 1.0, 1.0],
        },
        "coefficients": coefficients.tolist(),
        "mask": np.ones_like(coefficients).tolist(),
        "initializer": np.zeros((1, 6)).tolist(),
    }
    n = 31
    query = {
        "time": list(range(n)),
        "targets0": {"Gp": 190.0, "I": 50.0},
        "auxiliaries": {"Uii": [2.0] * n},
        "external_inputs": {
            "meal_rate_g_per_min": [0.0] * n,
            "insulin_pmol_per_kg_min": [0.0] * n,
        },
    }
    radau = nx.forecast(model, nx.blind_trajectory(query), "Radau")
    bdf = nx.forecast(model, nx.blind_trajectory(query), "BDF")
    t = np.arange(n)
    np.testing.assert_allclose(
        radau["Gp"], 150 + 40 * np.exp(-0.5 * t / 30.0), rtol=1e-6
    )
    np.testing.assert_allclose(radau["I"], bdf["I"], rtol=1e-5)


def test_the_noise_plan_is_frozen_with_the_default_settings():
    plan = json.loads((REPO / "configs" / "t2_noise_latent_sindy_v1.json").read_text())
    assert plan["status"] == "frozen_before_training"
    assert plan["settings"] == ls.LatentSINDyConfig().model_dump(mode="json")
    assert plan["seeds"] == [0, 1, 2]
    assert plan["cells"] == [f"acquisition_{i:02d}" for i in range(4)]
