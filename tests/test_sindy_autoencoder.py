"""The delay SINDy autoencoder: its windows, its replay, its fitting and its plan."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from autoformalism.baselines import latent_ode as lo
from autoformalism.baselines import latent_sindy as ls
from autoformalism.baselines import sindy_autoencoder as sa

REPO = Path(__file__).resolve().parents[1]


def test_windows_are_causal_and_start_at_rest():
    y = np.array([[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(
        sa.windows(y, 3)[0], [[1, 1, 1], [2, 1, 1], [3, 2, 1], [4, 3, 2]]
    )
    np.testing.assert_array_equal(
        sa.derivative_windows(y, np.arange(4.0), 3)[0],
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]],
    )


def test_the_horizon_marks_samples_past_the_record():
    values, exists = sa.horizon(np.array([[1.0, 2.0, 3.0]]), 2)
    np.testing.assert_array_equal(values[0], [[1, 2], [2, 3], [3, 3]])
    np.testing.assert_array_equal(exists, [[True, True], [True, True], [True, False]])


def _decay_model() -> dict:
    """z = [y, 0] from the window's newest entry; dz1/dt = -z1 + u."""
    coefficients = np.zeros((len(ls.library_terms(3)), 2))  # z1, z2, u
    coefficients[1, 0] = -1.0  # z1
    coefficients[3, 0] = 1.0  # u
    return {
        "targets": ["y"],
        "inputs": ["u"],
        "delays": 3,
        "standardization": {
            "target_mean": [0.0],
            "target_sd": [1.0],
            "input_mean": [0.0],
            "input_sd": [1.0],
        },
        # The hidden unit stays positive, where ELU is the identity.
        "encoder": [
            {"weight": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "bias": [10.0, 0.0]},
            {"weight": [[1.0, 0.0], [0.0, 0.0]], "bias": [-10.0, 0.0]},
        ],
        "decoder": [],
        "coefficients": coefficients.tolist(),
        "mask": np.ones_like(coefficients).tolist(),
    }


def test_the_replay_starts_at_rest_and_integrates_the_saved_equations():
    time = np.linspace(0.0, 4.0, 41)
    replay = sa.SindyAutoencoder(_decay_model())
    for method in ("Radau", "BDF"):
        y, h = replay.simulate(time, np.array([0.0]), np.ones((41, 1)), method=method)
        np.testing.assert_allclose(y[:, 0], 1 - np.exp(-time), rtol=1e-6, atol=1e-7)
        assert np.all(h == 0.0)
    assert replay.equations()["z1"] == "dz1/dt = -1*z1 +1*u"
    assert replay.equations()["z2"] == "dz2/dt = 0"


def test_the_replay_refuses_a_model_that_does_not_match_its_window():
    model = _decay_model()
    model["delays"] = 4
    with pytest.raises(ValueError, match="window"):
        sa.SindyAutoencoder(model)


def _toy(batch: int, seed: int) -> lo.Arrays:
    rng = np.random.default_rng(seed)
    time = np.round(np.arange(0.0, 6.0001, 0.2), 10)
    inputs = np.zeros((batch, len(time), 1))
    targets = np.zeros((batch, len(time), 1))
    for b in range(batch):
        inputs[b, time >= rng.uniform(1.0, 3.0), 0] = rng.uniform(-1.0, 1.0)
        y = 0.0
        for k in range(len(time)):
            targets[b, k, 0] = y
            if k + 1 < len(time):
                for _ in range(40):
                    y += 0.2 / 40 * -(y - inputs[b, k, 0])
    return lo.Arrays(time, targets, inputs)


def test_a_short_fit_thresholds_for_good_and_replays_as_it_trained():
    pytest.importorskip("torch")
    config = sa.SindyAutoencoderConfig(
        delays=(3,),
        widths=(8,),
        epochs=4,
        threshold=0.02,
        threshold_every=2,
        refinement_epochs=2,
        evaluate_every=1,
        batch_size=16,
    )
    train, validation = _toy(6, 0), _toy(3, 1)
    out = sa.train(train, validation, 2, 3, 0, config)
    kept = [h["kept_terms"] for h in out["history"] if "kept_terms" in h]
    assert kept == sorted(kept, reverse=True) and out["kept_terms"] == kept[-1]
    snapshot = out["best"]["snapshot"]
    assert out["best"]["epoch"] >= config.epochs and snapshot is not None
    mask = np.asarray(snapshot["mask"])
    assert int(mask.sum()) == out["kept_terms"]
    assert np.all(np.asarray(snapshot["coefficients"])[mask == 0] == 0)
    model = {
        "targets": ["y"],
        "inputs": ["u"],
        "delays": 3,
        "standardization": out["standardization"],
        **snapshot,
    }
    replay = sa.SindyAutoencoder(model)
    predicted = np.stack(
        [
            replay.simulate(
                validation.time, validation.targets[b, 0], validation.inputs[b]
            )[0]
            for b in range(3)
        ]
    )
    radau = lo.nmse(predicted, validation.targets, out["standardization"]["target_sd"])
    assert math.isclose(radau, out["best"]["validation_nmse_rk4"], rel_tol=1e-3)


def test_the_reactor_plan_is_frozen_with_the_default_settings():
    plan = json.loads(
        (REPO / "configs" / "cstr_hard_sindy_autoencoder_v1.json").read_text()
    )
    latent_ode = json.loads(
        (REPO / "configs" / "latent_ode_intervention_v1.json").read_text()
    )
    assert plan["status"] == "frozen_before_training"
    assert plan["settings"] == sa.SindyAutoencoderConfig().model_dump(mode="json")
    assert plan["seeds"] == [0, 1, 2]
    assert plan["cells"] == [
        "phase_c_cstr_controlled_reactor_mechanism_canonical_named_hard_reset_v1"
    ]
    for key in ("release_protocol", "release_summary_sha256"):
        assert plan[key] == latent_ode[key]
