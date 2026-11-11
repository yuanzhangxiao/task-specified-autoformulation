"""The latent neural ODE baseline: what it reads, how it trains, how it replays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autoformalism.baselines import latent_ode as lo
from autoformalism.expressions import ValidationContext

REPO = Path(__file__).resolve().parents[1]


def _toy(batch: int, seed: int) -> lo.Arrays:
    """y' = -(y - u) under random steps in u, on a grid of 0.1 over [0, 6]."""
    rng = np.random.default_rng(seed)
    time = np.round(np.arange(0.0, 6.0001, 0.1), 10)
    inputs = np.zeros((batch, len(time), 1))
    targets = np.zeros((batch, len(time), 1))
    for b in range(batch):
        on = rng.uniform(1.0, 3.0)
        inputs[b, time >= on, 0] = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-0.5, 0.5)
        for k in range(len(time)):
            targets[b, k, 0] = y
            if k + 1 < len(time):
                for _ in range(50):
                    y += 0.1 / 50 * -(y - inputs[b, k, 0])
    return lo.Arrays(time, targets, inputs)


def test_the_model_reads_targets_inputs_and_auxiliaries_only():
    context = ValidationContext.model_validate(
        {
            "targets": ["Gp", "I"],
            "auxiliaries": ["Uii"],
            "external_inputs": ["meal", "insulin"],
            "fixed_covariates": [],
        }
    )
    assert lo.channels(context) == (("Gp", "I"), ("meal", "insulin", "Uii"))
    lagged = context.model_copy(update={"lagged_targets": ("Gp",)})
    with pytest.raises(ValueError, match="lagged"):
        lo.channels(lagged)


def test_the_error_is_pooled_per_target_over_training_variance():
    observed = np.zeros((2, 3, 2))
    predicted = observed + np.array([1.0, 2.0])
    assert lo.nmse(predicted, observed, [1.0, 2.0]) == pytest.approx(1.0)
    assert lo.nmse(predicted, observed, [2.0, 2.0]) == pytest.approx(0.625)


def test_training_keeps_the_best_validation_checkpoint():
    pytest.importorskip("torch")
    config = lo.LatentODEConfig(
        latent_dimensions=(2,), width=16, iterations=60, evaluate_every=10
    )
    result = lo.train(_toy(8, 0), _toy(3, 1), 2, 0, config)
    scores = [h["validation_nmse_rk4"] for h in result["history"]]
    assert result["best"]["validation_nmse_rk4"] == min(scores)
    assert scores[-1] < scores[0]
    assert result["best"]["network"] is not None


def test_the_numpy_replay_is_the_trained_model():
    torch = pytest.importorskip("torch")
    config = lo.LatentODEConfig(latent_dimensions=(3,), width=16, iterations=5)
    data = _toy(2, 2)
    result = lo.train(data, data, 3, 1, config)
    model = {
        "targets": ["y"],
        "inputs": ["u"],
        "tau": result["tau"],
        "standardization": result["standardization"],
        "network": result["best"]["network"],
    }
    replayed = lo.LatentODE(model).simulate(
        data.time, data.targets[0, 0], data.inputs[0]
    )[0]
    net = lo._network(1, 1, 3, config, torch).double()
    state = net.state_dict()
    for part in ("dynamics", "encoder"):
        for i, layer in enumerate(model["network"][part]):
            state[f"{part}.{i}.weight"] = torch.tensor(layer["weight"])
            state[f"{part}.{i}.bias"] = torch.tensor(layer["bias"])
    net.load_state_dict(state)
    scale = model["standardization"]
    y, u = lo._standardized(data, scale, torch)
    with torch.no_grad():
        rk4 = lo._rk4_rollout(net, y[:1], u[:1], data.time, model["tau"], 16, torch)
    rk4 = rk4[0].numpy() * scale["target_sd"][0] + scale["target_mean"][0]
    assert np.max(np.abs(rk4[:, 0] - replayed[:, 0])) < 1e-6


def test_the_frozen_plan_uses_the_default_settings_on_the_study_cells():
    plan = json.loads(
        (REPO / "configs" / "latent_ode_intervention_v1.json").read_text()
    )
    assert plan["status"] == "frozen_before_training"
    assert plan["settings"] == lo.LatentODEConfig().model_dump(mode="json")
    assert plan["seeds"] == [0, 1, 2]
    assert plan["cells"] == [
        "phase_c_cstr_controlled_reactor_mechanism_canonical_named_hard_reset_v1",
        "phase_c_dalla_man_t2_canonical_named_hard_rates_v1",
    ]
    assert not plan["test_data_opened"] and not plan["private_reference_opened"]
