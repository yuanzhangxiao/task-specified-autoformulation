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


NINE_CASE_INPUTS = Path(
    "/Users/yuanzhangxiao/Projects/autoformalism/outputs/release/phase-b-nine-case-v1"
)


def test_the_phase_b_plan_copies_the_intervention_settings_on_eight_cells():
    phase_c = json.loads(
        (REPO / "configs" / "latent_ode_intervention_v1.json").read_text()
    )
    phase_b = json.loads(
        (REPO / "configs" / "latent_ode_phase_b_nine_case_v1.json").read_text()
    )
    assert phase_b["status"] == "frozen_before_training"
    assert phase_b["settings"] == phase_c["settings"]
    assert phase_b["seeds"] == phase_c["seeds"] == [0, 1, 2]
    assert len(phase_b["cells"]) == 8
    assert all(c.startswith("phase_b_") for c in phase_b["cells"])
    assert not any("cstr" in c for c in phase_b["cells"])
    assert phase_b["release_protocol"] == "phase-b-nine-case-inputs-1"
    assert phase_b["context"] == "raw_agent"
    assert not phase_b["test_data_opened"] and not phase_b["private_reference_opened"]


def test_the_runner_reads_a_phase_c_cell_unless_the_plan_names_the_nine_cases(
    monkeypatch,
):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "runner", REPO / "scripts" / "run_latent_ode_baseline.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls = []
    monkeypatch.setattr(runner, "load_cell", lambda r, n: calls.append(("c", n)))
    monkeypatch.setattr(
        runner.nine, "load_cell", lambda r, n, context: calls.append(("b", n, context))
    )
    phase_c = next(iter(runner.BASELINE_CELLS))
    runner.load(Path("r"), phase_c, None, None)
    runner.load(Path("r"), phase_c, "phase-c-development-2", None)
    runner.load(Path("r"), "phase_b_x", runner.nine.PROTOCOL, "raw_agent")
    assert calls == [("c", phase_c), ("c", phase_c), ("b", "phase_b_x", "raw_agent")]
    with pytest.raises(ValueError, match="not a Phase C baseline cell"):
        runner.load(Path("r"), "phase_b_x", None, None)


@pytest.mark.skipif(
    not NINE_CASE_INPUTS.exists(), reason="staged nine-case inputs absent"
)
def test_the_phase_b_plan_seals_every_cell_with_the_staged_receipt(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "runner", REPO / "scripts" / "run_latent_ode_baseline.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = REPO / "configs" / "latent_ode_phase_b_nine_case_v1.json"
    record = runner.plan(tmp_path / "root", NINE_CASE_INPUTS, config)
    frozen = json.loads(config.read_text())
    assert record["release_protocol"] == "phase-b-nine-case-inputs-1"
    assert set(record["cells"]) == set(frozen["cells"])
    assert len(record["tasks"]) == 8 * 3 * 3
    for cell in record["cells"].values():
        assert (
            cell["identity"]["inputs_receipt_sha256"] == frozen["inputs_receipt_sha256"]
        )
    t1 = record["cells"]["phase_b_dalla_man_t1_canonical_named_easy"]
    assert t1["targets"] == ["Gp"] and t1["inputs"][0] == "meal_event_g"
