"""Packaging and forecasting frozen baseline picks for the T2 noise handoff."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal import noise_exchange as nx
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate

CONTEXT = ValidationContext.model_validate(
    {
        "targets": ["Gp", "I"],
        "auxiliaries": ["Uii"],
        "external_inputs": ["meal_rate_g_per_min", "insulin_pmol_per_kg_min"],
        "fixed_covariates": [],
    }
)


def _query(n: int = 61) -> dict:
    time = np.arange(n, dtype=float)
    return {
        "time": time.tolist(),
        "targets0": {"Gp": 180.0, "I": 40.0},
        "auxiliaries": {"Uii": [2.0] * n},
        "external_inputs": {
            "meal_rate_g_per_min": [1.0 if 10 <= t < 20 else 0.0 for t in time],
            "insulin_pmol_per_kg_min": [0.5] * n,
        },
    }


def _continuous(equations: dict[str, str]) -> dict:
    return {
        "kind": "continuous",
        "candidate": equation_candidate("sindy", equations, CONTEXT).model_dump(
            mode="json"
        ),
        "context": CONTEXT.model_dump(mode="json"),
        "parameters": {},
        "initials": {},
    }


def test_a_blind_trajectory_carries_only_the_initial_readings():
    tr = nx.blind_trajectory(_query())
    assert all(np.all(v == v[0]) for v in tr.targets.values())
    assert tr.targets["Gp"][0] == 180.0 and tr.targets["I"][0] == 40.0
    assert tr.external_inputs["meal_rate_g_per_min"][15] == 1.0


def test_independent_solvers_agree_on_a_continuous_model():
    model = _continuous(
        {
            "Gp": "-0.05*(Gp - 150) + 3*meal_rate_g_per_min - 0.01*I - 0.1*Uii",
            "I": "-0.1*(I - 20) + 2*insulin_pmol_per_kg_min",
        }
    )
    tr = nx.blind_trajectory(_query())
    radau, bdf = nx.forecast(model, tr, "Radau"), nx.forecast(model, tr, "BDF")
    for name in ("Gp", "I"):
        assert radau[name][0] == pytest.approx(tr.targets[name][0])
        np.testing.assert_allclose(radau[name], bdf[name], rtol=1e-5, atol=1e-6)
    assert radau["Gp"][25] > radau["Gp"][9]  # the meal raises glucose
    with pytest.raises(ValueError, match="Radau or BDF"):
        nx.forecast(model, tr, "native")


def test_the_validation_metric_is_mse_over_training_variance():
    """A model that holds its initial values scores its own definition."""
    model = _continuous({"Gp": "0*Gp", "I": "0*I"})

    class Split:
        def __init__(self, trajectories):
            self.trajectories = trajectories

    rng = np.random.default_rng(0)

    def trajectory(i):
        time = np.arange(31, dtype=float)
        return Trajectory(
            trajectory_id=f"t{i}",
            time=time,
            targets={
                "Gp": 150 + rng.normal(0, 5, 31).cumsum(),
                "I": 30 + rng.normal(0, 2, 31).cumsum(),
            },
            auxiliaries={"Uii": np.full(31, 2.0)},
            external_inputs={
                "meal_rate_g_per_min": np.zeros(31),
                "insulin_pmol_per_kg_min": np.zeros(31),
            },
            fixed_covariates={},
            derivatives={},
        )

    train = [trajectory(i) for i in range(3)]
    validation = [trajectory(i) for i in range(3, 5)]

    class Cell:
        context = CONTEXT

        class dataset:
            pass

    Cell.dataset.train = Split(train)
    Cell.dataset.validation = Split(validation)
    expected = np.mean(
        [
            np.mean((v.targets[t] - v.targets[t][0]) ** 2)
            / np.var(np.concatenate([tr.targets[t] for tr in train]))
            for v in validation
            for t in ("Gp", "I")
        ]
    )
    assert nx.validation_nmse(model, Cell, "Radau") == pytest.approx(expected, rel=1e-6)


def test_a_native_map_runs_only_its_own_updates_on_its_own_grid():
    model = {
        "kind": "native_map",
        "candidate": {
            "candidate_id": "toy",
            "parent_candidate_id": None,
            "states": [
                {"name": "Gp", "kind": "observed"},
                {"name": "I", "kind": "observed"},
                {"name": "Uii", "kind": "observed"},
            ],
            "state_equations": [
                {"state": "Gp", "rhs": "-0.01*(Gp - 150) + meal_rate_g_per_min"},
                {"state": "I", "rhs": "-0.02*(I - 20)"},
                {"state": "Uii", "rhs": "0*Uii"},
            ],
            "observation_mappings": [
                {"channel": name, "expression": name} for name in ("Gp", "I")
            ],
            "initial_conditions": [
                {"state": name, "scope": "global", "expression": name}
                for name in ("Gp", "I", "Uii")
            ],
        },
        "context": CONTEXT.model_dump(mode="json"),
        "parameters": {},
        "networks": [],
        "network_weights": None,
        "step": 1.0,
    }
    tr = nx.blind_trajectory(_query())
    curves = nx.forecast(model, tr, "native")
    # x[k+1] = x[k] + f(x[k]): the first update is exact arithmetic.
    assert curves["Gp"][1] == pytest.approx(180.0 - 0.01 * 30.0)
    assert curves["I"][1] == pytest.approx(40.0 - 0.02 * 20.0)
    with pytest.raises(ValueError, match="one-step"):
        nx.forecast(model, tr, "Radau")
    coarse = _query()
    coarse["time"] = [2.0 * t for t in coarse["time"]]
    with pytest.raises(ValueError, match="sample interval"):
        nx.forecast(model, nx.blind_trajectory(coarse), "native")


def test_the_model_file_is_reproducible_and_typed(tmp_path):
    record = {"format": nx.FORMAT, "method": "m", "cell": "acquisition_01", "seed": 0}
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    nx._write_model(first, record)
    nx._write_model(second, record)
    assert first.read_bytes() == second.read_bytes()
    assert nx.read_model(first) == record
    nx._write_model(first, {**record, "format": "other"})
    with pytest.raises(ValueError, match="not a T2 noise"):
        nx.read_model(first)


def test_method_code_must_be_the_pinned_revision():
    with pytest.raises(ValueError, match="pinned"):
        nx.pinned({"implementation_revision": "0" * 40})


OUTPUTS = os.environ.get("AF_OUTPUTS")


@pytest.mark.skipif(
    not OUTPUTS, reason="set AF_OUTPUTS to rehearse on sealed T2-hard picks"
)
def test_readers_reproduce_recorded_validation_on_t2_hard():
    from autoformalism.rebuttal.phase_c_baselines import load_cell

    root = Path(OUTPUTS)
    cell = load_cell(
        root.parent / "artifacts" / "phase-c-development-final-v2",
        "phase_c_dalla_man_t2_canonical_named_hard_rates_v1",
    )
    for kind, method, run in (
        ("classical", "sindy", "phase-c-classical-v1"),
        ("vendored", "llm_ode", "js2-runs/phase-c-llm-ode-hosted-120b-v1"),
        ("d3_family", "d3", "js2-runs/phase-c-d3-hosted-120b-v2"),
        ("latent_ode", "latent_ode", "latent-ode-intervention-v1"),
    ):
        model, record = nx.READERS[kind](
            {"kind": kind, "method": method, "root": str(root / run)}, cell, 0
        )
        solver = "native" if model["kind"] == "native_map" else "Radau"
        assert nx.validation_nmse(model, cell, solver) == pytest.approx(
            record["recorded_validation"][1], rel=1e-6
        )
