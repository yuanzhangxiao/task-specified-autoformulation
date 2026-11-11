"""External baselines go through the intervention study's own scoring code."""

from __future__ import annotations

import numpy as np
import pytest

from autoformalism.benchmarks.phase_b_generation import PhaseBProtocol
from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate
from autoformalism.schemas import CandidateModel
from scripts import iclr_anchor_fidelity as anchor
from scripts import score_baseline_interventions as scoring

CSTR = anchor.fits.CSTR
CONTEXT = ValidationContext.model_validate(
    {
        "targets": ["T"],
        "auxiliaries": [],
        "external_inputs": ["Cf", "Tf", "Tjf"],
        "fixed_covariates": [],
    }
)
#: A toy reactor law that only the baseline grammar reads (``Max``).
LAW = "-0.8*(T - 350) + 6*Tjf - 4*Cf + Max(0*Tf, -1)"


def _row(semantics="continuous_time", **model):
    candidate = equation_candidate("sindy", {"T": LAW}, CONTEXT)
    return {
        "source_id": "sindy:toy:0",
        "source_protocol": scoring.PROTOCOL,
        "source_bundle_sha256": "0" * 64,
        "benchmark_id": CSTR,
        "method": "sindy",
        "repetition": 0,
        "status": "ready",
        "direct_readouts": {},
        "model": {
            "candidate": candidate.model_dump(mode="json"),
            "context": CONTEXT.model_dump(mode="json"),
            "parameters": {},
            "initials": {},
            "semantics": semantics,
            "grammar": "baseline",
            **model,
        },
    }


def _inputs(spec, time):
    """The schedule's input deviations on a grid, for the toy reference."""
    values = np.zeros((len(time), 3))
    if spec.get("kind") == "step":
        on = (time >= spec["start"]) & (time < spec["end"])
        values[on] = spec["value"]
    return values


def _reference(item, time):
    """The toy law itself, integrated finely, as the reference response."""
    u = _inputs(item["protocol"]["specification"], time)
    temperature = np.empty_like(time)
    temperature[0] = 350.0
    for i in range(len(time) - 1):
        x = temperature[i]
        for _ in range(100):
            x += (
                (time[i + 1] - time[i])
                / 100
                * (-0.8 * (x - 350) + 6 * u[i, 2] - 4 * u[i, 0])
            )
        temperature[i + 1] = x
    hidden = np.linspace(0.0, 1.0, len(time))
    return {
        "time": time.tolist(),
        "T": temperature.tolist(),
        **{h: hidden.tolist() for h in anchor.fits.HIDDEN[CSTR]},
        **{n: u[:, j].tolist() for j, n in enumerate(("Cf", "Tf", "Tjf"))},
    }


def _trajectory(time):
    return Trajectory(
        trajectory_id="toy",
        time=time,
        targets={"T": np.full(len(time), 350.0)},
        external_inputs={n: np.zeros(len(time)) for n in ("Cf", "Tf", "Tjf")},
        auxiliaries={},
        fixed_covariates={},
        derivatives={},
    )


def test_a_continuous_baseline_is_read_with_its_own_grammar():
    row = _row()
    with pytest.raises(ModelValidationError, match="Max"):
        compile_candidate(
            CandidateModel.model_validate(row["model"]["candidate"]), CONTEXT
        )
    time = np.linspace(0.0, 2.0, 21)
    curves = scoring.predict(row, _trajectory(time), "Radau")
    assert np.allclose(curves["T"], 350.0)


def test_a_native_map_runs_only_on_its_own_sample_interval():
    time = np.linspace(0.0, 2.0, 21)
    row = _row("native_increment", increment_step=0.1)
    curves = scoring.predict(row, _trajectory(time), "Radau")
    assert len(curves["T"]) == len(time)
    with pytest.raises(ValueError, match="sample interval"):
        scoring.predict(
            _row("native_increment", increment_step=0.2), _trajectory(time), "Radau"
        )


def test_the_study_scores_a_baseline_with_its_own_metrics(tmp_path):
    schedules = anchor.cases(CSTR, "confirmation")
    time = np.round(np.arange(0.0, 30.05, 0.1), 10)
    refs = {k: _reference(item, time) for k, item in schedules.items()}
    with scoring.baseline_predictions():
        result = anchor.evaluate_one(
            (tmp_path, 0, _row(), schedules, refs, "plan", {"T": 1.0})
        )
    assert anchor.predict.__name__ == "predict"  # restored afterwards
    targets = {r["case"]: r for r in result["scores"] if r["readout"] == "target"}
    assert set(targets) == {"feed_low", "feed_high", "jacket_cool", "jacket_warm"}
    for row in targets.values():
        assert row["status"] == "complete"
        assert row["direction_correct"] is True
        assert row["effect_rse"] < 1e-3
    hidden = {r["status"] for r in result["scores"] if r["readout"] != "target"}
    assert hidden == {"physical_correspondence_unavailable", "no_correspondence"}


def test_a_contract_mismatch_is_refused():
    row = _row()
    row["model"]["context"] = {**row["model"]["context"], "auxiliaries": ["C"]}
    with pytest.raises(ValueError, match="contract"):
        scoring.predict(row, _trajectory(np.linspace(0.0, 1.0, 11)), "Radau")


def test_freeze_refuses_unknown_methods_and_cells(tmp_path):
    with pytest.raises(ValueError, match="unknown methods"):
        scoring.freeze(tmp_path, tmp_path, {"oracle": tmp_path}, [CSTR])
    with pytest.raises(ValueError, match="unknown methods"):
        scoring.freeze(tmp_path, tmp_path, {"d3": tmp_path}, ["phase_c_other"])


def _latent_row(rng):
    """A small random latent ODE for the reactor cell: T plus two latents."""

    def layer(rows, cols):
        return {
            "weight": (0.1 * rng.standard_normal((rows, cols))).tolist(),
            "bias": (0.1 * rng.standard_normal(rows)).tolist(),
        }

    model = {
        "semantics": "latent_neural_ode",
        "context": CONTEXT.model_dump(mode="json"),
        "targets": ["T"],
        "inputs": ["Cf", "Tf", "Tjf"],
        "tau": 3.0,
        "standardization": {
            "target_mean": [350.0],
            "target_sd": [5.0],
            "input_mean": [0.0, 0.0, 0.0],
            "input_sd": [1.0, 1.0, 1.0],
        },
        "network": {
            "dynamics": [layer(8, 6), layer(8, 8), layer(3, 8)],
            "encoder": [layer(4, 4), layer(2, 4)],
        },
    }
    return {**_row(), "method": "latent_ode", "model": model}


def test_a_latent_ode_gives_its_targets_and_latent_coordinates():
    row = _latent_row(np.random.default_rng(0))
    time = np.linspace(0.0, 2.0, 21)
    radau = scoring.predict(row, _trajectory(time), "Radau")
    bdf = scoring.predict(row, _trajectory(time), "BDF")
    assert set(radau) == {"T", "latent.h0", "latent.h1"}
    assert radau["T"][0] == pytest.approx(350.0)
    assert np.max(np.abs(np.subtract(radau["T"], bdf["T"]))) < 1e-5


def test_the_study_probes_a_latent_ode_like_its_own_models(tmp_path):
    schedules = anchor.cases(CSTR, "confirmation")
    time = np.round(np.arange(0.0, 30.05, 0.1), 10)
    refs = {k: _reference(item, time) for k, item in schedules.items()}
    row = _latent_row(np.random.default_rng(1))
    with scoring.baseline_predictions():
        result = anchor.evaluate_one(
            (tmp_path, 0, row, schedules, refs, "plan", {"T": 1.0})
        )
    targets = [r for r in result["scores"] if r["readout"] == "target"]
    assert {r["status"] for r in targets} <= {"complete", "negligible_reference_effect"}
    probes = {r["status"] for r in result["scores"] if r["readout"] == "oracle_affine"}
    assert "calibration_unavailable" not in probes


def _suite():
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "cstr_mechanism_probes_v1.json"
    )
    return json.loads(path.read_text())


def test_the_probe_suite_copies_the_study_base_and_calibration():
    suite = _suite()
    table = scoring.suite_schedules(CSTR, suite)
    study = anchor.cases(CSTR, "confirmation")
    for name in suite["calibration"]:
        assert table[name] == study[name]
    base = study["train_zero"]["protocol"]
    probes = [k for k, v in table.items() if v["role"] == "contrast"]
    assert len(probes) == len(suite["probes"]) == 12
    for item in table.values():
        if item["role"] == "train_calibration":
            continue
        PhaseBProtocol.model_validate(item["protocol"])
        changed = {k for k in base if item["protocol"][k] != base[k]}
        assert changed <= {"protocol_id", "split", "specification"}
        if item["role"] == "contrast":
            assert item["control"] == "basal"
            low, high = item["window"]
            assert 0.0 <= low < high <= base["duration"]


def test_a_probe_never_shifts_a_hidden_initial_value():
    for probe_case in _suite()["probes"]:
        shift = probe_case["specification"].get("initial_shift", [0.0, 0.0, 0.0])
        assert shift[0] == shift[2] == 0.0


def test_an_unfrozen_suite_or_another_cell_is_refused():
    suite = _suite()
    with pytest.raises(ValueError, match="not frozen"):
        scoring.suite_schedules(CSTR, {**suite, "status": "draft"})
    with pytest.raises(ValueError, match="not frozen"):
        scoring.suite_schedules(anchor.fits.T2, suite)


def test_the_study_models_join_on_their_canonical_cell(tmp_path):
    from autoformalism.rebuttal.prefit_replay import sealed_write

    path = tmp_path / "frozen_models.json"
    sealed_write(
        path,
        {
            "protocol": "iclr-controlled-search-models-1",
            "rows": [
                {"benchmark_id": "sensor_00", "method": "full", "repetition": 0},
                {"benchmark_id": "acquisition_00", "method": "full", "repetition": 0},
            ],
        },
    )
    rows = scoring.study_rows(path, CSTR)
    assert [(r["benchmark_id"], r["source_benchmark_id"]) for r in rows] == [
        (CSTR, "sensor_00")
    ]
    assert rows[0]["source_protocol"] == "iclr-controlled-search-models-1"
