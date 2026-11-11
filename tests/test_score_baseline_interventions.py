"""External baselines go through the intervention study's own scoring code."""

from __future__ import annotations

import numpy as np
import pytest

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
