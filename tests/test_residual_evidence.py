"""Measured residual patterns are aligned, bounded and training-only."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.residual_feedback import ResidualSettings
from autoformalism.search.residual_evidence import (
    build_residual_evidence,
    validate_residual_evidence,
)

CONTEXT = ValidationContext(targets=("y",), external_inputs=("u",))
PARAMETERS = {"a": 0.5, "init_z": 2.0}


def fixture(count=4, times=None):
    times = np.arange(7.0) if times is None else np.asarray(times, dtype=float)
    data = DatasetSplit(
        SplitName.TRAIN,
        tuple(
            Trajectory(
                trajectory_id=f"train_{i}",
                time=times,
                targets={
                    "y": np.resize(
                        np.array([0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0]), len(times)
                    )
                    + i
                },
                external_inputs={
                    "u": np.zeros(len(times))
                    if i == 0
                    else np.arange(len(times), dtype=float)
                    if i == 1
                    else np.ones(len(times))
                },
                auxiliaries={},
                fixed_covariates={},
                derivatives={"forbidden_hidden": np.zeros(len(times))},
            )
            for i in range(count)
        ),
        "training-fixture",
    )
    current = {
        t.trajectory_id: {
            "time": t.time.tolist(),
            "predictions": {"y": (t.targets["y"] + i * 0.2 + 0.1).tolist()},
        }
        for i, t in enumerate(data.trajectories)
    }
    previous = {
        t.trajectory_id: {
            "time": t.time.tolist(),
            "predictions": {"y": (t.targets["y"] + 0.5).tolist()},
        }
        for t in data.trajectories
    }
    return data, current, previous


def packet(data, current, previous=None, **kwargs):
    return build_residual_evidence(
        data,
        CONTEXT,
        current,
        candidate_sha256="a" * 64,
        parameters=PARAMETERS,
        previous=previous,
        numerical_status={
            "feedback_status": "budget_limited_unresolved",
            "budget_exhausted": True,
            "validation": {"normalized_mse": 999},
            "hidden": "SECRET",
        },
        **kwargs,
    )


def test_known_bias_scale_regimes_and_two_fit_comparison():
    data, current, previous = fixture()
    result = packet(data, current, previous)
    scale = np.std(np.concatenate([t.targets["y"] for t in data.trajectories]))
    assert result.rows[0].normalized_mse == pytest.approx((0.1 / scale) ** 2)
    assert result.rows[0].normalized_signed_bias == pytest.approx(0.1 / scale)
    assert result.rows[0].previous_normalized_mse == pytest.approx((0.5 / scale) ** 2)
    assert [r.sampled_input_regime for r in result.rows] == [
        "zero",
        "varying",
        "constant_nonzero",
        "constant_nonzero",
    ]
    assert result.rows[0].observed.turning_point_count == 2
    assert result.rows[0].predicted.turning_point_count == 2
    for row in result.rows:
        assert (
            row.observed.common_change_tolerance
            == row.predicted.common_change_tolerance
        )
        assert sum(w.sample_count for w in row.windows) == row.sample_count
        reconstructed = (
            sum(w.sample_count * w.normalized_mse for w in row.windows)
            / row.sample_count
        )
        assert reconstructed == pytest.approx(row.normalized_mse)
    assert result.parameter_sha256 == public.content_sha256(PARAMETERS)
    assert result.training_content_sha256 == public.content_sha256(
        public.pack_split(data)
    )
    assert "SECRET" not in result.model_dump_json()
    assert "validation" not in result.numerical_status.model_dump()
    for d in result.details:
        for s in d.samples:
            assert s.normalized_residual == pytest.approx(
                (s.predicted - s.observed) / scale
            )
    validate_residual_evidence(
        result.model_dump(mode="json"),
        candidate_sha256="a" * 64,
        parameters=PARAMETERS,
        training_content_sha256=result.training_content_sha256,
    )


def test_predicted_flat_response_is_described_not_called_wrong_structure():
    data, current, previous = fixture(count=1)
    current["train_0"]["predictions"]["y"] = [1.0] * 7
    result = packet(data, current, previous)
    assert result.rows[0].predicted.range == 0
    assert result.rows[0].observed.range == 2
    assert result.rows[0].predicted.turning_point_count == 0
    assert not result.numerical_status.structural_failure_established


@pytest.mark.parametrize("split", [SplitName.TEST, SplitName.VALIDATION])
def test_heldout_rejected_before_reading_rows(split):
    with pytest.raises(ValueError, match="train split only"):
        packet(DatasetSplit(split, None, "heldout"), None)


@pytest.mark.parametrize(
    "change", ["time", "length", "nan", "hidden", "missing", "target"]
)
def test_bad_replays_do_not_become_partial_feedback(change):
    data, current, previous = fixture()
    if change == "time":
        current["train_0"]["time"][1] += 0.1
    elif change == "length":
        current["train_0"]["predictions"]["y"].pop()
    elif change == "nan":
        current["train_0"]["predictions"]["y"][0] = float("nan")
    elif change == "hidden":
        current["train_0"]["states"] = {"z": [1.0] * 7}
    elif change == "missing":
        current.pop("train_0")
    else:
        current["train_0"]["predictions"]["z"] = [1.0] * 7
    with pytest.raises(ValueError):
        packet(data, current, previous)


def test_previous_replay_failure_is_explicitly_absent():
    data, current, _ = fixture()
    result = packet(data, current)
    assert result.previous_normalized_mse is None
    assert all(r.previous_predicted is None for r in result.rows)
    assert all(s.previous_prediction is None for d in result.details for s in d.samples)


def test_nonuniform_time_windows_use_time_and_include_every_sample():
    data, current, previous = fixture(count=1, times=[0, 0.01, 0.1, 0.2, 0.3, 8, 10])
    result = packet(data, current, previous)
    windows = result.rows[0].windows
    assert [w.label for w in windows] == ["early", "late"]
    assert [w.sample_count for w in windows] == [5, 2]


def test_constant_two_sample_trace_and_noise_tolerance():
    data, current, _ = fixture(count=1, times=[0, 1])
    t = replace(data.trajectories[0], targets={"y": np.array([2.0, 2.0])})
    data = replace(data, trajectories=(t,))
    current["train_0"]["predictions"]["y"] = [2.0, 2.0 + 1e-10]
    result = packet(data, current)
    assert result.rows[0].observed.turning_point_count == 0
    assert result.rows[0].predicted.turning_point_count == 0
    assert result.rows[0].training_scale > 0


def test_bounded_selection_keeps_best_worst_zero_and_omission_counts():
    data, current, previous = fixture(count=20)
    result = packet(
        data,
        current,
        previous,
        settings=ResidualSettings(maximum_details=3, maximum_rows=5, maximum_samples=5),
    )
    assert result.total_rows == 20 and result.omitted_rows == 15
    assert len(result.details) == 3
    rows = {r.evidence_id: r for r in result.rows}
    assert {"train_0", "train_19"} <= {
        rows[d.row_id].trajectory_id for d in result.details
    }
    assert all(len(d.samples) == 5 and d.omitted_samples == 2 for d in result.details)
    assert all(
        d.samples[0].index == 0 and d.samples[-1].index == 6 for d in result.details
    )


@pytest.mark.parametrize("binding", ["packet", "parameter", "candidate", "data"])
def test_stale_or_tampered_feedback_rejected(binding):
    data, current, previous = fixture()
    value = packet(data, current, previous).model_dump(mode="json")
    params, candidate, digest = PARAMETERS, "a" * 64, value["training_content_sha256"]
    if binding == "packet":
        value["rows"][0]["normalized_mse"] += 1
    elif binding == "parameter":
        params = {**PARAMETERS, "init_z": 0.0}
    elif binding == "candidate":
        candidate = "b" * 64
    else:
        digest = "c" * 64
    with pytest.raises(ValueError, match="digest or candidate"):
        validate_residual_evidence(
            value,
            candidate_sha256=candidate,
            parameters=params,
            training_content_sha256=digest,
        )


def test_status_allowlist_rejects_nested_heldout_metadata():
    data, current, _ = fixture()
    with pytest.raises(ValueError):
        build_residual_evidence(
            data,
            CONTEXT,
            current,
            candidate_sha256="a" * 64,
            parameters=PARAMETERS,
            numerical_status={"feedback_status": {"validation": 5}},
        )


def test_shape_ids_and_selection_are_deterministic():
    data, current, previous = fixture()
    assert packet(data, current, previous) == packet(
        data, copy.deepcopy(current), copy.deepcopy(previous)
    )


def test_turn_counts_preserve_slow_dense_oscillations():
    from autoformalism.search.residual_evidence import _turns

    sparse = np.sin(np.linspace(0, 4 * np.pi, 81))
    dense = np.sin(np.linspace(0, 4 * np.pi, 10001))
    assert len(_turns(sparse, 0.02)) == len(_turns(dense, 0.02)) == 4
    assert _turns(np.linspace(0, 1, 10001), 0.02) == []
    assert _turns(np.array([0, 1, 0.995, 1.001, 0]), 0.02) == [3]
