"""Descriptive M38 extraction keeps sampling and mechanism ambiguity explicit."""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.fitting import interaction_feedback as bridge
from autoformalism.fitting import interaction_mismatches as mismatch
from autoformalism.fitting import interaction_response as response
from autoformalism.fitting import staged_feedback as staged
from tests.test_fitting_feedback import fitted as fitted_fixture
from tests.test_interaction_response import single_row, snapshot

fitted = fitted_fixture


def packet(source):
    """Synthetic trusted packet boundary; raw observation arrays use real summaries."""
    summary = response.build(source)
    return SimpleNamespace(
        response=summary,
        evidence_catalog={
            "numerics": "Numerics",
            **dict.fromkeys(summary.evidence_ids),
        },
    )


def one(observed, predicted=None, *, time=None, inputs=None):
    return packet(
        single_row(
            np.asarray(observed, dtype=float),
            inputs or {},
            time=time,
            predictions=observed if predicted is None else predicted,
        )
    )


def selected(summary, code, availability="measured"):
    return [
        row
        for row in summary.findings
        if row.code == code and row.availability == availability
    ]


def test_accurate_full_response_has_no_measured_discrepancy():
    result = mismatch.extract(one([0, 0, 4, 2, 0, 0, 0]))
    assert result.status == "available" and result.targets == ("y",)
    assert not result.findings
    assert result.coverage["y"].overview.normalized_mse == 0
    assert result.coverage["y"].selected_examples == 1
    assert not result.omitted_trajectory_target_examples
    assert not result.omitted_findings


@pytest.mark.parametrize("direction", [1, -1])
def test_signed_bias_uses_all_samples_and_training_scale(direction):
    observed = np.asarray([0, 1, 4, 2, 0, 0, 0], dtype=float) * direction
    result = mismatch.extract(one(observed, observed + 0.3 * direction))
    records = selected(result, "signed_bias")
    assert len(records) == 2
    assert {r.trajectory_id for r in records} == {None, "one"}
    for row in records:
        assert row.difference == pytest.approx(0.3 * direction / np.std(observed))
        assert row.unit == "training_sd"
        assert row.difference == row.predicted - row.observed
    assert not selected(result, "excursion_magnitude")


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("factor", [0.5, 1.5])
def test_magnitude_is_absolute_excursion_for_positive_and_negative_responses(
    direction, factor
):
    observed = np.asarray([0, 1, 4, 2, 0, 0, 0], dtype=float) * direction
    result = mismatch.extract(one(observed, factor * observed))
    record = selected(result, "excursion_magnitude")[0]
    assert record.observed == pytest.approx(4 / np.std(observed))
    assert record.predicted == pytest.approx(4 * factor / np.std(observed))
    assert (record.difference > 0) == (factor > 1)
    assert not selected(result, "excursion_direction")


def test_opposite_excursions_do_not_get_magnitude_or_timing_comparisons():
    observed = np.asarray([0, 1, 4, 2, 0, 0, 0], dtype=float)
    result = mismatch.extract(one(observed, -observed))
    assert len(selected(result, "excursion_direction")) == 1
    assert not selected(result, "excursion_magnitude")
    for code in ("peak_timing", "half_return"):
        assert not selected(result, code)
        withheld = selected(result, code, "withheld")[0]
        assert withheld.difference is None
        assert "EXCURSION_DIRECTIONS_NOT_COMPARABLE" in withheld.reason_codes


@pytest.mark.parametrize("observed_flat", [True, False])
def test_absent_or_spurious_excursion_is_not_called_opposite_direction(observed_flat):
    signal = [0, 0, 4, 2, 0, 0, 0]
    flat = np.zeros(7)
    result = mismatch.extract(one(flat, signal) if observed_flat else one(signal, flat))
    assert len(selected(result, "excursion_presence")) == 1
    assert not selected(result, "excursion_direction")
    assert not selected(result, "excursion_magnitude")
    assert len(selected(result, "peak_timing", "withheld")) == 1


@pytest.mark.parametrize(
    "signal,reason",
    [
        ([0, 0, 0, 0, 0, 0, 0], "FLAT"),
        ([0, 0, 4, 4, 0, 0, 0], "PLATEAU"),
        ([0, 1, 2, 3, 4, 5, 6], "BOUNDARY"),
    ],
)
def test_flat_tied_and_boundary_peaks_withhold_timing_without_inventing_zero(
    signal, reason
):
    result = mismatch.extract(one(signal))
    assert not [r for r in result.findings if r.availability == "measured"]
    for code in ("peak_timing", "half_return"):
        record = selected(result, code, "withheld")[0]
        assert record.difference is None
        assert record.observed is None and record.predicted is None
        assert f"OBSERVED_PEAK_{reason}" in record.reason_codes


def test_censored_half_return_is_not_a_zero_time_or_a_decay_constant():
    result = mismatch.extract(one([0, 0, 4, 2, 0, 0, 0], [0, 0, 4, 3, 3, 3, 3]))
    record = selected(result, "half_return", "withheld")[0]
    assert record.observed == 1
    assert record.predicted is None and record.difference is None
    assert record.reason_codes == ("PREDICTED_HALF_RETURN_NOT_REACHED",)
    assert any("not an exponential time constant" in x for x in record.caveats)


@pytest.mark.parametrize("direction", [1, -1])
def test_timing_differences_use_trajectory_duration_and_time_units(direction):
    observed = np.asarray([0, 0, 4, 2, 0, 0, 0], dtype=float) * direction
    predicted = np.asarray([0, 0, 0, 2, 4, 3, 0], dtype=float) * direction
    source = one(observed, predicted, time=[100, 101, 102, 103, 104, 105, 106])
    result = mismatch.extract(source)
    peak = selected(result, "peak_timing")[0]
    returned = selected(result, "half_return")[0]
    assert (peak.observed, peak.predicted, peak.difference) == (102, 104, 2)
    assert (returned.observed, returned.predicted, returned.difference) == (1, 2, 1)
    assert peak.threshold == pytest.approx(0.3)
    assert peak.unit == returned.unit == "trajectory_time"
    result = mismatch.extract(source, mismatch.Policy(time_difference_fraction=0.5))
    assert not selected(result, "peak_timing") and not selected(result, "half_return")


def test_sparse_time_windows_keep_exact_metric_and_empty_middle_explicit():
    source = one(
        [0, 1, 2, 3, 2, 1, 0],
        [0, 2, 4, 6, 6, 6, 6],
        time=[0, 0.01, 0.1, 0.2, 0.3, 8, 10],
    )
    result = mismatch.extract(source)
    windows = selected(result, "window_error")
    expected = source.response.examples[0].windows
    assert len(windows) == len(expected) == 2
    assert result.omitted_windows == 0
    assert result.coverage["y"].empty_selected_windows == {"one": ("middle",)}
    for record, window in zip(windows, expected, strict=True):
        assert record.difference == window.normalized_mse
        assert record.predicted == window.normalized_mse
        assert record.observed == 0
        assert record.evidence_ids == (window.evidence_id,)
        assert "input-specific" in " ".join(record.caveats)


@pytest.mark.parametrize(
    "inputs,expected",
    [
        ({"u": (0, 1, 0, 1, 0, 0, 0)}, "Repeated input"),
        (
            {"u": (0, 1, 0, 0, 0, 0, 0), "v": (0, 0, 1, 0, 0, 0, 0)},
            "Multiple changing",
        ),
    ],
)
def test_multiple_or_repeated_events_do_not_become_input_specific_delays(
    inputs, expected
):
    result = mismatch.extract(
        one([0, 0, 4, 2, 0, 0, 0], [0, 0, 0, 2, 4, 3, 0], inputs=inputs)
    )
    record = selected(result, "peak_timing")[0]
    assert record.difference == 2
    assert any(expected in caveat for caveat in record.caveats)
    assert "delay" not in record.interpretation


def test_all_targets_and_omitted_denominators_survive_selected_examples():
    source = packet(snapshot())
    result = mismatch.extract(source)
    assert result.targets == ("a", "z") and set(result.coverage) == {"a", "z"}
    assert result.omitted_trajectory_target_examples == 2
    for target, row in result.coverage.items():
        assert row.overview == source.response.target_overview[target]
        assert row.overview.trajectory_count == 3
        assert row.selected_examples == 2 and row.omitted_examples == 1
        records = [f for f in result.findings if f.target == target]
        assert row.measured_findings == sum(
            r.availability == "measured" for r in records
        )
    assert {r for f in result.findings for r in f.evidence_ids} <= set(
        source.evidence_catalog
    )
    assert result == mismatch.extract(source)
    mismatch.MismatchSummary.model_validate(result.model_dump(mode="json"))


def test_unavailable_has_no_mismatch_and_preserves_all_targets_and_reasons():
    source = snapshot()
    source.packet.diagnostics.output_status = "unavailable"
    result = mismatch.extract(packet(source))
    assert result.status == "unavailable" and not result.findings
    assert result.reason_codes == ("OUTPUT_NOT_VERIFIED",)
    assert result.targets == ("a", "z")
    assert all(r.selected_examples == 0 for r in result.coverage.values())
    assert all(r.overview.normalized_mse is None for r in result.coverage.values())
    assert result.omitted_trajectory_target_examples == 6


@pytest.mark.parametrize(
    "field",
    [
        "normalized_difference_threshold",
        "time_difference_fraction",
        "window_nmse_threshold",
    ],
)
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "0.1"])
def test_nonfinite_or_nonpositive_threshold_rejected(field, value):
    with pytest.raises(ValidationError):
        mismatch.Policy(**{field: value})


def test_timing_fraction_cannot_exceed_duration():
    with pytest.raises(ValidationError):
        mismatch.Policy(time_difference_fraction=1.1)


def test_summary_rejects_target_trimming_invented_evidence_and_false_counts():
    source = packet(snapshot())
    result = mismatch.extract(source).model_dump(mode="json")
    result["coverage"].pop("a")
    with pytest.raises(ValidationError, match="every target"):
        mismatch.MismatchSummary.model_validate(result)
    source.evidence_catalog.pop("numerics")
    with pytest.raises(ValueError, match="absent"):
        mismatch.extract(source)


def test_verified_real_fit_extracts_without_new_rollout_or_optimization(
    fitted, tmp_path, monkeypatch
):
    diagnostic = tmp_path / "diagnostic"
    staged.run(*fitted, diagnostic)
    exported = tmp_path / "exported"
    bridge.export_packet(diagnostic, exported)
    source = bridge.completed(diagnostic, exported)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}
    monkeypatch.setattr(
        staged, "evaluate_stage", lambda *a, **kw: pytest.fail("rollout")
    )
    monkeypatch.setattr(
        staged.checks, "Rollouts", lambda *a, **kw: pytest.fail("rollout")
    )
    monkeypatch.setattr(
        staged.checks, "least_squares", lambda *a, **kw: pytest.fail("optimization")
    )
    result = mismatch.extract(source)
    assert result.status == "available"
    assert result.coverage["y"].overview.normalized_mse is not None
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}
    assert not hasattr(result, "model_view")
