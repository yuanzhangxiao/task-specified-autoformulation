"""Offline M37 response evidence preserves complete training arrays and guards."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.expressions import ValidationContext
from autoformalism.fitting import interaction_response as response
from autoformalism.fitting import portable_feedback as portable
from autoformalism.fitting import staged_feedback as staged
from autoformalism.fitting import staged_feedback_fixtures as fixtures
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory


def snapshot(rows=None, errors=None, channels=("z", "a")):
    """Synthetic verified-source boundary with unequal lengths and reversed channels."""
    if rows is None:
        rows = tuple(
            PublicTrajectory(
                trajectory_id=tid,
                time=tuple(np.linspace(0, 6, n)),
                targets={
                    "a": tuple(np.linspace(1, 4, n)),
                    "z": tuple(np.sin(np.linspace(0, np.pi, n)) * 5 + 20),
                },
                external_inputs={"u": (0.0, 1.0, *([0.0] * (n - 2)))},
            )
            for tid, n in (("train_c", 4), ("train_a", 7), ("train_b", 3))
        )
    targets = tuple(sorted(rows[0].targets))
    training = PublicSplit(name="train", fingerprint="synthetic", rows=rows)
    scales = {
        c: max(float(np.std(np.concatenate([r.targets[c] for r in rows]))), 1e-12)
        for c in channels
    }
    errors = errors or {
        r.trajectory_id: {
            c: np.full(len(r.time), (i + 1) * (1 if c == "a" else -2), dtype=float)
            for c in channels
        }
        for i, r in enumerate(rows)
    }
    residual = np.concatenate(
        [
            np.column_stack([errors[r.trajectory_id][c] for c in channels]).ravel()
            for r in rows
        ]
    ).tolist()
    operation = {
        "status": "complete",
        "value": {
            "metadata": {"channels": list(channels), "scales": scales},
            "residual": residual,
            "sampled_states": {"DO_NOT_EXPOSE_HIDDEN_STATE": [999999]},
        },
    }
    return portable.Snapshot(
        problem=SimpleNamespace(
            training=training,
            request=SimpleNamespace(
                context=ValidationContext(
                    targets=targets, external_inputs=tuple(rows[0].external_inputs)
                )
            ),
        ),
        fit=None,
        backend={"DO_NOT_EXPOSE_VALIDATION": 12345},
        policy=staged.Policy(),
        packet=SimpleNamespace(diagnostics=SimpleNamespace(output_status="verified")),
        operations={"primary": operation, "alternate": deepcopy(operation)},
        plan=(),
        source_sha256="a" * 64,
        provenance={"DO_NOT_EXPOSE_REFERENCE": "forbidden"},
    )


def test_unequal_lengths_and_channel_order_preserve_full_array_errors():
    source = snapshot()
    result = response.build(source)
    assert result.targets == ("a", "z") and result.status == "available"
    assert len(result.examples) == 4
    assert result.omitted_trajectory_target_examples == 2
    assert result.omitted_windows == 6
    assert result.target_overview["a"].normalized_mse == pytest.approx(
        (4 + 7 * 4 + 3 * 9) / 14
    )
    assert result.target_overview["z"].normalized_signed_bias == pytest.approx(
        (-8 - 28 - 18) / 14
    )
    for target, factor in (("a", 1), ("z", -2)):
        worst, best = [r for r in result.examples if r.target == target]
        assert (worst.trajectory_id, best.trajectory_id) == ("train_b", "train_c")
        assert (worst.selected_as, best.selected_as) == ("worst", "contrasting_best")
        assert worst.normalized_mse == 9 * factor**2
        assert best.normalized_signed_bias == factor
        original = next(
            r
            for r in source.problem.training.rows
            if r.trajectory_id == best.trajectory_id
        )
        assert best.predicted.initial == pytest.approx(
            original.targets[target][0] + factor * best.training_scale
        )
        assert sum(w.sample_count for w in best.windows) == best.sample_count
    serialized = result.model_dump_json()
    assert "DO_NOT_EXPOSE" not in serialized
    assert result.evidence_ids == tuple(
        identity
        for row in result.examples
        for identity in (row.evidence_id, *(w.evidence_id for w in row.windows))
    )
    response.ResponseSummary.model_validate(result.model_dump(mode="json"))


@pytest.mark.parametrize("status", ["unverified", "unavailable", "skipped"])
def test_unverified_never_reads_or_fabricates_prediction_arrays(status):
    source = snapshot()
    source.packet.diagnostics.output_status = status
    source.operations["primary"]["value"] = None
    result = response.build(source)
    assert result.status == "unavailable"
    assert result.reason_codes == ("OUTPUT_NOT_VERIFIED",)
    assert not result.examples and not result.evidence_ids
    assert set(result.target_overview) == {"a", "z"}
    assert all(r.normalized_mse is None for r in result.target_overview.values())
    assert result.omitted_trajectory_target_examples == 6
    assert result.omitted_windows == 18


@pytest.mark.parametrize("stage", ["primary", "alternate"])
@pytest.mark.parametrize("status", ["missing", "interrupted", "unavailable"])
def test_incomplete_stage_is_explicit_unavailable(stage, status):
    source = snapshot()
    if status == "missing":
        source.operations.pop(stage)
    else:
        source.operations[stage] = {"status": status, "value": None}
    result = response.build(source)
    assert result.status == "unavailable"
    assert result.reason_codes == (f"{stage.upper()}_NOT_COMPLETE",)
    assert result.targets == ("a", "z") and not result.examples


@pytest.mark.parametrize("change", ["short", "long", "matrix", "nan", "inf"])
@pytest.mark.parametrize("stage", ["primary", "alternate"])
def test_malformed_complete_residuals_fail_closed(change, stage):
    source = snapshot()
    value = source.operations[stage]["value"]
    if change == "short":
        value["residual"].pop()
    elif change == "long":
        value["residual"].append(0)
    elif change == "matrix":
        value["residual"] = [value["residual"]]
    else:
        value["residual"][0] = float(change)
    with pytest.raises(ValueError, match="residual"):
        response.build(source)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True, "1", 123.0])
def test_invalid_or_unrelated_scale_is_rejected(scale):
    source = snapshot()
    source.operations["primary"]["value"]["metadata"]["scales"]["a"] = scale
    with pytest.raises(ValueError, match="scale"):
        response.build(source)


@pytest.mark.parametrize(
    "channels", [["a"], ["a", "a"], ["a", "hidden"], ["a", "z", "extra"]]
)
def test_metadata_cannot_trim_or_replace_public_targets(channels):
    source = snapshot()
    source.operations["primary"]["value"]["metadata"]["channels"] = channels
    with pytest.raises(ValueError, match="context target set"):
        response.build(source)


def test_training_channel_set_and_alternate_layout_must_match():
    source = snapshot()
    source.problem.request.context = ValidationContext(
        targets=("a",), external_inputs=("u",)
    )
    with pytest.raises(ValueError, match="training target set"):
        response.build(source)
    source = snapshot()
    source.operations["alternate"]["value"]["metadata"]["channels"].reverse()
    with pytest.raises(ValueError, match="layouts differ"):
        response.build(source)


@pytest.mark.parametrize("split", ["val", "test"])
def test_heldout_rejected_before_rows_are_read(split):
    class Heldout:
        name = split

        @property
        def rows(self):
            pytest.fail("heldout rows read")

    source = snapshot()
    source.problem.training = Heldout()
    with pytest.raises(ValueError, match="train split only"):
        response.build(source)


@pytest.mark.parametrize("per_target", [0, 3, -1, True, 1.5, "2"])
def test_example_bound_is_explicit(per_target):
    with pytest.raises(ValueError, match="per_target"):
        response.build(snapshot(), per_target=per_target)


def test_stable_ids_and_selection_under_input_order_and_packet_compaction():
    source = snapshot()
    full = response.build(source)
    rows = source.problem.training.rows
    errors = {
        r.trajectory_id: {
            c: np.full(len(r.time), (i + 1) * (1 if c == "a" else -2), dtype=float)
            for c in ("a", "z")
        }
        for i, r in enumerate(rows)
    }
    reordered = snapshot(tuple(reversed(rows)), errors, channels=("a", "z"))
    assert response.build(reordered) == full
    compact = response.build(source, per_target=1)
    assert compact.target_overview == full.target_overview
    assert compact.omitted_trajectory_target_examples == 4
    assert set(compact.evidence_ids) < set(full.evidence_ids)
    assert all(r.selected_as == "worst" for r in compact.examples)


def single_row(values, inputs, time=None, predictions=None):
    """Construct one observed output with independently specified predictions."""
    time = tuple(np.arange(len(values), dtype=float)) if time is None else tuple(time)
    row = PublicTrajectory(
        trajectory_id="one",
        time=time,
        targets={"y": tuple(values)},
        external_inputs=inputs,
    )
    scale = max(float(np.std(values)), 1e-12)
    residual = (
        np.zeros(len(values))
        if predictions is None
        else (np.asarray(predictions) - values) / scale
    )
    return snapshot((row,), {"one": {"y": residual}}, channels=("y",))


def test_nonuniform_windows_include_every_sample_and_do_not_fabricate_empty_middle():
    source = single_row(
        np.array([0, 1, 2, 3, 2, 1, 0], dtype=float),
        {},
        time=[0, 0.01, 0.1, 0.2, 0.3, 8, 10],
        predictions=[0, 2, 4, 6, 6, 6, 6],
    )
    result = response.build(source)
    row = result.examples[0]
    assert [w.label for w in row.windows] == ["early", "late"]
    assert [w.sample_count for w in row.windows] == [5, 2]
    assert row.empty_windows == ("middle",)
    assert not any(ref.endswith("middle") for ref in result.evidence_ids)
    assert row.normalized_mse == pytest.approx(
        sum(w.sample_count * w.normalized_mse for w in row.windows) / row.sample_count
    )
    assert row.normalized_signed_bias == pytest.approx(
        sum(w.sample_count * w.normalized_signed_bias for w in row.windows)
        / row.sample_count
    )


def test_common_observed_tolerance_peak_delay_and_half_return_definitions():
    source = single_row(
        np.array([0, 0, 4, 2, 0, 0, 0], dtype=float),
        {"u": (0, 1, 0, 0, 0, 0, 0)},
        predictions=[0, 0, 0, 2, 4, 2, 0],
    )
    row = response.build(source).examples[0]
    assert row.input_peak_delays == {"u": (1.0, 3.0)}
    assert row.peak_timing_error == 2
    assert row.observed.half_return_elapsed == 1
    assert row.predicted.half_return_elapsed == 1
    assert row.common_change_tolerance == 1e-8 + 0.04
    source = single_row(np.zeros(4), {}, predictions=[0, 1e-10, 0, 0])
    row = response.build(source).examples[0]
    assert row.training_scale == 1e-12
    assert row.observed.peak_status == row.predicted.peak_status == "flat"
    assert row.predicted.peak_time is None and row.peak_timing_error is None


@pytest.mark.parametrize(
    "inputs,repeated,multiple",
    [
        ({"u": (0, 1, 0, 1, 0, 0, 0)}, True, False),
        ({"u": (0, 1, 0, 0, 0, 0, 0), "v": (0, 0, 1, 0, 0, 0, 0)}, False, True),
    ],
)
def test_ambiguous_input_events_keep_flags_without_delays(inputs, repeated, multiple):
    source = single_row(np.array([0, 0, 2, 4, 2, 1, 0]), inputs)
    row = response.build(source).examples[0]
    assert not row.input_peak_delays
    assert row.inputs["u"].changing
    assert row.inputs["u"].repeated_events is repeated
    assert row.multiple_changing_inputs is multiple


def test_schema_rejects_invented_fields_missing_targets_or_unavailable_numbers():
    result = response.build(snapshot()).model_dump(mode="json")
    result["hidden_states"] = {}
    with pytest.raises(ValidationError, match="Extra inputs"):
        response.ResponseSummary.model_validate(result)
    result.pop("hidden_states")
    result["target_overview"].pop("a")
    with pytest.raises(ValidationError, match="every target"):
        response.ResponseSummary.model_validate(result)
    with pytest.raises(ValidationError, match="availability"):
        response.TargetOverview(
            status="unavailable", trajectory_count=1, sample_count=2, normalized_mse=0
        )


def test_real_portable_snapshot_needs_no_further_integration_or_optimization(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(staged.public, "_source_identity", lambda: "response-test")
    fitted = fixtures.fit_fixture("piecewise_inactive_guard", tmp_path / "fit")
    directory = tmp_path / "feedback"
    staged.run(
        staged.training_problem(fitted["problem"]),
        staged.adapter.FitResult.model_validate(fitted["fit"]),
        fitted["backend"],
        directory,
    )
    source = portable.load_snapshot(directory)
    monkeypatch.setattr(
        staged, "evaluate_stage", lambda *a, **kw: pytest.fail("integration")
    )
    monkeypatch.setattr(
        staged.checks, "Rollouts", lambda *a, **kw: pytest.fail("model execution")
    )
    monkeypatch.setattr(
        staged.checks, "least_squares", lambda *a, **kw: pytest.fail("optimization")
    )
    result = response.build(source)
    assert result.status == "available"
    assert result.examples[0].normalized_mse == pytest.approx(
        source.packet.diagnostics.replay_training_nmse
    )
    assert result == response.build(
        replace(source, fit=None, backend={}, provenance={})
    )
