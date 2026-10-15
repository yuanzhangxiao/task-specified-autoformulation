"""Bounded training-response evidence from a previously verified M32/M33 snapshot.

Call ``portable_feedback.load_snapshot`` first, then ``build(snapshot)``. This
adapter does not verify source seals itself and never integrates or optimizes a
model. It reconstructs observed-output predictions from the saved primary
normalized residual, retaining its trajectory/time/channel ordering and scales.
No sampled model states, held-out arrays, or evaluator references are accessed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.fitting.portable_feedback import SUMMARY_RTOL, Snapshot
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.schemas.public_fitting import PublicTrajectory
from autoformalism.schemas.response_evidence import InputResponse, ResponseShape
from autoformalism.search.response_evidence import shape

INTERPRETATION = (
    "Predictions are reconstructed from saved primary normalized residuals and "
    "observed training targets at the retained parameter vector. No integration "
    "or optimization is performed; reconstruction can introduce roundoff.",
    "NMSE averages squared normalized residuals over every sampled point. Signed "
    "bias is prediction minus observation in global training-standard-deviation "
    "units, with the diagnostic scale floor of 1e-12. Target overviews weight "
    "trajectories by sample count, including omitted examples.",
    "Early, middle and late windows divide elapsed trajectory time into thirds, "
    "with the final endpoint in late. Empty windows are absent, never zero error. "
    "Shape features use full arrays and one common observed-range tolerance.",
    "Peak means the largest absolute departure from the initial value; excursion "
    "is signed. Flat, tied/plateau and boundary peaks have no unique interior "
    "peak time. Times retain public trajectory units.",
    "Half-return is the first sampled return within half the peak departure "
    "after a unique interior peak. It is not a fitted exponential time constant; "
    "not_reached means censored, not infinity.",
    "Input delays are sampled associations, not causal effects. They require "
    "exactly one changing input with exactly one departure from its initial "
    "level; each response requires a unique interior peak after that departure. "
    "Repeated and overlapping events remain flagged without assigned delays.",
    "Training mismatches do not establish structural failure, a faulty equation, "
    "or identified mechanisms. No hidden-state trajectories, validation/test "
    "metrics or private reference are included.",
)


class ResponseWindow(StrictSchema):
    """A nonempty elapsed-time third, computed from all its sampled points."""

    evidence_id: str = Field(min_length=1)
    label: Literal["early", "middle", "late"]
    first_index: int = Field(ge=0)
    last_index: int = Field(ge=0)
    first_time: FiniteFloat
    last_time: FiniteFloat
    sample_count: int = Field(ge=1)
    normalized_mse: FiniteFloat = Field(ge=0)
    normalized_signed_bias: FiniteFloat


class ResponseInput(InputResponse):
    """Observed sampled input features; departure counts do not identify causes."""

    changing: bool
    repeated_events: bool


class ResponseExample(StrictSchema):
    """One selected trajectory/target pair, with no hidden-state arrays."""

    evidence_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    target: Identifier
    selected_as: Literal["worst", "contrasting_best"]
    sample_count: int = Field(ge=2)
    training_scale: FiniteFloat = Field(gt=0)
    normalized_mse: FiniteFloat = Field(ge=0)
    normalized_signed_bias: FiniteFloat
    common_change_tolerance: FiniteFloat = Field(gt=0)
    observed: ResponseShape
    predicted: ResponseShape
    windows: tuple[ResponseWindow, ...] = Field(min_length=1, max_length=3)
    empty_windows: tuple[Literal["early", "middle", "late"], ...]
    inputs: dict[Identifier, ResponseInput]
    multiple_changing_inputs: bool
    input_peak_delays: dict[Identifier, tuple[FiniteFloat | None, FiniteFloat | None]]
    peak_timing_error: FiniteFloat | None


class TargetOverview(StrictSchema):
    """Complete target coverage even when predicted evidence is unavailable."""

    status: Literal["available", "unavailable"]
    trajectory_count: int = Field(ge=1)
    sample_count: int = Field(ge=2)
    normalized_mse: FiniteFloat | None = Field(default=None, ge=0)
    normalized_signed_bias: FiniteFloat | None = None
    minimum_trajectory_nmse: FiniteFloat | None = Field(default=None, ge=0)
    maximum_trajectory_nmse: FiniteFloat | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def availability(self) -> TargetOverview:
        """Unavailable predictions must never masquerade as measured zeros."""
        values = (
            self.normalized_mse,
            self.normalized_signed_bias,
            self.minimum_trajectory_nmse,
            self.maximum_trajectory_nmse,
        )
        if (self.status == "available" and any(v is None for v in values)) or (
            self.status == "unavailable" and any(v is not None for v in values)
        ):
            raise ValueError("response overview availability differs from metrics")
        return self


class ResponseSummary(StrictSchema):
    """All-target overview and at most two examples per target, with exact refs."""

    protocol: Literal["interaction-training-response-1"] = (
        "interaction-training-response-1"
    )
    split: Literal["train"] = "train"
    status: Literal["available", "unavailable"]
    reason_codes: tuple[str, ...]
    targets: tuple[Identifier, ...] = Field(min_length=1)
    target_overview: dict[Identifier, TargetOverview]
    examples: tuple[ResponseExample, ...]
    omitted_trajectory_target_examples: int = Field(ge=0)
    omitted_windows: int = Field(ge=0)
    selection: Literal["Per target: worst, then contrasting best; stable ID ties."] = (
        "Per target: worst, then contrasting best; stable ID ties."
    )
    interpretation: tuple[str, ...] = INTERPRETATION

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Return only IDs of examples/windows actually present in this packet."""
        return tuple(
            evidence_id
            for row in self.examples
            for evidence_id in (row.evidence_id, *(w.evidence_id for w in row.windows))
        )

    @model_validator(mode="after")
    def coverage(self) -> ResponseSummary:
        """Reject target trimming, unavailable numeric evidence and unbounded rows."""
        if len(set(self.targets)) != len(self.targets) or set(self.targets) != set(
            self.target_overview
        ):
            raise ValueError("response summary must cover exactly every target")
        if any(v.status != self.status for v in self.target_overview.values()):
            raise ValueError("response summary and overview status differ")
        if self.status == "unavailable" and (self.examples or not self.reason_codes):
            raise ValueError("unavailable response requires a reason and no examples")
        if self.status == "available" and self.reason_codes:
            raise ValueError("available response must not carry unavailable reasons")
        if any(r.target not in self.targets for r in self.examples):
            raise ValueError("response example has an undeclared target")
        for target in self.targets:
            count = sum(r.target == target for r in self.examples)
            if count > 2 or (self.status == "available" and count == 0):
                raise ValueError("response requires one or two examples per target")
        total = sum(v.trajectory_count for v in self.target_overview.values())
        if total != len(self.examples) + self.omitted_trajectory_target_examples:
            raise ValueError("response example omission count differs")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("response evidence identifiers must be unique")
        return self


def _training(
    snapshot: Snapshot,
) -> tuple[tuple[str, ...], tuple[PublicTrajectory, ...]]:
    """Read public training observations, checking split before accessing rows."""
    training = snapshot.problem.training
    if training.name != "train":
        raise ValueError("response evidence requires train split only")
    context = snapshot.problem.request.context
    targets = tuple(sorted(context.targets))
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("response context requires unique nonempty targets")
    rows = tuple(training.rows)
    if not rows or len({r.trajectory_id for r in rows}) != len(rows):
        raise ValueError("response requires unique nonempty training trajectories")
    for row in rows:
        if set(row.targets) != set(targets):
            raise ValueError("training target set differs from public context")
        if set(row.external_inputs) != set(context.external_inputs):
            raise ValueError("training input set differs from public context")
        # Validate just arrays used below; never traverse auxiliaries or fit state.
        time = _array(row.time, len(row.time), "training time")
        if len(time) < 2 or np.any(np.diff(time) <= 0):
            raise ValueError("training times must be strictly increasing")
        for values in (*row.targets.values(), *row.external_inputs.values()):
            _array(values, len(time), "training channel")
    return targets, rows


def _array(values: object, size: int, label: str) -> np.ndarray:
    """Require a finite one-dimensional array of the complete expected length."""
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {label} array") from error
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"invalid {label} dimensions, length or finite values")
    return result


def _saved(
    value: Mapping, targets: tuple[str, ...], rows: tuple[PublicTrajectory, ...]
) -> tuple[tuple[str, ...], dict[str, float], np.ndarray]:
    """Validate complete residual layout and training-derived saved scales."""
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("complete response stage lacks metadata")
    channels = metadata.get("channels")
    if (
        not isinstance(channels, (list, tuple))
        or any(not isinstance(c, str) for c in channels)
        or len(channels) != len(targets)
        or set(channels) != set(targets)
    ):
        raise ValueError("response metadata channels differ from context target set")
    scales = metadata.get("scales")
    if not isinstance(scales, Mapping) or set(scales) != set(targets):
        raise ValueError("response metadata scale keys differ from context targets")
    checked = {}
    for target in targets:
        scale = scales[target]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ValueError("response scale must be a finite positive number")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("response scale must be a finite positive number")
        observed = np.concatenate([np.asarray(r.targets[target]) for r in rows])
        expected = max(float(np.std(observed)), 1e-12)
        if not np.isclose(scale, expected, rtol=SUMMARY_RTOL, atol=0):
            raise ValueError("response scale differs from observed training scale")
        checked[target] = float(scale)
    size = sum(len(r.time) for r in rows) * len(channels)
    return tuple(channels), checked, _array(value.get("residual"), size, "residual")


def _bands(time: np.ndarray) -> np.ndarray:
    """Assign every sample to an elapsed-time third, including the last endpoint."""
    return np.minimum(2, np.floor(3 * (time - time[0]) / (time[-1] - time[0])))


def _inputs(row: PublicTrajectory, time: np.ndarray) -> dict[str, ResponseInput]:
    """Summarize only declared, sampled external inputs and departure ambiguity."""
    result = {}
    for name, raw in sorted(row.external_inputs.items()):
        values = np.asarray(raw)
        tolerance = 1e-8 + 0.01 * float(np.ptp(values))
        active = np.abs(values - values[0]) > tolerance
        indices = np.flatnonzero(active)
        departures = int(np.sum(active & ~np.r_[False, active[:-1]]))
        result[name] = ResponseInput(
            initial=float(values[0]),
            final=float(values[-1]),
            minimum=float(values.min()),
            maximum=float(values.max()),
            first_sampled_change=float(time[indices[0]]) if len(indices) else None,
            departure_count=departures,
            changing=bool(len(indices)),
            repeated_events=departures > 1,
        )
    return result


def _example(
    row: PublicTrajectory,
    target: str,
    residual: np.ndarray,
    scale: float,
    evidence_id: str,
) -> ResponseExample:
    """Describe one full-array reconstruction without sampling hidden states."""
    time, observed = np.asarray(row.time), np.asarray(row.targets[target])
    with np.errstate(over="ignore", invalid="ignore"):
        predicted = observed + scale * residual
    if not np.isfinite(predicted).all():
        raise ValueError("nonfinite reconstructed response")
    tolerance = 1e-8 + 0.01 * float(np.ptp(observed))
    observed_shape, predicted_shape = (
        shape(time, values, tolerance) for values in (observed, predicted)
    )
    bands, windows, empty = _bands(time), [], []
    for band, label in enumerate(("early", "middle", "late")):
        indices = np.flatnonzero(bands == band)
        if not len(indices):
            empty.append(label)
            continue
        windows.append(
            ResponseWindow(
                evidence_id=f"{evidence_id}_{label}",
                label=label,
                first_index=int(indices[0]),
                last_index=int(indices[-1]),
                first_time=float(time[indices[0]]),
                last_time=float(time[indices[-1]]),
                sample_count=len(indices),
                normalized_mse=float(np.mean(residual[indices] ** 2)),
                normalized_signed_bias=float(np.mean(residual[indices])),
            )
        )
    inputs, delays = _inputs(row, time), {}
    changing = {k: v for k, v in inputs.items() if v.changing}
    if len(changing) == 1:
        name, inp = next(iter(changing.items()))
        if inp.departure_count == 1:
            delays[name] = tuple(
                s.peak_time - inp.first_sampled_change
                if s.peak_time is not None and s.peak_time >= inp.first_sampled_change
                else None
                for s in (observed_shape, predicted_shape)
            )
    return ResponseExample(
        evidence_id=evidence_id,
        trajectory_id=row.trajectory_id,
        target=target,
        selected_as="worst",
        sample_count=len(time),
        training_scale=scale,
        normalized_mse=float(np.mean(residual**2)),
        normalized_signed_bias=float(np.mean(residual)),
        common_change_tolerance=tolerance,
        observed=observed_shape,
        predicted=predicted_shape,
        windows=tuple(windows),
        empty_windows=tuple(empty),
        inputs=inputs,
        multiple_changing_inputs=len(changing) > 1,
        input_peak_delays=delays,
        peak_timing_error=predicted_shape.peak_time - observed_shape.peak_time
        if observed_shape.peak_time is not None
        and predicted_shape.peak_time is not None
        and observed_shape.excursion * predicted_shape.excursion > 0
        else None,
    )


def build(snapshot: Snapshot, *, per_target: int = 2) -> ResponseSummary:
    """Project a verified snapshot into bounded, target-balanced response evidence.

    The caller must obtain ``snapshot`` from ``portable_feedback.load_snapshot``.
    Incomplete/unverified outputs produce explicit unavailability for every
    public target. Malformed complete arrays/layout/scales fail closed. IDs use
    sorted public trajectory/target identities and never change with selection.
    """
    if isinstance(per_target, bool) or not isinstance(per_target, int):
        raise ValueError("per_target must be an integer from 1 to 2")
    if not 1 <= per_target <= 2:
        raise ValueError("per_target must be an integer from 1 to 2")
    targets, rows = _training(snapshot)
    total_rows = len(rows) * len(targets)
    total_windows = sum(len(np.unique(_bands(np.asarray(r.time)))) for r in rows)
    total_windows *= len(targets)
    reasons = []
    if snapshot.packet.diagnostics.output_status != "verified":
        reasons.append("OUTPUT_NOT_VERIFIED")
    for key in ("primary", "alternate"):
        operation = snapshot.operations.get(key)
        if not isinstance(operation, Mapping) or operation.get("status") != "complete":
            reasons.append(f"{key.upper()}_NOT_COMPLETE")
    if reasons:
        return ResponseSummary(
            status="unavailable",
            reason_codes=tuple(reasons),
            targets=targets,
            target_overview={
                target: TargetOverview(
                    status="unavailable",
                    trajectory_count=len(rows),
                    sample_count=sum(len(r.time) for r in rows),
                )
                for target in targets
            },
            examples=(),
            omitted_trajectory_target_examples=total_rows,
            omitted_windows=total_windows,
        )
    values = {}
    for key in ("primary", "alternate"):
        value = snapshot.operations[key].get("value")
        if not isinstance(value, Mapping):
            raise ValueError(f"complete {key} response stage lacks its value")
        values[key] = value
    channels, scales, residual = _saved(values["primary"], targets, rows)
    alternate_channels, alternate_scales, _ = _saved(values["alternate"], targets, rows)
    if channels != alternate_channels or scales != alternate_scales:
        raise ValueError("primary and alternate response layouts differ")
    # Decode in the exact original order before deterministic presentation sorting.
    trajectory_order = {
        tid: i for i, tid in enumerate(sorted(r.trajectory_id for r in rows))
    }
    examples, offset = [], 0
    for row in rows:
        size = len(row.time) * len(channels)
        matrix = residual[offset : offset + size].reshape(len(row.time), len(channels))
        offset += size
        for ci, channel in enumerate(channels):
            evidence_id = (
                f"response_trajectory_{trajectory_order[row.trajectory_id]:03d}"
                f"_target_{targets.index(channel):02d}"
            )
            examples.append(
                _example(row, channel, matrix[:, ci], scales[channel], evidence_id)
            )
    overview, selected = {}, []
    for target in targets:
        ranked = sorted(
            (r for r in examples if r.target == target),
            key=lambda r: (-r.normalized_mse, r.evidence_id),
        )
        count = sum(r.sample_count for r in ranked)
        overview[target] = TargetOverview(
            status="available",
            trajectory_count=len(ranked),
            sample_count=count,
            normalized_mse=sum(r.sample_count * r.normalized_mse for r in ranked)
            / count,
            normalized_signed_bias=sum(
                r.sample_count * r.normalized_signed_bias for r in ranked
            )
            / count,
            minimum_trajectory_nmse=ranked[-1].normalized_mse,
            maximum_trajectory_nmse=ranked[0].normalized_mse,
        )
        selected.append(ranked[0])
        if per_target == 2 and len(ranked) > 1:
            selected.append(
                ranked[-1].model_copy(update={"selected_as": "contrasting_best"})
            )
    return ResponseSummary(
        status="available",
        reason_codes=(),
        targets=targets,
        target_overview=overview,
        examples=tuple(selected),
        omitted_trajectory_target_examples=total_rows - len(selected),
        omitted_windows=total_windows - sum(len(r.windows) for r in selected),
    )
