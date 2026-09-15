"""Deterministic descriptions of fixed-parameter training-rollout mismatches."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from autoformalism.data import DatasetSplit, SplitName, TrainingScaler
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.residual_feedback import (
    NumericalQualification,
    ResidualDetail,
    ResidualEvidence,
    ResidualRow,
    ResidualSample,
    ResidualSettings,
    ResidualWindow,
    SignalShape,
)

INTERPRETATION = (
    "These measurements describe training mismatches at the retained parameters; "
    "they do not establish that the equation structure is incapable of fitting.",
    "When fitting is budget-limited, hypotheses about memory, feedback, nonlinearity "
    "or initialization are proposals to test, not certified causes or mandatory edits.",
    "Residual means prediction minus observation, divided by the target's global "
    "training standard deviation. Window and trajectory NMSEs use that same scale.",
    "Observed and predicted turning points use the same observed-range tolerance. "
    "They describe sampled reversals, not a proven oscillation, frequency "
    "or limit cycle.",
    "Zero refers only to sampled external inputs; initial state, supplied auxiliaries "
    "and unresolved between-sample behavior may still matter.",
    "Previous-fit comparisons use the identical data and scaling. A mismatch "
    "persisting across these two parameter sets is not proof of structural failure.",
    "Detail examples include worst, best and zero-input cases when available. "
    "Full row counts and omitted samples are explicit. "
    "IDs locate evidence, not model features.",
    "Replaying the saved parameters reproduces production scores; it is not an "
    "independent-solver accuracy check or a convergence test.",
    "No hidden-state trajectory errors, validation/test metrics, causal-effect "
    "estimates, confidence probabilities or significance tests are provided.",
)


def _turns(values: np.ndarray, tolerance: float) -> list[int]:
    """Count reversals beyond an amplitude threshold, even with very small steps."""
    direction, extreme_index, extreme = 0, 0, values[0]
    turns = []
    for index, value in enumerate(values[1:], start=1):
        if direction == 0:
            change = value - extreme
            if abs(change) > tolerance:
                direction = 1 if change > 0 else -1
                extreme_index, extreme = index, value
        elif direction * (value - extreme) >= 0:
            extreme_index, extreme = index, value
        elif direction * (extreme - value) > tolerance:
            turns.append(extreme_index)
            direction *= -1
            extreme_index, extreme = index, value
    return turns


def _shape(values: np.ndarray, tolerance: float) -> SignalShape:
    return SignalShape(
        initial=float(values[0]),
        final=float(values[-1]),
        mean=float(np.mean(values)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        range=float(np.ptp(values)),
        turning_point_count=len(_turns(values, tolerance)),
        common_change_tolerance=tolerance,
    )


def _aligned_predictions(
    train: DatasetSplit, context: ValidationContext, values: Mapping[str, dict]
) -> dict:
    ids = {t.trajectory_id for t in train.trajectories}
    if set(values) != ids:
        raise ValueError("replay must cover exactly the training trajectories")
    result = {}
    for trajectory in train.trajectories:
        record = values[trajectory.trajectory_id]
        if set(record) != {"time", "predictions"}:
            raise ValueError(
                "replay permits only times and observed-target predictions"
            )
        if not np.array_equal(np.asarray(record["time"]), trajectory.time):
            raise ValueError("replay times differ from observed training times")
        if set(record["predictions"]) != set(context.targets):
            raise ValueError("replay targets differ from the public context")
        predictions = {
            k: np.asarray(v, dtype=float) for k, v in record["predictions"].items()
        }
        if any(
            v.shape != trajectory.time.shape or not np.isfinite(v).all()
            for v in predictions.values()
        ):
            raise ValueError("replay predictions must be finite and aligned")
        result[trajectory.trajectory_id] = predictions
    return result


def _selection(
    rows: list[ResidualRow], maximum: int
) -> list[tuple[ResidualRow, tuple[str, ...]]]:
    """Include contrasting examples; tie breaks never depend on directory order."""
    ranked = sorted(rows, key=lambda row: (-row.normalized_mse, row.evidence_id))
    chosen: dict[str, tuple[ResidualRow, list[str]]] = {}

    def add(row, reason):
        if row.evidence_id in chosen:
            chosen[row.evidence_id][1].append(reason)
        elif len(chosen) < maximum:
            chosen[row.evidence_id] = (row, [reason])

    add(ranked[0], "largest_training_nmse")
    add(ranked[-1], "smallest_training_nmse")
    zero = [row for row in ranked if row.sampled_input_regime == "zero"]
    if zero:
        add(zero[0], "largest_nmse_with_zero_sampled_inputs")
    for row in ranked:
        add(row, "remaining_larger_training_nmse")
    return [(row, tuple(dict.fromkeys(reasons))) for row, reasons in chosen.values()]


def build_residual_evidence(
    train: DatasetSplit,
    context: ValidationContext,
    current: Mapping[str, dict],
    *,
    candidate_sha256: str,
    parameters: Mapping[str, float],
    numerical_status: dict,
    previous: Mapping[str, dict] | None = None,
    settings: ResidualSettings | None = None,
) -> ResidualEvidence:
    """Reject held-out inputs before access; compare only observed output channels."""
    if not isinstance(train, DatasetSplit) or train.name is not SplitName.TRAIN:
        raise ValueError("residual evidence requires the train split only")
    if context.lagged_targets:
        raise ValueError("residual evidence pilot requires open rollouts")
    settings = settings or ResidualSettings()
    packed = public.pack_split(train)
    if any(
        set(t.targets) != set(context.targets)
        or set(t.external_inputs) != set(context.external_inputs)
        or set(t.auxiliaries) != set(context.auxiliaries)
        for t in train.trajectories
    ):
        raise ValueError("training channel identities differ from public context")
    if not train.trajectories:
        raise ValueError("training trajectories must not be empty")
    # PublicSplit validation checks finite arrays, identities, dimensions and times.
    current_arrays = _aligned_predictions(train, context, current)
    old = None if previous is None else _aligned_predictions(train, context, previous)
    scales = TrainingScaler().fit(train).scales
    rows, residuals, old_residuals = [], [], []
    trajectories = {t.trajectory_id: t for t in train.trajectories}
    for ti, trajectory in enumerate(
        sorted(train.trajectories, key=lambda t: t.trajectory_id)
    ):
        times, inputs = trajectory.time, trajectory.external_inputs
        regime = (
            "no_inputs"
            if not inputs
            else "zero"
            if all(
                np.max(np.abs(v)) <= settings.absolute_tolerance
                for v in inputs.values()
            )
            else "constant_nonzero"
            if all(np.ptp(v) <= settings.absolute_tolerance for v in inputs.values())
            else "varying"
        )
        for ci, channel in enumerate(sorted(context.targets)):
            row_id = f"trajectory_{ti:03d}_target_{ci:02d}"
            observed = trajectory.targets[channel]
            predicted = current_arrays[trajectory.trajectory_id][channel]
            prior = None if old is None else old[trajectory.trajectory_id][channel]
            scale = scales[f"target:{channel}"].standard_deviation
            tolerance = (
                settings.absolute_tolerance
                + settings.relative_tolerance * float(np.ptp(observed))
            )
            residual = (predicted - observed) / scale
            earlier = None if prior is None else (prior - observed) / scale
            residuals.append(residual)
            if earlier is not None:
                old_residuals.append(earlier)
            bands = np.minimum(
                2, np.floor(3 * (times - times[0]) / (times[-1] - times[0])).astype(int)
            )
            windows = []
            for band, label in enumerate(("early", "middle", "late")):
                indices = np.flatnonzero(bands == band)
                if not len(indices):
                    continue
                windows.append(
                    ResidualWindow(
                        evidence_id=f"{row_id}_{label}",
                        label=label,
                        first_index=int(indices[0]),
                        last_index=int(indices[-1]),
                        first_time=float(times[indices[0]]),
                        last_time=float(times[indices[-1]]),
                        sample_count=len(indices),
                        normalized_mse=float(np.mean(residual[indices] ** 2)),
                        normalized_signed_bias=float(np.mean(residual[indices])),
                        previous_normalized_mse=None
                        if earlier is None
                        else float(np.mean(earlier[indices] ** 2)),
                    )
                )
            rows.append(
                ResidualRow(
                    evidence_id=row_id,
                    trajectory_id=trajectory.trajectory_id,
                    target=channel,
                    sample_count=len(times),
                    minimum_time_step=float(np.min(np.diff(times))),
                    maximum_time_step=float(np.max(np.diff(times))),
                    sampled_input_regime=regime,
                    input_ranges={
                        k: (float(v.min()), float(v.max()))
                        for k, v in sorted(inputs.items())
                    },
                    training_scale=scale,
                    normalized_mse=float(np.mean(residual**2)),
                    normalized_signed_bias=float(np.mean(residual)),
                    maximum_absolute_normalized_residual=float(
                        np.max(np.abs(residual))
                    ),
                    observed=_shape(observed, tolerance),
                    predicted=_shape(predicted, tolerance),
                    previous_normalized_mse=None
                    if earlier is None
                    else float(np.mean(earlier**2)),
                    previous_predicted=None
                    if prior is None
                    else _shape(prior, tolerance),
                    windows=tuple(windows),
                )
            )
    chosen = _selection(rows, settings.maximum_details)
    # Keep all detail references even when the compact row table is truncated.
    selected_rows = {row.evidence_id: row for row, _ in chosen}
    if len(selected_rows) > settings.maximum_rows:
        raise ValueError("maximum_rows must accommodate the detailed examples")
    for row in rows:
        if len(selected_rows) >= settings.maximum_rows:
            break
        selected_rows[row.evidence_id] = row
    details = []
    for row, reasons in chosen:
        trajectory = trajectories[row.trajectory_id]
        times, observed = trajectory.time, trajectory.targets[row.target]
        predicted = current_arrays[row.trajectory_id][row.target]
        residual = (predicted - observed) / row.training_scale
        events = []
        for values in (
            *trajectory.external_inputs.values(),
            *trajectory.auxiliaries.values(),
        ):
            tol = settings.absolute_tolerance + settings.relative_tolerance * np.ptp(
                values
            )
            changes = np.flatnonzero(np.abs(np.diff(values)) > tol)
            events.extend(int(i) for j in changes[:2] for i in (j, j + 1))
        ordered = [
            0,
            len(times) - 1,
            int(np.argmax(np.abs(residual))),
            int(np.argmax(observed)),
            int(np.argmin(observed)),
            *events,
            *_turns(observed, row.observed.common_change_tolerance)[:3],
            *_turns(predicted, row.observed.common_change_tolerance)[:3],
            *np.linspace(
                0, len(times) - 1, settings.maximum_samples, dtype=int
            ).tolist(),
        ]
        indices = sorted(list(dict.fromkeys(ordered))[: settings.maximum_samples])
        samples = tuple(
            ResidualSample(
                index=i,
                time=float(times[i]),
                observed=float(observed[i]),
                predicted=float(predicted[i]),
                normalized_residual=float(residual[i]),
                previous_prediction=None
                if old is None
                else float(old[row.trajectory_id][row.target][i]),
                inputs={
                    k: float(v[i])
                    for k, v in sorted(trajectory.external_inputs.items())
                },
                auxiliaries={
                    k: float(v[i]) for k, v in sorted(trajectory.auxiliaries.items())
                },
            )
            for i in indices
        )
        details.append(
            ResidualDetail(
                evidence_id=row.evidence_id + "_samples",
                row_id=row.evidence_id,
                reasons=reasons,
                samples=samples,
                omitted_samples=len(times) - len(samples),
            )
        )
    # Only this allowlist can reach the provider; raw fitter reports may contain
    # validation scores, paths and extensive diagnostic material.
    allowed = (
        "feedback_status",
        "native_optimizer_converged",
        "budget_exhausted",
        "residual_calls",
        "numerical_seconds",
        "previous_residual_calls",
        "recent_training_costs",
        "relative_training_cost_drop",
    )
    qualified = {k: numerical_status[k] for k in allowed if k in numerical_status}
    qualified.update(
        structural_failure_established=False, uncertainty="at_retained_parameters_only"
    )
    payload = {
        "protocol": "training-residual-evidence-1",
        "split": "train",
        "settings": settings.model_dump(mode="json"),
        "candidate_sha256": candidate_sha256,
        "parameter_sha256": public.content_sha256(dict(parameters)),
        "training_content_sha256": public.content_sha256(packed),
        "replay_sha256": public.content_sha256(
            {"current": current, "previous": previous}
        ),
        "normalized_mse": float(np.mean(np.concatenate(residuals) ** 2)),
        "previous_normalized_mse": None
        if old is None
        else float(np.mean(np.concatenate(old_residuals) ** 2)),
        "numerical_status": NumericalQualification.model_validate(qualified).model_dump(
            mode="json"
        ),
        "total_rows": len(rows),
        "omitted_rows": len(rows) - len(selected_rows),
        "rows": [
            r.model_dump(mode="json")
            for r in sorted(selected_rows.values(), key=lambda r: r.evidence_id)
        ],
        "details": [d.model_dump(mode="json") for d in details],
        "interpretation": INTERPRETATION,
    }
    return ResidualEvidence.model_validate(
        {**payload, "packet_sha256": public.content_sha256(payload)}
    )


def validate_residual_evidence(
    packet: dict,
    *,
    candidate_sha256: str,
    parameters: Mapping[str, float],
    training_content_sha256: str,
) -> dict:
    """Verify exact model/vector/data binding before attaching the provider packet."""
    value = ResidualEvidence.model_validate(packet).model_dump(mode="json")
    digest = value.pop("packet_sha256")
    if (
        public.content_sha256(value) != digest
        or value["candidate_sha256"] != candidate_sha256
        or value["parameter_sha256"] != public.content_sha256(dict(parameters))
        or value["training_content_sha256"] != training_content_sha256
    ):
        raise ValueError("residual evidence digest or candidate/vector/data differs")
    return {**value, "packet_sha256": digest}
