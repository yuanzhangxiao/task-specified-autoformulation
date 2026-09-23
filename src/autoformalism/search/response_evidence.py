"""Full-grid response summaries and deterministic, target-balanced presentation."""

from __future__ import annotations

import copy

import numpy as np

from autoformalism.data import SplitName
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.residual_feedback import ResidualEvidence
from autoformalism.schemas.response_evidence import (
    InputResponse,
    ResponseEvidence,
    ResponseRow,
    ResponseShape,
)
from autoformalism.search.residual_evidence import _aligned_predictions, _turns
from autoformalism.staged_topology import content_hash

INTERPRETATION = (
    "Training observations versus free rollout at retained fitted parameters; "
    "not proof of structural failure or identification of a faulty equation.",
    "Peak means the largest absolute departure from the signal's initial value; "
    "excursion is signed. Flat, tied/plateau and boundary peaks are not assigned "
    "a unique interior-peak delay. Times use the public trajectory time units.",
    "Half-return is the first sampled return within half the peak departure "
    "from the initial value, measured after a unique interior peak. It is not "
    "a fitted exponential time constant; not_reached is censored, not infinity.",
    "Input delays are sampled associations, not causal effects. They are shown "
    "as [observed, predicted] pairs, "
    "only with one changing input, one departure from its initial level, and a "
    "unique response peak after that departure. Repeated or overlapping inputs "
    "remain described by input summaries and trajectory-level response shapes.",
    "Bias is prediction minus observation in global training-SD units. Error "
    "uses every sampled point; shape summaries cannot capture every mismatch. "
    "No validation, test, hidden-state errors or private reference is included.",
)


def shape(time, values, tolerance: float) -> ResponseShape:
    """Measure a sampled dominant excursion; never invent a peak or decay rate."""
    delta = values - values[0]
    peak = int(np.argmax(np.abs(delta)))
    amplitude = float(delta[peak])
    tied = np.flatnonzero(
        np.isclose(np.abs(delta), abs(amplitude), atol=1e-12, rtol=1e-12)
    )
    status = (
        "flat"
        if abs(amplitude) <= tolerance
        else "plateau"
        if len(tied) > 1
        else "boundary"
        if peak in (0, len(time) - 1)
        else "interior"
    )
    returning = np.flatnonzero(np.abs(delta[peak + 1 :]) <= abs(amplitude) / 2)
    elapsed = (
        float(time[peak + 1 + returning[0]] - time[peak])
        if status == "interior" and len(returning)
        else None
    )
    return ResponseShape(
        initial=float(values[0]),
        final=float(values[-1]),
        minimum=float(values.min()),
        maximum=float(values.max()),
        excursion=amplitude,
        peak_time=float(time[peak]) if status == "interior" else None,
        peak_status=status,
        half_return_elapsed=elapsed,
        half_return_status="observed"
        if elapsed is not None
        else "not_reached"
        if status == "interior"
        else "undefined",
        turning_points=len(_turns(values, tolerance)),
    )


def build_response_evidence(
    train, context, predictions, packet: dict
) -> ResponseEvidence:
    """Use full common-grid predictions, never infer timing from sparse samples."""
    if train.name is not SplitName.TRAIN:
        raise ValueError("response evidence requires train split only")
    residual = ResidualEvidence.model_validate(packet)
    if (
        public.content_sha256(public.pack_split(train))
        != residual.training_content_sha256
    ):
        raise ValueError("response training identity differs")
    if (
        public.content_sha256({"current": predictions, "previous": None})
        != residual.replay_sha256
    ):
        raise ValueError("response predictions differ from the saved replay")
    aligned = _aligned_predictions(train, context, predictions)
    originals = {(r.trajectory_id, r.target): r for r in residual.rows}
    inputs, rows = {}, []
    for trajectory in sorted(train.trajectories, key=lambda t: t.trajectory_id):
        tid, time = trajectory.trajectory_id, trajectory.time
        inputs[tid] = {}
        for name, values in sorted(trajectory.external_inputs.items()):
            tolerance = 1e-8 + 0.01 * float(np.ptp(values))
            active = np.abs(values - values[0]) > tolerance
            indices = np.flatnonzero(active)
            inputs[tid][name] = InputResponse(
                initial=float(values[0]),
                final=float(values[-1]),
                minimum=float(values.min()),
                maximum=float(values.max()),
                first_sampled_change=float(time[indices[0]]) if len(indices) else None,
                departure_count=int(np.sum(active & ~np.r_[False, active[:-1]])),
            )
        changing = {
            k: v for k, v in inputs[tid].items() if v.first_sampled_change is not None
        }
        for target in sorted(context.targets):
            record = originals.get((tid, target))
            if record is None:
                raise ValueError(
                    "response packet must cover every trajectory and target"
                )
            tolerance = record.observed.common_change_tolerance
            observed = shape(time, trajectory.targets[target], tolerance)
            predicted = shape(time, aligned[tid][target], tolerance)
            delays = {}
            if len(changing) == 1:
                name, inp = next(iter(changing.items()))
                if inp.departure_count == 1:
                    delays[name] = tuple(
                        s.peak_time - inp.first_sampled_change
                        if s.peak_time is not None
                        and s.peak_time >= inp.first_sampled_change
                        else None
                        for s in (observed, predicted)
                    )
            rows.append(
                ResponseRow(
                    evidence_id=record.evidence_id,
                    trajectory_id=tid,
                    target=target,
                    sample_count=record.sample_count,
                    normalized_mse=record.normalized_mse,
                    normalized_signed_bias=record.normalized_signed_bias,
                    window_nmse={w.label: w.normalized_mse for w in record.windows},
                    observed=observed,
                    predicted=predicted,
                    input_peak_delays=delays,
                    peak_timing_error=predicted.peak_time - observed.peak_time
                    if observed.peak_time is not None
                    and predicted.peak_time is not None
                    and observed.excursion * predicted.excursion > 0
                    else None,
                )
            )
    return ResponseEvidence(
        candidate_sha256=residual.candidate_sha256,
        parameter_sha256=residual.parameter_sha256,
        training_content_sha256=residual.training_content_sha256,
        source_packet_sha256=content_hash(packet),
        replayed_packet_sha256=content_hash(packet),
        replay_sha256=residual.replay_sha256,
        rows=tuple(rows),
        inputs=inputs,
        numerical_status=residual.numerical_status.model_dump(mode="json"),
        interpretation=INTERPRETATION,
    )


def rounded(value):
    """Six significant digits for evidence presentation; fit artifacts stay exact."""
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value


def presentation(
    evidence: dict, packet: dict, *, per_target: int = 3, samples: bool = True
) -> tuple[dict, dict]:
    """Keep every target's overview, plus worst/best/input-regime contrasts."""
    full = ResponseEvidence.model_validate(evidence)
    residual = ResidualEvidence.model_validate(packet)
    if full.source_packet_sha256 != content_hash(packet):
        raise ValueError("response evidence and residual source differ")
    selected, overview = [], {}
    for target in sorted({r.target for r in full.rows}):
        rows = sorted(
            (r for r in full.rows if r.target == target),
            key=lambda r: (-r.normalized_mse, r.evidence_id),
        )
        n = sum(r.sample_count for r in rows)
        overview[target] = {
            "trajectories": len(rows),
            "samples": n,
            "nmse": sum(r.normalized_mse * r.sample_count for r in rows) / n,
            "signed_bias": sum(r.normalized_signed_bias * r.sample_count for r in rows)
            / n,
            "nmse_range": [rows[-1].normalized_mse, rows[0].normalized_mse],
        }
        choices = [rows[0], rows[-1]]
        regimes = set()
        for row in rows:
            regime = tuple(
                (k, v.departure_count)
                for k, v in full.inputs[row.trajectory_id].items()
            )
            if regime not in regimes:
                choices.append(row)
                regimes.add(regime)
        choices += rows
        seen = {}
        for row in choices:
            seen.setdefault(row.evidence_id, row)
        unique = list(seen.values())
        selected.extend(unique[:per_target])
    rows, refs, details = [], {}, []
    detail_by_id = {d.row_id: d for d in residual.details}
    for i, row in enumerate(selected, 1):
        name = f"R{i:03d}"
        refs[name] = row.evidence_id
        rows.append({**row.model_dump(mode="json"), "evidence_id": name})
        if samples and row.evidence_id in detail_by_id:
            d = detail_by_id[row.evidence_id]
            chosen = list(d.samples)
            # Keep at most three actually recorded samples, not reconstructed curves.
            indices = sorted(
                {
                    0,
                    len(chosen) - 1,
                    max(
                        range(len(chosen)),
                        key=lambda j: abs(chosen[j].normalized_residual),
                    ),
                }
            )
            details.append(
                {
                    "row": name,
                    "samples": [chosen[j].model_dump(mode="json") for j in indices],
                }
            )
    tids = sorted({r.trajectory_id for r in selected})
    result = {
        "protocol": full.protocol,
        "split": "train",
        "target_overview": overview,
        "numerical_status": full.numerical_status,
        "interpretation": full.interpretation,
        "examples": rows,
        "inputs": {
            t: {k: v.model_dump(mode="json") for k, v in full.inputs[t].items()}
            for t in tids
        },
        "diagnostic_samples": details,
        "omitted_trajectory_target_examples": len(full.rows) - len(rows),
        "selection": (
            "Per target: worst, best, then distinct input-departure patterns; "
            "deterministic ties."
        ),
        "display_significant_digits": 6,
        "source_response_sha256": content_hash(evidence),
    }
    return rounded(result), refs


def compact_user(user: dict, *, per_target: int, samples: bool) -> dict:
    """Shrink only optional evidence examples; preserve model and public contract."""
    value = copy.deepcopy(user)
    evidence = value.get("training_evidence", {})
    if evidence.get("protocol") != "training-response-evidence-1":
        return value
    kept, counts = [], {}
    for row in evidence["examples"]:
        target = row["target"]
        counts[target] = counts.get(target, 0) + 1
        if counts[target] <= per_target:
            kept.append(row)
    evidence["omitted_trajectory_target_examples"] += len(evidence["examples"]) - len(
        kept
    )
    evidence["examples"] = kept
    refs = {r["evidence_id"] for r in kept}
    tids = {r["trajectory_id"] for r in kept}
    evidence["inputs"] = {t: v for t, v in evidence["inputs"].items() if t in tids}
    evidence["diagnostic_samples"] = (
        [d for d in evidence["diagnostic_samples"] if d["row"] in refs]
        if samples
        else []
    )
    value["evidence_catalog"] = [
        v for v in value["evidence_catalog"] if v["ref"] in refs
    ]
    return value
