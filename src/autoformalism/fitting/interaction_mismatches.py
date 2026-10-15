"""M38 descriptive mismatches from a verified M37 training feedback packet.

The caller verifies the source using ``interaction_feedback.completed`` first.
This module does not execute equations, fit, integrate, or infer mechanisms. Its
thresholds decide what to display, not statistical significance or fit success.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from autoformalism.fitting.interaction_feedback import Packet
from autoformalism.fitting.interaction_response import ResponseExample, TargetOverview
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema

MismatchCode = Literal[
    "signed_bias",
    "excursion_presence",
    "excursion_direction",
    "excursion_magnitude",
    "peak_timing",
    "half_return",
    "window_error",
]


class Policy(StrictSchema):
    """Finite presentation thresholds; no tolerance is a hypothesis test."""

    protocol: Literal["interaction-mismatch-policy-1"] = "interaction-mismatch-policy-1"
    normalized_difference_threshold: FiniteFloat = Field(
        default=0.05, gt=0, strict=True
    )
    time_difference_fraction: FiniteFloat = Field(default=0.05, gt=0, le=1, strict=True)
    window_nmse_threshold: FiniteFloat = Field(default=0.01, gt=0, strict=True)


class Finding(StrictSchema):
    """A measured discrepancy or an explicitly unavailable feature comparison."""

    code: MismatchCode
    availability: Literal["measured", "withheld"]
    target: Identifier
    trajectory_id: str | None = None
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    observed: FiniteFloat | None = None
    predicted: FiniteFloat | None = None
    difference: FiniteFloat | None = None
    unit: Literal["training_sd", "target_native", "trajectory_time", "nmse"]
    threshold: FiniteFloat | None = Field(default=None, gt=0)
    interpretation: str
    caveats: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def coherent(self) -> Finding:
        """Censored timing never becomes a fabricated numerical difference."""
        if self.availability == "withheld":
            if self.difference is not None or not self.reason_codes:
                raise ValueError(
                    "withheld comparison requires reasons and no difference"
                )
        elif self.difference is None:
            raise ValueError("measured comparison requires a finite difference")
        return self


class TargetCoverage(StrictSchema):
    """Full target overview plus exact scope of the selected response examples."""

    overview: TargetOverview
    selected_examples: int = Field(ge=0, le=2)
    omitted_examples: int = Field(ge=0)
    empty_selected_windows: dict[str, tuple[Literal["early", "middle", "late"], ...]]
    measured_findings: int = Field(ge=0)
    withheld_comparisons: int = Field(ge=0)


class MismatchSummary(StrictSchema):
    protocol: Literal["interaction-training-mismatches-1"] = (
        "interaction-training-mismatches-1"
    )
    split: Literal["train"] = "train"
    status: Literal["available", "unavailable"]
    policy: Policy
    reason_codes: tuple[str, ...]
    targets: tuple[Identifier, ...]
    coverage: dict[Identifier, TargetCoverage]
    findings: tuple[Finding, ...]
    omitted_trajectory_target_examples: int = Field(ge=0)
    omitted_windows: int = Field(ge=0)
    omitted_findings: Literal[0] = 0
    interpretation: tuple[str, ...] = (
        "Thresholds are deterministic presentation heuristics, not statistical "
        "significance, fit-acceptance thresholds or evidence of a faulty mechanism.",
        "Target overviews use all training samples. Shape and window findings use "
        "only the displayed worst and, if supplied, contrasting-best examples; "
        "they do not establish consistency across omitted trajectories.",
        "Signed bias is prediction minus observation. Excursion magnitude compares "
        "absolute departures from each trajectory's own initial level, normalized "
        "by the common training standard deviation.",
        "Peak timing and sampled half-return require same-direction nonflat "
        "unique interior peaks. Half-return is not a decay time constant. "
        "Unobserved return is censored, not zero or infinity.",
        "Feature timing describes whole trajectories, not a response assigned to "
        "a particular input. Repeated or overlapping inputs remain ambiguous.",
        "No findings above these thresholds does not certify accurate trajectories, "
        "successful optimization, identifiable parameters or a correct mechanism.",
    )

    @model_validator(mode="after")
    def complete(self) -> MismatchSummary:
        """Keep all targets and expose selected-example and finding denominators."""
        if len(set(self.targets)) != len(self.targets) or set(self.targets) != set(
            self.coverage
        ):
            raise ValueError("mismatch summary must cover exactly every target")
        if any(f.target not in self.coverage for f in self.findings):
            raise ValueError("finding has an undeclared target")
        for target, row in self.coverage.items():
            found = [f for f in self.findings if f.target == target]
            if row.measured_findings != sum(
                f.availability == "measured" for f in found
            ) or row.withheld_comparisons != sum(
                f.availability == "withheld" for f in found
            ):
                raise ValueError("mismatch finding counts differ")
            if row.overview.status != self.status or (
                row.selected_examples + row.omitted_examples
                != row.overview.trajectory_count
            ):
                raise ValueError(
                    "mismatch coverage differs from full response overview"
                )
        if sum(r.omitted_examples for r in self.coverage.values()) != (
            self.omitted_trajectory_target_examples
        ):
            raise ValueError("mismatch omission counts differ")
        if self.status == "unavailable" and (self.findings or not self.reason_codes):
            raise ValueError(
                "unavailable mismatch summary requires reasons and no findings"
            )
        return self


def _caveats(row: ResponseExample | None = None) -> tuple[str, ...]:
    common = ("Descriptive mismatch; no causal or equation-fault attribution.",)
    if row is None:
        return (
            *common,
            "All training samples; trajectory weighting follows sample counts.",
        )
    return (
        *common,
        "Selected trajectory example, not all-training feature consistency.",
        *(
            ("Multiple changing inputs prevent an input-specific interpretation.",)
            if row.multiple_changing_inputs
            else ()
        ),
        *(
            ("Repeated input events prevent assigning this feature to one event.",)
            if any(v.repeated_events for v in row.inputs.values())
            else ()
        ),
    )


def _bias(
    target: str,
    bias: float,
    policy: Policy,
    row: ResponseExample | None = None,
) -> list[Finding]:
    if abs(bias) <= policy.normalized_difference_threshold:
        return []
    return [
        Finding(
            code="signed_bias",
            availability="measured",
            target=target,
            trajectory_id=row.trajectory_id if row else None,
            evidence_ids=(row.evidence_id,) if row else ("numerics",),
            observed=0,
            predicted=bias,
            difference=bias,
            unit="training_sd",
            threshold=policy.normalized_difference_threshold,
            interpretation="Prediction is higher on average than observation."
            if bias > 0
            else "Prediction is lower on average than observation.",
            caveats=_caveats(row),
        )
    ]


def _excursion(row: ResponseExample, policy: Policy) -> list[Finding]:
    observed, predicted = row.observed.excursion, row.predicted.excursion
    active = tuple(abs(v) > row.common_change_tolerance for v in (observed, predicted))
    common = {
        "availability": "measured",
        "target": row.target,
        "trajectory_id": row.trajectory_id,
        "evidence_ids": (row.evidence_id,),
        "caveats": _caveats(row),
    }
    if active[0] != active[1]:
        return [
            Finding(
                **common,
                code="excursion_presence",
                observed=observed,
                predicted=predicted,
                difference=predicted - observed,
                unit="target_native",
                threshold=row.common_change_tolerance,
                interpretation="Observed nonflat excursion is absent in prediction."
                if active[0]
                else "Prediction has a nonflat excursion absent from observations.",
            )
        ]
    if not all(active):
        return []
    if observed * predicted < 0:
        return [
            Finding(
                **common,
                code="excursion_direction",
                observed=observed,
                predicted=predicted,
                difference=predicted - observed,
                unit="target_native",
                interpretation="Observed and predicted excursions have opposite signs.",
            )
        ]
    obs, pred = abs(observed) / row.training_scale, abs(predicted) / row.training_scale
    if abs(pred - obs) <= policy.normalized_difference_threshold:
        return []
    return [
        Finding(
            **common,
            code="excursion_magnitude",
            observed=obs,
            predicted=pred,
            difference=pred - obs,
            unit="training_sd",
            threshold=policy.normalized_difference_threshold,
            interpretation="Predicted same-direction excursion is larger than observed."
            if pred > obs
            else "Predicted same-direction excursion is smaller than observed.",
        )
    ]


def _timing(row: ResponseExample, policy: Policy, half_return: bool) -> list[Finding]:
    observed, predicted = row.observed, row.predicted
    code = "half_return" if half_return else "peak_timing"
    duration = max(w.last_time for w in row.windows) - min(
        w.first_time for w in row.windows
    )
    if duration <= 0:
        raise ValueError("response windows must retain a positive trajectory duration")
    threshold = policy.time_difference_fraction * duration
    reasons = []
    for label, shape in (("OBSERVED", observed), ("PREDICTED", predicted)):
        if shape.peak_status != "interior":
            reasons.append(f"{label}_PEAK_{shape.peak_status.upper()}")
    if not reasons and observed.excursion * predicted.excursion <= 0:
        reasons.append("EXCURSION_DIRECTIONS_NOT_COMPARABLE")
    obs = observed.half_return_elapsed if half_return else observed.peak_time
    pred = predicted.half_return_elapsed if half_return else predicted.peak_time
    if half_return:
        for label, shape in (("OBSERVED", observed), ("PREDICTED", predicted)):
            if shape.half_return_status != "observed":
                reasons.append(
                    f"{label}_HALF_RETURN_{shape.half_return_status.upper()}"
                )
    if not reasons and (obs is None or pred is None):
        reasons.append("FEATURE_VALUE_UNAVAILABLE")
    common = {
        "code": code,
        "target": row.target,
        "trajectory_id": row.trajectory_id,
        "evidence_ids": (row.evidence_id,),
        "observed": obs,
        "predicted": pred,
        "unit": "trajectory_time",
        "threshold": threshold,
        "caveats": (
            *_caveats(row),
            "Sampled timing only; temporal resolution limits this comparison.",
            *(
                ("First sampled half-return is not an exponential time constant.",)
                if half_return
                else ()
            ),
        ),
    }
    if reasons:
        return [
            Finding(
                **common,
                availability="withheld",
                interpretation="Half-return difference is not available or comparable."
                if half_return
                else "Peak-time difference is unavailable or not comparable.",
                reason_codes=tuple(reasons),
            )
        ]
    if abs(pred - obs) <= threshold:
        return []
    feature = "sampled half-return after the peak" if half_return else "peak"
    return [
        Finding(
            **common,
            availability="measured",
            difference=pred - obs,
            interpretation=f"Predicted {feature} occurs later than observed."
            if pred > obs
            else f"Predicted {feature} occurs earlier than observed.",
        )
    ]


def _example(row: ResponseExample, policy: Policy) -> list[Finding]:
    result = [
        *_bias(row.target, row.normalized_signed_bias, policy, row),
        *_excursion(row, policy),
        *_timing(row, policy, half_return=False),
        *_timing(row, policy, half_return=True),
    ]
    for window in row.windows:
        if window.normalized_mse > policy.window_nmse_threshold:
            result.append(
                Finding(
                    code="window_error",
                    availability="measured",
                    target=row.target,
                    trajectory_id=row.trajectory_id,
                    evidence_ids=(window.evidence_id,),
                    observed=0,
                    predicted=window.normalized_mse,
                    difference=window.normalized_mse,
                    unit="nmse",
                    threshold=policy.window_nmse_threshold,
                    interpretation=f"{window.label.capitalize()} window NMSE exceeds "
                    f"the display threshold ({window.sample_count} sampled points; "
                    f"time {window.first_time:g} to {window.last_time:g}).",
                    caveats=(
                        *_caveats(row),
                        "Threshold exceedance does not show this window is worse than "
                        "other windows or identify an input-specific response.",
                    ),
                )
            )
    return result


def extract(packet: Packet, policy: Policy | None = None) -> MismatchSummary:
    """Extract deterministic descriptive findings without new numerical work.

    ``packet`` must already have been verified against its M37 source artifacts.
    No source seals, raw arrays, model equations or held-out data are read here.
    """
    selected = policy or Policy()
    response = packet.response
    findings = []
    coverage = {}
    for target in response.targets:
        overview = response.target_overview[target]
        examples = [r for r in response.examples if r.target == target]
        target_findings = []
        if response.status == "available":
            target_findings.extend(
                _bias(target, overview.normalized_signed_bias, selected)
            )
            for row in examples:
                target_findings.extend(_example(row, selected))
        coverage[target] = TargetCoverage(
            overview=overview,
            selected_examples=len(examples),
            omitted_examples=overview.trajectory_count - len(examples),
            empty_selected_windows={
                row.trajectory_id: row.empty_windows
                for row in examples
                if row.empty_windows
            },
            measured_findings=sum(
                f.availability == "measured" for f in target_findings
            ),
            withheld_comparisons=sum(
                f.availability == "withheld" for f in target_findings
            ),
        )
        findings.extend(target_findings)
    missing = {
        reference
        for finding in findings
        for reference in finding.evidence_ids
        if reference not in packet.evidence_catalog
    }
    if missing:
        raise ValueError("mismatch evidence reference is absent from the source packet")
    return MismatchSummary(
        status=response.status,
        policy=selected,
        reason_codes=response.reason_codes,
        targets=response.targets,
        coverage=coverage,
        findings=tuple(findings),
        omitted_trajectory_target_examples=response.omitted_trajectory_target_examples,
        omitted_windows=response.omitted_windows,
    )
