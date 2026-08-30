"""Deterministic aggregation for paired target-completeness judgments."""

from __future__ import annotations

from autoformalism.schemas import (
    AbsoluteVerdict,
    PairedTargetCompletenessJudgeResult,
    TargetCompletenessAssessment,
    TargetCompletenessJudgeResult,
)


def fail_dominant_verdict(
    first: AbsoluteVerdict,
    second: AbsoluteVerdict,
) -> AbsoluteVerdict:
    """Combine two orientations for a mandatory public requirement."""
    if AbsoluteVerdict.FAIL in (first, second):
        return AbsoluteVerdict.FAIL
    if first is AbsoluteVerdict.PASS and second is AbsoluteVerdict.PASS:
        return AbsoluteVerdict.PASS
    return AbsoluteVerdict.INDETERMINATE


def paired_target_question_consensus(
    forward: PairedTargetCompletenessJudgeResult,
    reverse: PairedTargetCompletenessJudgeResult,
) -> tuple[
    TargetCompletenessJudgeResult,
    TargetCompletenessJudgeResult,
    tuple[str, ...],
]:
    """Normalize A/B and B/A responses and aggregate by stable identity.

    ``forward`` presents baseline as A and mutated as B. ``reverse`` presents
    mutated as A and baseline as B. A failure in either orientation fails the
    corresponding candidate/target requirement; pass requires two passes.
    """
    forward_by_target = {
        item.target_id: item for item in forward.target_assessments
    }
    reverse_by_target = {
        item.target_id: item for item in reverse.target_assessments
    }
    if forward_by_target.keys() != reverse_by_target.keys():
        raise ValueError("paired target orientations contain different targets")

    baseline: list[TargetCompletenessAssessment] = []
    mutated: list[TargetCompletenessAssessment] = []
    disagreements: list[str] = []
    for target_id in sorted(forward_by_target):
        forward_item = forward_by_target[target_id]
        reverse_item = reverse_by_target[target_id]
        stable_assessments = (
            (
                "baseline",
                forward_item.candidate_a,
                reverse_item.candidate_b,
                baseline,
            ),
            (
                "mutated",
                forward_item.candidate_b,
                reverse_item.candidate_a,
                mutated,
            ),
        )
        for role, first, second, destination in stable_assessments:
            if first.verdict is not second.verdict:
                disagreements.append(f"{target_id}:{role}")
            destination.append(
                TargetCompletenessAssessment(
                    target_id=target_id,
                    verdict=fail_dominant_verdict(
                        first.verdict,
                        second.verdict,
                    ),
                    evidence=(
                        "A/B orientation: "
                        f"{first.evidence} B/A orientation: {second.evidence}"
                    ),
                )
            )

    return (
        TargetCompletenessJudgeResult(target_assessments=tuple(baseline)),
        TargetCompletenessJudgeResult(target_assessments=tuple(mutated)),
        tuple(disagreements),
    )
