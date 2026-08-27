#!/usr/bin/env python3
"""Summarize fit-free deterministic and scientific audits of raw-agent models."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from autoformalism.judging import (
    question_consensus,
    require_deterministic_orientation_consensus,
    reverse_hybrid_result,
    reverse_paired_assessments,
)
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    HybridJudgeResult,
    PairedAbsoluteAssessment,
    RelativeAssessment,
    RequirementEnforcement,
    RequirementRegistry,
)


def _result(row: dict[str, str]) -> HybridJudgeResult:
    return HybridJudgeResult(
        absolute_assessments=tuple(
            PairedAbsoluteAssessment.model_validate(item)
            for item in json.loads(row["absolute_assessments"])
        ),
        comparative_assessments=tuple(
            RelativeAssessment.model_validate(item)
            for item in json.loads(row["comparative_assessments"])
        ),
    )


def _deterministic(row: dict[str, str]) -> tuple[PairedAbsoluteAssessment, ...]:
    return tuple(
        PairedAbsoluteAssessment.model_validate(item)
        for item in json.loads(row["deterministic_assessments"])
    )


def _collapse_identical_candidate(
    assessment: PairedAbsoluteAssessment,
) -> dict[str, str]:
    left = assessment.candidate_a
    right = assessment.candidate_b
    verdict = (
        left.verdict
        if left.verdict is right.verdict
        else AbsoluteVerdict.INDETERMINATE
    )
    evidence = (
        left.evidence
        if left.verdict is right.verdict
        else "Duplicate candidate readings disagreed; withheld as indeterminate."
    )
    return {
        "criterion": assessment.criterion.value,
        "subject_id": assessment.subject_id,
        "verdict": verdict.value,
        "evidence": evidence,
    }


def summarize(
    pairs: tuple[AdversarialPair, ...], rows: list[dict[str, str]]
) -> dict[str, object]:
    """Return question-level audits without fitting or interpreting comparisons."""
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["pair_id"], int(row["repetition"]))][row["order"]] = row
    pair_index = {pair.pair_id: pair for pair in pairs}
    outcomes = []
    status_counts: Counter[str] = Counter()
    for (pair_id, repetition), orientations in sorted(grouped.items()):
        if set(orientations) != {"baseline_a", "baseline_b"}:
            outcomes.append(
                {
                    "pair_id": pair_id,
                    "repetition": repetition,
                    "status": "incomplete_orientations",
                    "available_orientations": sorted(orientations),
                }
            )
            continue
        forward = orientations["baseline_a"]
        reverse = orientations["baseline_b"]
        consensus, absolute_disagreements, _ = question_consensus(
            _result(forward), reverse_hybrid_result(_result(reverse))
        )
        deterministic = require_deterministic_orientation_consensus(
            _deterministic(forward),
            reverse_paired_assessments(_deterministic(reverse)),
        )
        requirements = RequirementRegistry.model_validate_json(
            forward["requirements"]
        )
        absolute = [
            _collapse_identical_candidate(item)
            for item in consensus.absolute_assessments
        ]
        runtime = [_collapse_identical_candidate(item) for item in deterministic]
        absolute_index = {
            (item["criterion"], item["subject_id"]): item["verdict"]
            for item in absolute
        }
        hard_verdicts = []
        for requirement in requirements.requirements:
            if requirement.enforcement is not RequirementEnforcement.HARD:
                continue
            for criterion in (
                AbsoluteCriterion.REQUIRED_MECHANISM_REPRESENTED,
                AbsoluteCriterion.REQUIRED_MECHANISM_CONNECTED,
            ):
                hard_verdicts.append(
                    absolute_index.get(
                        (criterion.value, requirement.requirement_id),
                        AbsoluteVerdict.INDETERMINATE.value,
                    )
                )
        runtime_verdicts = [item["verdict"] for item in runtime]
        if AbsoluteVerdict.FAIL.value in (*runtime_verdicts, *hard_verdicts):
            compliance = "fail"
        elif AbsoluteVerdict.INDETERMINATE.value in (
            *runtime_verdicts,
            *hard_verdicts,
        ):
            compliance = "indeterminate"
        else:
            compliance = "pass"
        status_counts[compliance] += 1
        pair = pair_index[pair_id]
        outcomes.append(
            {
                "pair_id": pair_id,
                "repetition": repetition,
                "status": "complete",
                "benchmark_id": pair.benchmark_id,
                "tier": pair.tier,
                "candidate_id": pair.valid_candidate.candidate_id,
                "task_compliance": compliance,
                "deterministic_assessments": runtime,
                "scientific_absolute_assessments": absolute,
                "public_requirements": requirements.model_dump(mode="json"),
                "absolute_disagreements": list(absolute_disagreements),
            }
        )
    complete = sum(item["status"] == "complete" for item in outcomes)
    expected = 2 * len(pairs)
    atomic_repairs = sum(
        int(row.get("atomic_missing_occurrence_repairs") or 0)
        + int(row.get("atomic_missing_repeat_repairs") or 0)
        for row in rows
    )
    return {
        "schema_version": "raw-agent-scientific-audit-summary-1",
        "parameter_fitting_used": False,
        "accuracy_claimed": False,
        "comparative_outcomes_interpreted": False,
        "pair_count": len(pairs),
        "successful_response_count": len(rows),
        "expected_response_count": expected,
        "response_success_rate": len(rows) / expected if expected else None,
        "paired_response_coverage": complete / len(pairs) if pairs else None,
        "responses_with_neutral_atomic_unit_repairs": sum(
            (
                int(row.get("atomic_missing_occurrence_repairs") or 0)
                + int(row.get("atomic_missing_repeat_repairs") or 0)
            )
            > 0
            for row in rows
        ),
        "neutral_atomic_unit_repair_count": atomic_repairs,
        "task_compliance_counts": dict(sorted(status_counts.items())),
        "outcomes": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    with args.scores.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize(pairs, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact = {key: value for key, value in summary.items() if key != "outcomes"}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
