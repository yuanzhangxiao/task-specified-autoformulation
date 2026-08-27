#!/usr/bin/env python3
"""Summarize unlabeled paired-question-consensus scientific comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from autoformalism.judging import (
    HybridScoringConfig,
    question_consensus,
    require_deterministic_orientation_consensus,
    reverse_hybrid_result,
    reverse_paired_assessments,
    score_hybrid_pair,
)
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import (
    HybridJudgeResult,
    PairedAbsoluteAssessment,
    RelativeAssessment,
    RequirementRegistry,
)


def _row_result(row: dict[str, str]) -> HybridJudgeResult:
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


def summarize(
    pairs: tuple[AdversarialPair, ...],
    rows: list[dict[str, str]],
) -> dict[str, object]:
    """Apply frozen identity-normalized question consensus descriptively."""
    by_key: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_key[(row["pair_id"], int(row["repetition"]))][row["order"]] = row
    pair_index = {pair.pair_id: pair for pair in pairs}
    outcomes: list[dict[str, object]] = []
    preference_counts: Counter[str] = Counter()
    for (pair_id, repetition), orientations in sorted(by_key.items()):
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
        consensus, absolute_disagreements, comparative_disagreements = (
            question_consensus(
                _row_result(forward),
                reverse_hybrid_result(_row_result(reverse)),
            )
        )
        deterministic = require_deterministic_orientation_consensus(
            _deterministic(forward),
            reverse_paired_assessments(_deterministic(reverse)),
        )
        requirements = RequirementRegistry.model_validate(
            json.loads(forward["requirements"])
        )
        reverse_requirements = RequirementRegistry.model_validate(
            json.loads(reverse["requirements"])
        )
        if requirements != reverse_requirements:
            raise ValueError(f"orientation requirement mismatch: {pair_id}")
        score = score_hybrid_pair(
            consensus,
            deterministic,
            requirements,
            HybridScoringConfig(
                partial_tiebreak_weight=0.05,
                comparative_weight=0.25,
                tie_threshold=0.05,
                comparative_indeterminate_policy="neutral_fixed_denominator",
            ),
        )
        preferred = {
            "candidate_a": "autoformalism",
            "candidate_b": "raw_agent",
        }.get(score.preferred, score.preferred)
        preference_counts[preferred] += 1
        pair = pair_index[pair_id]
        outcomes.append(
            {
                "pair_id": pair_id,
                "repetition": repetition,
                "status": "complete",
                "benchmark_id": pair.benchmark_id,
                "tier": pair.tier,
                "autoformalism_candidate_id": pair.valid_candidate.candidate_id,
                "raw_agent_candidate_id": pair.adversarial_candidate.candidate_id,
                "preferred": preferred,
                "decision_value_for_autoformalism": score.decision_value,
                "autoformalism_score": score.candidate_a.model_dump(mode="json"),
                "raw_agent_score": score.candidate_b.model_dump(mode="json"),
                "relative_preference_for_autoformalism": (
                    score.relative_preference_for_a
                ),
                "absolute_disagreements": list(absolute_disagreements),
                "comparative_disagreements": list(comparative_disagreements),
                "absolute_assessments": [
                    item.model_dump(mode="json")
                    for item in consensus.absolute_assessments
                ],
                "comparative_assessments": [
                    item.model_dump(mode="json")
                    for item in consensus.comparative_assessments
                ],
            }
        )
    return {
        "schema_version": "raw-data-agent-method-judge-summary-1",
        "pair_truth": "unlabeled",
        "accuracy_claimed": False,
        "pair_count": len(pairs),
        "complete_comparison_count": sum(
            item["status"] == "complete" for item in outcomes
        ),
        "preference_counts": dict(sorted(preference_counts.items())),
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
    result = summarize(pairs, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact = {key: result[key] for key in result if key != "outcomes"}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
