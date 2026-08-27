"""Compare symmetry-preserving aggregation rules on frozen hybrid calls.

This is a post-hoc development analysis. It never assigns a correct winner to
an unlabeled tradeoff and never makes a new LLM request.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

from autoformalism.judging import HybridScoringConfig, score_hybrid_pair
from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    HybridCalibrationLabels,
)
from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    CandidateAbsoluteAssessment,
    HybridJudgeResult,
    PairedAbsoluteAssessment,
    RelativeAssessment,
    RelativeCriterion,
    RelativeVerdict,
    RequirementRegistry,
)

RULE_FINAL_MEAN = "paired_final_decision_mean"
RULE_QUESTION_CONSENSUS = "paired_question_consensus"
RULE_UNCERTAINTY_ABSTENTION = "paired_uncertainty_abstention"
RULES = (
    RULE_FINAL_MEAN,
    RULE_QUESTION_CONSENSUS,
    RULE_UNCERTAINTY_ABSTENTION,
)


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _direction(value: float | None, threshold: float) -> str:
    if value is None:
        return ExpectedPairPreference.INDETERMINATE.value
    if value > threshold:
        return ExpectedPairPreference.BASELINE.value
    if value < -threshold:
        return ExpectedPairPreference.MUTATED.value
    return ExpectedPairPreference.TIE.value


def _uncertainty_direction(
    center: float | None,
    half_gap: float | None,
    threshold: float,
) -> str:
    """Return a verdict only when the orientation interval supports it."""
    if center is None or half_gap is None:
        return ExpectedPairPreference.INDETERMINATE.value
    lower = center - half_gap
    upper = center + half_gap
    if lower > threshold:
        return ExpectedPairPreference.BASELINE.value
    if upper < -threshold:
        return ExpectedPairPreference.MUTATED.value
    if lower >= -threshold and upper <= threshold:
        return ExpectedPairPreference.TIE.value
    return ExpectedPairPreference.INDETERMINATE.value


def _normalized_absolute(
    row: dict[str, str],
) -> tuple[PairedAbsoluteAssessment, ...]:
    """Normalize persisted A/B assessments to first/second pair identity."""
    baseline_is_a = row["baseline_position"] == "A"
    output = []
    for payload in json.loads(row["absolute_assessments"]):
        item = PairedAbsoluteAssessment.model_validate(payload)
        first, second = (
            (item.candidate_a, item.candidate_b)
            if baseline_is_a
            else (item.candidate_b, item.candidate_a)
        )
        output.append(
            PairedAbsoluteAssessment(
                criterion=item.criterion,
                subject_id=item.subject_id,
                candidate_a=first,
                candidate_b=second,
            )
        )
    return tuple(output)


def _normalized_deterministic(
    row: dict[str, str],
) -> tuple[PairedAbsoluteAssessment, ...]:
    baseline_is_a = row["baseline_position"] == "A"
    output = []
    for payload in json.loads(row["deterministic_assessments"]):
        item = PairedAbsoluteAssessment.model_validate(payload)
        first, second = (
            (item.candidate_a, item.candidate_b)
            if baseline_is_a
            else (item.candidate_b, item.candidate_a)
        )
        output.append(
            PairedAbsoluteAssessment(
                criterion=item.criterion,
                subject_id=item.subject_id,
                candidate_a=first,
                candidate_b=second,
            )
        )
    return tuple(output)


def _normalized_comparative(
    row: dict[str, str],
) -> tuple[RelativeAssessment, ...]:
    """Normalize comparative verdicts to first/second pair identity."""
    baseline_is_a = row["baseline_position"] == "A"
    output = []
    for payload in json.loads(row["comparative_assessments"]):
        item = RelativeAssessment.model_validate(payload)
        verdict = item.verdict
        if verdict is RelativeVerdict.CANDIDATE_A and not baseline_is_a:
            verdict = RelativeVerdict.CANDIDATE_B
        elif verdict is RelativeVerdict.CANDIDATE_B and not baseline_is_a:
            verdict = RelativeVerdict.CANDIDATE_A
        output.append(
            RelativeAssessment(
                criterion=item.criterion,
                verdict=verdict,
                evidence=item.evidence,
            )
        )
    return tuple(output)


def _absolute_consensus(
    left: tuple[PairedAbsoluteAssessment, ...],
    right: tuple[PairedAbsoluteAssessment, ...],
    *,
    fail_dominant_criteria: frozenset[AbsoluteCriterion] = frozenset(),
) -> tuple[tuple[PairedAbsoluteAssessment, ...], tuple[str, ...]]:
    """Keep matching verdicts and mark orientation disagreements unknown."""
    left_index = {(item.criterion, item.subject_id): item for item in left}
    right_index = {(item.criterion, item.subject_id): item for item in right}
    if left_index.keys() != right_index.keys():
        raise ValueError("orientation absolute-unit sets differ")
    output = []
    disagreements = []
    for key in sorted(left_index, key=lambda item: (item[0].value, item[1])):
        first = left_index[key]
        second = right_index[key]
        candidates = []
        for side in ("candidate_a", "candidate_b"):
            first_verdict = getattr(first, side).verdict
            second_verdict = getattr(second, side).verdict
            if first_verdict is second_verdict:
                verdict = first_verdict
                evidence = "Symmetric orientation consensus."
            elif (
                key[0] in fail_dominant_criteria
                and AbsoluteVerdict.FAIL in {first_verdict, second_verdict}
            ):
                verdict = AbsoluteVerdict.FAIL
                evidence = (
                    "Orientations disagreed; hard public requirement failed "
                    "closed because one orientation detected a violation."
                )
                disagreements.append(f"{key[0].value}:{key[1]}:{side}")
            else:
                verdict = AbsoluteVerdict.INDETERMINATE
                evidence = "Orientations disagreed; withheld as indeterminate."
                disagreements.append(f"{key[0].value}:{key[1]}:{side}")
            candidates.append(
                CandidateAbsoluteAssessment(
                    verdict=verdict,
                    evidence=evidence,
                )
            )
        output.append(
            PairedAbsoluteAssessment(
                criterion=key[0],
                subject_id=key[1],
                candidate_a=candidates[0],
                candidate_b=candidates[1],
            )
        )
    return tuple(output), tuple(disagreements)


def _comparative_consensus(
    left: tuple[RelativeAssessment, ...],
    right: tuple[RelativeAssessment, ...],
) -> tuple[tuple[RelativeAssessment, ...], tuple[str, ...]]:
    """Accept a comparative verdict only when both orientations agree."""
    left_index = {item.criterion: item for item in left}
    right_index = {item.criterion: item for item in right}
    if left_index.keys() != right_index.keys() or set(left_index) != set(
        RelativeCriterion
    ):
        raise ValueError("orientation comparative criteria differ")
    output = []
    disagreements = []
    for criterion in RelativeCriterion:
        first = left_index[criterion].verdict
        second = right_index[criterion].verdict
        if first is second:
            verdict = first
            evidence = "Symmetric orientations agree."
        else:
            verdict = RelativeVerdict.INDETERMINATE
            evidence = "Orientations disagreed; withheld as indeterminate."
            disagreements.append(criterion.value)
        output.append(
            RelativeAssessment(
                criterion=criterion,
                verdict=verdict,
                evidence=evidence,
            )
        )
    return tuple(output), tuple(disagreements)


def _deterministic_consensus(
    left: tuple[PairedAbsoluteAssessment, ...],
    right: tuple[PairedAbsoluteAssessment, ...],
) -> tuple[PairedAbsoluteAssessment, ...]:
    """Require deterministic facts to be exactly orientation-invariant."""
    left_index = {(item.criterion, item.subject_id): item for item in left}
    right_index = {(item.criterion, item.subject_id): item for item in right}
    if left_index.keys() != right_index.keys():
        raise ValueError("orientation deterministic-unit sets differ")
    for key, first in left_index.items():
        second = right_index[key]
        if (
            first.candidate_a.verdict is not second.candidate_a.verdict
            or first.candidate_b.verdict is not second.candidate_b.verdict
        ):
            raise ValueError(f"deterministic orientation mismatch: {key}")
    return tuple(left_index[key] for key in sorted(
        left_index, key=lambda item: (item[0].value, item[1])
    ))


def aggregate_trial(
    row_a: dict[str, str],
    row_b: dict[str, str],
    *,
    config: HybridScoringConfig,
) -> dict[str, object]:
    """Apply all three offline rules to one paired-orientation trial."""
    if row_a["order"] != "baseline_a" or row_b["order"] != "baseline_b":
        raise ValueError("aggregate_trial requires baseline_a then baseline_b")
    identity = ("pair_id", "judge_model", "repetition", "mutation_type")
    if any(row_a[key] != row_b[key] for key in identity):
        raise ValueError("orientation rows do not describe the same trial")
    requirements_a = RequirementRegistry.model_validate_json(row_a["requirements"])
    requirements_b = RequirementRegistry.model_validate_json(row_b["requirements"])
    if requirements_a != requirements_b:
        raise ValueError("orientation requirement registries differ")

    value_a = _optional_float(row_a["baseline_decision_value"])
    value_b = _optional_float(row_b["baseline_decision_value"])
    center = (
        None if value_a is None or value_b is None else mean((value_a, value_b))
    )
    half_gap = (
        None
        if value_a is None or value_b is None
        else abs(value_a - value_b) / 2.0
    )

    absolute, absolute_disagreements = _absolute_consensus(
        _normalized_absolute(row_a),
        _normalized_absolute(row_b),
        fail_dominant_criteria=(
            frozenset(
                {AbsoluteCriterion.TARGET_MAPPING_SEMANTICALLY_CONSISTENT}
            )
            if config.target_mapping_consensus == "fail_dominant"
            else frozenset()
        ),
    )
    comparative, comparative_disagreements = _comparative_consensus(
        _normalized_comparative(row_a),
        _normalized_comparative(row_b),
    )
    deterministic = _deterministic_consensus(
        _normalized_deterministic(row_a),
        _normalized_deterministic(row_b),
    )
    consensus_result = HybridJudgeResult(
        absolute_assessments=absolute,
        comparative_assessments=comparative,
    )
    consensus_score = score_hybrid_pair(
        consensus_result,
        deterministic,
        requirements_a,
        config,
    )
    consensus_value = consensus_score.decision_value

    return {
        "pair_id": row_a["pair_id"],
        "judge_model": row_a["judge_model"],
        "mutation_type": row_a["mutation_type"],
        "repetition": int(row_a["repetition"]),
        "orientation_values": {
            "baseline_a": value_a,
            "baseline_b": value_b,
        },
        "orientation_center": center,
        "orientation_half_gap": half_gap,
        "absolute_disagreements": list(absolute_disagreements),
        "comparative_disagreements": list(comparative_disagreements),
        "consensus_absolute_assessments": [
            item.model_dump(mode="json") for item in absolute
        ],
        "consensus_comparative_assessments": [
            item.model_dump(mode="json") for item in comparative
        ],
        "rules": {
            RULE_FINAL_MEAN: {
                "decision": center,
                "preference": _direction(center, config.tie_threshold),
            },
            RULE_QUESTION_CONSENSUS: {
                "decision": consensus_value,
                "preference": _direction(
                    consensus_value, config.tie_threshold
                ),
            },
            RULE_UNCERTAINTY_ABSTENTION: {
                "decision": center,
                "preference": _uncertainty_direction(
                    center,
                    half_gap,
                    config.tie_threshold,
                ),
            },
        },
    }


def analyze(
    rows: list[dict[str, str]],
    failures: list[dict[str, object]],
    labels: dict[str, HybridCalibrationLabels],
    *,
    config: HybridScoringConfig,
) -> dict[str, object]:
    """Compare symmetric rules without scoring unlabeled tradeoff winners."""
    grouped: dict[tuple[str, str, int], dict[str, dict[str, str]]] = defaultdict(
        dict
    )
    attempted = set()
    for row in rows:
        key = (row["judge_model"], row["pair_id"], int(row["repetition"]))
        grouped[key][row["order"]] = row
        attempted.add(key)
    for failure in failures:
        attempted.add(
            (
                str(failure["judge_model"]),
                str(failure["pair_id"]),
                int(failure["repetition"]),
            )
        )

    trials = []
    for orientations in grouped.values():
        if {"baseline_a", "baseline_b"} <= orientations.keys():
            trials.append(
                aggregate_trial(
                    orientations["baseline_a"],
                    orientations["baseline_b"],
                    config=config,
                )
            )

    rule_metrics = {}
    for rule in RULES:
        determined = [
            item
            for item in trials
            if item["rules"][rule]["preference"]
            != ExpectedPairPreference.INDETERMINATE.value
        ]
        labeled = [
            item
            for item in trials
            if labels[str(item["pair_id"])].overall_preference
            is not ExpectedPairPreference.UNLABELED
        ]
        labeled_determined = [item for item in labeled if item in determined]
        tradeoffs = [
            item
            for item in trials
            if labels[str(item["pair_id"])].overall_preference
            is ExpectedPairPreference.UNLABELED
        ]
        decisions_by_pair: dict[str, list[float]] = defaultdict(list)
        preferences_by_pair: dict[str, list[str]] = defaultdict(list)
        for item in trials:
            result = item["rules"][rule]
            decision = result["decision"]
            if decision is not None:
                decisions_by_pair[str(item["pair_id"])].append(float(decision))
            preferences_by_pair[str(item["pair_id"])].append(
                str(result["preference"])
            )
        repeat_sds = [
            pstdev(values)
            for values in decisions_by_pair.values()
            if len(values) > 1
        ]
        modal_consistency = [
            max(Counter(values).values()) / len(values)
            for values in preferences_by_pair.values()
            if values
        ]
        rule_metrics[rule] = {
            "decision_coverage": len(determined) / len(trials) if trials else None,
            "labeled_decision_coverage": (
                len(labeled_determined) / len(labeled) if labeled else None
            ),
            "labeled_accuracy_conditional": _rate(
                [
                    item["rules"][rule]["preference"]
                    == labels[str(item["pair_id"])].overall_preference.value
                    for item in labeled_determined
                ]
            ),
            "equivalence_tie_rate": _rate(
                [
                    item["rules"][rule]["preference"]
                    == ExpectedPairPreference.TIE.value
                    for item in labeled
                    if labels[str(item["pair_id"])].overall_preference
                    is ExpectedPairPreference.TIE
                ]
            ),
            "tradeoff_decision_coverage": _rate(
                [
                    item["rules"][rule]["preference"]
                    != ExpectedPairPreference.INDETERMINATE.value
                    for item in tradeoffs
                ]
            ),
            "tradeoff_preference_counts": dict(
                Counter(
                    str(item["rules"][rule]["preference"])
                    for item in tradeoffs
                )
            ),
            "mean_repeat_decision_sd": _mean(repeat_sds),
            "mean_pair_modal_preference_consistency": _mean(modal_consistency),
        }

    comparative_disagreements: Counter[tuple[str, str]] = Counter()
    absolute_disagreements: Counter[tuple[str, str]] = Counter()
    orientation_by_type: dict[str, list[float]] = defaultdict(list)
    for item in trials:
        mutation = str(item["mutation_type"])
        half_gap = item["orientation_half_gap"]
        if half_gap is not None:
            orientation_by_type[mutation].append(float(half_gap))
        for criterion in item["comparative_disagreements"]:
            comparative_disagreements[(mutation, str(criterion))] += 1
        for unit in item["absolute_disagreements"]:
            absolute_disagreements[(mutation, str(unit))] += 1

    return {
        "schema_version": "hybrid-symmetric-aggregation-analysis-1",
        "status": "posthoc_development_analysis",
        "new_llm_calls": 0,
        "target_mapping_consensus": config.target_mapping_consensus,
        "attempted_paired_trials": len(attempted),
        "complete_paired_trials": len(trials),
        "paired_response_coverage": len(trials) / len(attempted) if attempted else None,
        "rules": rule_metrics,
        "orientation_uncertainty_by_pair_type": {
            mutation: {
                "trials": len(values),
                "mean_half_gap": mean(values),
                "max_half_gap": max(values),
            }
            for mutation, values in sorted(orientation_by_type.items())
        },
        "comparative_disagreements": [
            {
                "mutation_type": mutation,
                "criterion": criterion,
                "count": count,
            }
            for (mutation, criterion), count in sorted(
                comparative_disagreements.items()
            )
        ],
        "absolute_disagreements": [
            {
                "mutation_type": mutation,
                "unit": unit,
                "count": count,
            }
            for (mutation, unit), count in sorted(absolute_disagreements.items())
        ],
        "interpretation_boundary": (
            "Only labeled equivalence trials have an overall accuracy target; "
            "tradeoff preference and coverage are descriptive."
        ),
        "trials": trials,
    }


def _summary(result: dict[str, object]) -> str:
    consensus_description = (
        "Question consensus fails a hard target requirement when either "
        "normalized orientation detects a violation; other disagreements "
        "remain indeterminate"
        if result.get("target_mapping_consensus") == "fail_dominant"
        else (
            "Question consensus marks any normalized orientation disagreement "
            "indeterminate"
        )
    )
    lines = [
        "# Symmetry-preserving hybrid aggregation",
        "",
        "Frozen calls are reused without new LLM requests. Unlabeled tradeoff "
        "winners are never counted as accuracy.",
        "",
        "| Rule | Coverage | Labeled coverage | Labeled accuracy | "
        "Equivalence tie | Tradeoff coverage | Repeat SD | Modal consistency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in RULES:
        item = result["rules"][name]

        def display(value: object) -> str:
            return "N/A" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {name} | {display(item['decision_coverage'])} | "
            f"{display(item['labeled_decision_coverage'])} | "
            f"{display(item['labeled_accuracy_conditional'])} | "
            f"{display(item['equivalence_tie_rate'])} | "
            f"{display(item['tradeoff_decision_coverage'])} | "
            f"{display(item['mean_repeat_decision_sd'])} | "
            f"{display(item['mean_pair_modal_preference_consistency'])} |"
        )
    lines.extend(
        (
            "",
            "Orientation uncertainty is retained as half the normalized "
            f"A/B decision gap. {consensus_description}; uncertainty abstention "
            "withholds a decision when the orientation interval crosses the "
            "tie boundary.",
        )
    )
    lines.extend(
        (
            "",
            "## Tradeoff outcomes by rule",
            "",
            "These counts are descriptive because tradeoff winners are unlabeled.",
            "",
            "| Rule | First | Second | Tie | Indeterminate |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for name in RULES:
        counts = result["rules"][name]["tradeoff_preference_counts"]
        lines.append(
            f"| {name} | {counts.get('baseline', 0)} | "
            f"{counts.get('mutated', 0)} | {counts.get('tie', 0)} | "
            f"{counts.get('indeterminate', 0)} |"
        )
    lines.extend(
        (
            "",
            "## Raw orientation uncertainty",
            "",
            "| Pair type | Trials | Mean half-gap | Maximum half-gap |",
            "|---|---:|---:|---:|",
        )
    )
    for mutation, item in result["orientation_uncertainty_by_pair_type"].items():
        lines.append(
            f"| {mutation} | {item['trials']} | "
            f"{item['mean_half_gap']:.3f} | {item['max_half_gap']:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Comparative orientation disagreements",
            "",
            "| Pair type | Criterion | Count |",
            "|---|---|---:|",
        )
    )
    for item in result["comparative_disagreements"]:
        lines.append(
            f"| {item['mutation_type']} | {item['criterion']} | "
            f"{item['count']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.scores.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    failures = [
        json.loads(line)
        for line in args.failures.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = {
        item.pair_id: item
        for item in (
            HybridCalibrationLabels.model_validate_json(line)
            for line in args.labels.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    protocol = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    scoring = protocol["protocol"]["scoring"]
    config = HybridScoringConfig(
        partial_tiebreak_weight=scoring["partial_tiebreak_weight"],
        comparative_weight=scoring["comparative_weight"],
        tie_threshold=scoring["tie_threshold"],
        comparative_indeterminate_policy=scoring.get(
            "comparative_indeterminate_policy",
            "exclude",
        ),
        include_model_semantics=scoring.get("include_model_semantics", False),
        include_target_mapping_semantics=scoring.get(
            "include_target_mapping_semantics", False
        ),
        include_initialization_semantics=scoring.get(
            "include_initialization_semantics", False
        ),
        target_mapping_enforcement=scoring.get(
            "target_mapping_enforcement", "soft"
        ),
        target_mapping_consensus=scoring.get(
            "target_mapping_consensus", "indeterminate"
        ),
    )
    result = analyze(rows, failures, labels, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = args.output.with_suffix(".md")
    summary.write_text(_summary(result), encoding="utf-8")
    print(summary.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
