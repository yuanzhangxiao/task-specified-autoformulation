"""Apply frozen gates to fresh-structure question-consensus validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    ExpectedVerdict,
    HybridCalibrationLabels,
)
from autoformalism.schemas import AbsoluteCriterion, PairedAbsoluteAssessment

if __package__:
    from scripts.analyze_hybrid_symmetric_aggregation import (
        RULE_QUESTION_CONSENSUS,
    )
else:
    from analyze_hybrid_symmetric_aggregation import RULE_QUESTION_CONSENSUS


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _direction(value: float | None, threshold: float) -> str:
    if value is None:
        return ExpectedPairPreference.INDETERMINATE.value
    if value > threshold:
        return ExpectedPairPreference.BASELINE.value
    if value < -threshold:
        return ExpectedPairPreference.MUTATED.value
    return ExpectedPairPreference.TIE.value


def _evaluate_gates(
    observed: dict[str, float | None],
    gates: dict[str, float],
) -> dict[str, dict[str, float | str | bool | None]]:
    output = {}
    for name, threshold in gates.items():
        value = observed[name]
        maximum = name.startswith("maximum_")
        passed = value is not None and (
            value <= threshold if maximum else value >= threshold
        )
        output[name] = {
            "observed": value,
            "threshold": threshold,
            "comparison": "maximum" if maximum else "minimum",
            "passed": passed,
        }
    return output


def _consensus_absolute_accuracy(
    trials: list[dict[str, object]],
    labels: dict[str, HybridCalibrationLabels],
) -> dict[AbsoluteCriterion, float | None]:
    outcomes: dict[AbsoluteCriterion, list[bool]] = defaultdict(list)
    for trial in trials:
        pair_labels = labels[str(trial["pair_id"])]
        actual = {
            (item.criterion, item.subject_id): item
            for item in (
                PairedAbsoluteAssessment.model_validate(payload)
                for payload in trial["consensus_absolute_assessments"]
            )
        }
        for expected in pair_labels.absolute_labels:
            if not expected.label_source.startswith("mutation_contract:"):
                continue
            item = actual.get((expected.criterion, expected.subject_id))
            if item is None:
                continue
            for gold, observed in (
                (expected.baseline, item.candidate_a.verdict.value),
                (expected.mutated, item.candidate_b.verdict.value),
            ):
                if gold is ExpectedVerdict.UNLABELED:
                    continue
                outcomes[expected.criterion].append(observed == gold.value)
    return {criterion: _rate(values) for criterion, values in outcomes.items()}


def evaluate_validation(
    symmetric: dict[str, object],
    labels: dict[str, HybridCalibrationLabels],
    *,
    response_success: float | None,
    tie_threshold: float,
    gates: dict[str, float],
) -> dict[str, object]:
    """Evaluate the predeclared primary aggregation rule and diagnostics."""
    trials = list(symmetric["trials"])
    primary = symmetric["rules"][RULE_QUESTION_CONSENSUS]
    labeled = [
        trial
        for trial in trials
        if labels[str(trial["pair_id"])].overall_preference
        is not ExpectedPairPreference.UNLABELED
    ]
    dominance = [
        trial
        for trial in labeled
        if labels[str(trial["pair_id"])].overall_preference
        in {ExpectedPairPreference.BASELINE, ExpectedPairPreference.MUTATED}
    ]
    determined_dominance = [
        trial
        for trial in dominance
        if trial["rules"][RULE_QUESTION_CONSENSUS]["preference"]
        != ExpectedPairPreference.INDETERMINATE.value
    ]
    dominance_by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for trial in dominance:
        dominance_by_pair[str(trial["pair_id"])].append(trial)
    pair_outcomes = []
    for pair_id, pair_trials in dominance_by_pair.items():
        decisions = [
            float(trial["rules"][RULE_QUESTION_CONSENSUS]["decision"])
            for trial in pair_trials
            if trial["rules"][RULE_QUESTION_CONSENSUS]["decision"] is not None
        ]
        prediction = _direction(mean(decisions) if decisions else None, tie_threshold)
        pair_outcomes.append(
            prediction == labels[pair_id].overall_preference.value
        )

    absolute = _consensus_absolute_accuracy(trials, labels)
    half_gaps = [
        float(trial["orientation_half_gap"])
        for trial in trials
        if trial["orientation_half_gap"] is not None
    ]
    equivalence_half_gaps = [
        float(trial["orientation_half_gap"])
        for trial in labeled
        if labels[str(trial["pair_id"])].overall_preference
        is ExpectedPairPreference.TIE
        and trial["orientation_half_gap"] is not None
    ]
    disagreement_count = sum(
        len(trial["comparative_disagreements"]) for trial in trials
    )
    comparative_question_count = len(trials) * 3
    observed = {
        "minimum_response_success": response_success,
        "minimum_paired_response_coverage": symmetric[
            "paired_response_coverage"
        ],
        "minimum_labeled_decision_coverage": primary[
            "labeled_decision_coverage"
        ],
        "minimum_labeled_accuracy": primary[
            "labeled_accuracy_conditional"
        ],
        "minimum_equivalence_tie_accuracy": primary[
            "equivalence_tie_rate"
        ],
        "minimum_known_dominance_accuracy": _rate(
            [
                trial["rules"][RULE_QUESTION_CONSENSUS]["preference"]
                == labels[str(trial["pair_id"])].overall_preference.value
                for trial in determined_dominance
            ]
        ),
        "minimum_known_dominance_pair_accuracy": _rate(pair_outcomes),
        "minimum_wrong_sink_absolute_accuracy": absolute.get(
            AbsoluteCriterion.SOURCE_ROLES_CONSISTENT
        ),
        "minimum_duplicate_absolute_accuracy": absolute.get(
            AbsoluteCriterion.SEMANTIC_FLUXES_NOT_DUPLICATED
        ),
        "minimum_accumulator_absolute_accuracy": absolute.get(
            AbsoluteCriterion.LATENT_ACCUMULATORS_JUSTIFIED
        ),
        "minimum_modal_preference_consistency": primary[
            "mean_pair_modal_preference_consistency"
        ],
        "maximum_mean_repeat_decision_sd": primary["mean_repeat_decision_sd"],
        "maximum_mean_orientation_half_gap": (
            mean(half_gaps) if half_gaps else None
        ),
        "maximum_equivalence_mean_orientation_half_gap": (
            mean(equivalence_half_gaps) if equivalence_half_gaps else None
        ),
        "maximum_comparative_disagreement_rate": (
            disagreement_count / comparative_question_count
            if comparative_question_count
            else None
        ),
    }
    checks = _evaluate_gates(observed, gates)
    unlabeled = [
        trial
        for trial in trials
        if labels[str(trial["pair_id"])].overall_preference
        is ExpectedPairPreference.UNLABELED
    ]
    return {
        "schema_version": "hybrid-judge-consensus-validation-result-1",
        "passed": all(bool(item["passed"]) for item in checks.values()),
        "primary_rule": RULE_QUESTION_CONSENSUS,
        "checks": checks,
        "known_dominance_trials": len(dominance),
        "known_dominance_determined_trials": len(determined_dominance),
        "known_dominance_pairs": len(dominance_by_pair),
        "unlabeled_tradeoff_trials": len(unlabeled),
        "unlabeled_tradeoff_preference_counts": dict(
            Counter(
                str(trial["rules"][RULE_QUESTION_CONSENSUS]["preference"])
                for trial in unlabeled
            )
        ),
        "consensus_absolute_accuracy_by_criterion": {
            criterion.value: value for criterion, value in absolute.items()
        },
        "comparative_disagreement_count": disagreement_count,
        "comparative_question_count": comparative_question_count,
        "interpretation_boundary": (
            "Unlabeled defect-tradeoff preferences do not contribute to any "
            "accuracy gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--symmetric-analysis", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.scores.open(encoding="utf-8", newline="") as handle:
        score_rows = list(csv.DictReader(handle))
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
    symmetric = json.loads(args.symmetric_analysis.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    if protocol["aggregation"]["primary_rule"] != RULE_QUESTION_CONSENSUS:
        raise ValueError("protocol does not freeze paired question consensus")
    total = len(score_rows) + len(failures)
    response_success = len(score_rows) / total if total else None
    result = {
        "judge_model": protocol["judge_model"],
        "protocol_status": protocol["status"],
        **evaluate_validation(
            symmetric,
            labels,
            response_success=response_success,
            tie_threshold=protocol["protocol"]["scoring"]["tie_threshold"],
            gates=protocol["validation_gate"],
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = args.output.with_suffix(".md")
    lines = [
        "# Frozen question-consensus validation",
        "",
        f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**.",
        "",
        "Unlabeled tradeoff winners are excluded from every accuracy gate.",
        "",
        "| Predeclared gate | Observed | Threshold | Result |",
        "|---|---:|---:|:---:|",
    ]
    for name, item in result["checks"].items():
        observed = item["observed"]
        observed_text = "N/A" if observed is None else f"{observed:.3f}"
        comparator = "≤" if item["comparison"] == "maximum" else "≥"
        lines.append(
            f"| {name} | {observed_text} | {comparator} "
            f"{item['threshold']:.3f} | "
            f"{'pass' if item['passed'] else 'fail'} |"
        )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
