"""Apply frozen gates to target-mapping and initialization judge validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    HybridCalibrationLabels,
)
from autoformalism.schemas import AbsoluteCriterion

if __package__:
    from scripts.analyze_hybrid_consensus_validation import (
        _consensus_absolute_accuracy,
        _direction,
        _evaluate_gates,
    )
    from scripts.analyze_hybrid_symmetric_aggregation import (
        RULE_QUESTION_CONSENSUS,
    )
else:
    from analyze_hybrid_consensus_validation import (
        _consensus_absolute_accuracy,
        _direction,
        _evaluate_gates,
    )
    from analyze_hybrid_symmetric_aggregation import RULE_QUESTION_CONSENSUS


def evaluate_model_semantics(
    symmetric: dict[str, object],
    labels: dict[str, HybridCalibrationLabels],
    *,
    response_success: float | None,
    tie_threshold: float,
    gates: dict[str, float],
) -> dict[str, object]:
    """Evaluate the frozen paired-question-consensus operating point."""
    trials = list(symmetric["trials"])
    primary = symmetric["rules"][RULE_QUESTION_CONSENSUS]
    labeled = [
        trial
        for trial in trials
        if labels[str(trial["pair_id"])].overall_preference
        is not ExpectedPairPreference.UNLABELED
    ]
    determined = [
        trial
        for trial in labeled
        if trial["rules"][RULE_QUESTION_CONSENSUS]["preference"]
        != ExpectedPairPreference.INDETERMINATE.value
    ]
    by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for trial in labeled:
        by_pair[str(trial["pair_id"])].append(trial)
    pair_outcomes = []
    for pair_id, pair_trials in by_pair.items():
        decisions = [
            float(trial["rules"][RULE_QUESTION_CONSENSUS]["decision"])
            for trial in pair_trials
            if trial["rules"][RULE_QUESTION_CONSENSUS]["decision"] is not None
        ]
        prediction = _direction(mean(decisions) if decisions else None, tie_threshold)
        pair_outcomes.append(prediction == labels[pair_id].overall_preference.value)

    absolute = _consensus_absolute_accuracy(trials, labels)
    half_gaps = [
        float(trial["orientation_half_gap"])
        for trial in trials
        if trial["orientation_half_gap"] is not None
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
        "minimum_pair_aggregate_accuracy": (
            sum(pair_outcomes) / len(pair_outcomes) if pair_outcomes else None
        ),
        "minimum_target_mapping_absolute_accuracy": absolute.get(
            AbsoluteCriterion.TARGET_MAPPING_SEMANTICALLY_CONSISTENT
        ),
        "minimum_initialization_absolute_accuracy": absolute.get(
            AbsoluteCriterion.INITIALIZATION_SEMANTICALLY_CONSISTENT
        ),
        "minimum_modal_preference_consistency": primary[
            "mean_pair_modal_preference_consistency"
        ],
        "maximum_mean_repeat_decision_sd": primary["mean_repeat_decision_sd"],
        "maximum_mean_orientation_half_gap": (
            mean(half_gaps) if half_gaps else None
        ),
        "maximum_comparative_disagreement_rate": (
            disagreement_count / comparative_question_count
            if comparative_question_count
            else None
        ),
    }
    checks = _evaluate_gates(observed, gates)
    return {
        "schema_version": "hybrid-judge-model-semantics-validation-result-1",
        "passed": all(bool(item["passed"]) for item in checks.values()),
        "primary_rule": RULE_QUESTION_CONSENSUS,
        "checks": checks,
        "labeled_trials": len(labeled),
        "determined_labeled_trials": len(determined),
        "labeled_pairs": len(by_pair),
        "consensus_absolute_accuracy_by_criterion": {
            criterion.value: value for criterion, value in absolute.items()
        },
        "comparative_disagreement_count": disagreement_count,
        "comparative_question_count": comparative_question_count,
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
    result = {
        "judge_model": protocol["judge_model"],
        "protocol_status": protocol["status"],
        **evaluate_model_semantics(
            symmetric,
            labels,
            response_success=len(score_rows) / total if total else None,
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
        "# Frozen model-semantics judge validation",
        "",
        f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**.",
        "",
        "The mutation contract is opened only after frozen judge calls.",
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
