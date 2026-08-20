"""Analyze hybrid judge outputs against question-level calibration labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    ExpectedVerdict,
    HybridCalibrationLabels,
)


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _direction(value: float | None, threshold: float) -> str:
    if value is None:
        return "indeterminate"
    if value > threshold:
        return "baseline"
    if value < -threshold:
        return "mutated"
    return "tie"


def _icc_one_one(groups: list[list[float]]) -> float | None:
    """Return one-way random-effects single-measure repeat ICC."""
    if len(groups) < 2 or not groups or len(groups[0]) < 2:
        return None
    repeat_count = len(groups[0])
    if any(len(group) != repeat_count for group in groups):
        return None
    row_means = [mean(group) for group in groups]
    grand_mean = mean(value for group in groups for value in group)
    between = repeat_count * sum(
        (row_mean - grand_mean) ** 2 for row_mean in row_means
    ) / (len(groups) - 1)
    within = sum(
        (value - row_mean) ** 2
        for group, row_mean in zip(groups, row_means, strict=True)
        for value in group
    ) / (len(groups) * (repeat_count - 1))
    denominator = between + (repeat_count - 1) * within
    return None if denominator == 0.0 else (between - within) / denominator


def _bootstrap_accuracy_ci(
    outcomes: dict[str, bool], *, repetitions: int = 2000
) -> list[float] | None:
    """Bootstrap a 95% interval by unique pair, never by repeated call."""
    if not outcomes:
        return None
    identifiers = sorted(outcomes)
    generator = random.Random(0)
    estimates = sorted(
        mean(
            outcomes[generator.choice(identifiers)]
            for _ in range(len(identifiers))
        )
        for _ in range(repetitions)
    )
    return [
        estimates[int(0.025 * (repetitions - 1))],
        estimates[int(0.975 * (repetitions - 1))],
    ]


def _normalized_absolute(
    rows: list[dict[str, object]], baseline_position: str
) -> dict[tuple[str, str], tuple[str, str]]:
    output = {}
    for item in rows:
        key = (str(item["criterion"]), str(item["subject_id"]))
        candidate_a = str(item["candidate_a"]["verdict"])
        candidate_b = str(item["candidate_b"]["verdict"])
        output[key] = (
            (candidate_a, candidate_b)
            if baseline_position == "A"
            else (candidate_b, candidate_a)
        )
    return output


def _normalized_relative(
    rows: list[dict[str, object]], baseline_position: str
) -> dict[str, str]:
    output = {}
    for item in rows:
        verdict = str(item["verdict"])
        if verdict in {"candidate_a", "candidate_b"}:
            position = "A" if verdict == "candidate_a" else "B"
            verdict = "baseline" if position == baseline_position else "mutated"
        output[str(item["criterion"])] = verdict
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie-threshold", type=float, default=0.05)
    args = parser.parse_args()
    labels = {
        item.pair_id: item
        for item in (
            HybridCalibrationLabels.model_validate_json(line)
            for line in args.labels.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    with args.scores.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    unknown_pairs = sorted({row["pair_id"] for row in rows} - set(labels))
    if unknown_pairs:
        raise SystemExit(f"scores have no question labels: {unknown_pairs}")

    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row["judge_model"]].append(row)
    metrics = {}
    for model, model_rows in sorted(by_model.items()):
        overall_correct: list[bool] = []
        absolute_correct: list[bool] = []
        absolute_pair_correct: list[bool] = []
        comparative_correct: list[bool] = []
        absolute_by_criterion: dict[str, list[bool]] = defaultdict(list)
        comparative_by_criterion: dict[str, list[bool]] = defaultdict(list)
        absolute_only_correct: list[bool] = []
        relative_only_correct: list[bool] = []
        order: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
        pair_decisions: dict[str, list[float]] = defaultdict(list)
        repeated: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in model_rows:
            gold = labels[row["pair_id"]]
            expected_overall = gold.overall_preference.value
            if expected_overall != ExpectedPairPreference.UNLABELED.value:
                overall_correct.append(row["baseline_preference"] == expected_overall)
            baseline_position = row["baseline_position"]
            absolute_rows = [
                *json.loads(row["deterministic_assessments"]),
                *json.loads(row["absolute_assessments"]),
            ]
            actual_absolute = _normalized_absolute(
                absolute_rows, baseline_position
            )
            for item in gold.absolute_labels:
                key = (item.criterion.value, item.subject_id)
                if key not in actual_absolute:
                    continue
                actual_baseline, actual_mutated = actual_absolute[key]
                item_results = []
                if item.baseline is not ExpectedVerdict.UNLABELED:
                    correct = actual_baseline == item.baseline.value
                    absolute_correct.append(correct)
                    absolute_by_criterion[item.criterion.value].append(correct)
                    item_results.append(correct)
                if item.mutated is not ExpectedVerdict.UNLABELED:
                    correct = actual_mutated == item.mutated.value
                    absolute_correct.append(correct)
                    absolute_by_criterion[item.criterion.value].append(correct)
                    item_results.append(correct)
                if len(item_results) == 2:
                    absolute_pair_correct.append(all(item_results))
            actual_relative = _normalized_relative(
                json.loads(row["comparative_assessments"]), baseline_position
            )
            for item in gold.comparative_labels:
                if item.preference is ExpectedPairPreference.UNLABELED:
                    continue
                correct = (
                    actual_relative.get(item.criterion.value)
                    == item.preference.value
                )
                comparative_correct.append(correct)
                comparative_by_criterion[item.criterion.value].append(correct)

            score_a = float(row["candidate_a_score"])
            score_b = float(row["candidate_b_score"])
            baseline_delta = (
                score_a - score_b
                if baseline_position == "A"
                else score_b - score_a
            )
            absolute_only_correct.append(
                _direction(baseline_delta, args.tie_threshold) == expected_overall
            )
            raw_relative = row["baseline_relative_preference"]
            relative_delta = (
                None
                if raw_relative == ""
                else 2.0 * float(raw_relative) - 1.0
            )
            relative_only_correct.append(
                _direction(relative_delta, args.tie_threshold) == expected_overall
            )
            order[(row["pair_id"], int(row["repetition"]))][row["order"]] = row[
                "baseline_preference"
            ]
            if row["baseline_decision_value"]:
                value = float(row["baseline_decision_value"])
                pair_decisions[row["pair_id"]].append(value)
                repeated[(row["pair_id"], row["order"])].append(value)

        order_pairs = [
            value
            for value in order.values()
            if {"baseline_a", "baseline_b"} <= value.keys()
        ]
        pair_outcomes = {
            pair_id: _direction(mean(values), args.tie_threshold)
            for pair_id, values in pair_decisions.items()
            if values
        }
        expected_repetitions = max(
            (len(values) for values in repeated.values()), default=0
        )
        complete_repeat_groups = [
            values
            for values in repeated.values()
            if len(values) == expected_repetitions
        ]
        pair_accuracy = {
            pair_id: outcome == labels[pair_id].overall_preference.value
            for pair_id, outcome in pair_outcomes.items()
            if labels[pair_id].overall_preference
            is not ExpectedPairPreference.UNLABELED
        }
        reviewed_absolute = sum(
            item.baseline is not ExpectedVerdict.UNLABELED
            or item.mutated is not ExpectedVerdict.UNLABELED
            for label in labels.values()
            for item in label.absolute_labels
        )
        total_absolute = sum(
            len(label.absolute_labels) for label in labels.values()
        )
        reviewed_comparative = sum(
            item.preference is not ExpectedPairPreference.UNLABELED
            for label in labels.values()
            for item in label.comparative_labels
        )
        total_comparative = sum(
            len(label.comparative_labels) for label in labels.values()
        )
        metrics[model] = {
            "comparison_count": len(model_rows),
            "combined_preference_accuracy": _rate(overall_correct),
            "absolute_only_preference_accuracy": _rate(absolute_only_correct),
            "relative_only_preference_accuracy": _rate(relative_only_correct),
            "pair_aggregated_accuracy": _rate(
                list(pair_accuracy.values())
            ),
            "pair_aggregated_accuracy_ci95": _bootstrap_accuracy_ci(
                pair_accuracy
            ),
            "absolute_verdict_accuracy": _rate(absolute_correct),
            "absolute_pair_exact_accuracy": _rate(absolute_pair_correct),
            "comparative_question_accuracy": _rate(comparative_correct),
            "order_consistency_rate": _rate(
                [
                    item["baseline_a"] == item["baseline_b"]
                    for item in order_pairs
                ]
            ),
            "repeat_expected_count": expected_repetitions,
            "repeat_complete_group_count": len(complete_repeat_groups),
            "repeat_icc_1_1": _icc_one_one(complete_repeat_groups),
            "mean_repeated_call_std": (
                mean(pstdev(values) for values in repeated.values())
                if repeated
                else None
            ),
            "reviewed_absolute_label_fraction": (
                reviewed_absolute / total_absolute if total_absolute else None
            ),
            "reviewed_comparative_label_fraction": (
                reviewed_comparative / total_comparative
                if total_comparative
                else None
            ),
            "absolute_by_criterion": {
                criterion: _rate(values)
                for criterion, values in sorted(absolute_by_criterion.items())
            },
            "comparative_by_criterion": {
                criterion: _rate(values)
                for criterion, values in sorted(comparative_by_criterion.items())
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(args.output.with_name("hybrid_judge_summary.md"), metrics)


def _format(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def _write_summary(path: Path, metrics: dict[str, object]) -> None:
    lines = [
        "# Hybrid scientific judge calibration",
        "",
        "| Judge | Combined | Absolute only | Comparative only | Pair aggregate | "
        "Atomic absolute | Pair absolute | Comparative questions | Order consistency | "
        "Repeat ICC | Repeat SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, raw in metrics.items():
        values = raw
        assert isinstance(values, dict)
        lines.append(
            "| "
            + " | ".join(
                (
                    model,
                    _format(values["combined_preference_accuracy"]),
                    _format(values["absolute_only_preference_accuracy"]),
                    _format(values["relative_only_preference_accuracy"]),
                    _format(values["pair_aggregated_accuracy"]),
                    _format(values["absolute_verdict_accuracy"]),
                    _format(values["absolute_pair_exact_accuracy"]),
                    _format(values["comparative_question_accuracy"]),
                    _format(values["order_consistency_rate"]),
                    _format(values["repeat_icc_1_1"]),
                    _format(values["mean_repeated_call_std"]),
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
