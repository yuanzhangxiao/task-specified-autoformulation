"""Analyze hybrid judge outputs against question-level calibration labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
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
    parser.add_argument("--failures", type=Path)
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
    failures = []
    if args.failures is not None and args.failures.is_file():
        failures = [
            json.loads(line)
            for line in args.failures.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    unknown_pairs = sorted(
        ({row["pair_id"] for row in rows} | {row["pair_id"] for row in failures})
        - set(labels)
    )
    if unknown_pairs:
        raise SystemExit(f"outcomes have no question labels: {unknown_pairs}")

    def outcome_key(row: dict[str, object]) -> tuple[str, str, int, str]:
        return (
            str(row["pair_id"]),
            str(row["judge_model"]),
            int(row["repetition"]),
            str(row["order"]),
        )

    success_keys = {outcome_key(row) for row in rows}
    failure_keys = [outcome_key(row) for row in failures]
    if len(failure_keys) != len(set(failure_keys)):
        raise SystemExit("duplicate persistent-failure keys")
    overlap = success_keys & set(failure_keys)
    if overlap:
        raise SystemExit(
            f"keys occur in both success and failure inputs: {len(overlap)}"
        )

    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row["judge_model"]].append(row)
    failures_by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in failures:
        failures_by_model[str(row["judge_model"])].append(row)
    metrics = {}
    for model in sorted(set(by_model) | set(failures_by_model)):
        model_rows = by_model[model]
        model_failures = failures_by_model[model]
        overall_correct: list[bool] = []
        runtime_correct: list[bool] = []
        runtime_pair_correct: list[bool] = []
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
            actual_runtime = _normalized_absolute(
                json.loads(row["deterministic_assessments"]), baseline_position
            )
            actual_absolute = _normalized_absolute(
                json.loads(row["absolute_assessments"]), baseline_position
            )
            for item in gold.absolute_labels:
                key = (item.criterion.value, item.subject_id)
                is_runtime = item.label_source == "deterministic_runtime"
                actual = actual_runtime if is_runtime else actual_absolute
                if key not in actual:
                    continue
                actual_baseline, actual_mutated = actual[key]
                item_results = []
                if item.baseline is not ExpectedVerdict.UNLABELED:
                    correct = actual_baseline == item.baseline.value
                    target = runtime_correct if is_runtime else absolute_correct
                    target.append(correct)
                    if not is_runtime:
                        absolute_by_criterion[item.criterion.value].append(correct)
                    item_results.append(correct)
                if item.mutated is not ExpectedVerdict.UNLABELED:
                    correct = actual_mutated == item.mutated.value
                    target = runtime_correct if is_runtime else absolute_correct
                    target.append(correct)
                    if not is_runtime:
                        absolute_by_criterion[item.criterion.value].append(correct)
                    item_results.append(correct)
                if len(item_results) == 2:
                    target_pairs = (
                        runtime_pair_correct
                        if is_runtime
                        else absolute_pair_correct
                    )
                    target_pairs.append(all(item_results))
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
            if expected_overall != ExpectedPairPreference.UNLABELED.value:
                absolute_only_correct.append(
                    _direction(baseline_delta, args.tie_threshold)
                    == expected_overall
                )
            raw_relative = row["baseline_relative_preference"]
            relative_delta = (
                None
                if raw_relative == ""
                else 2.0 * float(raw_relative) - 1.0
            )
            if expected_overall != ExpectedPairPreference.UNLABELED.value:
                relative_only_correct.append(
                    _direction(relative_delta, args.tie_threshold)
                    == expected_overall
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
        scored_absolute = sum(
            item.baseline is not ExpectedVerdict.UNLABELED
            for label in labels.values()
            for item in label.absolute_labels
            if item.label_source != "deterministic_runtime"
        ) + sum(
            item.mutated is not ExpectedVerdict.UNLABELED
            for label in labels.values()
            for item in label.absolute_labels
            if item.label_source != "deterministic_runtime"
        )
        possible_absolute = 2 * sum(
            item.label_source != "deterministic_runtime"
            for label in labels.values()
            for item in label.absolute_labels
        )
        scored_comparative = sum(
            item.preference is not ExpectedPairPreference.UNLABELED
            for label in labels.values()
            for item in label.comparative_labels
        )
        total_comparative = sum(
            len(label.comparative_labels) for label in labels.values()
        )
        metrics[model] = {
            "comparison_count": len(model_rows),
            "successful_comparison_count": len(model_rows),
            "failed_comparison_count": len(model_failures),
            "attempted_comparison_count": len(model_rows) + len(model_failures),
            "structured_response_success_rate": (
                len(model_rows) / (len(model_rows) + len(model_failures))
                if model_rows or model_failures
                else None
            ),
            "combined_preference_accuracy": _rate(overall_correct),
            "combined_preference_accuracy_conditional_on_response": _rate(
                overall_correct
            ),
            "combined_preference_accuracy_including_failures": (
                sum(overall_correct)
                / (
                    len(overall_correct)
                    + sum(
                        labels[str(row["pair_id"])].overall_preference
                        is not ExpectedPairPreference.UNLABELED
                        for row in model_failures
                    )
                )
                if overall_correct
                or any(
                    labels[str(row["pair_id"])].overall_preference
                    is not ExpectedPairPreference.UNLABELED
                    for row in model_failures
                )
                else None
            ),
            "absolute_only_preference_accuracy": _rate(absolute_only_correct),
            "relative_only_preference_accuracy": _rate(relative_only_correct),
            "pair_aggregated_accuracy": _rate(
                list(pair_accuracy.values())
            ),
            "pair_aggregated_accuracy_ci95": _bootstrap_accuracy_ci(
                pair_accuracy
            ),
            "runtime_certification_accuracy": _rate(runtime_correct),
            "runtime_pair_exact_accuracy": _rate(runtime_pair_correct),
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
            "gold_label_scope": "runtime_and_mutation_contract_only",
            "expert_review_used": False,
            "scored_absolute_verdict_count": scored_absolute,
            "possible_absolute_verdict_count": possible_absolute,
            "scored_absolute_label_fraction": (
                scored_absolute / possible_absolute
                if possible_absolute
                else None
            ),
            "scored_comparative_label_count": scored_comparative,
            "possible_comparative_label_count": total_comparative,
            "scored_comparative_label_fraction": (
                scored_comparative / total_comparative
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
            "failures_by_error_type": dict(
                sorted(
                    Counter(
                        str(row["error_type"]) for row in model_failures
                    ).items()
                )
            ),
            "failures_by_category": dict(
                sorted(
                    Counter(
                        str(row.get("failure_category") or "unknown")
                        for row in model_failures
                    ).items()
                )
            ),
            "responses_by_transport": dict(
                sorted(
                    Counter(
                        str(row.get("response_transport") or "unrecorded")
                        for row in model_rows
                    ).items()
                )
            ),
            "responses_with_tool_argument_key_repairs": sum(
                int(row.get("tool_argument_key_repairs") or 0) > 0
                for row in model_rows
            ),
            "tool_argument_key_repair_count": sum(
                int(row.get("tool_argument_key_repairs") or 0)
                for row in model_rows
            ),
            "responses_with_missing_atomic_unit_repairs": sum(
                (
                    int(row.get("atomic_missing_occurrence_repairs") or 0)
                    + int(row.get("atomic_missing_repeat_repairs") or 0)
                )
                > 0
                for row in model_rows
            ),
            "missing_atomic_occurrence_repair_count": sum(
                int(row.get("atomic_missing_occurrence_repairs") or 0)
                for row in model_rows
            ),
            "missing_atomic_repeat_repair_count": sum(
                int(row.get("atomic_missing_repeat_repairs") or 0)
                for row in model_rows
            ),
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
        "Gold labels are restricted to deterministic runtime facts and explicit "
        "mutation contracts; no domain-expert review is used. Untargeted questions "
        "are excluded from question-level accuracy.",
        "Provider/schema failures are retained as outcomes. Conditional accuracy "
        "uses valid structured responses only; end-to-end accuracy treats a failed "
        "response as an incorrect judge decision.",
        "",
        "| Judge | Response success | Combined (conditional) | Combined (end-to-end) | "
        "Absolute only | Comparative only | Pair aggregate | "
        "Runtime integrity | LLM semantic absolute | Pair semantic absolute | "
        "Comparative questions | Order consistency | Repeat ICC | Repeat SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, raw in metrics.items():
        values = raw
        assert isinstance(values, dict)
        lines.append(
            "| "
            + " | ".join(
                (
                    model,
                    _format(values["structured_response_success_rate"]),
                    _format(
                        values[
                            "combined_preference_accuracy_conditional_on_response"
                        ]
                    ),
                    _format(
                        values["combined_preference_accuracy_including_failures"]
                    ),
                    _format(values["absolute_only_preference_accuracy"]),
                    _format(values["relative_only_preference_accuracy"]),
                    _format(values["pair_aggregated_accuracy"]),
                    _format(values["runtime_certification_accuracy"]),
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
        repair_responses = int(
            values.get("responses_with_missing_atomic_unit_repairs") or 0
        )
        if repair_responses:
            lines.append(
                f"\n{model}: {repair_responses} responses required neutral "
                "missing-atomic-unit repair "
                f"({values['missing_atomic_occurrence_repair_count']} "
                "occurrences; "
                f"{values['missing_atomic_repeat_repair_count']} repeats)."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
