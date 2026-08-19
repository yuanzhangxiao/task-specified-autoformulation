"""Analyze blinded atomic pairwise judge accuracy and stability."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def _direction(
    value: float | None,
    *,
    indeterminate_rate: float = 0.0,
    not_applicable_rate: float = 0.0,
) -> str:
    if value is None:
        if not_applicable_rate == 1.0:
            return "not_applicable"
        return "indeterminate"
    if value > 0.5:
        return "baseline"
    if value < 0.5:
        return "mutated"
    return "tie"


def _baseline_atomic_verdict(verdict: str, baseline_position: str) -> str:
    if verdict in {"tie", "indeterminate", "not_applicable"}:
        return verdict
    chose_a = verdict == "candidate_a"
    chose_baseline = chose_a == (baseline_position == "A")
    return "baseline" if chose_baseline else "mutated"


def _rate(values: list[str], target: str) -> float | None:
    return sum(value == target for value in values) / len(values) if values else None


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.scores.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row["judge_model"]].append(row)
    metrics: dict[str, object] = {}
    for model, values in sorted(by_model.items()):
        preferences: list[float | None] = [
            float(row["baseline_preference"])
            if row["baseline_preference"]
            else None
            for row in values
        ]
        directions = [
            _direction(
                preference,
                indeterminate_rate=float(row["indeterminate_rate"]),
                not_applicable_rate=float(row["not_applicable_rate"]),
            )
            for row, preference in zip(values, preferences, strict=True)
        ]
        determined = [value for value in preferences if value is not None]
        repeated: dict[tuple[str, str], list[float]] = defaultdict(list)
        pair_aggregates: dict[str, list[float]] = defaultdict(list)
        reverse_pairs: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
        by_mutation: dict[str, list[str]] = defaultdict(list)
        atomic: dict[str, list[str]] = defaultdict(list)
        atomic_by_mutation: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row, preference, direction in zip(
            values, preferences, directions, strict=True
        ):
            if preference is not None:
                repeated[(row["pair_id"], row["order"])].append(preference)
                pair_aggregates[row["pair_id"]].append(preference)
            reverse_pairs[(row["pair_id"], int(row["repetition"]))][
                row["order"]
            ] = direction
            by_mutation[row["mutation_type"]].append(direction)
            for name, answer in json.loads(row["answers"]).items():
                outcome = _baseline_atomic_verdict(
                    answer["verdict"], row["baseline_position"]
                )
                atomic[name].append(outcome)
                atomic_by_mutation[row["mutation_type"]][name].append(outcome)
        order_consistency = [
            item["baseline_a"] == item["baseline_b"]
            for item in reverse_pairs.values()
            if {"baseline_a", "baseline_b"} <= item.keys()
        ]
        aggregate_directions = [
            _direction(mean(pair_values))
            for pair_values in pair_aggregates.values()
            if pair_values
        ]
        expected_repetitions = max(
            (len(group) for group in repeated.values()), default=0
        )
        complete_repeat_groups = [
            group
            for group in repeated.values()
            if len(group) == expected_repetitions
        ]
        metrics[model] = {
            "comparison_count": len(values),
            "determined_comparison_count": len(determined),
            "atomic_preference_accuracy": _rate(directions, "baseline"),
            "false_preference_rate": _rate(directions, "mutated"),
            "tie_rate": _rate(directions, "tie"),
            "fully_indeterminate_rate": _rate(directions, "indeterminate"),
            "fully_not_applicable_rate": _rate(directions, "not_applicable"),
            "mean_atomic_indeterminate_rate": mean(
                float(row["indeterminate_rate"]) for row in values
            ),
            "mean_atomic_not_applicable_rate": mean(
                float(row["not_applicable_rate"]) for row in values
            ),
            "mean_baseline_preference": mean(determined) if determined else None,
            "mean_baseline_margin": (
                mean(2.0 * (value - 0.5) for value in determined)
                if determined
                else None
            ),
            "order_consistency_rate": (
                sum(order_consistency) / len(order_consistency)
                if order_consistency
                else None
            ),
            "pair_aggregated_accuracy": _rate(
                aggregate_directions, "baseline"
            ),
            "pair_aggregated_false_preference_rate": _rate(
                aggregate_directions, "mutated"
            ),
            "repeat_expected_count": expected_repetitions,
            "repeat_complete_group_count": len(complete_repeat_groups),
            "repeat_incomplete_group_count": (
                len(repeated) - len(complete_repeat_groups)
            ),
            "repeat_icc_1_1": _icc_one_one(complete_repeat_groups),
            "mean_repeated_call_std": (
                mean(pstdev(group) for group in repeated.values())
                if repeated
                else None
            ),
            "by_question": {
                name: {
                    "baseline_preference_accuracy": _rate(outcomes, "baseline"),
                    "false_preference_rate": _rate(outcomes, "mutated"),
                    "tie_rate": _rate(outcomes, "tie"),
                    "indeterminate_rate": _rate(outcomes, "indeterminate"),
                    "not_applicable_rate": _rate(outcomes, "not_applicable"),
                }
                for name, outcomes in sorted(atomic.items())
            },
            "by_mutation": {
                mutation: {
                    "comparison_count": len(outcomes),
                    "baseline_preference_accuracy": _rate(outcomes, "baseline"),
                    "false_preference_rate": _rate(outcomes, "mutated"),
                    "tie_rate": _rate(outcomes, "tie"),
                    "indeterminate_rate": _rate(outcomes, "indeterminate"),
                    "not_applicable_rate": _rate(outcomes, "not_applicable"),
                }
                for mutation, outcomes in sorted(by_mutation.items())
            },
            "by_mutation_and_question": {
                mutation: {
                    name: {
                        "baseline_preference_accuracy": _rate(
                            outcomes, "baseline"
                        ),
                        "false_preference_rate": _rate(outcomes, "mutated"),
                        "tie_rate": _rate(outcomes, "tie"),
                        "indeterminate_rate": _rate(outcomes, "indeterminate"),
                        "not_applicable_rate": _rate(
                            outcomes, "not_applicable"
                        ),
                    }
                    for name, outcomes in sorted(question_outcomes.items())
                }
                for mutation, question_outcomes in sorted(
                    atomic_by_mutation.items()
                )
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary(args.output.with_name("comparative_judge_summary.md"), metrics)


def _write_summary(path: Path, metrics: dict[str, object]) -> None:
    lines = [
        "# Atomic comparative judge stress test",
        "",
        "| Judge | Accuracy | False preference | Pair-aggregate accuracy | "
        "Order consistency | Indeterminate | N/A | Repeat ICC | Repeat SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, raw_values in metrics.items():
        values = raw_values
        assert isinstance(values, dict)
        formatted = [
            model,
            _format(values["atomic_preference_accuracy"]),
            _format(values["false_preference_rate"]),
            _format(values["pair_aggregated_accuracy"]),
            _format(values["order_consistency_rate"]),
            _format(values["mean_atomic_indeterminate_rate"]),
            _format(values["mean_atomic_not_applicable_rate"]),
            _format(values["repeat_icc_1_1"]),
            _format(values["mean_repeated_call_std"]),
        ]
        lines.append("| " + " | ".join(formatted) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    main()
