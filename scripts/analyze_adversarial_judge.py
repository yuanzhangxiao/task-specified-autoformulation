"""Summarize paired judge preference, margins, AUROC, and variability."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev


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
    metrics = {}
    for model, values in sorted(by_model.items()):
        labels = {row["known_label"] for row in values}
        positive_label = "valid" if "valid" in labels else "baseline"
        negative_label = "adversarial" if "adversarial" in labels else "mutated"
        paired: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
        pair_mutations: dict[str, str] = {}
        paired_categories: dict[
            tuple[str, int], dict[str, dict[str, float]]
        ] = defaultdict(dict)
        repeated: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in values:
            score = float(row["aggregate_score"])
            paired[(row["pair_id"], int(row["repetition"]))][
                row["known_label"]
            ] = score
            pair_mutations[row["pair_id"]] = row["mutation_type"]
            paired_categories[(row["pair_id"], int(row["repetition"]))][
                row["known_label"]
            ] = {
                str(key): float(value)
                for key, value in json.loads(row["category_scores"]).items()
            }
            repeated[(row["pair_id"], row["known_label"])].append(score)
        margins = [
            item[positive_label] - item[negative_label]
            for item in paired.values()
            if {positive_label, negative_label} <= item.keys()
        ]
        valid_scores = [
            float(row["aggregate_score"])
            for row in values
            if row["known_label"] == positive_label
        ]
        adversarial_scores = [
            float(row["aggregate_score"])
            for row in values
            if row["known_label"] == negative_label
        ]
        category_margins: dict[str, list[float]] = defaultdict(list)
        for item in paired_categories.values():
            if {positive_label, negative_label} > item.keys():
                continue
            for category in (
                item[positive_label].keys() & item[negative_label].keys()
            ):
                category_margins[category].append(
                    item[positive_label][category]
                    - item[negative_label][category]
                )
        by_mutation: dict[str, list[float]] = defaultdict(list)
        for (pair_id, _), item in paired.items():
            if {positive_label, negative_label} <= item.keys():
                by_mutation[pair_mutations[pair_id]].append(
                    item[positive_label] - item[negative_label]
                )
        metrics[model] = {
            "paired_comparison_count": len(margins),
            "paired_preference_accuracy": (
                sum(value > 0 for value in margins) / len(margins)
                if margins
                else None
            ),
            "false_preference_rate": (
                sum(value < 0 for value in margins) / len(margins)
                if margins
                else None
            ),
            "tie_rate": (
                sum(value == 0 for value in margins) / len(margins)
                if margins
                else None
            ),
            "mean_score_margin": mean(margins) if margins else None,
            "median_score_margin": median(margins) if margins else None,
            "auroc": _auroc(valid_scores, adversarial_scores),
            "repeat_icc_1_1": _icc_one_one(list(repeated.values())),
            "mean_repeated_call_std": mean(
                pstdev(scores) for scores in repeated.values()
            ),
            "baseline_score_at_least_0_95_rate": (
                sum(score >= 0.95 for score in valid_scores) / len(valid_scores)
            ),
            "mutated_score_at_least_0_95_rate": (
                sum(score >= 0.95 for score in adversarial_scores)
                / len(adversarial_scores)
            ),
            "baseline_score_at_most_0_20_rate": (
                sum(score <= 0.20 for score in valid_scores) / len(valid_scores)
            ),
            "mean_category_score_margins": {
                category: mean(category_values)
                for category, category_values in sorted(category_margins.items())
            },
            "by_mutation": {
                mutation: {
                    "paired_comparison_count": len(mutation_margins),
                    "paired_preference_accuracy": sum(
                        value > 0 for value in mutation_margins
                    )
                    / len(mutation_margins),
                    "mean_score_margin": mean(mutation_margins),
                }
                for mutation, mutation_margins in sorted(by_mutation.items())
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary(args.output.with_name("adversarial_judge_summary.md"), metrics)


def _write_summary(path: Path, metrics: dict) -> None:
    lines = [
        "# Matched-pair judge stress test",
        "",
        "| Judge | Paired accuracy ↑ | False preference ↓ | AUROC ↑ | "
        "Mean margin ↑ | Median margin ↑ | Repeat ICC ↑ | Repeat SD ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, values in metrics.items():
        lines.append(
            "| "
            + " | ".join(
                (
                    model,
                    _format(values["paired_preference_accuracy"]),
                    _format(values["false_preference_rate"]),
                    _format(values["auroc"]),
                    _format(values["mean_score_margin"]),
                    _format(values["median_score_margin"]),
                    _format(values["repeat_icc_1_1"]),
                    _format(values["mean_repeated_call_std"]),
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _auroc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = sum(left > right for left in positive for right in negative)
    ties = sum(left == right for left in positive for right in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


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


if __name__ == "__main__":
    main()
