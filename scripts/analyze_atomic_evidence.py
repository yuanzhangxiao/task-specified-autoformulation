"""Analyze mutation-targeted atomic evidence without making new LLM calls."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _side_for_position(position: str) -> str:
    return "candidate_a" if position == "A" else "candidate_b"


def _occurrence_signature(item: dict[str, object]) -> tuple[object, ...]:
    return (
        item["equation_location"],
        item["governed_quantity"],
        item["unsigned_expression"],
        tuple(item["symbols"]),
    )


def _new_mutated_occurrence_ids(
    plan: dict[str, object],
    *,
    baseline_side: str,
    mutated_side: str,
) -> list[str]:
    occurrences = plan["signed_occurrences"]
    assert isinstance(occurrences, list)
    baseline_counts = Counter(
        _occurrence_signature(item)
        for item in occurrences
        if item["candidate_side"] == baseline_side
    )
    observed = Counter()
    new_ids = []
    for item in occurrences:
        if item["candidate_side"] != mutated_side:
            continue
        signature = _occurrence_signature(item)
        observed[signature] += 1
        if observed[signature] > baseline_counts[signature]:
            new_ids.append(str(item["occurrence_id"]))
    return new_ids


def _repeat_signature(item: dict[str, object]) -> tuple[object, ...]:
    return (
        item["equation_location"],
        item["governed_quantity"],
        item["unsigned_expression"],
    )


def _new_mutated_repeat_ids(
    plan: dict[str, object],
    *,
    baseline_side: str,
    mutated_side: str,
) -> list[str]:
    candidates = plan["exact_repeat_candidates"]
    assert isinstance(candidates, list)
    baseline_counts = Counter(
        _repeat_signature(item)
        for item in candidates
        if item["candidate_side"] == baseline_side
    )
    observed = Counter()
    new_ids = []
    for item in candidates:
        if item["candidate_side"] != mutated_side:
            continue
        signature = _repeat_signature(item)
        observed[signature] += 1
        if observed[signature] > baseline_counts[signature]:
            new_ids.append(str(item["repeat_pair_id"]))
    return new_ids


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.scores.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_model: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    unit_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        model = row["judge_model"]
        mutation = row["mutation_type"]
        baseline_side = _side_for_position(row["baseline_position"])
        mutated_side = (
            "candidate_b" if baseline_side == "candidate_a" else "candidate_a"
        )
        plan = json.loads(row["atomic_evidence_plan"])
        result = json.loads(row["atomic_assessments"])
        directions = {
            item["occurrence_id"]: item["expected_direction"]
            for item in result["signed_occurrence_assessments"]
        }
        relations = {
            item["repeat_pair_id"]: item["relation"]
            for item in result["repeated_contribution_assessments"]
        }
        if mutation == "wrong_meal_sink":
            identifiers = _new_mutated_occurrence_ids(
                plan,
                baseline_side=baseline_side,
                mutated_side=mutated_side,
            )
            unit_counts[model][mutation] += len(identifiers)
            by_model[model]["wrong_sink_expected_direction"].extend(
                directions[item] == "positive_contribution"
                for item in identifiers
            )
        elif mutation == "duplicated_gp_flux":
            identifiers = _new_mutated_repeat_ids(
                plan,
                baseline_side=baseline_side,
                mutated_side=mutated_side,
            )
            unit_counts[model][mutation] += len(identifiers)
            by_model[model]["duplicate_relation"].extend(
                relations[item] == "same_physical_contribution"
                for item in identifiers
            )

    metrics = {
        model: {
            "wrong_sink_expected_direction_accuracy": _rate(
                outcomes["wrong_sink_expected_direction"]
            ),
            "duplicate_relation_accuracy": _rate(
                outcomes["duplicate_relation"]
            ),
            "combined_atomic_mutation_accuracy": _rate(
                [
                    *outcomes["wrong_sink_expected_direction"],
                    *outcomes["duplicate_relation"],
                ]
            ),
            "scored_units_by_mutation": dict(unit_counts[model]),
        }
        for model, outcomes in sorted(by_model.items())
    }
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = args.output.with_suffix(".md")
    lines = [
        "# Atomic scientific-evidence analysis",
        "",
        (
            "Only mutation-added units are scored. The mutation contract remains "
            "hidden from the judge and is opened here after frozen calls."
        ),
        "",
        "| Judge | Wrong-sink expected direction | Exact-repeat relation | Combined |",
        "|---|---:|---:|---:|",
    ]
    for model, values in metrics.items():
        cells = []
        for key in (
            "wrong_sink_expected_direction_accuracy",
            "duplicate_relation_accuracy",
            "combined_atomic_mutation_accuracy",
        ):
            value = values[key]
            cells.append("N/A" if value is None else f"{value:.3f}")
        lines.append(f"| {model} | {' | '.join(cells)} |")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
