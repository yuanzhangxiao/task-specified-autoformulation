"""Analyze frozen equivalence and non-ordered tradeoff judge development calls."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

from autoformalism.rebuttal.hybrid_labels import (
    ExpectedVerdict,
    HybridCalibrationLabels,
)

EQUIVALENCE_TYPE = "algebraic_reordering_equivalent"


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _normalized_absolute(
    row: dict[str, str],
) -> dict[tuple[str, str], tuple[str, str]]:
    baseline_is_a = row["baseline_position"] == "A"
    output = {}
    for item in json.loads(row["absolute_assessments"]):
        candidate_a = item["candidate_a"]["verdict"]
        candidate_b = item["candidate_b"]["verdict"]
        output[(item["criterion"], item["subject_id"])] = (
            (candidate_a, candidate_b)
            if baseline_is_a
            else (candidate_b, candidate_a)
        )
    return output


def _order_consistency(rows: list[dict[str, str]]) -> float | None:
    by_trial: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for row in rows:
        by_trial[(row["pair_id"], int(row["repetition"]))][row["order"]] = row[
            "baseline_preference"
        ]
    paired = [
        outcomes
        for outcomes in by_trial.values()
        if {"baseline_a", "baseline_b"} <= outcomes.keys()
    ]
    return _rate(
        [
            outcomes["baseline_a"] == outcomes["baseline_b"]
            for outcomes in paired
        ]
    )


def _mean_repeat_sd(rows: list[dict[str, str]]) -> float | None:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["baseline_decision_value"]:
            groups[(row["pair_id"], row["order"])].append(
                float(row["baseline_decision_value"])
            )
    deviations = [pstdev(values) for values in groups.values() if len(values) > 1]
    return _mean(deviations)


def _mean_orientation_bias(rows: list[dict[str, str]]) -> float | None:
    trials: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["baseline_decision_value"]:
            trials[(row["pair_id"], int(row["repetition"]))][row["order"]] = (
                float(row["baseline_decision_value"])
            )
    differences = [
        abs(outcomes["baseline_a"] - outcomes["baseline_b"])
        for outcomes in trials.values()
        if {"baseline_a", "baseline_b"} <= outcomes.keys()
    ]
    return _mean(differences)


def _evaluate_gate(
    observed: dict[str, float | None],
    gate: dict[str, float],
) -> dict[str, dict[str, float | bool | None]]:
    checks = {}
    for name, threshold in gate.items():
        value = observed[name]
        is_maximum = name.startswith("maximum_")
        passed = value is not None and (
            value <= threshold if is_maximum else value >= threshold
        )
        checks[name] = {
            "observed": value,
            "threshold": threshold,
            "comparison": "maximum" if is_maximum else "minimum",
            "passed": passed,
        }
    return checks


def analyze(
    rows: list[dict[str, str]],
    failures: list[dict[str, object]],
    labels: dict[str, HybridCalibrationLabels],
    *,
    tie_threshold: float,
    gate: dict[str, float],
) -> dict[str, object]:
    """Compute predeclared reliability metrics without assigning tradeoff truth."""
    equivalence = [
        row for row in rows if row["mutation_type"] == EQUIVALENCE_TYPE
    ]
    tradeoffs = [
        row for row in rows if row["mutation_type"].startswith("tradeoff_")
    ]
    attempted = len(rows) + len(failures)
    equivalence_decisions: dict[str, list[float]] = defaultdict(list)
    for row in equivalence:
        if row["baseline_decision_value"]:
            equivalence_decisions[row["pair_id"]].append(
                float(row["baseline_decision_value"])
            )
    equivalence_pair_ties = [
        abs(mean(values)) <= tie_threshold
        for values in equivalence_decisions.values()
        if values
    ]
    equivalence_comparative_ties = [
        item["verdict"] == "tie"
        for row in equivalence
        for item in json.loads(row["comparative_assessments"])
    ]

    targeted_absolute = []
    for row in tradeoffs:
        actual = _normalized_absolute(row)
        for expected in labels[row["pair_id"]].absolute_labels:
            if not expected.label_source.startswith("mutation_contract:"):
                continue
            observed = actual.get((expected.criterion.value, expected.subject_id))
            if observed is None:
                continue
            for expected_verdict, actual_verdict in zip(
                (expected.baseline, expected.mutated), observed, strict=True
            ):
                if expected_verdict is not ExpectedVerdict.UNLABELED:
                    targeted_absolute.append(
                        actual_verdict == expected_verdict.value
                    )

    tradeoff_counts = Counter(row["baseline_preference"] for row in tradeoffs)
    per_type = {}
    for pair_type in sorted({row["mutation_type"] for row in tradeoffs}):
        subset = [row for row in tradeoffs if row["mutation_type"] == pair_type]
        decisions = [
            float(row["baseline_decision_value"])
            for row in subset
            if row["baseline_decision_value"]
        ]
        per_type[pair_type] = {
            "successful_calls": len(subset),
            "preference_counts": dict(
                Counter(row["baseline_preference"] for row in subset)
            ),
            "mean_decision_value_for_first_member": _mean(decisions),
            "mean_absolute_decision_margin": _mean(
                [abs(value) for value in decisions]
            ),
            "order_consistency": _order_consistency(subset),
        }

    contract_repairs = sum(
        int(row.get("redundant_absolute_unit_repairs") or 0) for row in rows
    )
    observed = {
        "minimum_response_success": len(rows) / attempted if attempted else None,
        "minimum_equivalence_call_tie_accuracy": _rate(
            [row["baseline_preference"] == "tie" for row in equivalence]
        ),
        "minimum_equivalence_pair_tie_accuracy": _rate(equivalence_pair_ties),
        "minimum_equivalence_comparative_tie_accuracy": _rate(
            equivalence_comparative_ties
        ),
        "minimum_tradeoff_targeted_absolute_accuracy": _rate(targeted_absolute),
        "minimum_overall_order_consistency": _order_consistency(rows),
        "maximum_tradeoff_orientation_bias": _mean_orientation_bias(tradeoffs),
        "maximum_mean_repeat_decision_sd": _mean_repeat_sd(rows),
    }
    checks = _evaluate_gate(observed, gate)
    return {
        "schema_version": "hybrid-judge-equivalence-tradeoff-analysis-1",
        "passed": all(bool(item["passed"]) for item in checks.values()),
        "attempted_calls": attempted,
        "successful_calls": len(rows),
        "failed_calls": len(failures),
        "redundant_absolute_unit_repairs": contract_repairs,
        "equivalence": {
            "successful_calls": len(equivalence),
            "call_tie_accuracy": observed[
                "minimum_equivalence_call_tie_accuracy"
            ],
            "pair_tie_accuracy": observed[
                "minimum_equivalence_pair_tie_accuracy"
            ],
            "comparative_question_tie_accuracy": observed[
                "minimum_equivalence_comparative_tie_accuracy"
            ],
            "mean_absolute_decision_margin": _mean(
                [
                    abs(float(row["baseline_decision_value"]))
                    for row in equivalence
                    if row["baseline_decision_value"]
                ]
            ),
        },
        "tradeoffs": {
            "successful_calls": len(tradeoffs),
            "targeted_absolute_accuracy": observed[
                "minimum_tradeoff_targeted_absolute_accuracy"
            ],
            "preference_counts": dict(tradeoff_counts),
            "tie_rate": _rate(
                [row["baseline_preference"] == "tie" for row in tradeoffs]
            ),
            "mean_orientation_bias": observed[
                "maximum_tradeoff_orientation_bias"
            ],
            "by_pair_type": per_type,
            "overall_preference_truth": "unlabeled",
        },
        "stability": {
            "overall_order_consistency": observed[
                "minimum_overall_order_consistency"
            ],
            "mean_repeat_decision_sd": observed[
                "maximum_mean_repeat_decision_sd"
            ],
        },
        "checks": checks,
        "interpretation_boundary": (
            "Tradeoff overall preferences are descriptive, not accuracy labels."
        ),
    }


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
    config = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    result = analyze(
        rows,
        failures,
        labels,
        tie_threshold=config["protocol"]["scoring"]["tie_threshold"],
        gate=config["development_gate"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = args.output.with_suffix(".md")
    lines = [
        "# Equivalence and tradeoff judge development",
        "",
        f"Overall predeclared gate: **{'PASS' if result['passed'] else 'FAIL'}**.",
        "",
        "Tradeoff overall preferences are intentionally unlabeled; they are "
        "reported descriptively and never counted as accuracy.",
        "",
        "| Gate | Observed | Threshold | Result |",
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
