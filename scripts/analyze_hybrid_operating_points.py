"""Estimate hybrid-judge call-budget operating points from frozen outcomes."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    HybridCalibrationLabels,
)

ORDERS = ("baseline_a", "baseline_b")


def _direction(value: float, threshold: float) -> str:
    if value > threshold:
        return "baseline"
    if value < -threshold:
        return "mutated"
    return "tie"


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _cluster_bootstrap_ci(
    values_by_pair: dict[str, list[float]],
    *,
    samples: int,
) -> list[float] | None:
    """Bootstrap pair-level means so repeated calls never act as new pairs."""
    if not values_by_pair:
        return None
    pair_means = {
        pair_id: mean(values)
        for pair_id, values in values_by_pair.items()
        if values
    }
    if not pair_means:
        return None
    identifiers = sorted(pair_means)
    generator = random.Random(0)
    estimates = sorted(
        mean(
            pair_means[generator.choice(identifiers)]
            for _ in range(len(identifiers))
        )
        for _ in range(samples)
    )
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    ]


def _configurations(repetitions: tuple[int, ...]) -> list[dict[str, object]]:
    """Return the predeclared one-order and order-averaged call budgets."""
    configurations: list[dict[str, object]] = [
        {
            "configuration": "one_order_a_one_call",
            "orders": ("baseline_a",),
            "repetitions_per_order": 1,
        },
        {
            "configuration": "one_order_b_one_call",
            "orders": ("baseline_b",),
            "repetitions_per_order": 1,
        },
    ]
    configurations.extend(
        {
            "configuration": f"both_orders_{count}_repetition"
            + ("s" if count != 1 else ""),
            "orders": ORDERS,
            "repetitions_per_order": count,
        }
        for count in range(1, len(repetitions) + 1)
    )
    return configurations


def _cyclic_repetitions(
    repetitions: tuple[int, ...], start_index: int
) -> tuple[int, ...]:
    """Rotate frozen seed identifiers without reusing one within a trial."""
    return repetitions[start_index:] + repetitions[:start_index]


def _adaptive_operating_points(
    *,
    model: str,
    repetitions: tuple[int, ...],
    pair_ids: tuple[str, ...],
    success_by_key: dict[tuple[str, str, int, str], float],
    labels: dict[str, HybridCalibrationLabels],
    tie_threshold: float,
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    """Retry only a missing orientation and require symmetric pair evidence."""
    results = []
    for attempt_limit in range(1, len(repetitions) + 1):
        accuracy_by_pair: dict[str, list[float]] = defaultdict(list)
        end_to_end_by_pair: dict[str, list[float]] = defaultdict(list)
        coverage_by_pair: dict[str, list[float]] = defaultdict(list)
        decisions_by_pair: dict[str, list[float]] = defaultdict(list)
        calls_used: list[float] = []
        retry_activated: list[float] = []
        order_agreements: list[float] = []
        for start_index in range(len(repetitions)):
            retry_sequence = _cyclic_repetitions(repetitions, start_index)[
                :attempt_limit
            ]
            for pair_id in pair_ids:
                selected: dict[str, float] = {}
                pair_calls = 0
                initial_missing = False
                for order in ORDERS:
                    for attempt_index, repetition in enumerate(retry_sequence):
                        pair_calls += 1
                        value = success_by_key.get(
                            (pair_id, model, repetition, order)
                        )
                        if value is not None:
                            selected[order] = value
                            break
                        if attempt_index == 0:
                            initial_missing = True
                calls_used.append(float(pair_calls))
                retry_activated.append(float(initial_missing))
                available = len(selected) == len(ORDERS)
                coverage_by_pair[pair_id].append(float(available))
                if not available:
                    end_to_end_by_pair[pair_id].append(0.0)
                    continue
                decision_value = mean(selected.values())
                decisions_by_pair[pair_id].append(decision_value)
                expected = labels[pair_id].overall_preference.value
                correct = _direction(decision_value, tie_threshold) == expected
                accuracy_by_pair[pair_id].append(float(correct))
                end_to_end_by_pair[pair_id].append(float(correct))
                order_agreements.append(
                    float(
                        _direction(selected[ORDERS[0]], tie_threshold)
                        == _direction(selected[ORDERS[1]], tie_threshold)
                    )
                )
        accuracy_values = [
            value for values in accuracy_by_pair.values() for value in values
        ]
        end_to_end_values = [
            value for values in end_to_end_by_pair.values() for value in values
        ]
        coverage_values = [
            value for values in coverage_by_pair.values() for value in values
        ]
        subset_sd_values = [
            pstdev(values) for values in decisions_by_pair.values() if values
        ]
        results.append(
            {
                "configuration": f"adaptive_both_orders_up_to_{attempt_limit}_seed"
                + ("s" if attempt_limit != 1 else ""),
                "max_attempts_per_orientation": attempt_limit,
                "initial_calls_per_pair": 2,
                "max_calls_per_pair": 2 * attempt_limit,
                "expected_calls_per_pair": _mean_or_none(calls_used),
                "retry_activation_rate": _mean_or_none(retry_activated),
                "paired_response_coverage": _mean_or_none(coverage_values),
                "pair_accuracy_conditional_on_paired_response": _mean_or_none(
                    accuracy_values
                ),
                "pair_accuracy_conditional_ci95": _cluster_bootstrap_ci(
                    accuracy_by_pair, samples=bootstrap_samples
                ),
                "end_to_end_pair_accuracy": _mean_or_none(end_to_end_values),
                "end_to_end_pair_accuracy_ci95": _cluster_bootstrap_ci(
                    end_to_end_by_pair, samples=bootstrap_samples
                ),
                "order_consistency_rate": _mean_or_none(order_agreements),
                "mean_pair_decision_sd_across_start_seeds": _mean_or_none(
                    subset_sd_values
                ),
            }
        )
    return results


def analyze_operating_points(
    rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    labels: dict[str, HybridCalibrationLabels],
    *,
    tie_threshold: float = 0.05,
    bootstrap_samples: int = 2000,
) -> dict[str, object]:
    """Aggregate frozen calls under progressively larger repetition budgets."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    success_by_key: dict[tuple[str, str, int, str], float] = {}
    failure_keys: set[tuple[str, str, int, str]] = set()
    models: set[str] = set()
    for row in rows:
        key = (
            str(row["pair_id"]),
            str(row["judge_model"]),
            int(row["repetition"]),
            str(row["order"]),
        )
        if key in success_by_key:
            raise ValueError(f"duplicate successful outcome key: {key}")
        raw_value = row.get("baseline_decision_value")
        if raw_value in {None, ""}:
            raise ValueError(f"successful outcome has no decision value: {key}")
        success_by_key[key] = float(raw_value)
        models.add(key[1])
    for row in failures:
        key = (
            str(row["pair_id"]),
            str(row["judge_model"]),
            int(row["repetition"]),
            str(row["order"]),
        )
        if key in failure_keys:
            raise ValueError(f"duplicate failed outcome key: {key}")
        failure_keys.add(key)
        models.add(key[1])
    overlap = set(success_by_key) & failure_keys
    if overlap:
        raise ValueError(f"keys occur in both successes and failures: {len(overlap)}")
    unknown_pairs = sorted(
        ({key[0] for key in success_by_key} | {key[0] for key in failure_keys})
        - set(labels)
    )
    if unknown_pairs:
        raise ValueError(f"outcomes have no labels: {unknown_pairs}")
    all_keys = set(success_by_key) | failure_keys
    if not all_keys:
        raise ValueError("no successful or failed outcomes were supplied")

    output: dict[str, object] = {}
    for model in sorted(models):
        model_keys = {key for key in all_keys if key[1] == model}
        repetition_ids = tuple(sorted({key[2] for key in model_keys}))
        labeled_pairs = tuple(
            sorted(
                pair_id
                for pair_id, label in labels.items()
                if label.overall_preference
                is not ExpectedPairPreference.UNLABELED
                and any(key[0] == pair_id for key in model_keys)
            )
        )
        expected_model_keys = {
            (pair_id, model, repetition, order)
            for pair_id in labeled_pairs
            for repetition in repetition_ids
            for order in ORDERS
        }
        missing_model_keys = expected_model_keys - model_keys
        if missing_model_keys:
            raise ValueError(
                f"model {model!r} has {len(missing_model_keys)} unrecorded "
                "pair/repetition/order outcomes"
            )
        model_results = []
        for configuration in _configurations(repetition_ids):
            orders = tuple(str(item) for item in configuration["orders"])
            repetition_count = int(configuration["repetitions_per_order"])
            subsets = tuple(itertools.combinations(repetition_ids, repetition_count))
            accuracy_by_pair: dict[str, list[float]] = defaultdict(list)
            strict_accuracy_by_pair: dict[str, list[float]] = defaultdict(list)
            coverage_by_pair: dict[str, list[float]] = defaultdict(list)
            completeness_by_pair: dict[str, list[float]] = defaultdict(list)
            decisions_by_pair: dict[str, list[float]] = defaultdict(list)
            requested_call_count = 0
            successful_call_count = 0
            order_agreements: list[float] = []
            for repetitions_subset in subsets:
                for pair_id in labeled_pairs:
                    expected = labels[pair_id].overall_preference.value
                    requested_keys = [
                        (pair_id, model, repetition, order)
                        for repetition in repetitions_subset
                        for order in orders
                    ]
                    values = [
                        success_by_key[key]
                        for key in requested_keys
                        if key in success_by_key
                    ]
                    requested_call_count += len(requested_keys)
                    successful_call_count += len(values)
                    complete = len(values) == len(requested_keys)
                    available = bool(values)
                    completeness_by_pair[pair_id].append(float(complete))
                    coverage_by_pair[pair_id].append(float(available))
                    if available:
                        decision_value = mean(values)
                        decisions_by_pair[pair_id].append(decision_value)
                        correct = _direction(decision_value, tie_threshold) == expected
                        accuracy_by_pair[pair_id].append(float(correct))
                        strict_accuracy_by_pair[pair_id].append(
                            float(correct and complete)
                        )
                    else:
                        strict_accuracy_by_pair[pair_id].append(0.0)
                    if len(orders) == 2:
                        for repetition in repetitions_subset:
                            first = success_by_key.get(
                                (pair_id, model, repetition, ORDERS[0])
                            )
                            second = success_by_key.get(
                                (pair_id, model, repetition, ORDERS[1])
                            )
                            if first is not None and second is not None:
                                order_agreements.append(
                                    float(
                                        _direction(first, tie_threshold)
                                        == _direction(second, tie_threshold)
                                    )
                                )
            conditional_accuracy_values = [
                value for values in accuracy_by_pair.values() for value in values
            ]
            strict_accuracy_values = [
                value
                for values in strict_accuracy_by_pair.values()
                for value in values
            ]
            coverage_values = [
                value for values in coverage_by_pair.values() for value in values
            ]
            complete_values = [
                value
                for values in completeness_by_pair.values()
                for value in values
            ]
            subset_sd_values = [
                pstdev(values) for values in decisions_by_pair.values() if values
            ]
            model_results.append(
                {
                    "configuration": configuration["configuration"],
                    "orders": list(orders),
                    "repetitions_per_order": repetition_count,
                    "calls_per_pair": len(orders) * repetition_count,
                    "repetition_subset_count": len(subsets),
                    "labeled_pair_count": len(labeled_pairs),
                    "response_success_rate": (
                        successful_call_count / requested_call_count
                        if requested_call_count
                        else None
                    ),
                    "complete_response_rate": _mean_or_none(complete_values),
                    "pair_decision_coverage": _mean_or_none(coverage_values),
                    "pair_accuracy_conditional_on_decision": _mean_or_none(
                        conditional_accuracy_values
                    ),
                    "pair_accuracy_conditional_ci95": _cluster_bootstrap_ci(
                        accuracy_by_pair, samples=bootstrap_samples
                    ),
                    "strict_end_to_end_pair_accuracy": _mean_or_none(
                        strict_accuracy_values
                    ),
                    "strict_end_to_end_pair_accuracy_ci95": _cluster_bootstrap_ci(
                        strict_accuracy_by_pair, samples=bootstrap_samples
                    ),
                    "order_consistency_rate": _mean_or_none(order_agreements),
                    "mean_pair_decision_sd_across_subsets": _mean_or_none(
                        subset_sd_values
                    ),
                }
            )
        output[model] = {
            "available_repetitions": list(repetition_ids),
            "labeled_pair_count": len(labeled_pairs),
            "operating_points": model_results,
            "adaptive_operating_points": _adaptive_operating_points(
                model=model,
                repetitions=repetition_ids,
                pair_ids=labeled_pairs,
                success_by_key=success_by_key,
                labels=labels,
                tie_threshold=tie_threshold,
                bootstrap_samples=bootstrap_samples,
            ),
        }
    return output


def _format(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def _write_summary(path: Path, metrics: dict[str, object]) -> None:
    lines = [
        "# Hybrid judge operating-point analysis",
        "",
        "Frozen successful and failed calls are reused without new LLM requests. "
        "Confidence intervals cluster by calibration pair. Strict end-to-end "
        "accuracy requires every requested call to succeed; conditional accuracy "
        "uses trials with at least one valid response.",
        "",
    ]
    for model, raw in metrics.items():
        values = raw
        assert isinstance(values, dict)
        lines.extend(
            [
                f"## {model}",
                "",
                "| Configuration | Calls/pair | Response success | Complete trials | "
                "Decision coverage | Conditional pair accuracy | Strict end-to-end | "
                "Order consistency | Decision SD |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        operating_points = values["operating_points"]
        assert isinstance(operating_points, list)
        for item in operating_points:
            assert isinstance(item, dict)
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(item["configuration"]),
                        str(item["calls_per_pair"]),
                        _format(item["response_success_rate"]),
                        _format(item["complete_response_rate"]),
                        _format(item["pair_decision_coverage"]),
                        _format(item["pair_accuracy_conditional_on_decision"]),
                        _format(item["strict_end_to_end_pair_accuracy"]),
                        _format(item["order_consistency_rate"]),
                        _format(item["mean_pair_decision_sd_across_subsets"]),
                    )
                )
                + " |"
            )
        lines.append("")
        lines.extend(
            [
                "### Symmetry-preserving adaptive retry",
                "",
                "A decision is emitted only after both orientations return a valid "
                "response. A failed orientation alone advances to the next frozen "
                "seed.",
                "",
                "| Maximum seeds/orientation | Expected calls/pair | Maximum calls | "
                "Retry activation | Paired coverage | Conditional pair accuracy | "
                "End-to-end pair accuracy | Order consistency | Decision SD |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        adaptive_points = values["adaptive_operating_points"]
        assert isinstance(adaptive_points, list)
        for item in adaptive_points:
            assert isinstance(item, dict)
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(item["max_attempts_per_orientation"]),
                        _format(item["expected_calls_per_pair"]),
                        str(item["max_calls_per_pair"]),
                        _format(item["retry_activation_rate"]),
                        _format(item["paired_response_coverage"]),
                        _format(
                            item[
                                "pair_accuracy_conditional_on_paired_response"
                            ]
                        ),
                        _format(item["end_to_end_pair_accuracy"]),
                        _format(item["order_consistency_rate"]),
                        _format(
                            item["mean_pair_decision_sd_across_start_seeds"]
                        ),
                    )
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie-threshold", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    with args.scores.open(encoding="utf-8", newline="") as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))
    failures = []
    if args.failures is not None and args.failures.is_file():
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
    metrics = analyze_operating_points(
        rows,
        failures,
        labels,
        tie_threshold=args.tie_threshold,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(
        args.output.with_name("hybrid_judge_operating_points.md"), metrics
    )


if __name__ == "__main__":
    main()
