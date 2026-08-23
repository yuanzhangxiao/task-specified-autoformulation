"""Diagnose frozen hybrid-judge errors without making new LLM calls."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    ExpectedVerdict,
    HybridCalibrationLabels,
)
from autoformalism.search.controller import _structural_hash

SCHEMA_VERSION = "hybrid-judge-diagnostics-1"
DEFAULT_COMPARATIVE_WEIGHTS = (0.0, 0.125, 0.25, 0.5, 1.0)
DEFAULT_TIE_THRESHOLDS = (0.0, 0.025, 0.05, 0.1)
ORDERS = ("baseline_a", "baseline_b")


@dataclass(frozen=True)
class PairMetadata:
    """Mutation and canonical baseline-structure identity for one pair."""

    mutation_type: str
    structure_id: str


def _direction(value: float | None, threshold: float) -> str:
    if value is None:
        return "indeterminate"
    if value > threshold:
        return "baseline"
    if value < -threshold:
        return "mutated"
    return "tie"


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _optional_bool(value: object) -> bool | None:
    if value in {True, "True", "true", "1"}:
        return True
    if value in {False, "False", "false", "0"}:
        return False
    if value in {None, "", "None", "null"}:
        return None
    raise ValueError(f"invalid optional Boolean CSV value: {value!r}")


def _baseline_absolute_delta(row: dict[str, object]) -> float:
    left = float(row["candidate_a_score"])
    right = float(row["candidate_b_score"])
    delta = left - right
    return delta if row["baseline_position"] == "A" else -delta


def _baseline_hard_statuses(
    row: dict[str, object],
) -> tuple[bool | None, bool | None]:
    left = _optional_bool(row.get("candidate_a_hard_status"))
    right = _optional_bool(row.get("candidate_b_hard_status"))
    return (left, right) if row["baseline_position"] == "A" else (right, left)


def _baseline_relative_delta(row: dict[str, object]) -> float:
    raw = row.get("baseline_relative_preference")
    return 0.0 if raw in {None, ""} else 2.0 * float(raw) - 1.0


def _reconstructed_decision(
    row: dict[str, object], *, comparative_weight: float
) -> float:
    baseline_hard, mutated_hard = _baseline_hard_statuses(row)
    if baseline_hard is True and mutated_hard is False:
        return 1.0
    if baseline_hard is False and mutated_hard is True:
        return -1.0
    return _baseline_absolute_delta(row) + comparative_weight * (
        _baseline_relative_delta(row)
    )


def _normalized_absolute(
    items: list[dict[str, Any]], baseline_position: str
) -> dict[tuple[str, str], tuple[str, str]]:
    output = {}
    for item in items:
        key = (str(item["criterion"]), str(item["subject_id"]))
        left = str(item["candidate_a"]["verdict"])
        right = str(item["candidate_b"]["verdict"])
        output[key] = (
            (left, right) if baseline_position == "A" else (right, left)
        )
    return output


def _normalized_relative(
    items: list[dict[str, Any]], baseline_position: str
) -> dict[str, str]:
    output = {}
    for item in items:
        verdict = str(item["verdict"])
        if verdict in {"candidate_a", "candidate_b"}:
            position = "A" if verdict == "candidate_a" else "B"
            verdict = "baseline" if position == baseline_position else "mutated"
        output[str(item["criterion"])] = verdict
    return output


def _signed_margin(value: float, expected: str, threshold: float) -> float | None:
    if expected == "baseline":
        return value - threshold
    if expected == "mutated":
        return -value - threshold
    if expected == "tie":
        return threshold - abs(value)
    return None


def _order_consistency(
    rows: list[dict[str, object]], *, weight: float, threshold: float
) -> tuple[int, float | None]:
    grouped: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for row in rows:
        key = (str(row["pair_id"]), int(row["repetition"]))
        grouped[key][str(row["order"])] = _direction(
            _reconstructed_decision(row, comparative_weight=weight), threshold
        )
    complete = [item for item in grouped.values() if set(ORDERS) <= set(item)]
    agreements = [item[ORDERS[0]] == item[ORDERS[1]] for item in complete]
    return len(complete), _rate(agreements)


def _question_performance(
    rows: list[dict[str, object]],
    labels: dict[str, HybridCalibrationLabels],
    metadata: dict[str, PairMetadata],
) -> list[dict[str, object]]:
    outcomes: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for row in rows:
        pair_id = str(row["pair_id"])
        gold = labels[pair_id]
        mutation = metadata[pair_id].mutation_type
        position = str(row["baseline_position"])
        runtime = _normalized_absolute(
            json.loads(str(row["deterministic_assessments"])), position
        )
        semantic = _normalized_absolute(
            json.loads(str(row["absolute_assessments"])), position
        )
        for item in gold.absolute_labels:
            key = (item.criterion.value, item.subject_id)
            layer = (
                "deterministic_runtime"
                if item.label_source == "deterministic_runtime"
                else "llm_semantic_absolute"
            )
            actual = runtime if layer == "deterministic_runtime" else semantic
            if key not in actual:
                continue
            baseline, mutated = actual[key]
            if item.baseline is not ExpectedVerdict.UNLABELED:
                outcomes[(layer, item.criterion.value, mutation)].append(
                    baseline == item.baseline.value
                )
            if item.mutated is not ExpectedVerdict.UNLABELED:
                outcomes[(layer, item.criterion.value, mutation)].append(
                    mutated == item.mutated.value
                )
        relative = _normalized_relative(
            json.loads(str(row["comparative_assessments"])), position
        )
        for item in gold.comparative_labels:
            if item.preference is ExpectedPairPreference.UNLABELED:
                continue
            outcomes[("llm_comparative", item.criterion.value, mutation)].append(
                relative.get(item.criterion.value) == item.preference.value
            )
    records = []
    for (layer, criterion, mutation), values in sorted(outcomes.items()):
        records.append(
            {
                "layer": layer,
                "criterion": criterion,
                "mutation_type": mutation,
                "correct": sum(values),
                "total": len(values),
                "accuracy": _rate(values),
            }
        )
    return records


def _pair_diagnostics(
    rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    labels: dict[str, HybridCalibrationLabels],
    metadata: dict[str, PairMetadata],
    *,
    weight: float,
    threshold: float,
) -> list[dict[str, object]]:
    success_by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    failure_by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        success_by_pair[str(row["pair_id"])].append(row)
    for row in failures:
        failure_by_pair[str(row["pair_id"])].append(row)
    records = []
    for pair_id in sorted(set(success_by_pair) | set(failure_by_pair)):
        pair_rows = success_by_pair[pair_id]
        expected = labels[pair_id].overall_preference.value
        decisions = [
            _reconstructed_decision(row, comparative_weight=weight)
            for row in pair_rows
        ]
        absolute = [_baseline_absolute_delta(row) for row in pair_rows]
        relative = [_baseline_relative_delta(row) for row in pair_rows]
        decision_mean = mean(decisions) if decisions else None
        predictions = [_direction(value, threshold) for value in decisions]
        order_means = {
            order: mean(values) if values else None
            for order in ORDERS
            for values in [
                [
                    value
                    for value, row in zip(decisions, pair_rows, strict=True)
                    if row["order"] == order
                ]
            ]
        }
        paired_count, consistency = _order_consistency(
            pair_rows, weight=weight, threshold=threshold
        )
        records.append(
            {
                "pair_id": pair_id,
                "structure_id": metadata[pair_id].structure_id,
                "mutation_type": metadata[pair_id].mutation_type,
                "expected": expected,
                "successful_calls": len(pair_rows),
                "failed_calls": len(failure_by_pair[pair_id]),
                "response_success_rate": len(pair_rows)
                / (len(pair_rows) + len(failure_by_pair[pair_id])),
                "conditional_call_accuracy": _rate(
                    [prediction == expected for prediction in predictions]
                ),
                "end_to_end_call_accuracy": sum(
                    prediction == expected for prediction in predictions
                )
                / (len(pair_rows) + len(failure_by_pair[pair_id])),
                "absolute_delta_mean": mean(absolute) if absolute else None,
                "comparative_delta_mean": mean(relative) if relative else None,
                "decision_mean": decision_mean,
                "decision_sd": pstdev(decisions) if decisions else None,
                "predicted": _direction(decision_mean, threshold),
                "correct": _direction(decision_mean, threshold) == expected,
                "signed_margin": (
                    _signed_margin(decision_mean, expected, threshold)
                    if decision_mean is not None
                    else None
                ),
                "baseline_a_mean": order_means["baseline_a"],
                "baseline_b_mean": order_means["baseline_b"],
                "paired_repetitions": paired_count,
                "order_consistency": consistency,
            }
        )
    return records


def _mutation_diagnostics(
    pair_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in pair_records:
        groups[str(record["mutation_type"])].append(record)
    output = []
    for mutation, records in sorted(groups.items()):
        successes = sum(int(item["successful_calls"]) for item in records)
        failures = sum(int(item["failed_calls"]) for item in records)
        complete_consistency = [
            float(item["order_consistency"])
            for item in records
            if item["order_consistency"] is not None
        ]
        output.append(
            {
                "mutation_type": mutation,
                "pair_count": len(records),
                "response_success_rate": successes / (successes + failures),
                "conditional_call_accuracy": mean(
                    float(item["conditional_call_accuracy"]) for item in records
                ),
                "pair_accuracy": _rate([bool(item["correct"]) for item in records]),
                "order_consistency": (
                    mean(complete_consistency) if complete_consistency else None
                ),
                "mean_signed_margin": mean(
                    float(item["signed_margin"]) for item in records
                ),
            }
        )
    return output


def _sensitivity_grid(
    rows: list[dict[str, object]],
    labels: dict[str, HybridCalibrationLabels],
    metadata: dict[str, PairMetadata],
    *,
    weights: tuple[float, ...],
    thresholds: tuple[float, ...],
    current_weight: float,
    current_threshold: float,
) -> list[dict[str, object]]:
    by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    output = []
    for weight, threshold in itertools.product(weights, thresholds):
        pair_outcomes: dict[str, bool] = {}
        structure_outcomes: dict[str, list[bool]] = defaultdict(list)
        misclassified = []
        for pair_id, pair_rows in by_pair.items():
            value = mean(
                _reconstructed_decision(row, comparative_weight=weight)
                for row in pair_rows
            )
            expected = labels[pair_id].overall_preference.value
            correct = _direction(value, threshold) == expected
            pair_outcomes[pair_id] = correct
            structure_outcomes[metadata[pair_id].structure_id].append(correct)
            if not correct:
                misclassified.append(pair_id)
        _, consistency = _order_consistency(
            rows, weight=weight, threshold=threshold
        )
        structure_accuracy = {
            structure: _rate(values)
            for structure, values in sorted(structure_outcomes.items())
        }
        output.append(
            {
                "comparative_weight": weight,
                "tie_threshold": threshold,
                "is_current": weight == current_weight
                and threshold == current_threshold,
                "pair_accuracy": _rate(list(pair_outcomes.values())),
                "mean_structure_accuracy": mean(
                    float(value) for value in structure_accuracy.values()
                ),
                "worst_structure_accuracy": min(
                    float(value) for value in structure_accuracy.values()
                ),
                "order_consistency": consistency,
                "structure_accuracy": structure_accuracy,
                "misclassified_pair_ids": sorted(misclassified),
            }
        )
    return output


def _leave_one_structure_out(
    sensitivity: list[dict[str, object]],
    metadata: dict[str, PairMetadata],
    labels: dict[str, HybridCalibrationLabels],
    rows: list[dict[str, object]],
    *,
    current_weight: float,
    current_threshold: float,
) -> dict[str, object]:
    by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    structures = sorted(
        {metadata[pair_id].structure_id for pair_id in by_pair}
    )
    folds = []
    for heldout in structures:
        train_structures = set(structures) - {heldout}

        def accuracy_for(config: dict[str, object], selected: set[str]) -> float:
            outcomes = []
            weight = float(config["comparative_weight"])
            threshold = float(config["tie_threshold"])
            for pair_id, pair_rows in by_pair.items():
                if metadata[pair_id].structure_id not in selected:
                    continue
                value = mean(
                    _reconstructed_decision(row, comparative_weight=weight)
                    for row in pair_rows
                )
                outcomes.append(
                    _direction(value, threshold)
                    == labels[pair_id].overall_preference.value
                )
            return float(_rate(outcomes) or 0.0)

        ranked = sorted(
            sensitivity,
            key=lambda item: (
                -accuracy_for(item, train_structures),
                abs(float(item["comparative_weight"]) - current_weight),
                abs(float(item["tie_threshold"]) - current_threshold),
                float(item["comparative_weight"]),
                float(item["tie_threshold"]),
            ),
        )
        selected = ranked[0]
        folds.append(
            {
                "heldout_structure_id": heldout,
                "training_structure_count": len(train_structures),
                "selected_comparative_weight": selected["comparative_weight"],
                "selected_tie_threshold": selected["tie_threshold"],
                "training_pair_accuracy": accuracy_for(selected, train_structures),
                "heldout_pair_accuracy": accuracy_for(selected, {heldout}),
            }
        )
    return {
        "status": "exploratory_posthoc_not_a_frozen_protocol_selection",
        "structure_count": len(structures),
        "folds": folds,
        "mean_heldout_pair_accuracy": mean(
            float(item["heldout_pair_accuracy"]) for item in folds
        ),
    }


def analyze_diagnostics(
    rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    labels: dict[str, HybridCalibrationLabels],
    metadata: dict[str, PairMetadata],
    *,
    current_weight: float = 0.25,
    current_threshold: float = 0.05,
    comparative_weights: tuple[float, ...] = DEFAULT_COMPARATIVE_WEIGHTS,
    tie_thresholds: tuple[float, ...] = DEFAULT_TIE_THRESHOLDS,
) -> dict[str, object]:
    """Return frozen pair diagnostics and exploratory aggregation sensitivity."""
    comparative_weights = tuple(
        sorted({*comparative_weights, current_weight})
    )
    tie_thresholds = tuple(sorted({*tie_thresholds, current_threshold}))
    outcome_pairs = {
        str(row["pair_id"]) for row in (*rows, *failures)
    }
    missing_labels = sorted(outcome_pairs - set(labels))
    missing_metadata = sorted(outcome_pairs - set(metadata))
    if missing_labels:
        raise ValueError(f"outcomes have no labels: {missing_labels}")
    if missing_metadata:
        raise ValueError(f"outcomes have no pair metadata: {missing_metadata}")
    if any(
        labels[pair_id].overall_preference is ExpectedPairPreference.UNLABELED
        for pair_id in outcome_pairs
    ):
        raise ValueError("diagnostic outcomes require labeled overall preferences")

    def outcome_key(row: dict[str, object]) -> tuple[str, str, int, str]:
        return (
            str(row["pair_id"]),
            str(row["judge_model"]),
            int(row["repetition"]),
            str(row["order"]),
        )

    success_keys = [outcome_key(row) for row in rows]
    failure_keys = [outcome_key(row) for row in failures]
    if len(success_keys) != len(set(success_keys)):
        raise ValueError("duplicate successful outcome keys")
    if len(failure_keys) != len(set(failure_keys)):
        raise ValueError("duplicate failed outcome keys")
    overlap = set(success_keys) & set(failure_keys)
    if overlap:
        raise ValueError(f"keys occur in successes and failures: {len(overlap)}")
    for row in rows:
        pair_id = str(row["pair_id"])
        mutation = str(row["mutation_type"])
        if mutation != metadata[pair_id].mutation_type:
            raise ValueError(
                "score mutation does not match frozen pair metadata: "
                f"pair={pair_id} score={mutation} "
                f"pair_file={metadata[pair_id].mutation_type}"
            )
        stored = float(row["baseline_decision_value"])
        reconstructed = _reconstructed_decision(
            row, comparative_weight=current_weight
        )
        if abs(stored - reconstructed) > 1e-9:
            raise ValueError(
                "stored decision does not match reconstructed frozen score: "
                f"pair={row['pair_id']} stored={stored} reconstructed={reconstructed}"
            )
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    failures_by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["judge_model"])].append(row)
    for row in failures:
        failures_by_model[str(row["judge_model"])].append(row)
    models = {}
    for model in sorted(set(by_model) | set(failures_by_model)):
        model_rows = by_model[model]
        model_failures = failures_by_model[model]
        pair_records = _pair_diagnostics(
            model_rows,
            model_failures,
            labels,
            metadata,
            weight=current_weight,
            threshold=current_threshold,
        )
        sensitivity = _sensitivity_grid(
            model_rows,
            labels,
            metadata,
            weights=comparative_weights,
            thresholds=tie_thresholds,
            current_weight=current_weight,
            current_threshold=current_threshold,
        )
        models[model] = {
            "successful_calls": len(model_rows),
            "failed_calls": len(model_failures),
            "response_success_rate": len(model_rows)
            / (len(model_rows) + len(model_failures)),
            "pair_diagnostics": pair_records,
            "mutation_diagnostics": _mutation_diagnostics(pair_records),
            "question_performance": _question_performance(
                model_rows, labels, metadata
            ),
            "aggregation_sensitivity": sensitivity,
            "leave_one_structure_out": _leave_one_structure_out(
                sensitivity,
                metadata,
                labels,
                model_rows,
                current_weight=current_weight,
                current_threshold=current_threshold,
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "exploratory_posthoc_diagnostic",
        "current_comparative_weight": current_weight,
        "current_tie_threshold": current_threshold,
        "models": models,
    }


def _format(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(payload: dict[str, object]) -> str:
    """Render a compact human-readable diagnostic report."""
    lines = [
        "# Hybrid judge diagnostic analysis",
        "",
        "This is a post-hoc diagnostic over frozen calls. Aggregation sensitivity "
        "is exploratory and cannot select a confirmatory protocol on the same data.",
    ]
    models = payload["models"]
    assert isinstance(models, dict)
    for model, raw in models.items():
        values = raw
        assert isinstance(values, dict)
        lines.extend(
            [
                "",
                f"## {model}",
                "",
                f"Responses: {values['successful_calls']} successful, "
                f"{values['failed_calls']} failed "
                f"({_format(values['response_success_rate'])} success).",
                "",
                "### Pair-level attribution",
                "",
                "| Pair | Structure | Mutation | Expected | Predicted | Correct | "
                "Decision | Signed margin | Call accuracy | A mean | B mean | "
                "Order consistency |",
                "|---|---|---|---|---|:---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in values["pair_diagnostics"]:
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(item["pair_id"]),
                        str(item["structure_id"]),
                        str(item["mutation_type"]),
                        str(item["expected"]),
                        str(item["predicted"]),
                        "yes" if item["correct"] else "no",
                        _format(item["decision_mean"]),
                        _format(item["signed_margin"]),
                        _format(item["conditional_call_accuracy"]),
                        _format(item["baseline_a_mean"]),
                        _format(item["baseline_b_mean"]),
                        _format(item["order_consistency"]),
                    )
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "### Mutation-level attribution",
                "",
                "| Mutation | Pairs | Response success | Call accuracy | "
                "Pair accuracy | "
                "Order consistency | Mean signed margin |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in values["mutation_diagnostics"]:
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(item["mutation_type"]),
                        str(item["pair_count"]),
                        _format(item["response_success_rate"]),
                        _format(item["conditional_call_accuracy"]),
                        _format(item["pair_accuracy"]),
                        _format(item["order_consistency"]),
                        _format(item["mean_signed_margin"]),
                    )
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "### Certified atomic-question performance",
                "",
                "| Layer | Criterion | Mutation | Correct | Total | Accuracy |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for item in values["question_performance"]:
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(item["layer"]),
                        str(item["criterion"]),
                        str(item["mutation_type"]),
                        str(item["correct"]),
                        str(item["total"]),
                        _format(item["accuracy"]),
                    )
                )
                + " |"
            )
        sensitivity = sorted(
            values["aggregation_sensitivity"],
            key=lambda item: (
                -float(item["worst_structure_accuracy"]),
                -float(item["pair_accuracy"]),
                -float(item["order_consistency"] or 0.0),
            ),
        )
        current = next(item for item in sensitivity if item["is_current"])
        shown = [current]
        shown.extend(item for item in sensitivity if item is not current)
        lines.extend(
            [
                "",
                "### Exploratory aggregation sensitivity",
                "",
                "The current configuration is shown first, followed by the strongest "
                "structure-robust grid points. These are not confirmatory selections.",
                "",
                "| Current | Comparative weight | Tie threshold | Pair accuracy | "
                "Worst-structure accuracy | Order consistency | Misclassified pairs |",
                "|:---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in shown[:10]:
            lines.append(
                "| "
                + " | ".join(
                    (
                        "yes" if item["is_current"] else "",
                        _format(item["comparative_weight"]),
                        _format(item["tie_threshold"]),
                        _format(item["pair_accuracy"]),
                        _format(item["worst_structure_accuracy"]),
                        _format(item["order_consistency"]),
                        ", ".join(item["misclassified_pair_ids"]) or "none",
                    )
                )
                + " |"
            )
        cross_validation = values["leave_one_structure_out"]
        lines.extend(
            [
                "",
                "### Leave-one-baseline-structure-out sensitivity",
                "",
                "Only baseline structure is used as the split unit. With few "
                "structures, "
                "this is diagnostic rather than a reliable tuning estimate.",
                "",
                "| Held-out structure | Selected weight | Selected threshold | "
                "Training accuracy | Held-out accuracy |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in cross_validation["folds"]:
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(item["heldout_structure_id"]),
                        _format(item["selected_comparative_weight"]),
                        _format(item["selected_tie_threshold"]),
                        _format(item["training_pair_accuracy"]),
                        _format(item["heldout_pair_accuracy"]),
                    )
                )
                + " |"
            )
        lines.append(
            "\nMean held-out pair accuracy: "
            f"{_format(cross_validation['mean_heldout_pair_accuracy'])}."
        )
    return "\n".join(lines) + "\n"


def _load_pairs(path: Path) -> dict[str, PairMetadata]:
    pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {
        pair.pair_id: PairMetadata(
            mutation_type=pair.mutation_type,
            structure_id=_structural_hash(pair.valid_candidate)[:12],
        )
        for pair in pairs
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--current-comparative-weight", type=float, default=0.25)
    parser.add_argument("--current-tie-threshold", type=float, default=0.05)
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
    payload = analyze_diagnostics(
        rows,
        failures,
        labels,
        _load_pairs(args.pairs),
        current_weight=args.current_comparative_weight,
        current_threshold=args.current_tie_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote diagnostics to {args.output} and {markdown_path}")


if __name__ == "__main__":
    main()
