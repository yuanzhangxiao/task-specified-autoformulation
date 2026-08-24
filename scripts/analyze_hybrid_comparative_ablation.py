"""Ablate comparative criteria and scoring constants on frozen hybrid outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    HybridCalibrationLabels,
)
from autoformalism.schemas import RelativeCriterion

_ALL_CRITERIA = tuple(RelativeCriterion)
_DEFAULT_LAMBDAS = (0.0, 0.05, 0.1, 0.25, 0.5)
_DEFAULT_THRESHOLDS = (0.0, 0.025, 0.05, 0.1, 0.2)


@dataclass(frozen=True)
class AblationDefinition:
    """One fixed comparative-criterion subset."""

    name: str
    criteria: tuple[RelativeCriterion, ...] | None
    diagnostic_only: bool = False


_ABLATIONS = (
    AblationDefinition("absolute_only", ()),
    AblationDefinition("all_three", _ALL_CRITERIA),
    AblationDefinition(
        "without_mechanistic_interpretability",
        (
            RelativeCriterion.PARSIMONY_WHILE_TASK_SUFFICIENT,
            RelativeCriterion.FEWER_UNSUPPORTED_ASSUMPTIONS,
        ),
    ),
    AblationDefinition(
        "without_parsimony",
        (
            RelativeCriterion.FEWER_UNSUPPORTED_ASSUMPTIONS,
            RelativeCriterion.MECHANISTIC_INTERPRETABILITY,
        ),
    ),
    AblationDefinition(
        "without_unsupported_assumptions",
        (
            RelativeCriterion.PARSIMONY_WHILE_TASK_SUFFICIENT,
            RelativeCriterion.MECHANISTIC_INTERPRETABILITY,
        ),
    ),
    AblationDefinition(
        "parsimony_only",
        (RelativeCriterion.PARSIMONY_WHILE_TASK_SUFFICIENT,),
    ),
    AblationDefinition(
        "unsupported_assumptions_only",
        (RelativeCriterion.FEWER_UNSUPPORTED_ASSUMPTIONS,),
    ),
    AblationDefinition(
        "mechanistic_interpretability_only",
        (RelativeCriterion.MECHANISTIC_INTERPRETABILITY,),
        diagnostic_only=True,
    ),
    AblationDefinition(
        "mutation_contract_labeled_only",
        None,
        diagnostic_only=True,
    ),
)


def _load_csv(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _load_jsonl(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            continue
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _load_labels(path: Path) -> dict[str, HybridCalibrationLabels]:
    return {
        item.pair_id: item
        for item in (
            HybridCalibrationLabels.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _parse_optional_bool(value: str) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _direction(value: float | None, threshold: float) -> str:
    if value is None:
        return ExpectedPairPreference.INDETERMINATE.value
    if value > threshold:
        return ExpectedPairPreference.BASELINE.value
    if value < -threshold:
        return ExpectedPairPreference.MUTATED.value
    return ExpectedPairPreference.TIE.value


def _comparative_values_for_baseline(row: dict[str, str]) -> dict[str, float | None]:
    baseline_position = row["baseline_position"]
    values: dict[str, float | None] = {}
    for item in json.loads(row["comparative_assessments"]):
        verdict = str(item["verdict"])
        if verdict == "tie":
            value = 0.5
        elif verdict == "candidate_a":
            value = 1.0 if baseline_position == "A" else 0.0
        elif verdict == "candidate_b":
            value = 1.0 if baseline_position == "B" else 0.0
        else:
            value = None
        values[str(item["criterion"])] = value
    return values


def _labeled_criteria(
    labels: HybridCalibrationLabels,
) -> tuple[RelativeCriterion, ...]:
    return tuple(
        item.criterion
        for item in labels.comparative_labels
        if item.preference is not ExpectedPairPreference.UNLABELED
    )


def _criteria_for_row(
    definition: AblationDefinition,
    labels: HybridCalibrationLabels,
) -> tuple[RelativeCriterion, ...]:
    return (
        _labeled_criteria(labels)
        if definition.criteria is None
        else definition.criteria
    )


def evaluate_row(
    row: dict[str, str],
    *,
    labels: HybridCalibrationLabels,
    definition: AblationDefinition,
    comparative_weight: float,
    tie_threshold: float,
) -> dict[str, object]:
    """Recompute one baseline-aligned decision for an ablation."""
    baseline_position = row["baseline_position"]
    score_a = float(row["candidate_a_score"])
    score_b = float(row["candidate_b_score"])
    absolute_delta = (
        score_a - score_b if baseline_position == "A" else score_b - score_a
    )
    hard_a = _parse_optional_bool(row["candidate_a_hard_status"])
    hard_b = _parse_optional_bool(row["candidate_b_hard_status"])
    baseline_hard, mutated_hard = (
        (hard_a, hard_b) if baseline_position == "A" else (hard_b, hard_a)
    )
    hard_override = (
        1.0
        if baseline_hard is True and mutated_hard is False
        else -1.0
        if baseline_hard is False and mutated_hard is True
        else None
    )
    values = _comparative_values_for_baseline(row)
    selected = _criteria_for_row(definition, labels)
    determined = [
        values[item.value]
        for item in selected
        if values.get(item.value) is not None
    ]
    relative = mean(determined) if determined else None
    comparative_delta = 0.0 if relative is None else 2.0 * relative - 1.0
    contribution = comparative_weight * comparative_delta
    decision = (
        hard_override
        if hard_override is not None
        else absolute_delta + contribution
    )
    predicted = _direction(decision, tie_threshold)
    expected = labels.overall_preference.value
    return {
        "pair_id": row["pair_id"],
        "mutation_type": row["mutation_type"],
        "judge_model": row["judge_model"],
        "repetition": int(row["repetition"]),
        "order": row["order"],
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "absolute_delta": absolute_delta,
        "selected_criteria": [item.value for item in selected],
        "determined_comparative_count": len(determined),
        "comparative_preference_for_baseline": relative,
        "weighted_comparative_contribution": contribution,
        "decision_for_baseline": decision,
        "hard_override": hard_override is not None,
    }


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _order_consistency(rows: list[dict[str, object]]) -> float | None:
    grouped: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["pair_id"]), int(row["repetition"]))][
            str(row["order"])
        ] = str(row["predicted"])
    paired = [values for values in grouped.values() if len(values) == 2]
    return _rate([len(set(values.values())) == 1 for values in paired])


def _pair_aggregate_accuracy(
    rows: list[dict[str, object]],
    *,
    threshold: float,
) -> float | None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    outcomes = []
    for pair_rows in grouped.values():
        decisions = [float(row["decision_for_baseline"]) for row in pair_rows]
        prediction = _direction(mean(decisions), threshold)
        outcomes.append(prediction == pair_rows[0]["expected"])
    return _rate(outcomes)


def _aggregate(
    rows: list[dict[str, object]],
    *,
    failure_count: int,
    threshold: float,
) -> dict[str, object]:
    correct = [bool(row["correct"]) for row in rows]
    attempted = len(rows) + failure_count
    predictions = [str(row["predicted"]) for row in rows]
    return {
        "successful_calls": len(rows),
        "failed_calls": failure_count,
        "response_success_rate": len(rows) / attempted if attempted else None,
        "conditional_accuracy": _rate(correct),
        "end_to_end_accuracy": sum(correct) / attempted if attempted else None,
        "pair_aggregate_accuracy": _pair_aggregate_accuracy(
            rows, threshold=threshold
        ),
        "order_consistency": _order_consistency(rows),
        "baseline_preference_rate": _rate(
            [value == "baseline" for value in predictions]
        ),
        "tie_rate": _rate([value == "tie" for value in predictions]),
        "mean_decision_for_baseline": (
            mean(float(row["decision_for_baseline"]) for row in rows)
            if rows
            else None
        ),
        "mean_determined_comparative_count": (
            mean(int(row["determined_comparative_count"]) for row in rows)
            if rows
            else None
        ),
    }


def analyze(
    raw_rows: list[dict[str, str]],
    failures: list[dict[str, object]],
    labels: dict[str, HybridCalibrationLabels],
    *,
    comparative_weight: float,
    tie_threshold: float,
    lambdas: tuple[float, ...] = _DEFAULT_LAMBDAS,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    """Return fixed ablations and all-three/no-interpretability sensitivity."""
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    failures_by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        by_model[row["judge_model"]].append(row)
    for row in failures:
        failures_by_model[str(row["judge_model"])].append(row)
    models = {}
    for model in sorted(set(by_model) | set(failures_by_model)):
        model_rows = by_model[model]
        failure_count = len(failures_by_model[model])
        ablations = {}
        for definition in _ABLATIONS:
            evaluated = [
                evaluate_row(
                    row,
                    labels=labels[row["pair_id"]],
                    definition=definition,
                    comparative_weight=comparative_weight,
                    tie_threshold=tie_threshold,
                )
                for row in model_rows
            ]
            ablations[definition.name] = {
                "criteria": (
                    "pair_specific_mutation_contract_labels"
                    if definition.criteria is None
                    else [item.value for item in definition.criteria]
                ),
                "diagnostic_only": definition.diagnostic_only,
                **_aggregate(
                    evaluated,
                    failure_count=failure_count,
                    threshold=tie_threshold,
                ),
            }
        sensitivity = {}
        for definition in (
            next(item for item in _ABLATIONS if item.name == "all_three"),
            next(
                item
                for item in _ABLATIONS
                if item.name == "without_mechanistic_interpretability"
            ),
        ):
            cells = []
            for weight in lambdas:
                for threshold in thresholds:
                    evaluated = [
                        evaluate_row(
                            row,
                            labels=labels[row["pair_id"]],
                            definition=definition,
                            comparative_weight=weight,
                            tie_threshold=threshold,
                        )
                        for row in model_rows
                    ]
                    cells.append(
                        {
                            "comparative_weight": weight,
                            "tie_threshold": threshold,
                            **_aggregate(
                                evaluated,
                                failure_count=failure_count,
                                threshold=threshold,
                            ),
                        }
                    )
            sensitivity[definition.name] = cells
        models[model] = {
            "fixed_ablation": ablations,
            "sensitivity": sensitivity,
        }
    return {
        "schema_version": "hybrid-comparative-ablation-1",
        "frozen_default": {
            "comparative_weight": comparative_weight,
            "tie_threshold": tie_threshold,
        },
        "sensitivity_grid": {
            "comparative_weights": list(lambdas),
            "tie_thresholds": list(thresholds),
        },
        "models": models,
    }


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render_markdown(metrics: dict[str, object]) -> str:
    lines = [
        "# Hybrid comparative-criterion ablation",
        "",
        (
            "Frozen rows are rescored without new LLM calls. The "
            "mutation-contract-labeled-only configuration is a diagnostic "
            "upper bound and is not available during ordinary judging."
        ),
    ]
    for model, model_metrics in metrics["models"].items():
        lines.extend(
            [
                "",
                f"## {model}",
                "",
                "### Fixed criterion ablation",
                "",
                (
                    "| Configuration | Diagnostic only | Conditional accuracy | "
                    "End-to-end | Pair aggregate | Order consistency | Baseline "
                    "preference | Tie rate | Mean D |"
                ),
                "|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, item in model_metrics["fixed_ablation"].items():
            lines.append(
                "| "
                + " | ".join(
                    (
                        name,
                        "yes" if item["diagnostic_only"] else "",
                        _fmt(item["conditional_accuracy"]),
                        _fmt(item["end_to_end_accuracy"]),
                        _fmt(item["pair_aggregate_accuracy"]),
                        _fmt(item["order_consistency"]),
                        _fmt(item["baseline_preference_rate"]),
                        _fmt(item["tie_rate"]),
                        _fmt(item["mean_decision_for_baseline"]),
                    )
                )
                + " |"
            )
        for configuration, cells in model_metrics["sensitivity"].items():
            thresholds = metrics["sensitivity_grid"]["tie_thresholds"]
            lines.extend(
                [
                    "",
                    f"### Sensitivity: {configuration}",
                    "",
                    (
                        "Cells are conditional accuracy; rows are comparative "
                        "weights and columns are tie thresholds."
                    ),
                    "",
                    "| λ \\ threshold | "
                    + " | ".join(_fmt(value) for value in thresholds)
                    + " |",
                    "|---:|" + "---:|" * len(thresholds),
                ]
            )
            by_weight: dict[float, dict[float, dict[str, object]]] = defaultdict(dict)
            for cell in cells:
                by_weight[float(cell["comparative_weight"])][
                    float(cell["tie_threshold"])
                ] = cell
            for weight in metrics["sensitivity_grid"]["comparative_weights"]:
                lines.append(
                    f"| {_fmt(weight)} | "
                    + " | ".join(
                        _fmt(
                            by_weight[float(weight)][float(threshold)][
                                "conditional_accuracy"
                            ]
                        )
                        for threshold in thresholds
                    )
                    + " |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--failures", type=Path, nargs="*", default=[])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparative-weight", type=float, default=0.25)
    parser.add_argument("--tie-threshold", type=float, default=0.05)
    args = parser.parse_args()
    metrics = analyze(
        _load_csv(args.scores),
        _load_jsonl(args.failures),
        _load_labels(args.labels),
        comparative_weight=args.comparative_weight,
        tie_threshold=args.tie_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = args.output.with_suffix(".md")
    markdown.write_text(_render_markdown(metrics), encoding="utf-8")
    print(f"wrote {args.output} and {markdown}")


if __name__ == "__main__":
    main()
