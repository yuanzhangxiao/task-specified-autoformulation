"""Decompose frozen hybrid-judge decisions without making new LLM calls."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from autoformalism.judging import HybridScoringConfig, score_hybrid_pair
from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    HybridCalibrationLabels,
)
from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    HybridJudgeResult,
    PairedAbsoluteAssessment,
    RequirementRegistry,
)

_ATOMIC_SENSITIVE_CRITERIA = frozenset(
    {
        AbsoluteCriterion.SOURCE_ROLES_CONSISTENT,
        AbsoluteCriterion.SINK_ROLES_CONSISTENT,
        AbsoluteCriterion.SEMANTIC_FLUXES_NOT_DUPLICATED,
    }
)


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _direction(value: float | None, threshold: float) -> str:
    if value is None:
        return ExpectedPairPreference.INDETERMINATE.value
    if value > threshold:
        return ExpectedPairPreference.BASELINE.value
    if value < -threshold:
        return ExpectedPairPreference.MUTATED.value
    return ExpectedPairPreference.TIE.value


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


def _hybrid_result(row: dict[str, str]) -> HybridJudgeResult:
    return HybridJudgeResult.model_validate(
        {
            "schema_version": "hybrid-1",
            "absolute_assessments": json.loads(row["absolute_assessments"]),
            "comparative_assessments": json.loads(
                row["comparative_assessments"]
            ),
        }
    )


def _deterministic_assessments(
    row: dict[str, str],
) -> tuple[PairedAbsoluteAssessment, ...]:
    return tuple(
        PairedAbsoluteAssessment.model_validate(item)
        for item in json.loads(row["deterministic_assessments"])
    )


def _without_atomic_sensitive_assessments(
    result: HybridJudgeResult,
) -> HybridJudgeResult:
    """Return the counterfactual with atomic-sensitive verdicts unresolved."""
    payload = result.model_dump(mode="json")
    for item in payload["absolute_assessments"]:
        if AbsoluteCriterion(item["criterion"]) not in _ATOMIC_SENSITIVE_CRITERIA:
            continue
        unresolved = {
            "verdict": AbsoluteVerdict.INDETERMINATE.value,
            "evidence": "Removed for atomic-sensitive marginal decomposition.",
        }
        item["candidate_a"] = unresolved
        item["candidate_b"] = unresolved
    return HybridJudgeResult.model_validate(payload)


def _baseline_value(
    left: float | None,
    right: float | None,
    baseline_position: str,
) -> tuple[float | None, float | None, float | None]:
    baseline, mutated = (
        (left, right) if baseline_position == "A" else (right, left)
    )
    delta = None if baseline is None or mutated is None else baseline - mutated
    return baseline, mutated, delta


def _normalized_comparative(
    result: HybridJudgeResult,
    baseline_position: str,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for item in result.comparative_assessments:
        verdict = item.verdict.value
        if verdict in {"candidate_a", "candidate_b"}:
            position = "A" if verdict == "candidate_a" else "B"
            verdict = "baseline" if position == baseline_position else "mutated"
        normalized[item.criterion.value] = verdict
    return normalized


def decompose_row(
    row: dict[str, str],
    *,
    labels: dict[str, HybridCalibrationLabels],
    config: HybridScoringConfig,
) -> dict[str, object]:
    """Recompute and decompose one persisted successful comparison."""
    result = _hybrid_result(row)
    deterministic = _deterministic_assessments(row)
    requirements = RequirementRegistry.model_validate(
        json.loads(row["requirements"])
    )
    full = score_hybrid_pair(result, deterministic, requirements, config)
    counterfactual = score_hybrid_pair(
        _without_atomic_sensitive_assessments(result),
        deterministic,
        requirements,
        config,
    )
    baseline_position = row["baseline_position"]
    baseline_score, mutated_score, absolute_delta = _baseline_value(
        full.candidate_a.shaped_score,
        full.candidate_b.shaped_score,
        baseline_position,
    )
    _, _, non_atomic_absolute_delta = _baseline_value(
        counterfactual.candidate_a.shaped_score,
        counterfactual.candidate_b.shaped_score,
        baseline_position,
    )
    atomic_marginal = (
        None
        if absolute_delta is None or non_atomic_absolute_delta is None
        else absolute_delta - non_atomic_absolute_delta
    )
    relative_a = full.relative_preference_for_a
    relative_baseline = (
        None
        if relative_a is None
        else relative_a
        if baseline_position == "A"
        else 1.0 - relative_a
    )
    comparative_delta = (
        None if relative_baseline is None else 2.0 * relative_baseline - 1.0
    )
    comparative_contribution = (
        None
        if comparative_delta is None
        else config.comparative_weight * comparative_delta
    )
    baseline_decision = (
        None
        if full.decision_value is None
        else full.decision_value
        if baseline_position == "A"
        else -full.decision_value
    )
    baseline_hard, mutated_hard = (
        (
            full.candidate_a.hard_requirement_status,
            full.candidate_b.hard_requirement_status,
        )
        if baseline_position == "A"
        else (
            full.candidate_b.hard_requirement_status,
            full.candidate_a.hard_requirement_status,
        )
    )
    hard_override = (
        "baseline"
        if baseline_hard is True and mutated_hard is False
        else "mutated"
        if baseline_hard is False and mutated_hard is True
        else None
    )
    if hard_override is None and baseline_decision is not None:
        expected_decision = (absolute_delta or 0.0) + (
            comparative_contribution or 0.0
        )
        if abs(baseline_decision - expected_decision) > 1e-9:
            raise ValueError(
                "persisted decision does not match reconstructed components: "
                f"pair={row['pair_id']} model={row['judge_model']}"
            )
    groups = []
    for left, right in zip(
        full.candidate_a.groups,
        full.candidate_b.groups,
        strict=True,
    ):
        if left.group_id != right.group_id:
            raise ValueError("candidate group registries differ")
        baseline_group, mutated_group = (
            (left, right) if baseline_position == "A" else (right, left)
        )
        groups.append(
            {
                "group_id": baseline_group.group_id,
                "kind": baseline_group.kind,
                "weight": baseline_group.weight,
                "baseline_complete": baseline_group.complete,
                "mutated_complete": mutated_group.complete,
                "complete_delta": (
                    None
                    if baseline_group.complete is None
                    or mutated_group.complete is None
                    else baseline_group.complete - mutated_group.complete
                ),
                "baseline_partial": baseline_group.partial,
                "mutated_partial": mutated_group.partial,
                "partial_delta": (
                    None
                    if baseline_group.partial is None
                    or mutated_group.partial is None
                    else baseline_group.partial - mutated_group.partial
                ),
            }
        )
    gold = labels[row["pair_id"]]
    expected = gold.overall_preference.value
    predicted = _direction(baseline_decision, config.tie_threshold)
    stored_decision = row.get("baseline_decision_value", "")
    if (
        stored_decision
        and baseline_decision is not None
        and abs(float(stored_decision) - baseline_decision) > 1e-9
    ):
        raise ValueError("persisted baseline decision does not match recomputation")
    return {
        "pair_id": row["pair_id"],
        "mutation_type": row["mutation_type"],
        "judge_model": row["judge_model"],
        "repetition": int(row["repetition"]),
        "order": row["order"],
        "baseline_position": baseline_position,
        "expected_preference": expected,
        "predicted_preference": predicted,
        "correct": predicted == expected,
        "baseline_shaped_score": baseline_score,
        "mutated_shaped_score": mutated_score,
        "absolute_delta_for_baseline": absolute_delta,
        "atomic_sensitive_marginal_for_baseline": atomic_marginal,
        "other_absolute_delta_for_baseline": non_atomic_absolute_delta,
        "baseline_comparative_preference": relative_baseline,
        "comparative_delta_for_baseline": comparative_delta,
        "weighted_comparative_contribution_for_baseline": (
            comparative_contribution
        ),
        "final_decision_for_baseline": baseline_decision,
        "hard_override": hard_override,
        "groups": groups,
        "comparative_assessments": _normalized_comparative(
            result, baseline_position
        ),
    }


def _accuracy(
    rows: list[dict[str, object]],
    key: str,
    *,
    threshold: float,
) -> float | None:
    labeled = [row for row in rows if row["expected_preference"] != "unlabeled"]
    if key == "final":
        return _mean([float(bool(row["correct"])) for row in labeled])
    values = []
    for row in labeled:
        value = row[key]
        predicted = _direction(
            None if value is None else float(value),
            threshold,
        )
        values.append(float(predicted == row["expected_preference"]))
    return _mean(values)


def _aggregate_model(
    rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    labels: dict[str, HybridCalibrationLabels],
    config: HybridScoringConfig,
) -> dict[str, object]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row[key] is not None]

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    comparative: dict[str, list[str]] = defaultdict(list)
    comparative_correct: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        for item in row["groups"]:
            groups[str(item["kind"])].append(item)
        gold = labels[str(row["pair_id"])]
        gold_relative = {item.criterion.value: item for item in gold.comparative_labels}
        for criterion, verdict in row["comparative_assessments"].items():
            comparative[criterion].append(verdict)
            expected = gold_relative.get(criterion)
            if (
                expected is not None
                and expected.preference is not ExpectedPairPreference.UNLABELED
            ):
                comparative_correct[criterion].append(
                    verdict == expected.preference.value
                )
    group_metrics = {}
    for kind, items in sorted(groups.items()):
        complete = [
            float(item["complete_delta"])
            for item in items
            if item["complete_delta"] is not None
        ]
        partial = [
            float(item["partial_delta"])
            for item in items
            if item["partial_delta"] is not None
        ]
        group_metrics[kind] = {
            "instance_count": len(items),
            "mean_complete_delta_for_baseline": _mean(complete),
            "complete_favors_baseline_rate": _mean(
                [float(value > 0.0) for value in complete]
            ),
            "complete_tie_rate": _mean([float(value == 0.0) for value in complete]),
            "mean_partial_delta_for_baseline": _mean(partial),
            "partial_favors_baseline_rate": _mean(
                [float(value > 0.0) for value in partial]
            ),
        }
    comparative_metrics = {
        criterion: {
            "counts": dict(Counter(verdicts)),
            "accuracy": _mean(
                [float(value) for value in comparative_correct.get(criterion, [])]
            ),
        }
        for criterion, verdicts in sorted(comparative.items())
    }
    examples = []
    by_mutation: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_mutation[str(row["mutation_type"])].append(row)
    for _mutation, mutation_rows in sorted(by_mutation.items()):
        center = median(
            float(row["final_decision_for_baseline"])
            for row in mutation_rows
            if row["final_decision_for_baseline"] is not None
        )
        selected = min(
            mutation_rows,
            key=lambda row: (
                abs(float(row["final_decision_for_baseline"]) - center),
                int(row["repetition"]),
                str(row["order"]),
            ),
        )
        examples.append(
            {
                key: selected[key]
                for key in (
                    "pair_id",
                    "mutation_type",
                    "repetition",
                    "order",
                    "baseline_shaped_score",
                    "mutated_shaped_score",
                    "absolute_delta_for_baseline",
                    "atomic_sensitive_marginal_for_baseline",
                    "other_absolute_delta_for_baseline",
                    "baseline_comparative_preference",
                    "weighted_comparative_contribution_for_baseline",
                    "final_decision_for_baseline",
                    "expected_preference",
                    "predicted_preference",
                )
            }
        )
    failure_stages = Counter(
        str(item.get("failure_stage") or "unknown") for item in failures
    )
    attempted = len(rows) + len(failures)
    return {
        "successful_calls": len(rows),
        "failed_calls": len(failures),
        "response_success_rate": len(rows) / attempted if attempted else None,
        "failure_stages": dict(failure_stages),
        "hard_override_count": sum(row["hard_override"] is not None for row in rows),
        "final_accuracy": _accuracy(
            rows,
            "final",
            threshold=config.tie_threshold,
        ),
        "absolute_only_accuracy": _accuracy(
            rows,
            "absolute_delta_for_baseline",
            threshold=config.tie_threshold,
        ),
        "comparative_only_accuracy": _accuracy(
            rows,
            "comparative_delta_for_baseline",
            threshold=config.tie_threshold,
        ),
        "mean_absolute_delta_for_baseline": _mean(
            values("absolute_delta_for_baseline")
        ),
        "mean_atomic_sensitive_marginal_for_baseline": _mean(
            values("atomic_sensitive_marginal_for_baseline")
        ),
        "atomic_sensitive_marginal_favors_baseline_rate": _mean(
            [
                float(value > 0.0)
                for value in values("atomic_sensitive_marginal_for_baseline")
            ]
        ),
        "mean_other_absolute_delta_for_baseline": _mean(
            values("other_absolute_delta_for_baseline")
        ),
        "mean_baseline_comparative_preference": _mean(
            values("baseline_comparative_preference")
        ),
        "mean_weighted_comparative_contribution_for_baseline": _mean(
            values("weighted_comparative_contribution_for_baseline")
        ),
        "mean_final_decision_for_baseline": _mean(
            values("final_decision_for_baseline")
        ),
        "group_metrics": group_metrics,
        "comparative_metrics": comparative_metrics,
        "representative_examples": examples,
    }


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render_markdown(metrics: dict[str, object]) -> str:
    lines = [
        "# Hybrid decision decomposition",
        "",
        (
            "Frozen successful and failed calls are reused without new LLM "
            "requests."
        ),
        (
            "The atomic-sensitive marginal removes source-role, sink-role, and "
            "semantic-duplication assessments, then recomputes the nonlinear "
            "group score."
        ),
        "",
        (
            "| Judge | Success | Final accuracy | Absolute only | Comparative "
            "only | Mean absolute Δ | Mean atomic-sensitive marginal | Mean "
            "other-absolute Δ | Mean weighted comparative | Mean final D |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    models = metrics["models"]
    for model, item in models.items():
        lines.append(
            "| "
            + " | ".join(
                (
                    model,
                    _fmt(item["response_success_rate"]),
                    _fmt(item["final_accuracy"]),
                    _fmt(item["absolute_only_accuracy"]),
                    _fmt(item["comparative_only_accuracy"]),
                    _fmt(item["mean_absolute_delta_for_baseline"]),
                    _fmt(item["mean_atomic_sensitive_marginal_for_baseline"]),
                    _fmt(item["mean_other_absolute_delta_for_baseline"]),
                    _fmt(item["mean_weighted_comparative_contribution_for_baseline"]),
                    _fmt(item["mean_final_decision_for_baseline"]),
                )
            )
            + " |"
        )
    for model, item in models.items():
        lines.extend(
            [
                "",
                f"## {model}",
                "",
                (
                    "Failures by stage: "
                    f"`{json.dumps(item['failure_stages'], sort_keys=True)}`. "
                    f"Hard overrides: `{item['hard_override_count']}`."
                ),
                "",
                "### Group saturation",
                "",
                (
                    "| Group kind | Instances | Mean complete Δ | Complete favors "
                    "baseline | Complete tie | Mean partial Δ | Partial favors "
                    "baseline |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for kind, group in item["group_metrics"].items():
            lines.append(
                "| "
                + " | ".join(
                    (
                        kind,
                        str(group["instance_count"]),
                        _fmt(group["mean_complete_delta_for_baseline"]),
                        _fmt(group["complete_favors_baseline_rate"]),
                        _fmt(group["complete_tie_rate"]),
                        _fmt(group["mean_partial_delta_for_baseline"]),
                        _fmt(group["partial_favors_baseline_rate"]),
                    )
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "### Direct comparative questions",
                "",
                (
                    "| Criterion | Baseline/mutated/tie/indeterminate | "
                    "Certified accuracy |"
                ),
                "|---|---|---:|",
            ]
        )
        for criterion, comparison in item["comparative_metrics"].items():
            counts = json.dumps(comparison["counts"], sort_keys=True)
            lines.append(
                f"| {criterion} | `{counts}` | "
                f"{_fmt(comparison['accuracy'])} |"
            )
        lines.extend(
            [
                "",
                "### Representative successful calls",
                "",
                (
                    "| Mutation | Pair/repetition/order | Baseline score | Mutated "
                    "score | Absolute Δ | Atomic-sensitive marginal | "
                    "Other-absolute Δ | Baseline comparative p | Weighted "
                    "comparative | Final D | Prediction |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for example in item["representative_examples"]:
            identifier = (
                f"{example['pair_id']}/r{example['repetition']}/"
                f"{example['order']}"
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(example["mutation_type"]),
                        identifier,
                        _fmt(example["baseline_shaped_score"]),
                        _fmt(example["mutated_shaped_score"]),
                        _fmt(example["absolute_delta_for_baseline"]),
                        _fmt(example["atomic_sensitive_marginal_for_baseline"]),
                        _fmt(example["other_absolute_delta_for_baseline"]),
                        _fmt(example["baseline_comparative_preference"]),
                        _fmt(example["weighted_comparative_contribution_for_baseline"]),
                        _fmt(example["final_decision_for_baseline"]),
                        str(example["predicted_preference"]),
                    )
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
    parser.add_argument("--partial-tiebreak-weight", type=float, default=0.05)
    parser.add_argument("--comparative-weight", type=float, default=0.25)
    parser.add_argument("--tie-threshold", type=float, default=0.05)
    args = parser.parse_args()
    config = HybridScoringConfig(
        partial_tiebreak_weight=args.partial_tiebreak_weight,
        comparative_weight=args.comparative_weight,
        tie_threshold=args.tie_threshold,
    )
    labels = _load_labels(args.labels)
    raw_rows = _load_csv(args.scores)
    raw_failures = _load_jsonl(args.failures)
    decomposed = [
        decompose_row(row, labels=labels, config=config) for row in raw_rows
    ]
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    failures_by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in decomposed:
        by_model[str(row["judge_model"])].append(row)
    for row in raw_failures:
        failures_by_model[str(row["judge_model"])].append(row)
    model_names = sorted(set(by_model) | set(failures_by_model))
    metrics = {
        "schema_version": "hybrid-decision-decomposition-1",
        "scoring_config": {
            "partial_tiebreak_weight": config.partial_tiebreak_weight,
            "comparative_weight": config.comparative_weight,
            "tie_threshold": config.tie_threshold,
        },
        "atomic_sensitive_criteria": sorted(
            item.value for item in _ATOMIC_SENSITIVE_CRITERIA
        ),
        "models": {
            model: _aggregate_model(
                by_model[model],
                failures_by_model[model],
                labels,
                config,
            )
            for model in model_names
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_render_markdown(metrics), encoding="utf-8")
    print(f"wrote {args.output} and {markdown_path}")


if __name__ == "__main__":
    main()
