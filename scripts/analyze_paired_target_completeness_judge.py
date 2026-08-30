"""Analyze frozen V8 paired target-only outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from autoformalism.rebuttal.adversarial import AdversarialPair


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _passed_minimum(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _target_verdicts(result: dict[str, Any], candidate: str) -> dict[str, str]:
    assessments = result["target_assessments"]
    return {
        str(item["target_id"]): str(item[candidate]["verdict"])
        for item in assessments
    }


def _consensus_verdicts(result: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["target_id"]): str(item["verdict"])
        for item in result["target_assessments"]
    }


def evaluate_paired_target_completeness(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    pairs: tuple[AdversarialPair, ...],
    *,
    repetitions: int,
    evaluated_target_id: str,
    seed_attempt_limit: int,
    gates: dict[str, float],
) -> dict[str, object]:
    """Evaluate paired, normalized, fail-dominant target judgments."""
    expected_keys = {
        (pair.pair_id, repetition)
        for pair in pairs
        for repetition in range(repetitions)
    }
    success_by_key = {
        (str(row["pair_id"]), int(row["repetition"])): row for row in rows
    }
    failure_by_key = {
        (str(row["pair_id"]), int(row["repetition"])): row
        for row in failures
    }
    if success_by_key.keys() & failure_by_key.keys():
        raise ValueError("paired trials occur in both scores and failures")
    observed = success_by_key.keys() | failure_by_key.keys()
    if observed - expected_keys:
        raise ValueError("outcome contains an unexpected paired-trial key")

    candidate_correct = 0
    candidate_total = 0
    complete_correct = 0
    target_correct = 0
    target_total = 0
    evaluated_correct = 0
    evaluated_total = 0
    valid_passes = 0
    valid_total = 0
    incomplete_failures = 0
    incomplete_total = 0
    orientation_matches = 0
    orientation_total = 0
    attempted_orientation_calls = 0
    successful_orientation_calls = 0
    retried_trials = 0
    first_seed_successes = 0
    verdicts_by_candidate: dict[tuple[str, str], list[str]] = defaultdict(list)
    error_types: Counter[str] = Counter()

    for row in rows:
        selected_attempt = int(row["selected_seed_attempt"])
        first_seed_successes += int(selected_attempt == 1)
        retried_trials += int(selected_attempt > 1)
        attempted_orientation_calls += 2 * selected_attempt
        successful_orientation_calls += 2
        for failed_seed in row["prior_seed_failures"]:
            successful_orientation_calls += int(
                failed_seed["discarded_successful_orientation_count"]
            )
            for error in failed_seed["orientation_errors"]:
                error_types[str(error["error_type"])] += 1

        forward = row["forward"]["result"]
        reverse = row["reverse"]["result"]
        forward_baseline = _target_verdicts(forward, "candidate_a")
        forward_mutated = _target_verdicts(forward, "candidate_b")
        reverse_baseline = _target_verdicts(reverse, "candidate_b")
        reverse_mutated = _target_verdicts(reverse, "candidate_a")
        requested = {str(item) for item in row["requested_target_ids"]}
        for verdicts in (
            forward_baseline,
            forward_mutated,
            reverse_baseline,
            reverse_mutated,
        ):
            if verdicts.keys() != requested:
                raise ValueError("stored orientation target units differ")
        for first, second in (
            (forward_baseline, reverse_baseline),
            (forward_mutated, reverse_mutated),
        ):
            for target_id in requested:
                orientation_matches += int(first[target_id] == second[target_id])
                orientation_total += 1

        consensus = row["consensus"]
        complete_correct += int(
            consensus["baseline_overall_verdict"] == "pass"
            and consensus["mutated_overall_verdict"] == "fail"
        )
        for role in ("baseline", "mutated"):
            expected_overall = "pass" if role == "baseline" else "fail"
            actual_overall = str(consensus[f"{role}_overall_verdict"])
            candidate_correct += int(actual_overall == expected_overall)
            candidate_total += 1
            verdicts_by_candidate[(str(row["pair_id"]), role)].append(
                actual_overall
            )
            by_target = _consensus_verdicts(consensus[role])
            if by_target.keys() != requested:
                raise ValueError("stored consensus target units differ")
            for target_id, actual in by_target.items():
                expected = (
                    "fail"
                    if role == "mutated" and target_id == evaluated_target_id
                    else "pass"
                )
                target_correct += int(actual == expected)
                target_total += 1
                if target_id == evaluated_target_id:
                    evaluated_correct += int(actual == expected)
                    evaluated_total += 1
                    if role == "baseline":
                        valid_passes += int(actual == "pass")
                        valid_total += 1
                    else:
                        incomplete_failures += int(actual == "fail")
                        incomplete_total += 1

    for row in failures:
        retried_trials += 1
        attempted_orientation_calls += 2 * seed_attempt_limit
        for failed_seed in row["seed_failures"]:
            successful_orientation_calls += int(
                failed_seed["discarded_successful_orientation_count"]
            )
            for error in failed_seed["orientation_errors"]:
                error_types[str(error["error_type"])] += 1

    pair_correct = 0
    pair_total = 0
    repeat_consistencies: list[float] = []
    for pair in pairs:
        aggregated: dict[str, str | None] = {}
        for role in ("baseline", "mutated"):
            verdicts = verdicts_by_candidate[(pair.pair_id, role)]
            if not verdicts:
                aggregated[role] = None
                continue
            counts = Counter(verdicts)
            top = max(counts.values())
            modes = sorted(value for value, count in counts.items() if count == top)
            aggregated[role] = modes[0] if len(modes) == 1 else None
            repeat_consistencies.append(top / len(verdicts))
        if None in aggregated.values():
            continue
        pair_total += 1
        pair_correct += int(
            aggregated == {"baseline": "pass", "mutated": "fail"}
        )

    requested_trials = len(expected_keys)
    metrics = {
        "requested_paired_trial_count": requested_trials,
        "successful_paired_trial_count": len(rows),
        "terminal_failed_trial_count": len(failures),
        "paired_trial_coverage": _ratio(len(rows), requested_trials),
        "first_seed_paired_success": _ratio(first_seed_successes, requested_trials),
        "retry_activation_rate": _ratio(retried_trials, requested_trials),
        "attempted_orientation_call_count": attempted_orientation_calls,
        "successful_orientation_call_count": successful_orientation_calls,
        "orientation_response_success": _ratio(
            successful_orientation_calls, attempted_orientation_calls
        ),
        "candidate_verdict_accuracy": _ratio(candidate_correct, candidate_total),
        "complete_trial_accuracy": _ratio(complete_correct, len(rows)),
        "pair_aggregate_accuracy": _ratio(pair_correct, pair_total),
        "all_target_unit_accuracy": _ratio(target_correct, target_total),
        "evaluated_target_accuracy": _ratio(evaluated_correct, evaluated_total),
        "valid_target_pass_rate": _ratio(valid_passes, valid_total),
        "incomplete_target_fail_rate": _ratio(
            incomplete_failures, incomplete_total
        ),
        "orientation_verdict_consistency": _ratio(
            orientation_matches, orientation_total
        ),
        "mean_repeat_modal_consistency": (
            mean(repeat_consistencies) if repeat_consistencies else None
        ),
        "failures_by_error_type": dict(sorted(error_types.items())),
    }
    gate_metric = {
        "minimum_paired_trial_coverage": "paired_trial_coverage",
        "minimum_candidate_verdict_accuracy": "candidate_verdict_accuracy",
        "minimum_complete_trial_accuracy": "complete_trial_accuracy",
        "minimum_pair_aggregate_accuracy": "pair_aggregate_accuracy",
        "minimum_all_target_unit_accuracy": "all_target_unit_accuracy",
        "minimum_evaluated_target_accuracy": "evaluated_target_accuracy",
        "minimum_valid_target_pass_rate": "valid_target_pass_rate",
        "minimum_incomplete_target_fail_rate": "incomplete_target_fail_rate",
        "minimum_orientation_verdict_consistency": (
            "orientation_verdict_consistency"
        ),
        "minimum_repeat_modal_consistency": "mean_repeat_modal_consistency",
    }
    checks: dict[str, dict[str, object]] = {}
    for gate, metric_name in gate_metric.items():
        observed_value = metrics[metric_name]
        assert observed_value is None or isinstance(observed_value, float)
        checks[gate] = {
            "observed": observed_value,
            "threshold": gates[gate],
            "passed": _passed_minimum(observed_value, gates[gate]),
        }
    return {
        "schema_version": "paired-target-completeness-analysis-1",
        "passed": all(bool(item["passed"]) for item in checks.values()),
        "evaluated_target_id": evaluated_target_id,
        "metrics": metrics,
        "checks": checks,
        "interpretation": (
            "Each seed is a two-orientation transaction. A malformed orientation "
            "discards the whole seed; successful orientations are normalized to "
            "stable candidate identities and combined fail-dominantly per target."
        ),
    }


def _format(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def summary_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Paired target-only completeness validation",
        "",
        f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**.",
        "",
        "Both candidates are visible, but each target is assessed absolutely. "
        "There are no atomic occurrence, repeat, comparative, preference, or "
        "numeric-score questions.",
        "",
        "| Predeclared gate | Observed | Threshold | Result |",
        "|---|---:|---:|:---:|",
    ]
    checks = result["checks"]
    assert isinstance(checks, dict)
    for name, raw in checks.items():
        assert isinstance(raw, dict)
        lines.append(
            f"| {name} | {_format(raw['observed'])} | "
            f"≥ {_format(raw['threshold'])} | "
            f"{'pass' if raw['passed'] else 'fail'} |"
        )
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    lines.extend(
        [
            "",
            "## Transport diagnostics",
            "",
            "- First-seed paired success: "
            f"{_format(metrics['first_seed_paired_success'])}",
            f"- Retry activation: {_format(metrics['retry_activation_rate'])}",
            "- Orientation response success: "
            f"{_format(metrics['orientation_response_success'])}",
            "- Failures by error type: `"
            f"{json.dumps(metrics['failures_by_error_type'], sort_keys=True)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.scores.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures = [
        json.loads(line)
        for line in args.failures.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = evaluate_paired_target_completeness(
        rows,
        failures,
        pairs,
        repetitions=int(config["protocol"]["repetitions"]),
        evaluated_target_id=str(config["pair_construction"]["target_channel"]),
        seed_attempt_limit=int(config["protocol"]["max_paired_seed_attempts"]),
        gates=config["validation_gate"],
    )
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(summary_markdown(result), encoding="utf-8")
    print(summary_markdown(result), end="")


if __name__ == "__main__":
    main()
