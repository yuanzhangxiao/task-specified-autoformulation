"""Analyze candidate-specific target-completeness calibration outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from autoformalism.rebuttal.adversarial import AdversarialPair


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _passed_minimum(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def evaluate_target_completeness(
    rows: list[dict[str, str]],
    failures: list[dict[str, object]],
    pairs: tuple[AdversarialPair, ...],
    *,
    repetitions: int,
    evaluated_target_id: str,
    gates: dict[str, float],
) -> dict[str, object]:
    """Evaluate frozen absolute calls without pairwise or orientation logic."""
    expected_call_keys = {
        (pair.pair_id, repetition, candidate_role)
        for pair in pairs
        for repetition in range(repetitions)
        for candidate_role in ("baseline", "mutated")
    }
    success_by_key = {
        (row["pair_id"], int(row["repetition"]), row["candidate_role"]): row
        for row in rows
    }
    failure_keys = {
        (
            str(row["pair_id"]),
            int(row["repetition"]),
            str(row["candidate_role"]),
        )
        for row in failures
    }
    if success_by_key.keys() & failure_keys:
        raise ValueError("logical calls occur in both scores and failures")
    observed = success_by_key.keys() | failure_keys
    if observed - expected_call_keys:
        raise ValueError("outcome contains an unexpected logical-call key")

    complete_trials = 0
    correct_complete_trials = 0
    call_correct = 0
    target_correct = 0
    target_total = 0
    evaluated_target_correct = 0
    evaluated_target_total = 0
    valid_target_passes = 0
    valid_target_total = 0
    incomplete_target_failures = 0
    incomplete_target_total = 0
    verdicts_by_candidate: dict[tuple[str, str], list[str]] = defaultdict(list)

    for key, row in success_by_key.items():
        _pair_id, _repetition, candidate_role = key
        expected_overall = "pass" if candidate_role == "baseline" else "fail"
        actual_overall = row["overall_verdict"]
        call_correct += int(actual_overall == expected_overall)
        verdicts_by_candidate[(row["pair_id"], candidate_role)].append(
            actual_overall
        )
        assessments = json.loads(row["target_assessments"])
        requested = set(json.loads(row["requested_target_ids"]))
        by_target = {item["target_id"]: item["verdict"] for item in assessments}
        if by_target.keys() != requested:
            raise ValueError("stored target assessment units differ from request")
        for target_id, actual in by_target.items():
            expected = (
                "fail"
                if candidate_role == "mutated"
                and target_id == evaluated_target_id
                else "pass"
            )
            target_correct += int(actual == expected)
            target_total += 1
            if target_id == evaluated_target_id:
                evaluated_target_correct += int(actual == expected)
                evaluated_target_total += 1
                if candidate_role == "baseline":
                    valid_target_passes += int(actual == "pass")
                    valid_target_total += 1
                else:
                    incomplete_target_failures += int(actual == "fail")
                    incomplete_target_total += 1

    for pair in pairs:
        for repetition in range(repetitions):
            baseline = success_by_key.get((pair.pair_id, repetition, "baseline"))
            mutated = success_by_key.get((pair.pair_id, repetition, "mutated"))
            if baseline is None or mutated is None:
                continue
            complete_trials += 1
            correct_complete_trials += int(
                baseline["overall_verdict"] == "pass"
                and mutated["overall_verdict"] == "fail"
            )

    pair_aggregate_correct = 0
    pair_aggregate_total = 0
    repeat_consistencies = []
    for pair in pairs:
        aggregate: dict[str, str | None] = {}
        for candidate_role in ("baseline", "mutated"):
            verdicts = verdicts_by_candidate[(pair.pair_id, candidate_role)]
            if not verdicts:
                aggregate[candidate_role] = None
                continue
            counts = Counter(verdicts)
            most_common = counts.most_common()
            top_count = most_common[0][1]
            modes = sorted(
                verdict for verdict, count in most_common if count == top_count
            )
            aggregate[candidate_role] = modes[0] if len(modes) == 1 else None
            repeat_consistencies.append(top_count / len(verdicts))
        if None in aggregate.values():
            continue
        pair_aggregate_total += 1
        pair_aggregate_correct += int(
            aggregate == {"baseline": "pass", "mutated": "fail"}
        )

    metrics = {
        "requested_call_count": len(expected_call_keys),
        "successful_call_count": len(success_by_key),
        "failed_call_count": len(failure_keys),
        "response_success": _ratio(len(success_by_key), len(expected_call_keys)),
        "joint_candidate_coverage": _ratio(
            complete_trials, len(pairs) * repetitions
        ),
        "candidate_verdict_accuracy": _ratio(call_correct, len(success_by_key)),
        "complete_trial_accuracy": _ratio(
            correct_complete_trials, complete_trials
        ),
        "pair_aggregate_accuracy": _ratio(
            pair_aggregate_correct, pair_aggregate_total
        ),
        "all_target_unit_accuracy": _ratio(target_correct, target_total),
        "evaluated_target_accuracy": _ratio(
            evaluated_target_correct, evaluated_target_total
        ),
        "valid_target_pass_rate": _ratio(valid_target_passes, valid_target_total),
        "incomplete_target_fail_rate": _ratio(
            incomplete_target_failures, incomplete_target_total
        ),
        "mean_repeat_modal_consistency": (
            mean(repeat_consistencies) if repeat_consistencies else None
        ),
        "failure_types": dict(
            sorted(Counter(str(row["error_type"]) for row in failures).items())
        ),
    }
    checks = {
        "minimum_response_success": {
            "observed": metrics["response_success"],
            "threshold": gates["minimum_response_success"],
            "passed": _passed_minimum(
                metrics["response_success"], gates["minimum_response_success"]
            ),
        },
        "minimum_joint_candidate_coverage": {
            "observed": metrics["joint_candidate_coverage"],
            "threshold": gates["minimum_joint_candidate_coverage"],
            "passed": _passed_minimum(
                metrics["joint_candidate_coverage"],
                gates["minimum_joint_candidate_coverage"],
            ),
        },
        "minimum_candidate_verdict_accuracy": {
            "observed": metrics["candidate_verdict_accuracy"],
            "threshold": gates["minimum_candidate_verdict_accuracy"],
            "passed": _passed_minimum(
                metrics["candidate_verdict_accuracy"],
                gates["minimum_candidate_verdict_accuracy"],
            ),
        },
        "minimum_complete_trial_accuracy": {
            "observed": metrics["complete_trial_accuracy"],
            "threshold": gates["minimum_complete_trial_accuracy"],
            "passed": _passed_minimum(
                metrics["complete_trial_accuracy"],
                gates["minimum_complete_trial_accuracy"],
            ),
        },
        "minimum_pair_aggregate_accuracy": {
            "observed": metrics["pair_aggregate_accuracy"],
            "threshold": gates["minimum_pair_aggregate_accuracy"],
            "passed": _passed_minimum(
                metrics["pair_aggregate_accuracy"],
                gates["minimum_pair_aggregate_accuracy"],
            ),
        },
        "minimum_all_target_unit_accuracy": {
            "observed": metrics["all_target_unit_accuracy"],
            "threshold": gates["minimum_all_target_unit_accuracy"],
            "passed": _passed_minimum(
                metrics["all_target_unit_accuracy"],
                gates["minimum_all_target_unit_accuracy"],
            ),
        },
        "minimum_evaluated_target_accuracy": {
            "observed": metrics["evaluated_target_accuracy"],
            "threshold": gates["minimum_evaluated_target_accuracy"],
            "passed": _passed_minimum(
                metrics["evaluated_target_accuracy"],
                gates["minimum_evaluated_target_accuracy"],
            ),
        },
        "minimum_valid_target_pass_rate": {
            "observed": metrics["valid_target_pass_rate"],
            "threshold": gates["minimum_valid_target_pass_rate"],
            "passed": _passed_minimum(
                metrics["valid_target_pass_rate"],
                gates["minimum_valid_target_pass_rate"],
            ),
        },
        "minimum_incomplete_target_fail_rate": {
            "observed": metrics["incomplete_target_fail_rate"],
            "threshold": gates["minimum_incomplete_target_fail_rate"],
            "passed": _passed_minimum(
                metrics["incomplete_target_fail_rate"],
                gates["minimum_incomplete_target_fail_rate"],
            ),
        },
        "minimum_repeat_modal_consistency": {
            "observed": metrics["mean_repeat_modal_consistency"],
            "threshold": gates["minimum_repeat_modal_consistency"],
            "passed": _passed_minimum(
                metrics["mean_repeat_modal_consistency"],
                gates["minimum_repeat_modal_consistency"],
            ),
        },
    }
    return {
        "schema_version": "target-completeness-judge-analysis-1",
        "passed": all(bool(item["passed"]) for item in checks.values()),
        "evaluated_target_id": evaluated_target_id,
        "metrics": metrics,
        "checks": checks,
        "interpretation": (
            "Every LLM call assesses one candidate independently. Pairwise "
            "orientation and comparative preference are not part of this protocol."
        ),
    }


def _format(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def summary_markdown(result: dict[str, object]) -> str:
    """Render the predeclared validation checks."""
    lines = [
        "# Candidate-specific target-completeness validation",
        "",
        f"Overall result: **{'PASS' if result['passed'] else 'FAIL'}**.",
        "",
        "Each call evaluates one candidate; there are no A/B orientations, "
        "atomic signed-occurrence questions, comparative questions, or scores.",
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

    with args.scores.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
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
    result = evaluate_target_completeness(
        rows,
        failures,
        pairs,
        repetitions=int(config["protocol"]["repetitions"]),
        evaluated_target_id=str(config["pair_construction"]["target_channel"]),
        gates=config["validation_gate"],
    )
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(summary_markdown(result), encoding="utf-8")
    print(summary_markdown(result), end="")


if __name__ == "__main__":
    main()
