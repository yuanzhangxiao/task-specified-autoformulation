"""Select a paired-question-consensus call budget from frozen trials.

The analysis makes no LLM requests. It rotates the five frozen seed identifiers
so each seed serves as a possible production start, retries an incomplete paired
orientation trial only at a distinct seed, and never scores unlabeled tradeoff
winners as accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    HybridCalibrationLabels,
)

if __package__:
    from scripts.analyze_hybrid_symmetric_aggregation import (
        RULE_QUESTION_CONSENSUS,
    )
else:
    from analyze_hybrid_symmetric_aggregation import RULE_QUESTION_CONSENSUS


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


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


def _cyclic(values: tuple[int, ...], start: int) -> tuple[int, ...]:
    return values[start:] + values[:start]


def _cluster_bootstrap_ci(
    values_by_pair: dict[str, list[float]],
    *,
    samples: int,
) -> list[float] | None:
    """Bootstrap pair means so repeated seeds are not independent items."""
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


def _checks(
    metrics: dict[str, float | None],
    gates: dict[str, float],
) -> dict[str, dict[str, float | bool | str | None]]:
    output = {}
    for name, threshold in gates.items():
        observed = metrics[name]
        maximum = name.startswith("maximum_")
        passed = observed is not None and (
            observed <= threshold if maximum else observed >= threshold
        )
        output[name] = {
            "observed": observed,
            "threshold": threshold,
            "comparison": "maximum" if maximum else "minimum",
            "passed": passed,
        }
    return output


def _configuration(
    *,
    target_complete: int,
    attempt_limit: int,
    seed_ids: tuple[int, ...],
    pair_ids: tuple[str, ...],
    trials: dict[tuple[str, int], dict[str, object]],
    labels: dict[str, HybridCalibrationLabels],
    tie_threshold: float,
    stages_per_attempt: int,
    judgments_per_attempt: int,
    bootstrap_samples: int,
    gates: dict[str, float],
) -> dict[str, object]:
    accuracy_by_pair: dict[str, list[float]] = defaultdict(list)
    end_to_end_by_pair: dict[str, list[float]] = defaultdict(list)
    decisions_by_pair: dict[str, list[float]] = defaultdict(list)
    preferences_by_pair: dict[str, list[str]] = defaultdict(list)
    pair_aggregate_accuracy: list[bool] = []
    decision_coverage: list[bool] = []
    labeled_coverage: list[bool] = []
    equivalence_accuracy: list[bool] = []
    dominance_accuracy: list[bool] = []
    attempts_used: list[float] = []
    replacement_activated: list[bool] = []
    tradeoff_preferences: Counter[str] = Counter()

    for start_index in range(len(seed_ids)):
        sequence = _cyclic(seed_ids, start_index)[:attempt_limit]
        for pair_id in pair_ids:
            decisions = []
            attempts = 0
            for repetition in sequence:
                attempts += 1
                trial = trials.get((pair_id, repetition))
                if trial is None:
                    continue
                raw = trial["rules"][RULE_QUESTION_CONSENSUS]["decision"]
                if raw is None:
                    continue
                decisions.append(float(raw))
                if len(decisions) == target_complete:
                    break
            attempts_used.append(float(attempts))
            replacement_activated.append(attempts > target_complete)
            available = len(decisions) == target_complete
            decision_coverage.append(available)
            label = labels[pair_id].overall_preference
            if label is not ExpectedPairPreference.UNLABELED:
                labeled_coverage.append(available)
            if not available:
                if label is not ExpectedPairPreference.UNLABELED:
                    end_to_end_by_pair[pair_id].append(0.0)
                continue
            decision = mean(decisions)
            preference = _direction(decision, tie_threshold)
            decisions_by_pair[pair_id].append(decision)
            preferences_by_pair[pair_id].append(preference)
            if label is ExpectedPairPreference.UNLABELED:
                tradeoff_preferences[preference] += 1
                continue
            correct = preference == label.value
            accuracy_by_pair[pair_id].append(float(correct))
            end_to_end_by_pair[pair_id].append(float(correct))
            if label is ExpectedPairPreference.TIE:
                equivalence_accuracy.append(correct)
            else:
                dominance_accuracy.append(correct)

    for pair_id in pair_ids:
        label = labels[pair_id].overall_preference
        if label in {
            ExpectedPairPreference.UNLABELED,
            ExpectedPairPreference.TIE,
        }:
            continue
        decisions = decisions_by_pair[pair_id]
        if decisions:
            pair_aggregate_accuracy.append(
                _direction(mean(decisions), tie_threshold) == label.value
            )

    labeled_accuracy = [
        bool(value)
        for values in accuracy_by_pair.values()
        for value in values
    ]
    end_to_end_accuracy = [
        bool(value)
        for values in end_to_end_by_pair.values()
        for value in values
    ]
    repeat_sds = [
        pstdev(values) for values in decisions_by_pair.values() if values
    ]
    modal_consistency = [
        max(Counter(values).values()) / len(values)
        for values in preferences_by_pair.values()
        if values
    ]
    gate_metrics = {
        "minimum_decision_coverage": _rate(decision_coverage),
        "minimum_labeled_decision_coverage": _rate(labeled_coverage),
        "minimum_labeled_accuracy": _rate(labeled_accuracy),
        "minimum_equivalence_tie_accuracy": _rate(equivalence_accuracy),
        "minimum_known_dominance_accuracy": _rate(dominance_accuracy),
        "minimum_known_dominance_pair_accuracy": _rate(
            pair_aggregate_accuracy
        ),
        "minimum_modal_preference_consistency": _mean(modal_consistency),
        "maximum_mean_repeat_decision_sd": _mean(repeat_sds),
    }
    checks = _checks(gate_metrics, gates)
    expected_attempts = _mean(attempts_used)
    return {
        "configuration": (
            f"target_{target_complete}_complete_paired_seed"
            f"{'s' if target_complete != 1 else ''}_within_{attempt_limit}"
        ),
        "target_complete_paired_seeds": target_complete,
        "maximum_distinct_seed_attempts": attempt_limit,
        "expected_paired_seed_attempts": expected_attempts,
        "maximum_judge_operations_per_pair": (
            judgments_per_attempt * attempt_limit
        ),
        "expected_judge_operations_per_pair": (
            None
            if expected_attempts is None
            else judgments_per_attempt * expected_attempts
        ),
        "maximum_logical_llm_stages_per_pair": stages_per_attempt * attempt_limit,
        "expected_logical_llm_stages_per_pair": (
            None
            if expected_attempts is None
            else stages_per_attempt * expected_attempts
        ),
        "response_replacement_activation_rate": _rate(replacement_activated),
        "decision_coverage": gate_metrics["minimum_decision_coverage"],
        "labeled_decision_coverage": gate_metrics[
            "minimum_labeled_decision_coverage"
        ],
        "labeled_accuracy_conditional": gate_metrics["minimum_labeled_accuracy"],
        "labeled_accuracy_ci95": _cluster_bootstrap_ci(
            accuracy_by_pair,
            samples=bootstrap_samples,
        ),
        "strict_end_to_end_labeled_accuracy": _rate(end_to_end_accuracy),
        "strict_end_to_end_labeled_accuracy_ci95": _cluster_bootstrap_ci(
            end_to_end_by_pair,
            samples=bootstrap_samples,
        ),
        "equivalence_tie_accuracy": gate_metrics[
            "minimum_equivalence_tie_accuracy"
        ],
        "known_dominance_accuracy": gate_metrics[
            "minimum_known_dominance_accuracy"
        ],
        "known_dominance_pair_accuracy": gate_metrics[
            "minimum_known_dominance_pair_accuracy"
        ],
        "mean_pair_modal_preference_consistency": gate_metrics[
            "minimum_modal_preference_consistency"
        ],
        "mean_repeat_decision_sd": gate_metrics[
            "maximum_mean_repeat_decision_sd"
        ],
        "unlabeled_tradeoff_preference_counts": dict(tradeoff_preferences),
        "checks": checks,
        "passes_all_gates": all(bool(item["passed"]) for item in checks.values()),
    }


def analyze_operating_points(
    symmetric: dict[str, object],
    labels: dict[str, HybridCalibrationLabels],
    protocol: dict[str, object],
    *,
    bootstrap_samples: int = 2000,
) -> dict[str, object]:
    """Evaluate and select the cheapest passing frozen-seed configuration."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if protocol["source_primary_rule"] != RULE_QUESTION_CONSENSUS:
        raise ValueError("operating-point source rule is not question consensus")
    if protocol["new_llm_calls"] != 0:
        raise ValueError("offline operating-point analysis cannot request LLM calls")
    seed_ids = tuple(int(value) for value in protocol["available_seed_ids"])
    if not seed_ids or len(seed_ids) != len(set(seed_ids)):
        raise ValueError("available seed identifiers must be nonempty and unique")
    grid = protocol["candidate_grid"]
    targets = tuple(int(value) for value in grid["target_complete_paired_seeds"])
    attempt_limits = tuple(
        int(value) for value in grid["maximum_distinct_seed_attempts"]
    )
    if any(value < 1 or value > len(seed_ids) for value in targets):
        raise ValueError("target complete-seed counts exceed the frozen seed set")
    if any(value < 1 or value > len(seed_ids) for value in attempt_limits):
        raise ValueError("maximum seed attempts exceed the frozen seed set")
    scoring = protocol.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("operating-point protocol must freeze scoring")
    tie_threshold = float(scoring["tie_threshold"])
    cost = protocol["cost_accounting"]
    judgments = int(cost["judge_operations_per_paired_seed_attempt"])
    stages = judgments * int(cost["logical_llm_stages_per_judge_operation"])
    gates = {
        str(key): float(value)
        for key, value in protocol["selection_gate"].items()
    }

    trials: dict[tuple[str, int], dict[str, object]] = {}
    for trial in symmetric["trials"]:
        pair_id = str(trial["pair_id"])
        repetition = int(trial["repetition"])
        key = (pair_id, repetition)
        if key in trials:
            raise ValueError(f"duplicate symmetric trial: {key}")
        if pair_id not in labels:
            raise ValueError(f"symmetric trial has no labels: {pair_id}")
        if repetition not in seed_ids:
            raise ValueError(f"symmetric trial uses unconfigured seed: {repetition}")
        trials[key] = trial
    pair_ids = tuple(sorted(labels))
    if not pair_ids:
        raise ValueError("no labeled pair identities were supplied")

    configurations = [
        _configuration(
            target_complete=target,
            attempt_limit=attempt_limit,
            seed_ids=seed_ids,
            pair_ids=pair_ids,
            trials=trials,
            labels=labels,
            tie_threshold=tie_threshold,
            stages_per_attempt=stages,
            judgments_per_attempt=judgments,
            bootstrap_samples=bootstrap_samples,
            gates=gates,
        )
        for target in targets
        for attempt_limit in attempt_limits
        if attempt_limit >= target
    ]
    passing = [item for item in configurations if item["passes_all_gates"]]
    selected = min(
        passing,
        key=lambda item: (
            float(item["expected_logical_llm_stages_per_pair"]),
            int(item["maximum_logical_llm_stages_per_pair"]),
            int(item["target_complete_paired_seeds"]),
            int(item["maximum_distinct_seed_attempts"]),
        ),
        default=None,
    )
    return {
        "schema_version": "hybrid-judge-consensus-operating-point-result-1",
        "status": "posthoc_development_analysis",
        "new_llm_calls": 0,
        "source_symmetric_schema_version": symmetric["schema_version"],
        "source_complete_paired_trials": len(trials),
        "expected_paired_trials": len(pair_ids) * len(seed_ids),
        "configuration_count": len(configurations),
        "passing_configuration_count": len(passing),
        "selected_operating_point": selected,
        "configurations": configurations,
        "interpretation_boundary": protocol["interpretation_boundary"],
    }


def _display(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def _summary(result: dict[str, object]) -> str:
    lines = [
        "# Paired-question-consensus operating points",
        "",
        "Frozen fresh-structure calls are reused without new LLM requests. "
        "Unlabeled tradeoff winners are excluded from accuracy and selection.",
        "",
        "| Selected | Complete paired seeds | Maximum seed attempts | "
        "Expected judge operations | Maximum judge operations | Coverage | "
        "Labeled accuracy | Strict end-to-end | Equivalence tie | Dominance | "
        "Pair dominance | Repeat SD | Modal consistency |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selected = result["selected_operating_point"]
    selected_name = None if selected is None else selected["configuration"]
    for item in result["configurations"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    "yes" if item["configuration"] == selected_name else "",
                    str(item["target_complete_paired_seeds"]),
                    str(item["maximum_distinct_seed_attempts"]),
                    _display(item["expected_judge_operations_per_pair"]),
                    str(item["maximum_judge_operations_per_pair"]),
                    _display(item["decision_coverage"]),
                    _display(item["labeled_accuracy_conditional"]),
                    _display(item["strict_end_to_end_labeled_accuracy"]),
                    _display(item["equivalence_tie_accuracy"]),
                    _display(item["known_dominance_accuracy"]),
                    _display(item["known_dominance_pair_accuracy"]),
                    _display(item["mean_repeat_decision_sd"]),
                    _display(item["mean_pair_modal_preference_consistency"]),
                )
            )
            + " |"
        )
    lines.extend(("", "## Selected production candidate", ""))
    if selected is None:
        lines.append("No candidate configuration passed every development gate.")
    else:
        lines.extend(
            (
                f"`{selected_name}` passed every gate.",
                "",
                "This remains a search-integration development choice, not an "
                "independent scientific validation result.",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symmetric-analysis", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    symmetric_bytes = args.symmetric_analysis.read_bytes()
    symmetric = json.loads(symmetric_bytes)
    label_bytes = args.labels.read_bytes()
    labels = {
        item.pair_id: item
        for item in (
            HybridCalibrationLabels.model_validate_json(line)
            for line in label_bytes.decode("utf-8").splitlines()
            if line.strip()
        )
    }
    protocol_bytes = args.protocol_config.read_bytes()
    protocol = json.loads(protocol_bytes)
    result = analyze_operating_points(
        symmetric,
        labels,
        protocol,
        bootstrap_samples=args.bootstrap_samples,
    )
    result["protocol_config_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    result["source_symmetric_analysis_sha256"] = hashlib.sha256(
        symmetric_bytes
    ).hexdigest()
    result["source_labels_sha256"] = hashlib.sha256(label_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = args.output.with_suffix(".md")
    summary.write_text(_summary(result), encoding="utf-8")
    print(summary.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
