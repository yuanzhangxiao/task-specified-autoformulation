#!/usr/bin/env python3
"""Summarize the matched public-only proposer refinement pilot."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.rebuttal.proposer_refinement_pilot import (
    build_refinement_pilot_tasks,
    load_refinement_pilot_plan,
)
from autoformalism.schemas import CandidateModel
from autoformalism.targets import PublicTargetContract, evaluate_public_targets


def collect_report(
    plan_path: Path,
    search_root: Path,
    *,
    target_contract_root: Path,
    mechanism_spec_root: Path,
) -> dict[str, object]:
    """Collect operational and public-development endpoints for every task."""
    plan = load_refinement_pilot_plan(plan_path)
    rows: list[dict[str, object]] = []
    by_trial: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for task in build_refinement_pilot_tasks(plan):
        run = (
            search_root
            / "searches"
            / task.arm_id
            / "runs"
            / f"{task.benchmark_id}_{task.tier}_seed{task.repetition}"
        )
        summary = _object(run / "summary.json")
        rounds = [
            payload
            for path in sorted((run / "checkpoints").glob("round_*.json"))
            if (payload := _object(path)) is not None
        ]
        events = _events(run / "proposer_events.jsonl")
        responses = [
            item
            for item in events
            if item.get("event") == "llm_response"
            and item.get("role") == "proposer"
        ]
        proposer_failures = [
            item
            for item in events
            if item.get("event") == "llm_failure"
            and item.get("role") == "proposer"
        ]
        valid_rounds = [item for item in rounds if item.get("valid") is True]
        candidate_payload = (
            None if summary is None else summary.get("selected_candidate")
        )
        candidate = (
            None
            if not isinstance(candidate_payload, dict)
            else CandidateModel.model_validate(candidate_payload)
        )
        target_pass = None
        mechanism_compliance = None
        mechanism_complete = None
        if candidate is not None:
            target_contract = PublicTargetContract.model_validate_json(
                (
                    target_contract_root
                    / "specs"
                    / f"{task.benchmark_id}.json"
                ).read_text(encoding="utf-8")
            )
            mechanism_spec = MechanismEvaluationSpec.model_validate_json(
                (
                    mechanism_spec_root
                    / "specs"
                    / f"{task.benchmark_id}.json"
                ).read_text(encoding="utf-8")
            )
            target_pass = evaluate_public_targets(candidate, target_contract).passed
            mechanism = evaluate_mechanisms(candidate, mechanism_spec)
            mechanism_compliance = mechanism.mechanism_compliance
            mechanism_complete = mechanism.mechanism_compliance_complete
        fit_attempts = [
            attempt
            for round_payload in rounds
            for attempt in round_payload.get("fit_attempts", [])
            if isinstance(attempt, dict)
        ]
        accepted_refinements = sum(
            _challenger_selected(item) for item in valid_rounds[1:]
        )
        selection_hash = None if summary is None else summary.get("selection_hash")
        selected_round = _selected_round(valid_rounds, selection_hash)
        selected_record = (
            None
            if selected_round is None
            else selected_round.get("record")
        )
        selected_fit = (
            None
            if not isinstance(selected_record, dict)
            else selected_record.get("pruned_fit")
        )
        selected_training_nmse = _nested_number(
            selected_fit,
            "training_metrics",
            "normalized_mse",
        )
        proposer_resource = _llm_resource(responses)
        judge_resource = _llm_resource(
            [
                item
                for item in _events(run / "hybrid_pair_events.jsonl")
                if item.get("event") == "llm_response"
            ]
        )
        runtime = _object(run / "task_runtime.json") or {}
        complexity = _complexity(candidate)
        row = {
            "arm_id": task.arm_id,
            "benchmark_id": task.benchmark_id,
            "tier": task.tier,
            "repetition": task.repetition,
            "source_complete": summary is not None,
            "attempted_round_count": len(rounds),
            "valid_round_count": len(valid_rounds),
            "valid_postinitial_round_count": sum(
                int(item.get("round_index", 0)) > 0 for item in valid_rounds
            ),
            "accepted_postinitial_challenger_count": accepted_refinements,
            "fit_attempt_count": len(fit_attempts),
            "fit_retry_activation_count": sum(
                isinstance(item.get("fit_attempts"), list)
                and len(item["fit_attempts"]) > 1
                for item in rounds
            ),
            "proposer_response_count": len(responses),
            "proposer_failure_event_count": len(proposer_failures),
            "initial_proposer_request_hash": (
                None if not responses else responses[0].get("request_hash")
            ),
            "initial_proposer_cache_hit": (
                None if not responses else responses[0].get("cache_hit")
            ),
            "proposer_logical_calls": proposer_resource["logical_calls"],
            "proposer_provider_attempts": proposer_resource[
                "provider_attempts"
            ],
            "proposer_logical_input_tokens": proposer_resource[
                "logical_input_tokens"
            ],
            "proposer_logical_output_tokens": proposer_resource[
                "logical_output_tokens"
            ],
            "proposer_provider_input_tokens": proposer_resource[
                "provider_input_tokens"
            ],
            "proposer_provider_output_tokens": proposer_resource[
                "provider_output_tokens"
            ],
            "proposer_provider_latency_seconds": proposer_resource[
                "provider_latency_seconds"
            ],
            "judge_logical_calls": judge_resource["logical_calls"],
            "judge_provider_attempts": judge_resource["provider_attempts"],
            "judge_provider_input_tokens": judge_resource[
                "provider_input_tokens"
            ],
            "judge_provider_output_tokens": judge_resource[
                "provider_output_tokens"
            ],
            "judge_provider_latency_seconds": judge_resource[
                "provider_latency_seconds"
            ],
            "allocated_cpu_core_hours": runtime.get(
                "allocated_cpu_core_hours"
            ),
            "allocated_gpu_hours": runtime.get("allocated_gpu_hours"),
            "selection_validation_normalized_mse": (
                None
                if summary is None
                else summary.get("selection_validation_normalized_mse")
            ),
            "selection_training_normalized_mse": selected_training_nmse,
            "selected_round_index": (
                None if selected_round is None else selected_round["round_index"]
            ),
            "public_target_pass": target_pass,
            "public_mechanism_compliance": mechanism_compliance,
            "public_mechanism_compliance_complete": mechanism_complete,
            **complexity,
        }
        rows.append(row)
        by_trial[(task.benchmark_id, task.tier, task.repetition)].append(row)
    matched = [_matched_pair(key, value) for key, value in sorted(by_trial.items())]
    groups = [_group(arm.arm_id, rows) for arm in plan.arms]
    return {
        "schema_version": "phase-b-proposer-refinement-pilot-report-1",
        "status": "complete",
        "development_only": True,
        "test_data_opened": False,
        "private_reference_opened": False,
        "weighted_overall_score_defined": False,
        "task_count": len(rows),
        "source_completion_count": sum(
            item["source_complete"] is True for item in rows
        ),
        "matched_round_zero_count": sum(
            item["round_zero_request_match"] is True for item in matched
        ),
        "groups": groups,
        "matched_trials": matched,
        "tasks": rows,
    }


def write_report(report: dict[str, object], output_root: Path) -> None:
    """Write JSON, CSV, and concise Markdown reports."""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "proposer_refinement_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = report["tasks"]
    assert isinstance(rows, list)
    with (output_root / "proposer_refinement_tasks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = list(rows[0]) if rows else ["arm_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Feedback-rich incumbent refinement pilot",
        "",
        "Public train/validation evidence only; test and private references "
        "remained closed.",
        "",
        "| Arm | Complete | Valid rounds | Accepted refinements | Target pass | "
        "Mechanism compliance | Median validation NMSE |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        assert isinstance(group, dict)
        lines.append(
            f"| {group['arm_id']} | {group['source_completion_rate']:.3f} | "
            f"{group['mean_valid_round_count']:.3f} | "
            f"{group['mean_accepted_postinitial_challenger_count']:.3f} | "
            f"{_format(group['public_target_pass_rate'])} | "
            f"{_format(group['mean_public_mechanism_compliance'])} | "
            f"{_format(group['median_validation_normalized_mse'])} |"
        )
    lines.extend(
        [
            "",
            "Matched round-zero requests: "
            f"{report['matched_round_zero_count']}/"
            f"{len(report['matched_trials'])}.",
            "",
            "No scalar winner is declared automatically; fit, mechanism "
            "compliance, validity, and resources remain separate endpoints.",
        ]
    )
    (output_root / "proposer_refinement_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _group(arm_id: str, rows: list[dict[str, object]]) -> dict[str, object]:
    selected = [item for item in rows if item["arm_id"] == arm_id]
    training_losses = [
        float(item["selection_training_normalized_mse"])
        for item in selected
        if isinstance(item["selection_training_normalized_mse"], (int, float))
    ]
    losses = [
        float(item["selection_validation_normalized_mse"])
        for item in selected
        if isinstance(item["selection_validation_normalized_mse"], (int, float))
    ]
    compliance = [
        float(item["public_mechanism_compliance"])
        for item in selected
        if isinstance(item["public_mechanism_compliance"], (int, float))
    ]
    target_results = [
        item["public_target_pass"]
        for item in selected
        if isinstance(item["public_target_pass"], bool)
    ]
    compliance_complete = [
        item["public_mechanism_compliance_complete"]
        for item in selected
        if isinstance(item["public_mechanism_compliance_complete"], bool)
    ]
    return {
        "arm_id": arm_id,
        "trial_count": len(selected),
        "source_completion_rate": mean(
            item["source_complete"] is True for item in selected
        ),
        "mean_valid_round_count": mean(
            int(item["valid_round_count"]) for item in selected
        ),
        "mean_accepted_postinitial_challenger_count": mean(
            int(item["accepted_postinitial_challenger_count"])
            for item in selected
        ),
        "fit_retry_activation_rate": mean(
            int(item["fit_retry_activation_count"]) > 0
            for item in selected
        ),
        "public_target_pass_rate": (
            None if not target_results else mean(target_results)
        ),
        "mean_public_mechanism_compliance": (
            None if not compliance else mean(compliance)
        ),
        "public_mechanism_complete_assessment_rate": (
            None if not compliance_complete else mean(compliance_complete)
        ),
        "median_training_normalized_mse": (
            None if not training_losses else median(training_losses)
        ),
        "median_validation_normalized_mse": None if not losses else median(losses),
        "mean_validation_normalized_mse": None if not losses else mean(losses),
        "mean_state_count": _mean_optional(
            item["state_count"] for item in selected
        ),
        "mean_latent_state_count": _mean_optional(
            item["latent_state_count"] for item in selected
        ),
        "mean_parameter_count": _mean_optional(
            item["parameter_count"] for item in selected
        ),
        "mean_expression_ast_node_count": _mean_optional(
            item["total_expression_ast_node_count"] for item in selected
        ),
        "proposer_failure_event_count": sum(
            int(item["proposer_failure_event_count"]) for item in selected
        ),
        "proposer_provider_attempts": sum(
            int(item["proposer_provider_attempts"]) for item in selected
        ),
        "proposer_provider_input_tokens": sum(
            int(item["proposer_provider_input_tokens"]) for item in selected
        ),
        "proposer_provider_output_tokens": sum(
            int(item["proposer_provider_output_tokens"])
            for item in selected
        ),
        "judge_provider_attempts": sum(
            int(item["judge_provider_attempts"]) for item in selected
        ),
        "judge_provider_input_tokens": sum(
            int(item["judge_provider_input_tokens"]) for item in selected
        ),
        "judge_provider_output_tokens": sum(
            int(item["judge_provider_output_tokens"]) for item in selected
        ),
        "allocated_cpu_core_hours": _sum_optional(
            item["allocated_cpu_core_hours"] for item in selected
        ),
        "allocated_gpu_hours": _sum_optional(
            item["allocated_gpu_hours"] for item in selected
        ),
    }


def _matched_pair(
    key: tuple[str, str, int], rows: list[dict[str, object]]
) -> dict[str, object]:
    by_arm = {str(item["arm_id"]): item for item in rows}
    exploratory = by_arm["rich_exploratory"]
    refinement = by_arm["rich_incumbent_refinement"]
    hashes = (
        exploratory["initial_proposer_request_hash"],
        refinement["initial_proposer_request_hash"],
    )
    covered = all(item is not None for item in hashes)
    matched = covered and hashes[0] == hashes[1]
    if covered and not matched:
        raise ValueError(f"round-zero proposer request differs across arms: {key}")
    if matched and refinement["initial_proposer_cache_hit"] is not True:
        raise ValueError(f"refinement arm did not reuse round zero: {key}")
    return {
        "benchmark_id": key[0],
        "tier": key[1],
        "repetition": key[2],
        "both_sources_complete": (
            exploratory["source_complete"] is True
            and refinement["source_complete"] is True
        ),
        "round_zero_request_match": matched,
        "refinement_round_zero_cache_hit": (
            refinement["initial_proposer_cache_hit"]
        ),
    }


def _challenger_selected(round_payload: dict[str, object]) -> bool:
    challenge = round_payload.get("incumbent_challenge")
    record = round_payload.get("record")
    if not isinstance(challenge, dict) or not isinstance(record, dict):
        return False
    return challenge.get("selected_hash") == record.get("structural_hash")


def _selected_round(
    rounds: list[dict[str, Any]], selection_hash: object
) -> dict[str, Any] | None:
    if not isinstance(selection_hash, str):
        return None
    for payload in rounds:
        record = payload.get("record")
        if (
            isinstance(record, dict)
            and record.get("structural_hash") == selection_hash
        ):
            return payload
    return None


def _nested_number(payload: object, *keys: str) -> float | None:
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _llm_resource(events: list[dict[str, Any]]) -> dict[str, int | float]:
    provider_events = [item for item in events if item.get("cache_hit") is not True]
    return {
        "logical_calls": sum(int(item.get("logical_calls", 1)) for item in events),
        "provider_attempts": sum(
            int(item.get("provider_attempts", 1)) for item in provider_events
        ),
        "logical_input_tokens": _token_sum(events, "input_tokens"),
        "logical_output_tokens": _token_sum(events, "output_tokens"),
        "provider_input_tokens": _token_sum(provider_events, "input_tokens"),
        "provider_output_tokens": _token_sum(provider_events, "output_tokens"),
        "provider_latency_seconds": sum(
            float(item.get("latency_ms", 0.0)) for item in provider_events
        )
        / 1000.0,
    }


def _token_sum(events: list[dict[str, Any]], key: str) -> int:
    total = 0
    for item in events:
        usage = item.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get(key), int):
            total += int(usage[key])
    return total


def _complexity(candidate: CandidateModel | None) -> dict[str, int | None]:
    if candidate is None:
        return {
            "state_count": None,
            "latent_state_count": None,
            "process_count": None,
            "parameter_count": None,
            "state_equation_additive_term_count": None,
            "total_expression_ast_node_count": None,
        }
    expressions = [
        *(item.rhs for item in candidate.state_equations),
        *(item.expression for item in candidate.processes),
        *(item.expression for item in candidate.observation_mappings),
        *(
            item.expression
            for item in candidate.initial_conditions
            if item.expression is not None
        ),
    ]
    return {
        "state_count": len(candidate.states),
        "latent_state_count": sum(
            item.kind.value == "latent" for item in candidate.states
        ),
        "process_count": len(candidate.processes),
        "parameter_count": len(candidate.parameters),
        "state_equation_additive_term_count": sum(
            _additive_term_count(item.rhs) for item in candidate.state_equations
        ),
        "total_expression_ast_node_count": sum(
            sum(1 for _ in ast.walk(ast.parse(source, mode="eval")))
            for source in expressions
        ),
    }


def _additive_term_count(source: str) -> int:
    return _additive_node_count(ast.parse(source, mode="eval").body)


def _additive_node_count(node: ast.AST) -> int:
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _additive_node_count(node.left) + _additive_node_count(node.right)
    return 1


def _mean_optional(values: Iterable[object]) -> float | None:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
    ]
    return None if not numeric else mean(numeric)


def _sum_optional(values: Iterable[object]) -> float | None:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
    ]
    return None if not numeric else sum(numeric)


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _format(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--target-contract-root", type=Path, required=True)
    parser.add_argument("--mechanism-spec-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = collect_report(
        args.plan,
        args.search_root,
        target_contract_root=args.target_contract_root,
        mechanism_spec_root=args.mechanism_spec_root,
    )
    write_report(report, args.output_root)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "tasks"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
