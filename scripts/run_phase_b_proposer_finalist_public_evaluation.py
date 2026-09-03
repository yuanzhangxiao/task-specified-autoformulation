#!/usr/bin/env python3
"""Evaluate one repaired proposer finalist using public development data only."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path
from time import monotonic, process_time
from typing import Any

from autoformalism.execution import (
    ExecutionArguments,
    _context,
    _load_inputs,
    _load_public_target_contract,
)
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import FitConfig, fit_candidate
from autoformalism.rebuttal.final_evaluation import certify_runtime_validity
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.rebuttal.proposer_finalist_evaluation import (
    canonical_plan_sha256,
    load_proposer_finalist_evaluation_plan,
    task_identity,
)
from autoformalism.schemas import CandidateModel
from autoformalism.targets import evaluate_public_targets


def main() -> None:
    """Validate all frozen inputs, fit one candidate, and checkpoint its result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-replay-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-contract-root", type=Path, required=True)
    parser.add_argument("--mechanism-spec-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()

    output = args.output_root / "tasks" / f"task_{args.task_index:03d}.json"
    if output.is_file():
        _validate_checkpoint(output, args.config, args.task_index)
        print(f"resumed verified checkpoint: {output}")
        return

    plan = load_proposer_finalist_evaluation_plan(args.config)
    condition, cell, repetition, source_task_index = task_identity(
        plan, args.task_index
    )
    source_rows, ledger = _validate_source_replay(
        args.source_replay_root, plan
    )
    row_key = (
        condition.reasoning_effort,
        condition.max_output_tokens,
        source_task_index,
    )
    source_row = source_rows.get(row_key)
    if source_row is None:
        raise ValueError(f"source replay row is absent: {row_key}")
    expected_identity = (cell.benchmark_id, cell.tier, repetition)
    observed_identity = (
        source_row.get("benchmark_id"),
        source_row.get("tier"),
        source_row.get("repetition"),
    )
    if observed_identity != expected_identity:
        raise ValueError(
            "source replay row identity differs: "
            f"expected={expected_identity}, actual={observed_identity}"
        )

    candidate_relative = (
        Path("finalists")
        / condition.directory_name
        / f"task_{source_task_index:03d}.json"
    )
    candidate_path = args.source_replay_root / candidate_relative
    _verify_ledger_artifact(candidate_relative, candidate_path, ledger)
    candidate = CandidateModel.model_validate_json(
        candidate_path.read_text(encoding="utf-8")
    )
    candidate_sha256 = hashlib.sha256(
        candidate.model_dump_json().encode("utf-8")
    ).hexdigest()
    if candidate_sha256 != source_row.get("candidate_sha256"):
        raise ValueError("canonical finalist candidate SHA-256 differs")

    dataset, context, prompt, target_contract = _public_inputs(
        data_root=args.data_root,
        target_contract_root=args.target_contract_root,
        benchmark_id=cell.benchmark_id,
        tier=cell.tier,
        repetition=repetition,
        scratch_root=args.output_root / ".context",
    )
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if prompt_sha256 != cell.public_prompt_sha256:
        raise ValueError("public proposer prompt SHA-256 differs")
    target_path = (
        args.target_contract_root / "specs" / f"{cell.benchmark_id}.json"
    )
    if _sha256(target_path) != cell.public_target_contract_sha256:
        raise ValueError("public target contract SHA-256 differs")
    mechanism_path = (
        args.mechanism_spec_root / "specs" / f"{cell.benchmark_id}.json"
    )
    if _sha256(mechanism_path) != cell.public_mechanism_spec_sha256:
        raise ValueError("public mechanism specification SHA-256 differs")
    mechanism_spec = MechanismEvaluationSpec.model_validate_json(
        mechanism_path.read_text(encoding="utf-8")
    )
    if (
        mechanism_spec.benchmark_id,
        mechanism_spec.tier,
        mechanism_spec.public_prompt_sha256,
    ) != (cell.benchmark_id, cell.tier, prompt_sha256):
        raise ValueError("public mechanism specification provenance differs")

    runtime = certify_runtime_validity(candidate, context)
    target_evaluation = evaluate_public_targets(candidate, target_contract)
    mechanism_evaluation = evaluate_mechanisms(candidate, mechanism_spec)
    attempts: list[dict[str, Any]] = []
    selected_fit: dict[str, Any] | None = None
    if runtime.valid:
        compiled = compile_candidate(candidate, context)
        for attempt_index, profile in enumerate(plan.fit_profiles):
            settings = FitConfig(
                number_of_starts=profile.number_of_starts,
                random_seed=repetition,
                integration_backend=profile.integration_backend,
                maximum_function_evaluations=(
                    profile.maximum_function_evaluations
                ),
                maximum_wall_time_seconds=profile.maximum_wall_time_seconds,
                allow_derivative_regression=False,
                parameter_fit_strategy="bounded_nonlinear",
            )
            wall_started = monotonic()
            cpu_started = process_time()
            try:
                fitted = fit_candidate(
                    compiled,
                    dataset.train,
                    dataset.validation,
                    settings,
                )
            except Exception as exc:  # untrusted candidates may fail numerically
                attempt = {
                    "attempt_index": attempt_index,
                    "profile_id": profile.profile_id,
                    "fit_config": settings.model_dump(mode="json"),
                    "success": False,
                    "wall_seconds": monotonic() - wall_started,
                    "process_cpu_seconds": process_time() - cpu_started,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                }
            else:
                attempt = _fit_payload(
                    fitted,
                    attempt_index=attempt_index,
                    profile_id=profile.profile_id,
                    fit_config=settings,
                    wall_seconds=monotonic() - wall_started,
                    process_cpu_seconds=process_time() - cpu_started,
                )
                if fitted.success:
                    selected_fit = attempt
            attempts.append(attempt)
            if selected_fit is not None and plan.stop_after_first_success:
                break

    payload = {
        "schema_version": "phase-b-proposer-finalist-public-evaluation-task-1",
        "status": "complete",
        "development_only": True,
        "new_llm_calls_made": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "weighted_overall_score_defined": False,
        "plan_sha256": canonical_plan_sha256(plan),
        "task_index": args.task_index,
        "source_task_index": source_task_index,
        "condition": condition.model_dump(mode="json"),
        "benchmark_id": cell.benchmark_id,
        "tier": cell.tier,
        "repetition": repetition,
        "source_candidate_path": str(candidate_relative),
        "source_candidate_file_sha256": _sha256(candidate_path),
        "candidate_sha256": candidate_sha256,
        "train_fingerprint": dataset.train.fingerprint,
        "validation_fingerprint": dataset.validation.fingerprint,
        "runtime": runtime.model_dump(mode="json"),
        "public_target": target_evaluation.model_dump(mode="json"),
        "public_mechanism": mechanism_evaluation.model_dump(mode="json"),
        "complexity": _complexity(candidate),
        "fit_success": selected_fit is not None,
        "selected_fit_profile": (
            None if selected_fit is None else selected_fit["profile_id"]
        ),
        "selected_fit": selected_fit,
        "fit_attempts": attempts,
        "allocated_cpus": _optional_int(os.environ.get("SLURM_CPUS_PER_TASK")),
    }
    _write_once_json(output, payload)
    print(
        f"{condition.directory_name} {cell.benchmark_id} rep={repetition} "
        f"fit_success={payload['fit_success']} "
        "graph_mechanism_compliance="
        f"{mechanism_evaluation.graph_mechanism_compliance:.3f} "
        "annotation_compliance="
        f"{mechanism_evaluation.mechanism_annotation_compliance:.3f}",
        flush=True,
    )


def _public_inputs(
    *,
    data_root: Path,
    target_contract_root: Path,
    benchmark_id: str,
    tier: str,
    repetition: int,
    scratch_root: Path,
) -> tuple[object, object, str, object]:
    contract_path = target_contract_root / "specs" / f"{benchmark_id}.json"
    arguments = ExecutionArguments(
        data_root=data_root,
        benchmark_id=benchmark_id,
        tier=tier,
        seed=repetition,
        proposer_model=None,
        judge_model=None,
        iteration_budget=1,
        beam_size=1,
        output_root=scratch_root,
        resume=False,
        dry_run=True,
        mock_llm=True,
        use_clean_observations=False,
        use_judge=False,
        development_only=True,
        public_target_contract=contract_path,
    )
    dataset, _test_loader, prompt, _judge_prompt = _load_inputs(arguments)
    context = _context(arguments, dataset)
    contract = _load_public_target_contract(
        arguments,
        proposer_prompt=prompt,
        targets=dataset.roles.targets,
    )
    if contract is None:
        raise ValueError(f"missing public target contract: {benchmark_id}")
    return dataset, context, prompt, contract


def _validate_source_replay(
    root: Path, plan: object
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    source = plan.source_replay
    manifest_path = root / "proposer_repair_replay.json"
    manifest = _read_object(manifest_path)
    expected = {
        "schema_version": source.manifest_schema_version,
        "status": source.required_status,
        "source_plan_sha256": source.source_plan_sha256,
        "replay_plan_sha256": source.replay_plan_sha256,
        "artifact_ledger_sha256": source.artifact_ledger_sha256,
        "replay_result_count": source.replay_result_count,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"source replay manifest differs at {key}")
    digest_path = root / "proposer_repair_replay.json.sha256"
    fields = digest_path.read_text(encoding="utf-8").strip().split()
    if fields != [_sha256(manifest_path), "proposer_repair_replay.json"]:
        raise ValueError("source replay manifest companion digest differs")
    ledger_path = root / "artifact_ledger.jsonl"
    if _sha256(ledger_path) != source.artifact_ledger_sha256:
        raise ValueError("source replay artifact ledger SHA-256 differs")
    ledger: dict[str, dict[str, Any]] = {}
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        relative_path = Path(str(item["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"unsafe source replay ledger path: {relative_path}"
            )
        relative = str(relative_path)
        if relative in ledger:
            raise ValueError(f"duplicate source replay ledger path: {relative}")
        ledger[relative] = item
        _verify_ledger_artifact(Path(relative), root / relative, ledger)

    rows_relative = Path("repair_replay_rows.jsonl")
    rows_path = root / rows_relative
    _verify_ledger_artifact(rows_relative, rows_path, ledger)
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = (
            str(item["reasoning_effort"]),
            int(item["max_output_tokens"]),
            int(item["task_index"]),
        )
        if key in rows:
            raise ValueError(f"duplicate source replay row: {key}")
        if not item.get("deterministic_valid") or not item.get(
            "public_target_passed"
        ):
            raise ValueError(f"source replay finalist did not pass: {key}")
        rows[key] = item
    if len(rows) != source.replay_result_count:
        raise ValueError("source replay row count differs")
    expected_keys = {
        (condition.reasoning_effort, condition.max_output_tokens, task)
        for condition in plan.conditions
        for task in range(len(plan.cells) * len(plan.repetitions))
    }
    if set(rows) != expected_keys:
        raise ValueError("source replay finalist matrix differs")
    return rows, ledger


def _verify_ledger_artifact(
    relative: Path,
    path: Path,
    ledger: dict[str, dict[str, Any]],
) -> None:
    item = ledger.get(str(relative))
    if item is None:
        raise ValueError(f"source replay artifact is absent from ledger: {relative}")
    if not path.is_file() or _sha256(path) != item.get("sha256"):
        raise ValueError(f"source replay artifact differs: {relative}")
    if path.stat().st_size != item.get("size_bytes"):
        raise ValueError(f"source replay artifact size differs: {relative}")


def _fit_payload(
    fitted: object,
    *,
    attempt_index: int,
    profile_id: str,
    fit_config: FitConfig,
    wall_seconds: float,
    process_cpu_seconds: float,
) -> dict[str, Any]:
    return {
        "attempt_index": attempt_index,
        "profile_id": profile_id,
        "fit_config": fit_config.model_dump(mode="json"),
        "success": fitted.success,
        "message": fitted.message,
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": process_cpu_seconds,
        "training_normalized_mse": fitted.training_metrics.normalized_mse,
        "training_per_target_normalized_mse": dict(
            fitted.training_metrics.per_target_normalized_mse
        ),
        "training_failed_trajectories": list(
            fitted.training_metrics.failed_trajectories
        ),
        "validation_normalized_mse": fitted.validation_metrics.normalized_mse,
        "validation_per_target_normalized_mse": dict(
            fitted.validation_metrics.per_target_normalized_mse
        ),
        "validation_failed_trajectories": list(
            fitted.validation_metrics.failed_trajectories
        ),
        "global_parameters": dict(fitted.global_parameters),
        "global_initial_conditions": dict(fitted.global_initial_conditions),
        "function_evaluations": sum(
            item.function_evaluations for item in fitted.diagnostics
        ),
        "integration_failures": sum(
            item.integration_failures for item in fitted.diagnostics
        ),
        "diagnostics": [
            {
                "start_index": item.start_index,
                "success": item.success,
                "status": item.status,
                "message": item.message,
                "cost": item.cost,
                "function_evaluations": item.function_evaluations,
                "integration_failures": item.integration_failures,
                "backend": item.backend,
                "integration_failure_messages": list(
                    item.integration_failure_messages
                ),
            }
            for item in fitted.diagnostics
        ],
    }


def _complexity(candidate: CandidateModel) -> dict[str, int]:
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
    node = ast.parse(source, mode="eval").body
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _additive_node_count(node)
    return 1


def _additive_node_count(node: ast.AST) -> int:
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _additive_node_count(node.left) + _additive_node_count(node.right)
    return 1


def _validate_checkpoint(path: Path, config: Path, task_index: int) -> None:
    plan = load_proposer_finalist_evaluation_plan(config)
    payload = _read_object(path)
    if (
        payload.get("schema_version")
        != "phase-b-proposer-finalist-public-evaluation-task-1"
        or payload.get("status") != "complete"
        or payload.get("task_index") != task_index
        or payload.get("plan_sha256") != canonical_plan_sha256(plan)
    ):
        raise ValueError(f"existing finalist checkpoint differs: {path}")


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_once_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing finalist checkpoint differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optional_int(value: str | None) -> int | None:
    return None if value is None else int(value)


if __name__ == "__main__":
    main()
