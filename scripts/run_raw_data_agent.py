#!/usr/bin/env python3
"""Run one checkpointed raw-data frontier-agent baseline repetition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from autoformalism.baselines.raw_data_agent import (
    RawAgentConfig,
    RawAgentInputs,
    RawAgentOutputContract,
    RawAgentProvider,
    evaluate_raw_agent_candidate,
    evaluate_raw_agent_fitted_model,
    evaluation_metrics_payload,
    fit_result_payload,
    raw_agent_validation_context,
    repair_raw_data_agent_candidate,
    run_raw_data_agent,
)
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.fitting import FitConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--tier", choices=("easy", "medium", "hard"), required=True)
    parser.add_argument("--provider", choices=("openai", "gemini"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=30000)
    parser.add_argument("--llm-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--llm-max-attempts", type=int, default=2)
    parser.add_argument("--contract-repair-attempts", type=int, default=1)
    parser.add_argument(
        "--output-contract",
        choices=tuple(item.value for item in RawAgentOutputContract),
        default=RawAgentOutputContract.STRUCTURE_ONLY.value,
    )
    parser.add_argument("--fit-max-nfev", type=int, default=50)
    parser.add_argument("--fit-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.repetition < 0:
        raise SystemExit("--repetition must be nonnegative")
    if args.contract_repair_attempts < 0:
        raise SystemExit("--contract-repair-attempts must be nonnegative")
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    registry = BenchmarkRegistry()
    spec = registry.get(args.benchmark_id)
    data_config = DataConfig(
        root=data_root,
        benchmark_id=args.benchmark_id,
        tier=args.tier,
    )
    dataset = BenchmarkLoader(registry).load_development(data_config)
    prompt_root = (data_root / spec.relative_root).resolve()
    if spec.data_layout == "legacy_split_files":
        prompt_root = (
            prompt_root / spec.tier_directory_template.format(tier=args.tier)
        ).resolve()
    if not prompt_root.is_relative_to(data_root):
        raise SystemExit("benchmark prompt path escapes the public data root")
    prompt_path = prompt_root / "proposer_prompt.txt"
    train_path = _tidy_path(prompt_root, spec, "train")
    validation_path = _tidy_path(prompt_root, spec, "validation")
    for path in (prompt_path, train_path, validation_path):
        if not path.is_file():
            raise SystemExit(f"required public input is missing: {path}")

    context = raw_agent_validation_context(dataset, spec)
    inputs = RawAgentInputs(
        benchmark_id=args.benchmark_id,
        tier=args.tier,
        public_prompt=prompt_path.read_text(encoding="utf-8"),
        train_path=train_path,
        validation_path=validation_path,
        targets=context.targets,
        auxiliaries=context.auxiliaries,
        external_inputs=context.external_inputs,
        fixed_covariates=context.fixed_covariates,
        lagged_targets=context.lagged_targets,
    )
    config = RawAgentConfig(
        provider=RawAgentProvider(args.provider),
        model=args.model,
        repetition=args.repetition,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.llm_timeout_seconds,
        max_output_tokens=args.max_output_tokens,
        max_tool_calls=args.max_tool_calls,
        max_attempts=args.llm_max_attempts,
        output_contract=RawAgentOutputContract(args.output_contract),
    )
    run_directory = output_root / _run_name(config, inputs)
    run_directory.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema_version": "raw-data-agent-run-config-1",
        "benchmark_id": inputs.benchmark_id,
        "tier": inputs.tier,
        "provider": config.provider.value,
        "model": config.model,
        "repetition": config.repetition,
        "agent_config": config.model_dump(mode="json"),
        "fit_config": (
            None
            if config.output_contract is RawAgentOutputContract.FITTED_MODEL
            else {
                "number_of_starts": 1,
                "random_seed": args.repetition,
                "integration_backend": "solve_ivp",
                "allow_derivative_regression": False,
                "maximum_function_evaluations": args.fit_max_nfev,
                "maximum_wall_time_seconds": args.fit_timeout_seconds,
            }
        ),
        "fixed_model_evaluation_config": (
            {
                "integration_backend": "solve_ivp",
                "parameter_optimization": False,
                "fit_trajectory_initial_conditions": False,
                "target_scales_fit_on": "train",
            }
            if config.output_contract is RawAgentOutputContract.FITTED_MODEL
            else None
        ),
        "public_input_hashes": {
            **inputs.file_hashes,
            "proposer_prompt.txt": _sha256(prompt_path),
        },
        "split_fingerprints": {
            "train": dataset.train.fingerprint,
            "validation": dataset.validation.fingerprint,
        },
        "test_data_opened": False,
        "pruning_applied": False,
        "parameter_refit_applied": (
            config.output_contract is RawAgentOutputContract.STRUCTURE_ONLY
        ),
        "contract_repair_attempts": args.contract_repair_attempts,
    }
    _write_json(run_directory / "run_config.json", run_config)
    if args.dry_run:
        _write_json(
            run_directory / "status.json",
            {"status": "dry_run", "run_directory": str(run_directory)},
        )
        print(json.dumps(run_config, indent=2, sort_keys=True))
        return

    _require_key(config.provider)
    try:
        artifact = run_raw_data_agent(
            config=config,
            inputs=inputs,
            output_directory=run_directory,
        )
        evaluated_artifact = artifact
        for repair_index in range(args.contract_repair_attempts + 1):
            try:
                if config.output_contract is RawAgentOutputContract.FITTED_MODEL:
                    (
                        candidate,
                        repairs,
                        training_metrics,
                        validation_metrics,
                        warnings,
                    ) = evaluate_raw_agent_fitted_model(
                        artifact=evaluated_artifact,
                        dataset=dataset,
                        context=context,
                        simulation_config=FitConfig(
                            random_seed=args.repetition,
                            integration_backend="solve_ivp",
                            allow_derivative_regression=False,
                        ),
                    )
                else:
                    candidate, repairs, fit, warnings = evaluate_raw_agent_candidate(
                        artifact=evaluated_artifact,
                        dataset=dataset,
                        context=context,
                        fit_config=FitConfig(**run_config["fit_config"]),
                    )
                break
            except ModelValidationError as exc:
                if repair_index >= args.contract_repair_attempts:
                    raise
                evaluated_artifact = repair_raw_data_agent_candidate(
                    config=config,
                    inputs=inputs,
                    original=evaluated_artifact,
                    diagnostics=str(exc),
                    repair_index=repair_index + 1,
                    output_directory=run_directory,
                )
        if config.output_contract is RawAgentOutputContract.FITTED_MODEL:
            rollout_success = not (
                training_metrics.failed_trajectories
                or validation_metrics.failed_trajectories
            )
            evaluation = {
                "schema_version": "raw-data-agent-fitted-evaluation-1",
                "candidate_id": candidate.candidate_id,
                "repairs": list(repairs),
                "validation_warnings": list(warnings),
                "parameter_source": "provider_agent",
                "fitted_parameter_values": {
                    item.name: (evaluated_artifact.fitted_parameter_values or {})[
                        item.name
                    ]
                    for item in candidate.parameters
                },
                "fit_method_summary": evaluated_artifact.fit_method_summary,
                "parameter_refit_applied": False,
                "training_metrics": evaluation_metrics_payload(training_metrics),
                "validation_metrics": evaluation_metrics_payload(
                    validation_metrics
                ),
                "test_data_opened": False,
                "pruning_applied": False,
            }
            status = {
                "status": "complete" if rollout_success else "rollout_failed",
                "candidate_id": candidate.candidate_id,
                "validation_normalized_mse": validation_metrics.normalized_mse,
                "training_normalized_mse": training_metrics.normalized_mse,
                "parameter_refit_applied": False,
            }
        else:
            evaluation = {
                "schema_version": "raw-data-agent-evaluation-1",
                "candidate_id": candidate.candidate_id,
                "repairs": list(repairs),
                "validation_warnings": list(warnings),
                "fit": fit_result_payload(fit),
                "parameter_refit_applied": True,
                "test_data_opened": False,
                "pruning_applied": False,
            }
            status = {
                "status": "complete" if fit.success else "fit_failed",
                "candidate_id": candidate.candidate_id,
                "validation_normalized_mse": fit.validation_metrics.normalized_mse,
                "training_normalized_mse": fit.training_metrics.normalized_mse,
                "parameter_refit_applied": True,
            }
        _write_json(run_directory / "candidate.json", candidate.model_dump(mode="json"))
        _write_json(run_directory / "evaluation.json", evaluation)
        status.update({
            "agent_latency_seconds": artifact.latency_seconds,
            "tool_call_count": artifact.tool_call_count,
            "requested_max_tool_calls": config.max_tool_calls,
            "provider_reported_max_tool_calls": (
                artifact.provider_reported_max_tool_calls
            ),
            "tool_call_limit_exceeded": (
                artifact.tool_call_count > config.max_tool_calls
            ),
            "usage": None if artifact.usage is None else artifact.usage.model_dump(),
        })
        _write_json(run_directory / "status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True))
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": _redact(exc),
        }
        _write_json(run_directory / "status.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


def _tidy_path(prompt_root: Path, spec: Any, split: str) -> Path:
    if spec.data_layout != "tidy_split_file":
        raise SystemExit("the raw-data pilot currently supports tidy Phase-B cells")
    return prompt_root / spec.split_filename_template.format(split=split)


def _run_name(config: RawAgentConfig, inputs: RawAgentInputs) -> str:
    safe_model = "".join(char if char.isalnum() else "-" for char in config.model)
    return (
        f"{config.provider.value}_{safe_model}_{inputs.benchmark_id}_"
        f"{inputs.tier}_rep{config.repetition}"
    )


def _require_key(provider: RawAgentProvider) -> None:
    names = (
        ("OPENAI_API_KEY",)
        if provider is RawAgentProvider.OPENAI
        else ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    if not any(os.environ.get(name) for name in names):
        raise SystemExit(f"missing API key environment variable: {' or '.join(names)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _redact(error: Exception) -> str:
    rendered = str(error)
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            rendered = rendered.replace(secret, "[REDACTED]")
    return rendered[:4000]


if __name__ == "__main__":
    main()
