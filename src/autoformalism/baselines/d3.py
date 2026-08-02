"""Leakage-safe D3 workflow and optional external bridge adapter."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from autoformalism.baselines.core import evaluate_equations, target_scales
from autoformalism.baselines.models import BaselineConfig, BaselineResult
from autoformalism.data import (
    DatasetSplit,
    DevelopmentDataset,
    SplitName,
    Trajectory,
)
from autoformalism.expressions import (
    ModelValidationError,
    ValidationContext,
    compile_candidate,
    repair_protected_declarations,
)
from autoformalism.fitting import FitConfig, evaluate_fitted_candidate, fit_candidate
from autoformalism.llm import LLMClient
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.schemas import CandidateModel, ParameterScope, StateKind

D3_UPSTREAM_REVISION = "ee86212dfd5935bb0c9626eaa0570223ff7ecf1c"


class D3AdapterError(RuntimeError):
    """Raised when the external D3 bridge violates the exchange contract."""


def run_d3_no_tools(
    config: BaselineConfig,
    dataset: DevelopmentDataset,
    test_loader,
    context: ValidationContext,
    *,
    task_prompt: str,
    command: Sequence[str] = (),
    work_directory: Path,
    llm_client: LLMClient | None = None,
) -> BaselineResult:
    """Run safe D3 locally or delegate to an isolated compatibility bridge."""
    if command:
        return _run_external_bridge(
            config,
            dataset,
            test_loader,
            context,
            task_prompt=task_prompt,
            command=command,
            work_directory=work_directory,
        )
    if llm_client is None:
        raise D3AdapterError("D3 requires --model or --d3-command")
    return _run_safe_d3(
        config,
        dataset,
        test_loader,
        context,
        task_prompt=task_prompt,
        llm_client=llm_client,
        work_directory=work_directory,
    )


def _run_external_bridge(
    config: BaselineConfig,
    dataset: DevelopmentDataset,
    test_loader: Callable[[], DatasetSplit],
    context: ValidationContext,
    *,
    task_prompt: str,
    command: Sequence[str],
    work_directory: Path,
) -> BaselineResult:
    """Run a version-pinned D3 compatibility process without exposing test data."""
    work_directory.mkdir(parents=True, exist_ok=True)
    request_path = work_directory / "d3_request.json"
    response_path = work_directory / "d3_response.json"
    request = {
        "protocol_version": "1",
        "external_tools_enabled": False,
        "candidate_submission_enabled": True,
        "task_prompt": task_prompt,
        "targets": list(context.targets),
        "auxiliaries": list(context.auxiliaries),
        "external_inputs": list(context.external_inputs),
        "fixed_covariates": list(context.fixed_covariates),
        "training": _split_payload(dataset.train),
        "validation": _split_payload(dataset.validation),
        "seed": config.seed,
    }
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    completed = subprocess.run(
        [*command, str(request_path), str(response_path)],
        cwd=work_directory,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise D3AdapterError(
            f"D3 bridge failed with code {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    try:
        response = json.loads(response_path.read_text(encoding="utf-8"))
        equations = {str(k): str(v) for k, v in response["equations"].items()}
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise D3AdapterError(f"invalid D3 response: {exc}") from exc
    if set(equations) != set(context.targets):
        raise D3AdapterError("D3 response must contain one equation per target")
    scales = target_scales(dataset.train, context.targets)
    training = evaluate_equations(
        equations, context, dataset.train, scales, identifier="d3_train"
    )
    validation = evaluate_equations(
        equations, context, dataset.validation, scales, identifier="d3_validation"
    )
    test = evaluate_equations(
        equations, context, test_loader(), scales, identifier="d3_test"
    )
    return BaselineResult(
        method=config.method,
        benchmark_id=dataset.benchmark_id,
        tier=dataset.tier,
        seed=config.seed,
        equations=equations,
        selected_hyperparameters={
            "external_tools_enabled": "false",
            "candidate_submission_enabled": "true",
            "adaptation": "external_bridge",
            "upstream_revision": D3_UPSTREAM_REVISION,
        },
        training_normalized_mse=training.normalized_mse,
        validation_normalized_mse=validation.normalized_mse,
        test_normalized_mse=test.normalized_mse,
        test_per_target_normalized_mse=dict(test.per_target_normalized_mse),
    )


def _run_safe_d3(
    config: BaselineConfig,
    dataset: DevelopmentDataset,
    test_loader: Callable[[], DatasetSplit],
    context: ValidationContext,
    *,
    task_prompt: str,
    llm_client: LLMClient,
    work_directory: Path,
) -> BaselineResult:
    """Adapt D3's propose-fit-reflect loop to the restricted model grammar."""
    work_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_directory / "d3_checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path, config, dataset)
    records: list[dict[str, Any]] = list(checkpoint["records"])
    start = len(records)
    best_validation = min(
        (
            float(item["validation_mse"])
            for item in records
            if item.get("validation_mse") is not None
        ),
        default=float("inf"),
    )
    stagnant = int(checkpoint.get("stagnant", 0))
    fit_config = FitConfig(random_seed=config.seed)

    for generation in range(start, config.d3_generations):
        feedback = _d3_feedback(records)
        try:
            proposed = llm_client.propose(
                system_prompt=_d3_system_prompt(task_prompt, context),
                user_prompt=json.dumps(
                    {
                        "generation": generation,
                        "total_generations": config.d3_generations,
                        "previous_results": feedback,
                        "instruction": (
                            "Propose a new or improved differential model. Use only "
                            "training/validation feedback; external acquisition tools "
                            "are unavailable."
                        ),
                    },
                    sort_keys=True,
                ),
            ).parsed
            candidate, repairs = repair_protected_declarations(proposed, context)
            candidate, projection_repairs = _project_d3_structure(
                candidate, context
            )
            repairs = (*repairs, *projection_repairs)
            _validate_d3_structure(candidate, context)
            compiled = compile_candidate(candidate, context)
            fitted = fit_candidate(
                compiled, dataset.train, dataset.validation, fit_config
            )
            if not fitted.success:
                raise D3AdapterError("numerical parameter fitting failed")
            validation_mse = fitted.validation_metrics.normalized_mse
            record = {
                "generation": generation,
                "candidate": candidate.model_dump(mode="json"),
                "repairs": list(repairs),
                "parameters": dict(fitted.global_parameters),
                "training_mse": fitted.training_metrics.normalized_mse,
                "validation_mse": validation_mse,
                "error": None,
            }
            if validation_mse < best_validation - 1e-12:
                best_validation = validation_mse
                stagnant = 0
            else:
                stagnant += 1
        except (
            D3AdapterError,
            LLMProviderError,
            LLMResponseError,
            ModelValidationError,
            TypeError,
            ValueError,
        ) as exc:
            record = {
                "generation": generation,
                "candidate": None,
                "parameters": {},
                "training_mse": None,
                "validation_mse": None,
                "error": f"{type(exc).__name__}: {str(exc)[:2000]}",
            }
            stagnant += 1
        records.append(record)
        _write_checkpoint(
            checkpoint_path, config, dataset, records, stagnant=stagnant
        )
        if stagnant >= config.d3_patience:
            break

    eligible = [item for item in records if item.get("validation_mse") is not None]
    if not eligible:
        failures = "; ".join(
            f"generation {item['generation']}: {item.get('error')}"
            for item in records
        )
        raise D3AdapterError(
            "D3 produced no valid fitted candidates. " + failures[-6000:]
        )
    selected = min(eligible, key=lambda item: float(item["validation_mse"]))
    candidate = CandidateModel.model_validate(selected["candidate"])
    compiled = compile_candidate(candidate, context)
    combined = DatasetSplit(
        SplitName.TRAIN,
        (*dataset.train.trajectories, *dataset.validation.trajectories),
        f"{dataset.train.fingerprint}+{dataset.validation.fingerprint}",
    )
    refitted = fit_candidate(compiled, combined, dataset.validation, fit_config)
    if not refitted.success:
        raise D3AdapterError("D3 train-plus-validation refit failed")
    _, test_metrics = evaluate_fitted_candidate(
        compiled,
        test_loader(),
        global_parameters=refitted.global_parameters,
        global_initial_conditions=refitted.global_initial_conditions,
        target_scales=refitted.target_scales,
        config=fit_config,
        fit_trajectory_initial_conditions=False,
    )
    equations = {
        equation.state: equation.rhs for equation in candidate.state_equations
    }
    return BaselineResult(
        method=config.method,
        benchmark_id=dataset.benchmark_id,
        tier=dataset.tier,
        seed=config.seed,
        equations=equations,
        selected_hyperparameters={
            "generations_completed": len(records),
            "selected_generation": int(selected["generation"]),
            "external_tools_enabled": "false",
            "adaptation": "restricted_schema",
            "upstream_revision": D3_UPSTREAM_REVISION,
            "selected_parameters": json.dumps(
                dict(refitted.global_parameters), sort_keys=True
            ),
        },
        training_normalized_mse=float(selected["training_mse"]),
        validation_normalized_mse=float(selected["validation_mse"]),
        test_normalized_mse=test_metrics.normalized_mse,
        test_per_target_normalized_mse=dict(
            test_metrics.per_target_normalized_mse
        ),
    )


def _validate_d3_structure(
    candidate: CandidateModel, context: ValidationContext
) -> None:
    """Keep D3 on its fixed observed-state skeleton and global parameters."""
    states = {item.name for item in candidate.states}
    if states != set(context.targets):
        raise D3AdapterError(
            "D3 candidate states must be exactly the observed target channels"
        )
    if any(item.kind is not StateKind.OBSERVED for item in candidate.states):
        raise D3AdapterError("D3 candidate states must all be observed")
    if any(
        item.scope is not ParameterScope.GLOBAL for item in candidate.parameters
    ):
        raise D3AdapterError("D3 parameters must be global")


def _project_d3_structure(
    candidate: CandidateModel, context: ValidationContext
) -> tuple[CandidateModel, tuple[str, ...]]:
    """Project over-declared supplied trajectories onto D3's fixed state skeleton."""
    target_set = set(context.targets)
    removed_states = [
        item.name for item in candidate.states if item.name not in target_set
    ]
    if not removed_states:
        return candidate, ()
    supplied = {
        *context.auxiliaries,
        *context.external_inputs,
        *context.fixed_covariates,
    }
    unsupported = sorted(set(removed_states) - supplied)
    if unsupported:
        raise D3AdapterError(
            "D3 proposed non-target states that are not supplied channels: "
            f"{unsupported}"
        )
    payload = candidate.model_dump(mode="json")
    payload["states"] = [
        item for item in payload["states"] if item["name"] in target_set
    ]
    payload["state_equations"] = [
        item for item in payload["state_equations"] if item["state"] in target_set
    ]
    payload["initial_conditions"] = [
        item for item in payload["initial_conditions"] if item["state"] in target_set
    ]
    payload["observation_mappings"] = [
        item
        for item in payload["observation_mappings"]
        if item["channel"] in target_set
    ]
    expression_text = " ".join(
        [item["rhs"] for item in payload["state_equations"]]
        + [item["expression"] for item in payload["processes"]]
        + [item["expression"] for item in payload["observation_mappings"]]
    )
    used = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression_text))
    payload["parameters"] = [
        item for item in payload["parameters"] if item["name"] in used
    ]
    declared = {
        *(item["name"] for item in payload["states"]),
        *(item["name"] for item in payload["processes"]),
        *(item["name"] for item in payload["parameters"]),
    }
    payload["constraints"] = [
        item for item in payload["constraints"] if item["subject"] in declared
    ]
    return CandidateModel.model_validate(payload), (
        "treated supplied channels declared as states as exogenous trajectories: "
        + ", ".join(sorted(removed_states)),
    )


def _d3_system_prompt(task_prompt: str, context: ValidationContext) -> str:
    available = tuple(
        dict.fromkeys(
            (
                *context.targets,
                *context.auxiliaries,
                *context.external_inputs,
                *context.fixed_covariates,
                context.time_symbol,
            )
        )
    )
    unavailable_meal_aliases = sorted(
        {"meal_amount", "meal_time", "meal_pulse"} - set(available)
    )
    return (
        "You are the D3 dynamical-system discovery agent. Iteratively propose an "
        "interpretable differential model, use fitted validation feedback to refine "
        "it, and return the required structured candidate. External feature-"
        "acquisition and execution tools are disabled. Candidate submission remains "
        "enabled through the structured response. To preserve D3's fixed simulator "
        f"state skeleton, declare exactly these observed states: {context.targets}. "
        "Each state must contain its derivative RHS. You may declare safe algebraic "
        "processes and bounded global parameters, but no latent states or trajectory-"
        "specific parameters. Use only symbols explicitly available in the task.\n\n"
        f"Benchmark task:\n{task_prompt}\n\n"
        f"Exact target state names: {context.targets}. Exact supplied auxiliary "
        f"trajectory names: {context.auxiliaries}. Exact external input names: "
        f"{context.external_inputs}. Exact fixed covariate names: "
        f"{context.fixed_covariates}. The complete allowed data-symbol list is: "
        f"{available}. Do not invent aliases for these names. In particular, these "
        f"meal aliases are unavailable in this tier and must not appear: "
        f"{unavailable_meal_aliases}. Supplied auxiliaries are RHS inputs, not extra "
        "states. For an observed state, set observed_channel equal to its name and "
        "omit initial. If uncertain, return a simple target equation using only the "
        "listed supplied symbols and bounded global parameters."
    )


def _d3_feedback(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact train/validation feedback without test information."""
    return [
        {
            "generation": item["generation"],
            "candidate": item.get("candidate"),
            "fitted_parameters": item.get("parameters", {}),
            "training_mse": item.get("training_mse"),
            "validation_mse": item.get("validation_mse"),
            "failure": item.get("error"),
        }
        for item in records[-4:]
    ]


def _load_checkpoint(
    path: Path, config: BaselineConfig, dataset: DevelopmentDataset
) -> dict[str, Any]:
    if not path.exists():
        return {"records": [], "stagnant": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != _checkpoint_fingerprint(config, dataset):
        raise D3AdapterError("D3 checkpoint fingerprint does not match this run")
    return payload


def _write_checkpoint(
    path: Path,
    config: BaselineConfig,
    dataset: DevelopmentDataset,
    records: Sequence[dict[str, Any]],
    *,
    stagnant: int,
) -> None:
    payload = {
        "protocol_version": "2",
        "fingerprint": _checkpoint_fingerprint(config, dataset),
        "records": list(records),
        "stagnant": stagnant,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_fingerprint(
    config: BaselineConfig, dataset: DevelopmentDataset
) -> str:
    import hashlib

    payload = {
        "benchmark_id": dataset.benchmark_id,
        "tier": dataset.tier,
        "seed": config.seed,
        "generations": config.d3_generations,
        "patience": config.d3_patience,
        "llm_model": config.llm_model,
        "adapter_revision": "3",
        "train": dataset.train.fingerprint,
        "validation": dataset.validation.fingerprint,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _split_payload(split: DatasetSplit) -> list[dict[str, object]]:
    return [_trajectory_payload(item) for item in split.trajectories]


def _trajectory_payload(item: Trajectory) -> dict[str, object]:
    return {
        "trajectory_id": item.trajectory_id,
        "time": item.time.tolist(),
        "targets": {key: value.tolist() for key, value in item.targets.items()},
        "auxiliaries": {
            key: value.tolist() for key, value in item.auxiliaries.items()
        },
        "external_inputs": {
            key: value.tolist() for key, value in item.external_inputs.items()
        },
        "fixed_covariates": dict(item.fixed_covariates),
        "derivatives": {
            key: value.tolist() for key, value in item.derivatives.items()
        },
    }
