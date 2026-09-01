"""Leakage-safe D3-native-no-tools workflow."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from autoformalism.baselines.d3_native import (
    NativeD3Error,
    evaluate_native_d3,
    fit_native_d3,
    validate_native_candidate,
)
from autoformalism.baselines.models import (
    BaselineConfig,
    BaselineDevelopmentResult,
    BaselineResult,
)
from autoformalism.data import DatasetSplit, DevelopmentDataset
from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.llm import LLMClient
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.schemas import CandidateModel

D3_UPSTREAM_REVISION = "ee86212dfd5935bb0c9626eaa0570223ff7ecf1c"


class D3AdapterError(RuntimeError):
    """Raised when the external D3 bridge violates the exchange contract."""


def run_d3_native_no_tools(
    config: BaselineConfig,
    dataset: DevelopmentDataset,
    test_loader,
    context: ValidationContext,
    *,
    task_prompt: str,
    work_directory: Path,
    llm_client: LLMClient | None = None,
) -> BaselineResult:
    """Run native-fit D3 with external feature-acquisition tools disabled."""
    if llm_client is None:
        raise D3AdapterError("D3-native-no-tools requires --model")
    return _run_native_d3(
        config,
        dataset,
        test_loader,
        context,
        task_prompt=task_prompt,
        llm_client=llm_client,
        work_directory=work_directory,
    )


def run_d3_native_no_tools_development(
    config: BaselineConfig,
    dataset: DevelopmentDataset,
    context: ValidationContext,
    *,
    task_prompt: str,
    work_directory: Path,
    llm_client: LLMClient | None = None,
) -> BaselineDevelopmentResult:
    """Run D3 selection using train/validation while leaving test sealed."""
    if llm_client is None:
        raise D3AdapterError("D3-native-no-tools requires --model")
    result = _run_native_d3(
        config,
        dataset,
        None,
        context,
        task_prompt=task_prompt,
        llm_client=llm_client,
        work_directory=work_directory,
    )
    assert isinstance(result, BaselineDevelopmentResult)
    return result

def _run_native_d3(
    config: BaselineConfig,
    dataset: DevelopmentDataset,
    test_loader: Callable[[], DatasetSplit] | None,
    context: ValidationContext,
    *,
    task_prompt: str,
    llm_client: LLMClient,
    work_directory: Path,
) -> BaselineResult | BaselineDevelopmentResult:
    """Run D3's propose-fit-reflect loop with native Adam/Euler fitting."""
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
    observed_channels = (*context.targets, *context.auxiliaries)
    available_inputs = _numeric_available_inputs(dataset, context)

    for generation in range(start, config.d3_generations):
        feedback = _d3_feedback(records)
        try:
            proposed = llm_client.propose(
                system_prompt=_d3_system_prompt(
                    task_prompt, context, available_inputs=available_inputs
                ),
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
            candidate = proposed
            validate_native_candidate(
                candidate, observed_channels, available_inputs
            )
            fitted = fit_native_d3(
                candidate,
                dataset.train,
                dataset.validation,
                targets=context.targets,
                seed=config.seed + generation,
            )
            validation_mse = fitted.validation_mse
            record = {
                "generation": generation,
                "candidate": candidate.model_dump(mode="json"),
                "parameters": dict(fitted.parameters),
                "training_mse": fitted.training_mse,
                "validation_mse": validation_mse,
                "epochs_completed": fitted.epochs_completed,
                "target_scales": dict(fitted.target_scales),
                "error": None,
            }
            if validation_mse < best_validation - 1e-12:
                best_validation = validation_mse
                stagnant = 0
            else:
                stagnant += 1
        except (
            ArithmeticError,
            D3AdapterError,
            NativeD3Error,
            LLMProviderError,
            LLMResponseError,
            ModelValidationError,
            RuntimeError,
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
    equations = {
        equation.state: equation.rhs for equation in candidate.state_equations
    }
    hyperparameters: dict[str, float | int | str] = {
        "generations_completed": len(records),
        "selected_generation": int(selected["generation"]),
        "external_tools_enabled": "false",
        "adaptation": "native_adam_teacher_forced_euler",
        "upstream_revision": D3_UPSTREAM_REVISION,
        "selected_parameters": json.dumps(
            dict(selected["parameters"]), sort_keys=True
        ),
        "parameter_fitting": "pytorch_adam",
        "state_update": "teacher_forced_forward_euler",
        "learning_rate": 1e-2,
        "maximum_epochs": 2_000,
        "validation_interval": 10,
        "early_stopping_patience_checks": 100,
        "optimizer_device": "cpu",
        "parameter_initialization": "midpoint_of_declared_initialization_range",
        "modeled_observed_channels": json.dumps(observed_channels),
        "selected_epochs_completed": int(selected["epochs_completed"]),
    }
    if test_loader is None:
        return BaselineDevelopmentResult(
            method=config.method,
            benchmark_id=dataset.benchmark_id,
            tier=dataset.tier,
            seed=config.seed,
            equations=equations,
            selected_hyperparameters=hyperparameters,
            selection_payload={
                "candidate": candidate.model_dump(mode="json"),
                "parameters": dict(selected["parameters"]),
                "target_scales": dict(selected["target_scales"]),
                "selected_generation": int(selected["generation"]),
            },
            training_normalized_mse=float(selected["training_mse"]),
            validation_normalized_mse=float(selected["validation_mse"]),
            test_data_opened=False,
        )
    test_mse, test_per_target = evaluate_native_d3(
        candidate,
        test_loader(),
        dict(selected["parameters"]),
        context.targets,
        dict(selected["target_scales"]),
    )
    return BaselineResult(
        method=config.method,
        benchmark_id=dataset.benchmark_id,
        tier=dataset.tier,
        seed=config.seed,
        equations=equations,
        selected_hyperparameters=hyperparameters,
        training_normalized_mse=float(selected["training_mse"]),
        validation_normalized_mse=float(selected["validation_mse"]),
        test_normalized_mse=test_mse,
        test_per_target_normalized_mse=test_per_target,
    )


def _d3_system_prompt(
    task_prompt: str,
    context: ValidationContext,
    *,
    available_inputs: tuple[str, ...] | None = None,
) -> str:
    observed_states = (*context.targets, *context.auxiliaries)
    scalar_inputs = (
        (*context.external_inputs, *context.fixed_covariates)
        if available_inputs is None
        else available_inputs
    )
    scalar_fixed_covariates = tuple(
        name for name in scalar_inputs if name not in context.external_inputs
    )
    scalar_external_inputs = tuple(
        name for name in scalar_inputs if name in context.external_inputs
    )
    available = tuple(
        dict.fromkeys(
            (
                *context.targets,
                *context.auxiliaries,
                *scalar_inputs,
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
        "enabled through the structured response. The native D3 simulator has a "
        "fixed state vector consisting of these observed channels: "
        f"{observed_states}. Return one change equation for every listed channel. "
        "A channel need not appear in other equations when the proposed model does "
        "not use it. Additional named quantities may be safe algebraic features "
        "computed from available states, inputs, covariates, and global parameters. "
        "Use only symbols explicitly available in the task.\n\n"
        f"Benchmark task:\n{task_prompt}\n\n"
        f"Exact scored target names: {context.targets}. Additional observed dynamic "
        f"state names with supplied derivative labels: {context.auxiliaries}. "
        f"Exact numeric external input names: "
        f"{scalar_external_inputs}. Exact scalar fixed covariate names: "
        f"{scalar_fixed_covariates}. The complete allowed data-symbol list is: "
        f"{available}. Do not invent aliases for these names. In particular, these "
        f"meal aliases are unavailable in this tier and must not appear: "
        f"{unavailable_meal_aliases}. Give one state equation for every listed "
        "observed state. Observation mappings need only cover scored targets. "
        "Parameters must be global. Initial values are teacher-forced from measured "
        "states and are not fitted. If uncertain, return simple equations using only "
        "the listed symbols and global parameters."
    )


def _numeric_available_inputs(
    dataset: DevelopmentDataset, context: ValidationContext
) -> tuple[str, ...]:
    """Return external/fixed inputs usable numerically by native D3."""
    trajectories = (*dataset.train.trajectories, *dataset.validation.trajectories)
    numeric: list[str] = []
    for name in (*context.external_inputs, *context.fixed_covariates):
        valid = True
        for trajectory in trajectories:
            raw_value: object
            if name in trajectory.external_inputs:
                raw_value = trajectory.external_inputs[name]
            elif name in trajectory.fixed_covariates:
                raw_value = trajectory.fixed_covariates[name]
            else:
                valid = False
                break
            try:
                values = np.asarray(raw_value, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                valid = False
                break
            if not len(values) or not np.isfinite(values).all():
                valid = False
                break
        if valid:
            numeric.append(name)
    return tuple(numeric)


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
        "adapter_revision": "5-native-interface-wording",
        "train": dataset.train.fingerprint,
        "validation": dataset.validation.fingerprint,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
