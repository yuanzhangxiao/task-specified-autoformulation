"""Opt-in, public-only fitting qualification of every returned construction.

The campaign freezes one declared start and a common diagnostic prefix. Scores
never select models, alter their equations, or initiate further construction.
Each numerical operation has one durable attempt; interruption cannot renew it.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from collections.abc import Callable
from pathlib import Path
from time import monotonic, process_time
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_verification as verification
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-fit-qualification-1"


class Policy(StrictSchema):
    """Predeclare budgets, information access, and a diagnostic score label."""

    protocol: Literal["construction-fit-qualification-policy-1"] = (
        "construction-fit-qualification-policy-1"
    )
    population: Literal["returned-eight", "synthetic-smoke"] = "returned-eight"
    prefix: PrefixPolicy = Field(default_factory=lambda: PrefixPolicy(duration=8))
    training: kernel.NumericalPolicy = Field(
        default_factory=lambda: kernel.NumericalPolicy(
            maximum_calls=300, seconds=600, verification_seconds=60
        )
    )
    training_execution: execution.SearchExecutionPolicy = Field(
        default_factory=lambda: execution.SearchExecutionPolicy(
            search_overall_seconds=720, verification_overall_seconds=90
        )
    )
    inference: kernel.NumericalPolicy = Field(
        default_factory=lambda: kernel.NumericalPolicy(
            maximum_calls=100, seconds=60, verification_seconds=30
        )
    )
    inference_execution: execution.SearchExecutionPolicy = Field(
        default_factory=lambda: execution.SearchExecutionPolicy(
            search_overall_seconds=90, verification_overall_seconds=60
        )
    )
    forecast_execution: verification.ExecutionPolicy = Field(
        default_factory=lambda: verification.ExecutionPolicy(
            numerical_seconds=60, overall_seconds=90
        )
    )
    reconstruction_execution: verification.ExecutionPolicy = Field(
        default_factory=lambda: verification.ExecutionPolicy(
            numerical_seconds=60, overall_seconds=90
        )
    )
    poor_fit_nmse: FiniteFloat = Field(default=1.0, gt=0)


def _validate(data: dict, policy: Policy) -> list[dict]:
    """Validate complete public splits and retain every packet in stable order."""
    if (
        set(data)
        != {
            "protocol",
            "records",
            "training",
            "validation",
            "source",
            "test_data_opened",
        }
        or data["protocol"] != "construction-fit-inputs-1"
    ):
        raise ValueError("construction fitting input schema differs")
    if data["test_data_opened"] is not False:
        raise ValueError("test data prohibited")
    training = PublicSplit.model_validate(data["training"])
    validation = PublicSplit.model_validate(data["validation"])
    if training.name != "train" or validation.name != "val":
        raise ValueError("training/validation split boundary differs")
    if {r.trajectory_id for r in training.rows} & {
        r.trajectory_id for r in validation.rows
    }:
        raise ValueError("training/validation trajectory overlap")
    if policy.population == "returned-eight" and (
        len(data["records"]) != 8
        or len(training.rows) != 16
        or len(validation.rows) != 4
    ):
        raise ValueError("all eight models and all 16/4 public rows are required")
    for role, allowed in (
        ("inputs", policy.prefix.future_inputs),
        ("auxiliaries", policy.prefix.future_auxiliaries),
    ):
        supplied = data["source"].get(f"permitted_future_{role}")
        if supplied is None and policy.population == "returned-eight":
            raise ValueError("source future forcing authorization is required")
        if supplied is not None and set(allowed) - set(supplied):
            raise ValueError("policy future forcing exceeds source authorization")
    for split in (training, validation):
        if not split.rows:
            raise ValueError("empty public split")
        for row in split.rows:
            if row.time[-1] <= row.time[0] + policy.prefix.duration:
                raise ValueError("common validation suffix must be nonempty")
            kernel.make_prefix(row, policy.prefix)
            kernel.make_forecast_schedule(row, policy.prefix)
    tasks = []
    for task_id, record in sorted(data["records"].items()):
        if not task_id or Path(task_id).name != task_id or task_id in {".", ".."}:
            raise ValueError("invalid task identifier")
        problem = PrefixFitProblem.model_validate(record["problem"])
        if set(policy.prefix.future_inputs) - set(
            problem.context.external_inputs
        ) or set(policy.prefix.future_auxiliaries) - set(problem.context.auxiliaries):
            raise ValueError("policy future forcing exceeds model context")
        roots = sum(
            rule.initial.mode == "value" and rule.initial.scope == "trajectory"
            for rule in problem.graph.rules.values()
        )
        if any(
            set(row.targets) != set(problem.context.targets)
            for split in (training, validation)
            for row in split.rows
        ):
            raise ValueError("model and public targets differ")
        packet_hash = record["source_packet_sha256"]
        if len(packet_hash) != 64 or any(
            c not in "0123456789abcdef" for c in packet_hash
        ):
            raise ValueError("invalid source packet hash")
        tasks.append(
            {
                "task_id": task_id,
                "source_packet_sha256": packet_hash,
                "problem_sha256": public.content_sha256(problem),
                "initializer_kind": "trajectory_roots" if roots else "analytic_map",
                "root_count": roots,
            }
        )
    if policy.population == "returned-eight" and (
        sum(t["root_count"] > 0 for t in tasks) != 5
    ):
        raise ValueError("expected three analytic maps and five root models")
    return tasks


def prepare(root: Path, inputs: dict, policy: Policy) -> dict:
    """Seal source identities, all public rows, policy, and the entire roster."""
    tasks = _validate(inputs, policy)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(inputs),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": tasks,
        "test_data_opened": False,
        "private_data_opened": False,
        "live_llm_calls": 0,
        "automatic_promotion": False,
        "automatic_revision": False,
        "score_based_preselection": False,
        "prefix_selection": "predeclared_diagnostic_not_optimized",
        "parameter_start": "one_existing_declared_start_no_best_fit_selection",
        "normalization": "training_channel_std_floor_1e-12",
        "validation_horizon": "strictly_after_same_elapsed_prefix_for_all_models",
    }
    with public._lock(root):
        seal(root / "inputs.json", inputs)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(tasks)}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Return (plan, inputs); execution checks current code and numerical runtime."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    policy = Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _validate(data, policy)
        or any(
            plan[key] is not False
            for key in (
                "test_data_opened",
                "private_data_opened",
                "automatic_promotion",
                "automatic_revision",
                "score_based_preselection",
            )
        )
        or plan["live_llm_calls"] != 0
    ):
        raise ValueError("construction fitting plan identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("construction fitting source/runtime differs")
    return plan, data


def _read_operation(folder: Path, identity: dict) -> dict:
    """Bind a receipt to its immutable reservation and final progress content."""
    value = read_seal(folder / "result.json")
    if value["identity"] != identity or read_seal(folder / "started.json") != identity:
        raise ValueError("operation identity differs")
    progress = (
        public._read(folder / "progress.json")
        if (folder / "progress.json").exists()
        else {}
    )
    if value["progress_sha256"] != public.content_sha256(progress):
        raise ValueError("operation checkpoint differs")
    return value | {"observed_progress": progress}


def _operation(
    folder: Path,
    identity: dict,
    name: str,
    seconds: float,
    work: Callable[[Callable[[dict], None]], dict],
) -> dict:
    """Execute once and preserve charged interruptions without silent retries."""
    directory = folder / "operations" / name
    bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
    with public._lock(directory):
        start, path = directory / "started.json", directory / "result.json"
        if start.exists() and read_seal(start) != bound:
            raise ValueError("operation reservation differs")
        if path.exists():
            return _read_operation(directory, bound)
        interrupted = None
        progress_path = directory / "progress.json"
        if start.exists():
            result = {
                "status": "interrupted",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": seconds,
                "accounting_complete": False,
            }
        else:
            seal(start, bound)
            wall, cpu = monotonic(), process_time()

            def save(value: dict) -> None:
                public._write(progress_path, value)

            try:
                result = {"status": "complete", "value": work(save)}
            except Exception as error:
                result = {
                    "status": "unavailable",
                    "value": None,
                    "error": f"{type(error).__name__}: {error}"[-2000:],
                }
            except BaseException as error:
                interrupted = error
                receipt = getattr(error, "execution_receipt", None) or getattr(
                    error, "verification_receipt", None
                )
                if receipt is not None:
                    save(receipt)
                result = {"status": "interrupted", "value": receipt}
            elapsed = monotonic() - wall
            result.update(
                wall_seconds=elapsed,
                cpu_seconds=process_time() - cpu,
                budget_charge_seconds=elapsed,
                accounting_complete=True,
            )
        progress = public._read(progress_path) if progress_path.exists() else {}
        result.update(
            identity=bound,
            budget_restarted=False,
            progress_sha256=public.content_sha256(progress),
        )
        seal(path, public._jsonable(result))
        if interrupted is not None:
            raise interrupted
        return _read_operation(directory, bound)


def _reconstruct(
    problem: PrefixFitProblem,
    training: PublicSplit,
    fit: dict,
    policy: Policy,
    checkpoint: Callable[[dict], None],
) -> dict:
    """Independently replay frozen training coordinates for per-target errors."""
    budget = verification._Budget(policy.reconstruction_execution)
    oracle = kernel.TrainingOracle(problem, training, policy.training)
    vector = oracle.vector(fit["parameters"])
    result = verification._base_result(policy.reconstruction_execution) | {
        "fit_success": False,
        "no_search_performed": True,
        "fit_sha256": public.content_sha256(fit),
    }

    def evaluate(method: str, deadline: float) -> np.ndarray:
        return oracle.evaluate(vector, deadline, jacobian=False, method=method)[0]

    def qualify(solutions: list[np.ndarray]) -> None:
        difference = float(np.max(abs(solutions[0] - solutions[1]), initial=0))
        channels = tuple(oracle.system.channels)
        squared = solutions[0].reshape(-1, len(channels)) ** 2
        suffix = np.concatenate(
            [
                np.asarray(row.time) > row.time[0] + policy.prefix.duration
                for row in training.rows
            ]
        )
        late = squared[suffix]
        result.update(
            fit_success=difference <= policy.training.solver_tolerance,
            solver_difference=difference,
            normalized_mse=float(np.mean(squared)),
            per_target_normalized_mse=dict(
                zip(channels, np.mean(squared, axis=0).tolist(), strict=True)
            ),
            sample_count=len(squared),
            objective_difference=abs(float(np.mean(squared)) - fit["normalized_mse"]),
            suffix_score={
                "normalized_mse": float(np.mean(late)),
                "per_target_normalized_mse": dict(
                    zip(channels, np.mean(late, axis=0).tolist(), strict=True)
                ),
                "sample_count": len(late),
                "scoring_rule": "strictly_after_prefix_cutoff",
                "conditioning": "training_reconstruction_roots_fitted_on_full_training",
            },
        )
        if not result["fit_success"]:
            result["verification_error"] = (
                "independent reconstruction integrators disagree"
            )

    return verification._run(result, budget, checkpoint, evaluate, qualify)


def _state(operation: dict | None, success: str, threshold: float) -> str:
    """Separate availability, numerical failure, and a diagnostic poor-fit label."""
    if operation is None:
        return "not_attempted"
    if operation["status"] != "complete":
        return operation["status"]
    value = operation.get("value") or {}
    if value.get(success):
        score = value.get("normalized_mse")
        return "poor_fit" if score is not None and score > threshold else "available"
    if any(
        (value.get(k) or {}).get("timeout_reason")
        for k in ("timing", "search_timing", "verification_timing")
    ):
        return "budget_exhausted"
    if value.get("verification_calls_failed") or (
        value.get("search_calls_failed") and not value.get("search", {}).get("best")
    ):
        return "integration_failure"
    if value.get("verification_error"):
        return "verification_failure"
    return "unavailable"


def _cost(operations: dict) -> dict:
    """Report actual costs, nested callback costs, and unknown interrupted totals."""
    result = {
        "operations": len(operations),
        "accounting_complete": all(
            v["accounting_complete"] and v["status"] == "complete"
            for v in operations.values()
        ),
        "timing_accounting_complete": all(
            v["accounting_complete"] for v in operations.values()
        ),
        "budget_charge_seconds": sum(
            v["budget_charge_seconds"] for v in operations.values()
        ),
        "scope": "Operation wall/CPU includes setup and progress writes; excludes "
        "final receipt sealing, reports, and scheduler wait. Callback and numerical "
        "times are nested subcategories, not additional charges. Calls count whole "
        "oracle evaluations, not individual trajectories or RHS evaluations.",
    }
    for field in ("wall_seconds", "cpu_seconds"):
        values = [v.get(field) for v in operations.values()]
        lower_bounds = []
        for operation, value in zip(operations.values(), values, strict=True):
            progress = operation.get("observed_progress") or {}
            timing = progress.get("overall_timing") or progress.get("timing") or {}
            lower_bounds.append(
                value if value is not None else timing.get("elapsed_" + field, 0)
            )
        result[field + "_observed"] = sum(lower_bounds)
        result[field + "_total"] = (
            sum(values) if all(v is not None for v in values) else None
        )
    for field in (
        "search_calls_attempted",
        "search_calls_started",
        "search_calls_completed",
        "search_calls_failed",
        "verification_calls_attempted",
        "verification_calls_started",
        "verification_calls_completed",
        "verification_calls_failed",
    ):
        values, observed = [], []
        for operation in operations.values():
            value = operation.get("value") or operation.get("observed_progress") or {}
            observed.append(value.get(field, 0))
            values.append(
                value.get(field, 0) if operation["status"] == "complete" else None
            )
        result[field + "_observed"] = sum(observed)
        result[field + "_total"] = (
            sum(values) if all(v is not None for v in values) else None
        )
    for field in (
        "callback_wall_seconds",
        "callback_cpu_seconds",
        "numerical_active_wall_seconds",
    ):
        total = 0.0
        complete = True
        for operation in operations.values():
            value = operation.get("value") or operation.get("observed_progress") or {}
            total += sum(
                (value.get(k) or {}).get(field, 0)
                for k in ("search_timing", "verification_timing", "timing")
                if k != "timing" or "verification_timing" not in value
            )
            complete &= operation["status"] == "complete"
        result[field + "_observed"] = total
        result[field + "_total"] = total if complete else None
    return result


def _assemble(identity: dict, data: dict, operations: dict, policy: Policy) -> dict:
    """Recompute reporting from receipts without selecting or modifying models."""
    fit = operations.get("training", {}).get("value") or {}
    reconstruction = operations.get("training_reconstruction", {}).get("value") or {}
    rows, scores = [], []
    for index, row in enumerate(PublicSplit.model_validate(data["validation"]).rows):
        inference = operations.get(f"val_{index}_inference")
        forecast = operations.get(f"val_{index}_forecast")
        prediction = (forecast or {}).get("value") or {}
        score = None
        if (
            forecast
            and forecast["status"] == "complete"
            and prediction.get("forecast_success")
        ):
            score = kernel.score_forecast(
                prediction,
                row,
                row.time[0] + policy.prefix.duration,
                fit["training_scales"],
            )
            scores.append(score)
        rows.append(
            {
                "trajectory_id": row.trajectory_id,
                "inference_status": _state(
                    inference, "fit_success", policy.poor_fit_nmse
                ),
                "forecast_status": _state(
                    forecast, "forecast_success", policy.poor_fit_nmse
                ),
                "prefix_rank": ((inference or {}).get("value") or {}).get(
                    "rank_diagnostic"
                ),
                "score": score,
            }
        )
    aggregate = None
    if len(scores) == len(rows):
        count = sum(s["sample_count"] for s in scores)
        channels = sorted(scores[0]["per_target_normalized_mse"])
        per_target = {
            channel: sum(
                s["sample_count"] * s["per_target_normalized_mse"][channel]
                for s in scores
            )
            / count
            for channel in channels
        }
        aggregate = {
            "normalized_mse": float(np.mean(list(per_target.values()))),
            "per_target_normalized_mse": per_target,
            "sample_count": count,
            "aggregation": "sample_weighted_per_target_then_equal_target_mean",
        }
    return {
        "protocol": PROTOCOL,
        "identity": identity,
        "operations": operations,
        "training_status": _state(
            operations.get("training"), "fit_success", policy.poor_fit_nmse
        ),
        "training_reconstruction_status": _state(
            operations.get("training_reconstruction"),
            "fit_success",
            policy.poor_fit_nmse,
        ),
        "training_score": {
            key: reconstruction.get(key)
            for key in (
                "normalized_mse",
                "per_target_normalized_mse",
                "sample_count",
                "objective_difference",
            )
        }
        if reconstruction.get("fit_success")
        else None,
        "training_suffix_score": reconstruction.get("suffix_score")
        if reconstruction.get("fit_success")
        else None,
        "training_fit_normalized_mse": fit.get("normalized_mse"),
        "training_rank": fit.get("rank_diagnostic"),
        "validation_rows": rows,
        "validation_available_rows": len(scores),
        "validation_score": aggregate,
        "validation_status": "unavailable"
        if aggregate is None
        else (
            "poor_fit"
            if aggregate["normalized_mse"] > policy.poor_fit_nmse
            else "available"
        ),
        "cost": _cost(operations),
        "model_retained": True,
        "automatic_promotion": False,
        "automatic_revision": False,
    }


def _completed(folder: Path, identity: dict, data: dict, policy: Policy) -> dict:
    """Verify source receipts and deterministically reconstruct a completed report."""
    saved = read_seal(folder / "result.json")
    if saved["identity"] != identity:
        raise ValueError("task identity differs")
    operations = _journal_snapshot(folder, identity, data, policy)
    if saved != public._jsonable(_assemble(identity, data, operations, policy)):
        raise ValueError("task report differs from retained numerical receipts")
    return saved


def _journal_snapshot(
    folder: Path,
    identity: dict,
    data: dict,
    policy: Policy,
) -> dict:
    """Read complete or unfinished operations without resetting their reservations."""
    allowances = {
        "training": policy.training_execution.search_overall_seconds
        + policy.training_execution.verification_overall_seconds,
        "training_reconstruction": policy.reconstruction_execution.overall_seconds,
    }
    for index in range(len(data["validation"]["rows"])):
        allowances[f"val_{index}_inference"] = (
            policy.inference_execution.search_overall_seconds
            + policy.inference_execution.verification_overall_seconds
        )
        allowances[f"val_{index}_forecast"] = policy.forecast_execution.overall_seconds
    operations = {}
    directory = folder / "operations"
    if not directory.exists():
        return operations
    for entry in sorted(directory.iterdir()):
        if not entry.is_dir():
            continue
        if entry.is_symlink() or entry.name not in allowances:
            raise ValueError("unexpected operation journal")
        start = entry / "started.json"
        if not start.exists():
            # A process may hold the directory lock before reserving its work.
            if (entry / "result.json").exists() or (entry / "progress.json").exists():
                raise ValueError("operation receipt lacks a reservation")
            continue
        bound = {
            "campaign": identity,
            "stage": entry.name,
            "allowance_seconds": allowances[entry.name],
        }
        if read_seal(start) != bound:
            raise ValueError("task operation identity differs")
        if (entry / "result.json").exists():
            operations[entry.name] = _read_operation(entry, bound)
        else:
            progress = (
                public._read(entry / "progress.json")
                if (entry / "progress.json").exists()
                else {}
            )
            operations[entry.name] = {
                "identity": bound,
                "status": "running_or_interrupted",
                "value": None,
                "observed_progress": progress,
                "progress_sha256": public.content_sha256(progress),
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": allowances[entry.name],
                "accounting_complete": False,
                "budget_restarted": False,
            }
    return operations


def run_task(root: Path, index: int) -> dict:
    """Fit one frozen model, then infer validation roots without suffix targets."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown construction fitting task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    policy = Policy.model_validate(plan["policy"])
    problem = PrefixFitProblem.model_validate(
        data["records"][task["task_id"]]["problem"]
    )
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return _completed(folder, identity, data, policy)
        operations = {}

        def perform(name: str, seconds: float, work: Callable) -> dict | None:
            operations[name] = _operation(folder, identity, name, seconds, work)
            operation = operations[name]
            return operation["value"] if operation["status"] == "complete" else None

        training = PublicSplit.model_validate(data["training"])
        fit = perform(
            "training",
            policy.training_execution.search_overall_seconds
            + policy.training_execution.verification_overall_seconds,
            lambda save: execution.fit_training(
                problem, training, policy.training, policy.training_execution, save
            ),
        )
        if fit and fit.get("fit_success"):
            perform(
                "training_reconstruction",
                policy.reconstruction_execution.overall_seconds,
                lambda save: _reconstruct(problem, training, fit, policy, save),
            )
            for row_index, row in enumerate(
                PublicSplit.model_validate(data["validation"]).rows
            ):
                prefix = kernel.make_prefix(row, policy.prefix)
                inference = perform(
                    f"val_{row_index}_inference",
                    policy.inference_execution.search_overall_seconds
                    + policy.inference_execution.verification_overall_seconds,
                    lambda save, prefix=prefix: execution.infer_prefix(
                        problem,
                        fit,
                        prefix,
                        policy.inference,
                        policy.inference_execution,
                        save,
                    ),
                )
                if inference and inference.get("fit_success"):
                    schedule = kernel.make_forecast_schedule(row, policy.prefix)
                    perform(
                        f"val_{row_index}_forecast",
                        policy.forecast_execution.overall_seconds,
                        lambda save,
                        inference=inference,
                        schedule=schedule: verification.forecast(
                            problem,
                            fit,
                            inference,
                            schedule,
                            policy.inference,
                            policy.forecast_execution,
                            save,
                        ),
                    )
        result = public._jsonable(_assemble(identity, data, operations, policy))
        seal(folder / "result.json", result)
        return result


def report(root: Path) -> dict:
    """Show availability and descriptive ranks for every task, including failures."""
    plan, data = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    records, results, completed = [], [], 0
    for task in plan["tasks"]:
        folder = root / "results" / task["task_id"]
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        result = (
            _completed(folder, identity, data, policy)
            if (folder / "result.json").exists()
            else None
        )
        terminal = result is not None
        completed += int(terminal)
        if result is None:
            operations = _journal_snapshot(folder, identity, data, policy)
            if operations:
                result = _assemble(identity, data, operations, policy)
        if result:
            results.append(result)
        records.append(
            task
            | {
                "status": "complete"
                if terminal
                else "partial"
                if result
                else "pending",
                "training_status": result["training_status"]
                if result
                else "not_attempted",
                "training_score": result["training_score"] if result else None,
                "training_suffix_score": result["training_suffix_score"]
                if result
                else None,
                "training_fit_normalized_mse": result["training_fit_normalized_mse"]
                if result
                else None,
                "validation_status": result["validation_status"]
                if result
                else "not_attempted",
                "validation_score": result["validation_score"] if result else None,
                "validation_available_rows": result["validation_available_rows"]
                if result
                else 0,
                "training_rank": result["training_rank"] if result else None,
                "model_retained": True,
            }
        )
    for key in ("training", "training_suffix", "validation"):
        values = sorted(
            {r[f"{key}_score"]["normalized_mse"] for r in records if r[f"{key}_score"]}
        )
        for record in records:
            score = record[f"{key}_score"]
            record[f"{key}_descriptive_rank"] = (
                1 + values.index(score["normalized_mse"]) if score else None
            )
    operations = {
        f"{i}:{name}": op
        for i, result in enumerate(results)
        for name, op in result["operations"].items()
    }
    output = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "records": records,
        "completed_tasks": completed,
        "model_count": len(records),
        "cost": _cost(operations),
        "nmse_distribution": {
            key: _distribution(records, key)
            for key in ("training", "training_suffix", "validation")
        },
        "prefix": policy.prefix.model_dump(mode="json"),
        "prefix_selection": "predeclared_diagnostic_not_optimized",
        "ranking_scope": "descriptive_only_available_complete_scores_no_selection",
        "automatic_promotion": False,
        "automatic_revision": False,
        "test_data_opened": False,
        "private_data_opened": False,
        "live_llm_calls": 0,
        "limitations": [
            "One start and bounded search do not establish global fit optimality.",
            "The fixed prefix is diagnostic; its duration is not optimized.",
            "Numerical agreement certifies neither science nor identifiability.",
            "Original parameter redundancies remain; rank diagnostics are conditional.",
            "Incomplete validation has no aggregate rank; all models remain visible.",
            "Cooperative deadlines cannot preempt symbolic setup or callbacks.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "report" / "summary.json", output)
        (root / "SUMMARY.md").write_text(_markdown(output))
    return output


def _distribution(records: list[dict], key: str) -> dict:
    """Summarize available models with median and unscaled median absolute deviation."""
    scores = [record[f"{key}_score"] for record in records if record[f"{key}_score"]]

    def summarize(values: list[float]) -> dict:
        median = float(np.median(values)) if values else None
        return {
            "median": median,
            "mad": float(np.median(np.abs(np.asarray(values) - median)))
            if values
            else None,
        }

    channels = sorted({name for s in scores for name in s["per_target_normalized_mse"]})
    return {
        "available_models": len(scores),
        "total_models": len(records),
        "unavailable_models": len(records) - len(scores),
        "mad_definition": "median_absolute_deviation_from_median_unscaled",
        "aggregate": summarize([s["normalized_mse"] for s in scores]),
        "per_target": {
            channel: summarize(
                [s["per_target_normalized_mse"][channel] for s in scores]
            )
            for channel in channels
        },
    }


def _markdown(summary: dict) -> str:
    """Render all models, channel scores, availability, and exact measured costs."""

    def number(value: float | None) -> str:
        return "unavailable" if value is None else f"{value:.6g}"

    def score(record: dict, key: str) -> str:
        return number((record.get(key + "_score") or {}).get("normalized_mse"))

    lines = [
        "# Construction fitting qualification",
        "",
        f"Plan: `{summary['plan_sha256']}`. Completed "
        f"{summary['completed_tasks']}/{summary['model_count']} tasks.",
        "",
        f"The prefix is a predeclared diagnostic duration of "
        f"{summary['prefix']['duration']:g} time units. All suffix scores use "
        "strictly later samples. Training reconstruction uses roots fitted on full "
        "training trajectories; validation roots use only their permitted prefix. "
        "Normalization uses training channel standard deviations with a 1e-12 floor. "
        "No score selects, removes, promotes, or revises a model.",
        "",
        "| Model | Initializer | Train status | Train full NMSE | Train suffix NMSE | "
        "Validation status | Validation suffix NMSE | Available rows | "
        "Validation rank |",
        "|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for index, record in enumerate(summary["records"], 1):
        lines.append(
            f"| {index}: `{record['task_id']}` | {record['initializer_kind']} | "
            f"{record['training_status']} | {score(record, 'training')} | "
            f"{score(record, 'training_suffix')} | {record['validation_status']} | "
            f"{score(record, 'validation')} | {record['validation_available_rows']} | "
            f"{record['validation_descriptive_rank'] or 'unavailable'} |"
        )
    lines += [
        "",
        "## Median and MAD across models",
        "",
        "MAD is the unscaled median absolute deviation; unavailable models "
        "have no substituted score.",
        "",
        "| Score | Available / total models | Median NMSE | MAD |",
        "|---|---:|---:|---:|",
    ]
    for key, value in summary["nmse_distribution"].items():
        for channel, metric in {
            "aggregate": value["aggregate"],
            **value["per_target"],
        }.items():
            lines.append(
                f"| {key}: {channel} | "
                f"{value['available_models']}/{value['total_models']} | "
                f"{number(metric['median'])} | {number(metric['mad'])} |"
            )
    lines += [
        "",
        "## Per-target NMSE",
        "",
        "| Model | Score | Target | NMSE |",
        "|---|---|---|---:|",
    ]
    for index, record in enumerate(summary["records"], 1):
        for key in ("training", "training_suffix", "validation"):
            values = (record.get(key + "_score") or {}).get("per_target_normalized_mse")
            for channel, value in (values or {"all": None}).items():
                lines.append(f"| {index} | {key} | {channel} | {number(value)} |")
    cost = summary["cost"]
    lines += [
        "",
        "## Cost and limits",
        "",
        cost["scope"],
        "",
        f"Accounting complete: {cost['accounting_complete']}. "
        f"Operations: {cost['operations']}. Budget charge: "
        f"{number(cost['budget_charge_seconds'])} seconds.",
        "",
        "| Quantity | Exact total | Observed lower bound |",
        "|---|---:|---:|",
    ]
    for key in (
        "wall_seconds",
        "cpu_seconds",
        "search_calls_started",
        "verification_calls_started",
        "callback_wall_seconds",
        "numerical_active_wall_seconds",
    ):
        lines.append(
            f"| {key} | {number(cost[key + '_total'])} | "
            f"{number(cost[key + '_observed'])} |"
        )
    lines += [
        "",
        *[f"- {item}" for item in summary["limitations"]],
        "",
        "No test/private data or live LLM calls were used. "
        "Detailed numerical receipts and adverse outcomes remain in the results.",
        "",
    ]
    return "\n".join(lines)


def _archive(root: Path) -> Path:
    """Package allowlisted regular artifacts deterministically, excluding symlinks."""
    paths = [
        root / name
        for name in (
            "plan.json",
            "inputs.json",
            "SUMMARY.md",
            "launcher_execution.json",
        )
    ]
    for name in ("results", "report", "submissions", "logs", "runtime"):
        directory = root / name
        if directory.is_symlink():
            raise ValueError("inspection cannot include a symlink directory")
        paths.extend(path for path in directory.rglob("*") if path.is_file())
    entries = []
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("inspection cannot include external files or symlinks")
        if path.is_file() and not path.name.startswith("."):
            entries.append(path)
    destination = root / "inspection.tar.gz"
    temporary = root / "inspection.tar.gz.tmp"
    with (
        temporary.open("wb") as stream,
        gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in sorted(entries):
            value = path.read_bytes()
            info = tarfile.TarInfo(path.relative_to(root).as_posix())
            info.size, info.mode, info.mtime = len(value), 0o644, 0
            archive.addfile(info, io.BytesIO(value))
    temporary.replace(destination)
    return destination


def inspect(root: Path) -> dict:
    """Inspect the frozen plan and launcher artifacts without starting work."""
    plan, data = verify(root, runtime=False)
    paths = [root / "launcher_execution.json"]
    paths += [
        p
        for name in ("submissions", "logs", "runtime")
        for p in (root / name).rglob("*")
        if p.is_file()
    ]
    summary = report(root)
    archive = _archive(root)
    return {
        "plan": plan,
        "source": data["source"],
        "launcher_artifacts": [str(p.relative_to(root)) for p in paths if p.exists()],
        "report": summary,
        "inspection_archive": str(archive.resolve()),
    }


def smoke(root: Path) -> dict:
    """Exercise analytic-map and trajectory-root paths on tiny synthetic controls."""
    from autoformalism.fitting.initialization_qualification import fixture

    examples = {name: fixture(name, 0) for name in ("shared_root", "trajectory_root")}
    first = examples["shared_root"]
    # Both models see exactly the same observations, irrespective of fit quality.
    inputs = {
        "protocol": "construction-fit-inputs-1",
        "records": {
            name: {
                "problem": record["problem"],
                "source_packet_sha256": public.content_sha256(record["problem"]),
                "source_model_id": name,
                "proposal_seed": 0,
                "metadata": {"synthetic_control": True},
            }
            for name, record in examples.items()
        },
        "training": first["training"],
        "validation": first["validation"],
        "source": {"kind": "synthetic-smoke-no-benchmark-data"},
        "test_data_opened": False,
    }
    policy = Policy(
        population="synthetic-smoke",
        prefix=PrefixPolicy(duration=1),
        training=kernel.NumericalPolicy(maximum_calls=60, seconds=20),
        inference=kernel.NumericalPolicy(maximum_calls=40, seconds=10),
    )
    prepare(root, inputs, policy)
    for index in range(2):
        run_task(root, index)
    return report(root)
