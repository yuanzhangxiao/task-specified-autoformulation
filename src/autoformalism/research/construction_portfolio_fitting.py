"""Training-only qualification of a complete, authenticated construction portfolio.

Each stage has one durable reservation. Restart resumes unattempted stages and
reuses authenticated receipts; it never renews an interrupted search allowance.
All model identities remain visible, including failures. Prefix forecasts infer
new roots from prefixes after shared parameters are frozen.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_continuation as coordinates
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_portfolio as numerical
from autoformalism.fitting import prefix_verification as forecasting
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_portfolio_intake as intake
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-portfolio-fitting-1"


class Policy(StrictSchema):
    """Common bounded numerical work, with no score-selected starts or models."""

    population: Literal["scaled-readout-23", "synthetic-smoke"] = "scaled-readout-23"
    qualification_seconds: FiniteFloat = Field(default=180, gt=0, le=180)
    search_seconds: FiniteFloat = Field(default=900, gt=0, le=900)
    verification_seconds: FiniteFloat = Field(default=180, gt=0, le=180)
    diagnostic_seconds: FiniteFloat = Field(default=180, gt=0, le=180)
    maximum_calls: int = Field(default=1000, ge=1, le=1000)
    numerical: kernel.NumericalPolicy = Field(default_factory=kernel.NumericalPolicy)
    inference: kernel.NumericalPolicy = Field(
        default_factory=lambda: kernel.NumericalPolicy(
            seconds=30, maximum_calls=100, verification_seconds=15
        )
    )
    inference_execution: execution.SearchExecutionPolicy = Field(
        default_factory=lambda: execution.SearchExecutionPolicy(
            search_overall_seconds=60, verification_overall_seconds=45
        )
    )
    forecast_execution: forecasting.ExecutionPolicy = Field(
        default_factory=lambda: forecasting.ExecutionPolicy(
            numerical_seconds=60, overall_seconds=90
        )
    )


def _tasks(data: dict, policy: Policy) -> list[dict]:
    count = 23 if policy.population == "scaled-readout-23" else len(data["models"])
    intake.validate_inputs(data, expected_count=count)
    return [
        {
            "index": index,
            "task_id": model["candidate_id"],
            "candidate_id": model["candidate_id"],
            "problem_sha256": model["fitting_problem_sha256"],
        }
        for index, model in enumerate(data["models"])
    ]


def prepare(root: Path, source: Path, policy: Policy) -> dict:
    """Authenticate the delivery once, then freeze a compact portable intake."""
    data = intake.load_delivery(source)
    return prepare_inputs(root, data, policy)


def prepare_inputs(root: Path, data: dict, policy: Policy) -> dict:
    """Seal validated inputs; synthetic fixtures use the same campaign boundary."""
    tasks = _tasks(data, policy)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": tasks,
        "prefix": data["prefix"],
        "start_policy": "original_declared_start_only",
        "normalization": "training_channel_std_floor_1e-12",
        "validation_status": "not_bundled_matching_frozen_split_required",
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
    }
    if reservation_ceiling(plan) > 6000:
        raise ValueError("campaign reservations exceed the two-hour worker envelope")
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(tasks)}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Authenticate data and task identities; workers additionally pin runtime."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    policy = Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data, policy)
        or plan["prefix"] != data["prefix"]
        or plan["test_data_opened"] is not False
        or plan["automatic_revision"] is not False
        or plan["automatic_promotion"] is not False
        or plan["live_llm_calls"] != 0
    ):
        raise ValueError("portfolio plan identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("portfolio source/runtime differs")
    return plan, data


def _allowances(policy: Policy, rows: int) -> dict[str, float]:
    return {
        "setup": 120,
        "start_verification": policy.verification_seconds + 60,
        "qualification": policy.qualification_seconds + 60,
        "search": policy.search_seconds + 120,
        "endpoint_verification": policy.verification_seconds + 60,
        "diagnosis": policy.diagnostic_seconds + 60,
    } | {
        f"train_{i:02d}_{stage}": seconds
        for i in range(rows)
        for stage, seconds in (
            (
                "inference",
                policy.inference_execution.search_overall_seconds
                + policy.inference_execution.verification_overall_seconds,
            ),
            ("forecast", policy.forecast_execution.overall_seconds),
        )
    }


def reservation_ceiling(plan: dict, rows: int = 16) -> float:
    """Worst-case inclusive operation reservations for one model, not an ETA."""
    return sum(_allowances(Policy.model_validate(plan["policy"]), rows).values())


def _value(operation: dict | None, *, partial: bool = False) -> dict:
    if not operation:
        return {}
    if operation["status"] == "complete":
        return operation.get("value") or {}
    return operation.get("observed_progress") or {} if partial else {}


def _verified(operation: dict | None) -> bool:
    value = _value(operation)
    return bool(
        value.get("original_equations_verified")
        and value.get("status", "complete") == "complete"
        and np.isfinite(value.get("training_nmse", np.nan))
    )


def _retained(operations: dict) -> tuple[str | None, dict]:
    eligible = [
        (name, _value(operations.get(name)))
        for name in ("start_verification", "endpoint_verification")
        if _verified(operations.get(name))
    ]
    return (
        min(eligible, key=lambda pair: pair[1]["training_nmse"])
        if eligible
        else (None, {})
    )


def _fit_record(problem, training, original, endpoint) -> dict:
    """Freeze a verified original-coordinate fit for prefix-only inference."""
    parameters = endpoint["parameters"]
    original.vector(parameters)
    return {
        "protocol": "prefix-training-fit-1",
        "problem_sha256": public.content_sha256(problem),
        "training_sha256": public.content_sha256(training),
        "fit_success": True,
        "parameters": deepcopy(parameters),
        "shared_parameters": {name: parameters[name] for name in original.shared},
        "training_scales": dict(original.scales),
        "normalized_mse": endpoint["training_nmse"],
        "verification_sha256": public.content_sha256(endpoint),
        "convergence_claimed": False,
        "selection_data": "train_only",
    }


def _operations(folder: Path, identity: dict, policy: Policy, rows: int) -> dict:
    """Read reservations and completed receipts without retrying unfinished work."""
    allowed = _allowances(policy, rows)
    operations = {}
    for directory in sorted((folder / "operations").glob("*")):
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or directory.name not in allowed
        ):
            raise ValueError("unknown portfolio operation")
        if not (directory / "started.json").exists():
            if any(
                (directory / name).exists() for name in ("result.json", "progress.json")
            ):
                raise ValueError("portfolio operation lacks a reservation")
            continue
        bound = {
            "campaign": identity,
            "stage": directory.name,
            "allowance_seconds": allowed[directory.name],
        }
        if (directory / "started.json").exists() and read_seal(
            directory / "started.json"
        ) != bound:
            raise ValueError("portfolio operation reservation differs")
        if (directory / "result.json").exists():
            operations[directory.name] = journal._read_operation(directory, bound)
        else:
            progress_path = directory / "progress.json"
            progress = public._read(progress_path) if progress_path.exists() else {}
            operations[directory.name] = {
                "identity": bound,
                "status": "running_or_interrupted",
                "value": None,
                "observed_progress": progress,
                "progress_sha256": public.content_sha256(progress),
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": allowed[directory.name],
                "accounting_complete": False,
                "budget_restarted": False,
            }
    return operations


def _assemble(identity: dict, model: dict, data: dict, operations: dict) -> dict:
    """Summarize evidence without feeding suffix scores into parameter selection."""
    origin, retained = _retained(operations)
    inference_rows, available = [], []
    prefix_policy = PrefixPolicy.model_validate(data["prefix"])
    training = PublicSplit.model_validate(data["training"])
    for index, row in enumerate(training.rows):
        infer = _value(operations.get(f"train_{index:02d}_inference"))
        prediction = _value(operations.get(f"train_{index:02d}_forecast"))
        score = None
        if infer.get("fit_success") and prediction.get("forecast_success"):
            score = kernel.score_forecast(
                prediction,
                row,
                row.time[0] + prefix_policy.duration,
                infer["training_scales"],
            )
            if score["available"]:
                available.append(score)
        inference_rows.append(
            {
                "trajectory_id": row.trajectory_id,
                "inference_success": bool(infer.get("fit_success")),
                "forecast_success": bool(prediction.get("forecast_success")),
                "score": score,
            }
        )
    suffix = None
    if len(available) == len(training.rows):
        count = sum(s["sample_count"] for s in available)
        per_target = {
            name: sum(
                s["per_target_normalized_mse"][name] * s["sample_count"]
                for s in available
            )
            / count
            for name in training.rows[0].targets
        }
        suffix = {
            "normalized_mse": float(np.mean(list(per_target.values()))),
            "per_target_normalized_mse": per_target,
            "sample_count": count,
            "conditioning": "fresh_prefix_only_roots_with_frozen_shared_parameters",
            "scoring_rule": "strictly_after_prefix_cutoff",
        }
    qualification = _value(operations.get("qualification"))
    search = _value(operations.get("search"), partial=True)
    assessment = _value(operations.get("diagnosis")).get("assessment")
    fitting_status = (
        "not_started"
        if not operations
        else "interface_failed"
        if not _value(operations.get("setup"))
        else "qualification_pending"
        if "qualification" not in operations
        else "derivative_or_initial_rollout_unqualified"
        if not qualification.get("passed")
        else "search_pending"
        if "search" not in operations
        else "search_interrupted_or_unavailable"
        if operations["search"]["status"] != "complete"
        else "no_verified_endpoint"
        if not retained
        else "budget_stopped"
        if search.get("budget_exhausted")
        else "verified_endpoint_convergence_separate"
    )
    return {
        "status": "complete",
        "identity": identity,
        "candidate_id": model["candidate_id"],
        "family": model["family"],
        "fitting_status": fitting_status,
        "retained_origin": origin,
        "retained": retained or None,
        "setup": _value(operations.get("setup")),
        "qualification": qualification,
        "search_stop_reason": search.get("stop_reason"),
        "search_budget_exhausted": search.get("budget_exhausted"),
        "assessment": assessment,
        "training_reconstruction_nmse": retained.get("training_nmse"),
        "training_reconstruction_per_target": retained.get("per_target_normalized_mse"),
        "training_prefix_forecast": suffix,
        "training_prefix_rows": inference_rows,
        "training_prefix_available_rows": len(available),
        "validation_status": "not_bundled_matching_frozen_split_required",
        "operation_statuses": {
            name: value["status"] for name, value in operations.items()
        },
        "operation_receipts": {
            name: public.content_sha256(value) for name, value in operations.items()
        },
        "cost": journal._cost(operations),
        "cost_exclusions": [
            "Runtime/source verification, reporting and scheduler wait.",
            "Adapter reconstruction on partial resume; no search allowance renewed.",
        ],
        "model_retained": True,
        "scientific_adequacy": "not_established",
        "global_optimum_certified": False,
        "automatic_revision": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
    }


def run_task(root: Path, index: int) -> dict:
    """Execute pending stages once; verify useful endpoints before forecasting."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown portfolio task index")
    task, model = plan["tasks"][index], data["models"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    policy = Policy.model_validate(plan["policy"])
    problem = PrefixFitProblem.model_validate(model["problem"])
    training = PublicSplit.model_validate(data["training"])
    prefix_policy = PrefixPolicy.model_validate(data["prefix"])
    folder = root / "results" / task["task_id"]
    allowances = _allowances(policy, len(training.rows))
    with public._lock(folder):
        operations = _operations(folder, identity, policy, len(training.rows))
        if (folder / "result.json").exists():
            expected = _assemble(identity, model, data, operations)
            if read_seal(folder / "result.json") != expected:
                raise ValueError("portfolio result differs from operation evidence")
            return expected
        compiled = None

        def oracles():
            nonlocal compiled
            if compiled is None:
                compiled = numerical.make_oracle(problem, training, policy.numerical)
            return compiled

        def perform(name: str, function: Callable) -> dict:
            operations[name] = journal._operation(
                folder, identity, name, allowances[name], function
            )
            return _value(operations[name])

        def setup(save):
            original, reduced = oracles()
            result = {
                "routing": original.portfolio_audit,
                "reductions": reduced.portfolio_audit,
                "original_start": original.start,
                "effective_start": reduced.start,
                "original_coordinates": len(original.names),
                "optimizer_coordinates": len(reduced.names),
                "training_scales": original.scales,
            }
            save(result)
            return result

        setup_result = perform("setup", setup)
        if setup_result:
            original, reduced = oracles()
            start = reduced.start
            perform(
                "start_verification",
                lambda save: numerical.verify_endpoint(
                    original, reduced, start, policy.verification_seconds, save
                ),
            )
            qualified = perform(
                "qualification",
                lambda save: numerical.qualification(
                    original, reduced, start, policy.qualification_seconds, save
                ),
            )
            if qualified.get("passed"):
                perform(
                    "search",
                    lambda save: numerical.search(
                        reduced,
                        start,
                        policy.search_seconds,
                        policy.maximum_calls,
                        save,
                    ),
                )
                search = _value(operations.get("search"), partial=True)
                best = search.get("best")
                if best:
                    point = best["parameters"]
                    perform(
                        "endpoint_verification",
                        lambda save: numerical.verify_endpoint(
                            original, reduced, point, policy.verification_seconds, save
                        ),
                    )
            origin, endpoint = _retained(operations)
            if endpoint:
                effective = coordinates.to_effective(
                    original, reduced, endpoint["parameters"]
                )
                history = _value(operations.get("search"), partial=True)
                if origin != "endpoint_verification":
                    history = {
                        "stop_reason": "retained_initial_point",
                        "accepted_steps": [],
                    }
                perform(
                    "diagnosis",
                    lambda save: numerical.diagnose(
                        original,
                        reduced,
                        effective,
                        endpoint,
                        history,
                        policy.diagnostic_seconds,
                        save,
                    ),
                )
                fit = _fit_record(problem, training, original, endpoint)
                for row_index, row in enumerate(training.rows):
                    prefix = kernel.make_prefix(row, prefix_policy)
                    inference = perform(
                        f"train_{row_index:02d}_inference",
                        lambda save, prefix=prefix: execution.infer_prefix(
                            problem,
                            fit,
                            prefix,
                            policy.inference,
                            policy.inference_execution,
                            save,
                        ),
                    )
                    if inference.get("fit_success"):
                        schedule = kernel.make_forecast_schedule(row, prefix_policy)
                        perform(
                            f"train_{row_index:02d}_forecast",
                            lambda save,
                            inference=inference,
                            schedule=schedule: forecasting.forecast(
                                problem,
                                fit,
                                inference,
                                schedule,
                                policy.inference,
                                policy.forecast_execution,
                                save,
                            ),
                        )
        result = public._jsonable(_assemble(identity, model, data, operations))
        seal(folder / "result.json", result)
        return result


def _distribution(values: list[float]) -> dict:
    median = float(np.median(values)) if values else None
    return {
        "available_models": len(values),
        "median": median,
        "mad": float(np.median(np.abs(np.asarray(values) - median)))
        if values
        else None,
        "mad_scaling": "unscaled",
    }


def report(root: Path) -> dict:
    """Report every roster member from receipts; no fitting or implicit retry."""
    plan, data = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    records = []
    for task, model in zip(plan["tasks"], data["models"], strict=True):
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = root / "results" / task["task_id"]
        operations = _operations(
            folder, identity, policy, len(data["training"]["rows"])
        )
        result = _assemble(identity, model, data, operations)
        terminal = (folder / "result.json").exists()
        if terminal and read_seal(folder / "result.json") != result:
            raise ValueError(
                "portfolio terminal result differs from operation evidence"
            )
        records.append(
            result
            | {
                "status": "complete"
                if terminal
                else "partial"
                if operations
                else "missing"
            }
        )
    completed = sum(r["status"] == "complete" for r in records)
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if completed == len(records) else "incomplete",
        "expected": len(records),
        "recorded": completed,
        "status_counts": dict(Counter(r["fitting_status"] for r in records)),
        "rows": records,
        "distributions": {
            "training_reconstruction": _distribution(
                [
                    r["training_reconstruction_nmse"]
                    for r in records
                    if r["training_reconstruction_nmse"] is not None
                ]
            ),
            "training_prefix_forecast": _distribution(
                [
                    r["training_prefix_forecast"]["normalized_mse"]
                    for r in records
                    if r["training_prefix_forecast"] is not None
                ]
            ),
        },
        "validation_status": plan["validation_status"],
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "limitations": [
            "One original start and a bounded search do not certify a global optimum.",
            "Numerical replay, convergence, identifiability and science are separate.",
            "Prefix forecasts reuse training-fitted shared parameters; they are not "
            "independent validation.",
            "Validation is not bundled; no validation metrics are claimed.",
            "Interrupted allowances are never automatically renewed.",
            "Deadlines are cooperative; filesystem waits may exceed them.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", summary)
        lines = [
            "# Construction portfolio fitting",
            "",
            f"{completed}/{len(records)} tasks recorded.",
            "",
            "| Model | Status | Reconstruction NMSE | Prefix forecast NMSE |",
            "| --- | --- | ---: | ---: |",
        ]
        for row in records:
            suffix = (row["training_prefix_forecast"] or {}).get("normalized_mse")
            lines.append(
                f"| {row['candidate_id']} | {row['fitting_status']} | "
                f"{row['training_reconstruction_nmse']} | {suffix} |"
            )
        (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def inspection(root: Path) -> Path:
    """Package the compact frozen inputs, results and journals for local review."""
    report(root)
    files = [
        root / name
        for name in ("inputs.json", "plan.json", "summary.json", "REPORT.md")
    ]
    files += sorted((root / "results").rglob("*.json"))
    files += sorted((root / "submissions").rglob("*.json"))
    files += sorted((root / "submissions").rglob("*.id"))
    files += [
        p
        for p in (root / "launcher_execution.json", root / "submission_manifest.json")
        if p.exists()
    ]
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("inspection source escapes campaign root")
            payload = path.read_bytes()
            item = tarfile.TarInfo(str(path.relative_to(root)))
            item.size, item.mode = len(payload), 0o600
            archive.addfile(item, io.BytesIO(payload))
    destination = root / "inspection.tar.gz"
    temporary = root / ".inspection.tar.gz.tmp"
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(destination)
    return destination
