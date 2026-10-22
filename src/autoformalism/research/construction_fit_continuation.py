"""Paired, training-only continuation of all eight frozen construction models.

Durable stage reservations support deterministic resume without renewing an
interrupted allowance. Runtime qualification gates numerical search; no receipt
triggers a proposer or changes the original scientific model.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path
from time import monotonic
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_continuation as numerical
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_verification_campaign as source_campaign
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-affine-continuation-1"
ARMS = ("affine_original", "affine_reduced")
STARTS = ("inherited", "declared", "perturbed_declared")


class Policy(StrictSchema):
    """Matched arms and fixed starts; numerical work never chooses its own budget."""

    population: Literal["returned-eight", "synthetic-smoke"] = "returned-eight"
    search_seconds: FiniteFloat = Field(default=600, gt=0, le=1800)
    restart_seconds: FiniteFloat = Field(default=300, gt=0, le=1800)
    diagnostic_seconds: FiniteFloat = Field(default=120, gt=0, le=600)
    verification_seconds: FiniteFloat = Field(default=180, gt=0, le=600)
    maximum_calls: int = Field(default=1000, ge=1, le=10000)
    restart_count: int = Field(default=2, ge=0, le=2)


class SourceTask(StrictSchema):
    """Only bounded provenance metadata, never additional observational payloads."""

    task_id: str = Field(min_length=1)
    problem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_packet_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    initializer_kind: Literal["trajectory_roots", "analytic_map"] | None = None
    root_count: int | None = Field(default=None, ge=0)


def export_inputs(source: Path, destination: Path) -> dict:
    """Authenticate original operations, then export only training and endpoints."""
    plan, data, candidates = source_campaign._source(source.resolve())
    payload = {
        "protocol": "construction-continuation-inputs-1",
        "source_plan_sha256": public.content_sha256(plan),
        "source_inputs_sha256": public.content_sha256(data),
        "training": data["training"],
        "numerical_policy": plan["policy"]["training"],
        "models": [
            {
                "task_id": c["task"]["task_id"],
                "problem": data["records"][c["task"]["task_id"]]["problem"],
                "source_task": c["task"],
                "parameters": c["parameters"],
                "parameters_sha256": c["parameters_sha256"],
                "source_search_nmse": c["search_training_nmse"],
            }
            for c in candidates
        ],
        "test_data_opened": False,
        "validation_exported": False,
    }
    _validate(payload, Policy())
    destination.parent.mkdir(parents=True, exist_ok=True)
    seal(destination, payload)
    return {"inputs_sha256": public.content_sha256(payload), "models": len(candidates)}


def _validate(data: dict, policy: Policy) -> None:
    """A positive allowlist prevents held-out scores/arrays entering the worker."""
    if (
        set(data)
        != {
            "protocol",
            "source_plan_sha256",
            "source_inputs_sha256",
            "training",
            "numerical_policy",
            "models",
            "test_data_opened",
            "validation_exported",
        }
        or data["protocol"] != "construction-continuation-inputs-1"
    ):
        raise ValueError("continuation input schema differs")
    if (
        data["test_data_opened"] is not False
        or data["validation_exported"] is not False
    ):
        raise ValueError("held-out data prohibited")
    train = PublicSplit.model_validate(data["training"])
    if train.name != "train" or not train.rows:
        raise ValueError("nonempty training split required")
    kernel.NumericalPolicy.model_validate(data["numerical_policy"])
    if not data["models"] or (
        policy.population == "returned-eight"
        and (len(data["models"]) != 8 or len(train.rows) != 16)
    ):
        raise ValueError("all eight models and sixteen training rows required")
    names = []
    for model in data["models"]:
        if set(model) != {
            "task_id",
            "problem",
            "source_task",
            "parameters",
            "parameters_sha256",
            "source_search_nmse",
        }:
            raise ValueError("model input schema differs")
        name = model["task_id"]
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("invalid model task id")
        names.append(name)
        SourceTask.model_validate(model["source_task"])
        problem = PrefixFitProblem.model_validate(model["problem"])
        if any(set(row.targets) != set(problem.context.targets) for row in train.rows):
            raise ValueError("public targets differ")
        if (
            model["source_task"]["task_id"] != name
            or model["source_task"]["problem_sha256"] != public.content_sha256(problem)
            or model["parameters_sha256"] != public.content_sha256(model["parameters"])
        ):
            raise ValueError("source model or endpoint identity differs")
    if len(set(names)) != len(names):
        raise ValueError("duplicate task ids")


def prepare(root: Path, data: dict, policy: Policy) -> dict:
    """Freeze a new training-only experiment without altering historical records."""
    _validate(data, policy)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "policy": policy.model_dump(mode="json"),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "tasks": [
            {"model": index, "task_id": model["task_id"], "arm": arm}
            for index, model in enumerate(data["models"])
            for arm in ARMS
        ],
        "starts": list(STARTS[: policy.restart_count + 1]),
        "start_rule": "original_declared_coordinates_seed1729_no_score_selection",
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Check immutable scientific inputs and the exact executable numerical tree."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    policy = Policy.model_validate(plan["policy"])
    _validate(data, policy)
    expected_tasks = [
        {"model": i, "task_id": m["task_id"], "arm": arm}
        for i, m in enumerate(data["models"])
        for arm in ARMS
    ]
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != expected_tasks
        or plan["starts"] != list(STARTS[: policy.restart_count + 1])
        or any(
            plan[name] is not False
            for name in (
                "test_data_opened",
                "validation_data_opened",
                "automatic_revision",
                "automatic_promotion",
            )
        )
        or plan["llm_calls"] != 0
    ):
        raise ValueError("continuation plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("continuation source/runtime differs")
    return plan, data


def _allowances(policy: Policy) -> dict:
    stages = {
        "qualification": policy.diagnostic_seconds + 90,
        "retained_control_verification": policy.verification_seconds + 90,
    }
    for index, name in enumerate(STARTS[: policy.restart_count + 1]):
        stages[f"{name}_search"] = (
            policy.search_seconds if index == 0 else policy.restart_seconds
        ) + 150
        stages[f"{name}_verification"] = policy.verification_seconds + 90
        stages[f"{name}_diagnosis"] = policy.diagnostic_seconds + 90
    return stages


def _operations(folder, identity, policy):
    """Read stage receipts without numerical execution, preserving missing stages."""
    result = {}
    for stage, seconds in _allowances(policy).items():
        directory = folder / "operations" / stage
        bound = {"campaign": identity, "stage": stage, "allowance_seconds": seconds}
        if (directory / "result.json").exists():
            result[stage] = journal._read_operation(directory, bound)
        elif (directory / "started.json").exists():
            if read_seal(directory / "started.json") != bound:
                raise ValueError("stage reservation differs")
            result[stage] = {
                "status": "running_or_interrupted",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": seconds,
                "accounting_complete": False,
            }
    return result


def _assemble(identity, operations, *, terminal=False):
    """Retain only independently verified improvements; keep inherited control."""
    retained, origin = None, None
    for stage, operation in operations.items():
        if not stage.endswith("verification"):
            continue
        point = operation.get("value") or {}
        if not point.get("original_equations_verified"):
            continue
        if retained is None or max(
            point["training_nmse"], point["alternate_training_nmse"]
        ) < min(retained["training_nmse"], retained["alternate_training_nmse"]):
            retained, origin = point, stage.removesuffix("_verification")
    diagnosis = (
        (operations.get(f"{origin}_diagnosis", {}).get("value") or {}) if origin else {}
    )
    restart_summary = []
    for name in STARTS[1:]:
        attempt = operations.get(f"{name}_search")
        if attempt is None:
            continue
        fit = _search_checkpoint(attempt)
        point = operations.get(f"{name}_verification", {}).get("value") or {}
        restart_summary.append(
            {
                "start": name,
                "operation_status": attempt["status"],
                "stop_reason": fit.get("stop_reason"),
                "checkpoint_recovered": fit.get(
                    "checkpoint_recovered_after_interruption", False
                ),
                "verified_training_nmse": point.get("training_nmse")
                if point.get("original_equations_verified")
                else None,
            }
        )
    return {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "retained_origin": origin,
        "retained": retained,
        "fitting_diagnosis": diagnosis.get("assessment"),
        "diagnosis_scope": "selected endpoint and its own accepted optimizer history; "
        "other predeclared attempts are reported separately in restart_summary",
        "restart_summary": restart_summary,
        "optimization_status": (diagnosis.get("assessment") or {}).get(
            "classification", "optimization_unresolved"
        ),
        "operations": operations,
        "cost": journal._cost(operations),
        "diagnostic_cost": _diagnostic_cost(operations),
        "automatic_revision": False,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
    }


def _diagnostic_cost(operations):
    """Expose derivative work separately; interrupted work is a lower bound."""
    stages = {
        name: op
        for name, op in operations.items()
        if name == "qualification" or name.endswith("_diagnosis")
    }
    checks = [
        (
            (op.get("value") or op.get("observed_progress") or {}).get(
                "derivative_check"
            )
            or {}
        )
        for op in stages.values()
    ]
    return {
        "derivative_calls_started_observed": sum(
            c.get("calls_started", 0) for c in checks
        ),
        "derivative_calls_completed_observed": sum(
            c.get("calls_completed", 0) for c in checks
        ),
        "stage_wall_seconds_observed": sum(
            op.get("wall_seconds") or 0 for op in stages.values()
        ),
        "complete": bool(stages)
        and all(
            op["status"] == "complete" and bool(check)
            for op, check in zip(stages.values(), checks, strict=True)
        ),
        "scope": "Derivative calls only; stage wall includes setup, lift checks "
        "and assessment. Nested within total cost, not an additional charge; "
        "interruptions give lower bounds.",
    }


def _search_checkpoint(operation):
    """Recover a saved best without renewing a failed or interrupted search."""
    if operation.get("status") == "complete":
        return operation.get("value") or {}
    saved = operation.get("observed_progress") or operation.get("value") or {}
    return saved | {
        "stop_reason": "interrupted",
        "budget_exhausted": True,
        "checkpoint_recovered_after_interruption": bool(saved.get("best")),
    }


def run_task(root: Path, index: int) -> dict:
    """Run each reserved stage once; replay and missing-stage continuation are safe."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown task index")
    task = plan["tasks"][index]
    model = data["models"][task["model"]]
    policy = Policy.model_validate(plan["policy"])
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"] / task["arm"]
    problem = PrefixFitProblem.model_validate(model["problem"])
    training = PublicSplit.model_validate(data["training"])
    settings = kernel.NumericalPolicy.model_validate(data["numerical_policy"])
    allowances = _allowances(policy)

    def build(arm):
        return numerical.make_oracle(problem, training, settings, arm)

    def operate(name, work):
        def call(save):
            deadline = monotonic() + allowances[name]

            def check():
                if monotonic() > deadline:
                    raise TimeoutError("inclusive stage allowance exhausted")

            def checked_save(value):
                # Save a useful checkpoint even at a late boundary; no late
                # operation may become a completed scientific certificate.
                save(value)
                check()

            original, oracle = build(
                "affine_original"
                if name == "retained_control_verification"
                else task["arm"]
            )
            check()
            result = work(original, oracle, checked_save)
            check()
            return result

        return journal._operation(folder, identity, name, allowances[name], call)

    with public._lock(folder):
        terminal = folder / "result.json"
        if terminal.exists():
            saved = read_seal(terminal)
            rebuilt = _assemble(
                identity, _operations(folder, identity, policy), terminal=True
            )
            if saved != rebuilt:
                raise ValueError("continuation result differs from stage receipts")
            return saved
        qualification = operate(
            "qualification",
            lambda o, a, save: numerical.qualification(
                o, a, model["parameters"], policy.diagnostic_seconds, save
            ),
        )
        control = operate(
            "retained_control_verification",
            lambda o, a, save: numerical.verify_endpoint(
                o,
                o,
                model["parameters"],
                policy.verification_seconds,
                save,
            ),
        )
        if (qualification.get("value") or {}).get("passed") and (
            control.get("value") or {}
        ).get("original_equations_verified"):
            for start_index, name in enumerate(plan["starts"]):
                seconds = (
                    policy.search_seconds
                    if start_index == 0
                    else policy.restart_seconds
                )

                def start_search(o, a, save, n=start_index, limit=seconds):
                    starts = numerical.predetermined_starts(o, model["parameters"])
                    effective = numerical.to_effective(o, a, starts[n]["parameters"])
                    value = numerical.search(
                        a, effective, limit, policy.maximum_calls, save
                    )
                    value["original_start"] = starts[n]
                    save(value)
                    return value

                operation = operate(f"{name}_search", start_search)
                fit = _search_checkpoint(operation)
                best = fit.get("best")
                if not best:
                    continue
                checked = operate(
                    f"{name}_verification",
                    lambda o, a, save, b=best: numerical.verify_endpoint(
                        o, a, b["parameters"], policy.verification_seconds, save
                    ),
                )
                value = checked.get("value") or {}
                if value.get("original_equations_verified"):
                    operate(
                        f"{name}_diagnosis",
                        lambda o, a, save, b=best, v=value, f=fit: numerical.diagnose(
                            a, b["parameters"], v, f, policy.diagnostic_seconds, save
                        ),
                    )
        result = _assemble(
            identity, _operations(folder, identity, policy), terminal=True
        )
        seal(terminal, result)
        return result


def report(root: Path) -> dict:
    """Summarize every arm, without interpreting absent records as failed science."""
    plan, _ = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    rows = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = root / "results" / task["task_id"] / task["arm"]
        terminal = (folder / "result.json").exists()
        result = _assemble(
            identity, _operations(folder, identity, policy), terminal=terminal
        )
        if terminal and read_seal(folder / "result.json") != result:
            raise ValueError("terminal summary differs from operations")
        retained = result["retained"] or {}
        qualified = result["operations"].get("qualification", {}).get("value") or {}
        rows.append(
            {
                **task,
                "status": result["status"],
                "qualification_passed": qualified.get("passed", False),
                "original_coordinates": qualified.get("original_coordinates"),
                "optimizer_coordinates": qualified.get("optimizer_coordinates"),
                "retained_origin": result["retained_origin"],
                "training_nmse": retained.get("training_nmse"),
                "per_target_normalized_mse": retained.get("per_target_normalized_mse"),
                "optimization_status": result["optimization_status"],
                "fitting_diagnosis": result["fitting_diagnosis"],
                "diagnosis_scope": result["diagnosis_scope"],
                "restart_summary": result["restart_summary"],
                "cost": result["cost"],
                "diagnostic_cost": result["diagnostic_cost"],
            }
        )
    complete = sum(row["status"] == "complete" for row in rows)
    summary = {
        "protocol": PROTOCOL,
        "status": "complete" if complete == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": complete,
        "rows": rows,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
        "limitation": (
            "Local convergence and identification are separate. "
            "No global optimum or scientific-correctness claim."
        ),
    }
    with public._lock(root / "report"):
        public._write(root / "report/summary.json", summary)
    return summary


def inspect(root: Path) -> dict:
    """Export inputs and receipts without fitting or model selection."""
    summary = report(root)
    path = root / "inspection.tar.gz"
    temporary = path.with_suffix(".tmp")
    paths = [root / "plan.json", root / "inputs.json"]
    if (root / "launcher_execution.json").exists():
        paths.append(root / "launcher_execution.json")
    for name in ("results", "report", "submissions", "logs"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file())
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as archive,
    ):
        for source in sorted(paths):
            if source.is_symlink() or not source.resolve().is_relative_to(
                root.resolve()
            ):
                raise ValueError("archive path escapes campaign")
            if source.name.startswith("."):
                continue
            blob = source.read_bytes()
            entry = tarfile.TarInfo(source.relative_to(root).as_posix())
            entry.mode, entry.size = 0o644, len(blob)
            archive.addfile(entry, io.BytesIO(blob))
    temporary.replace(path)
    return {"status": summary["status"], "archive": str(path.resolve())}
