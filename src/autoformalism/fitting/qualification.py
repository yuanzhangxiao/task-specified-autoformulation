"""Phase C milestone 1: matched coordinate/domain and known-block diagnostics."""

from __future__ import annotations

import fcntl
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.benchmarks.fitting_qualification_inputs import experiment_request
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.fitting.coordinates import NumericalCoordinates, training_coordinates
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "phase-c-fitting-qualification-1"
ARMS = {
    "control": (False, False),
    "scaled": (True, False),
    "domain": (False, True),
    "scaled_domain": (True, True),
}


class QualificationConfig(StrictSchema):
    """All numerical allowances and roster choices are frozen before execution."""

    protocol: Literal["phase-c-fitting-qualification-1"] = PROTOCOL
    starts: int = Field(default=3, ge=1, le=8)
    fitter: CollocationSensitivityConfig = CollocationSensitivityConfig(
        initializer_seconds=120,
        refinement_seconds=180,
        maximum_function_evaluations=240,
        collocation_node_start="rollout_or_observed",
        node_warmup_seconds=5,
        recovery_policy="feasible",
        collocation_diagnostics=True,
        least_squares_ftol=None,
        poll_policy="local",
    )
    replay_seconds: float = Field(default=240, gt=0, le=1200)
    fit_nmse_target: float = Field(default=0.01, gt=0)
    reference_nmse_limit: float = Field(default=1e-6, gt=0)
    solver_agreement_limit: float = Field(default=1e-4, gt=0)


@contextmanager
def _report_lock(root: Path):
    with (root / ".report.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def prepare(inputs_path: Path, root: Path, config: QualificationConfig) -> dict:
    """Freeze a portable campaign; original data and equations are immutable."""
    if inputs_path.resolve().is_relative_to(root.resolve()):
        raise ValueError("source inputs must be outside campaign root")
    inputs = read_seal(inputs_path)
    if inputs["protocol"] != "phase-c-fitting-inputs-1" or set(inputs["cases"]) != {
        "cstr_easy",
        "cstr_hard",
        "basin_coupled",
    }:
        raise ValueError("expected the three fixed qualification cases")
    tasks = []
    for name, case in sorted(inputs["cases"].items()):
        training = PublicSplit.model_validate(case["training"])
        validation = PublicSplit.model_validate(case["validation"])
        scopes = (
            ("parameters_only", "initials_only", "joint")
            if name == "cstr_hard"
            else ("parameters_only",)
        )
        for scope in scopes:
            for seed in range(config.starts):
                for arm, (scaled, domains) in ARMS.items():
                    if domains and (name != "cstr_hard" or scope == "parameters_only"):
                        continue  # No duplicate no-op domain arms.
                    request, truth = experiment_request(case, scope, seed, domains)
                    public._check_data(request, training, validation)
                    model, start, _ = public._lower(request)
                    coordinates = (
                        training_coordinates(
                            model,
                            public.unpack_split(training),
                            start,
                            state_channels=case["state_channel_proxies"],
                        )
                        if scaled
                        else None
                    )
                    tasks.append(
                        {
                            "task_id": f"{name}_{scope}_s{seed}_{arm}",
                            "case": name,
                            "scope": scope,
                            "seed": seed,
                            "arm": arm,
                            "request": request.model_dump(mode="json"),
                            "reference_parameters": truth,
                            "start": start,
                            "coordinates": coordinates.model_dump(mode="json")
                            if coordinates
                            else None,
                            "domain_policy": "shared_initial_concentration_nonnegative"
                            if domains
                            else "original_parameter_domains",
                        }
                    )
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(inputs),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "config": config.model_dump(mode="json"),
        "tasks": tasks,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "gpus": 0,
    }
    with public._lock(root):
        seal(root / "inputs.json", inputs)
        seal(root / "plan.json", plan)
    report(root)
    return {"identity": public.content_sha256(plan), "tasks": len(tasks), "gpus": 0}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Reject changed source, inputs or environment before granting any fit budget."""
    plan, inputs = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    if plan["protocol"] != PROTOCOL or plan["inputs_sha256"] != public.content_sha256(
        inputs
    ):
        raise ValueError("qualification identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("qualification source/runtime differs; use frozen environment")
    return plan, inputs


def _replay_summary(rows: list[dict], channels: list[str]) -> dict:
    """Never turn incomplete coverage into a partial-data accuracy score."""
    metrics = {}
    for name in ("train", "val"):
        selected = [r for r in rows if r["split"] == name]
        metrics[name] = (
            float(
                np.mean(
                    [
                        sum(r["samples"] * r["nmse"][c] for r in selected)
                        / sum(r["samples"] for r in selected)
                        for c in channels
                    ]
                )
            )
            if all(r["complete"] for r in selected)
            else None
        )
    return {
        "complete": all(r["complete"] for r in rows),
        "metrics": metrics,
        "maximum_solver_difference": max(
            (r["solver_difference"] for r in rows if r["complete"]), default=None
        ),
        "rows": rows,
        "validation_initials_fitted": False,
        "test_data_opened": False,
    }


def replay(request, parameters, train, val, seconds: float, *, journal=None) -> dict:
    """Bounded independent scoring, optionally journaled without a resumed budget.

    A terminal journal is reused. An interrupted journal is closed as unavailable;
    previously completed rows survive but no additional integration is performed.
    """
    model, _, _ = public._lower(request)
    if set(parameters) != set(model.parameter_names):
        raise ValueError("incomplete replay parameters")
    public._check_data(request, train, val)
    scales = {
        c: max(float(np.std(np.concatenate([r.targets[c] for r in train.rows]))), 1e-12)
        for c in request.context.targets
    }
    trajectories = [
        (split.name, trajectory)
        for split in (train, val)
        for trajectory in public.unpack_split(split).trajectories
    ]
    identity = public.content_sha256(
        {
            "request": request.model_dump(mode="json"),
            "parameters": parameters,
            "train": train.model_dump(mode="json"),
            "val": val.model_dump(mode="json"),
            "seconds": seconds,
        }
    )
    rows = []

    def empty(name, trajectory, reason=None):
        return {
            "split": name,
            "trajectory": trajectory.trajectory_id,
            "samples": trajectory.number_of_rows,
            "complete": False,
            "errors": [reason] if reason else [],
            "nmse": {},
            "solver_difference": None,
            "solver_scores": {},
        }

    def persist(result=None):
        if journal is not None:
            public._write(
                journal,
                {
                    "identity": identity,
                    "rows": rows,
                    "result": result,
                    "budget_restarted": False,
                },
            )

    if journal is not None and journal.exists():
        saved = public._read(journal)
        if saved["identity"] != identity:
            raise ValueError("replay journal identity differs")
        if saved.get("result") is not None:
            return saved["result"]
        prior = {(r["split"], r["trajectory"]): r for r in saved["rows"]}
        for name, trajectory in trajectories:
            row = prior.get((name, trajectory.trajectory_id))
            if row is None:
                row = empty(name, trajectory, "replay interrupted before this row")
            elif not row["complete"]:
                row["errors"].append("replay interrupted; budget not restarted")
            rows.append(row)
        result = _replay_summary(rows, list(scales))
        result.update(stop_reason="replay_interrupted", budget_restarted=False)
        persist(result)
        return result
    deadline = monotonic() + seconds
    persist()
    timed_out = False
    for name, trajectory in trajectories:
        row = empty(name, trajectory)
        rows.append(row)
        predictions = []
        for method in ("Radau", "DOP853"):
            row["pending_method"] = method
            persist()
            if timed_out or monotonic() >= deadline:
                timed_out = True
                row["errors"].append(f"{method}: replay wall-clock limit reached")
                continue
            try:
                sim = simulate_trajectory(
                    model,
                    trajectory,
                    parameters,
                    {},
                    FitConfig(
                        integration_method=method,
                        relative_tolerance=1e-9,
                        absolute_tolerance=1e-11,
                    ),
                    reset_observed_states=False,
                    deadline=deadline,
                )
            except (TimeoutError, RuntimeError, ValueError, ArithmeticError) as error:
                timed_out |= isinstance(error, TimeoutError)
                row["errors"].append(f"{method}: {type(error).__name__}: {error}")
                continue
            if sim.success:
                predictions.append(sim.predictions)
                row["solver_scores"][method] = {
                    c: float(
                        np.mean(
                            ((sim.predictions[c] - trajectory.targets[c]) / scales[c])
                            ** 2
                        )
                    )
                    for c in scales
                }
            else:
                row["errors"].append(f"{method}: {sim.message}")
            row.pop("pending_method", None)
            persist()
        row.pop("pending_method", None)
        row["complete"] = len(predictions) == 2
        if row["complete"]:
            row["nmse"] = row["solver_scores"]["Radau"]
            row["solver_difference"] = max(
                float(np.max(np.abs(predictions[0][c] - predictions[1][c]))) / scales[c]
                for c in scales
            )
        persist()
    result = _replay_summary(rows, list(scales))
    result.update(
        stop_reason="replay_wall_budget_exhausted"
        if timed_out
        else "complete"
        if result["complete"]
        else "replay_solver_failed",
        budget_exhausted=timed_out,
        budget_restarted=False,
    )
    persist(result)
    return result


def qualify(root: Path) -> dict:
    """Replay the known witness once per case before enabling the fitting array."""
    plan, inputs = verify(root)
    config = QualificationConfig.model_validate(plan["config"])
    with public._lock(root / "qualification"):
        results = {}
        for name, case in inputs["cases"].items():
            path = root / "qualification" / f"{name}.json"
            if not path.exists():
                result = replay(
                    PublicFitRequest.model_validate(case["request"]),
                    case["reference_parameters"],
                    PublicSplit.model_validate(case["training"]),
                    PublicSplit.model_validate(case["validation"]),
                    config.replay_seconds,
                )
                seal(path, result)
            results[name] = read_seal(path)
        passed = all(
            r["complete"]
            and max(r["metrics"].values()) <= config.reference_nmse_limit
            and r["maximum_solver_difference"] <= config.solver_agreement_limit
            for r in results.values()
        )
        result = {
            "passed": passed,
            "cases": results,
            "plan_sha256": public.content_sha256(plan),
        }
        seal(root / "qualification" / "result.json", result)
    return {"passed": passed, "cases": len(results)}


def run_task(root: Path, index: int) -> dict:
    """One durable attempt; an interrupted optimizer never receives a fresh budget."""
    plan, inputs = verify(root)
    qualified = read_seal(root / "qualification/result.json")
    if not qualified["passed"] or qualified["plan_sha256"] != public.content_sha256(
        plan
    ):
        raise ValueError("reference qualification must pass before fitting")
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("task index out of range")
    task = plan["tasks"][index]
    directory = root / "results" / task["task_id"]
    config = QualificationConfig.model_validate(plan["config"])
    with public._lock(directory):
        completed = directory / "result.json"
        if completed.exists():
            return read_seal(completed)
        started, backend_path = directory / "started.json", directory / "backend.json"
        if started.exists() and not backend_path.exists():
            result = {
                "task_id": task["task_id"],
                "status": "interrupted",
                "budget_restarted": False,
            }
            seal(completed, result)
            return result
        request = PublicFitRequest.model_validate(task["request"])
        case = inputs["cases"][task["case"]]
        train, val = (
            PublicSplit.model_validate(case["training"]),
            PublicSplit.model_validate(case["validation"]),
        )
        if not backend_path.exists():
            seal(
                started,
                {
                    "plan_sha256": public.content_sha256(plan),
                    "task_sha256": public.content_sha256(task),
                },
            )
            model, start, _ = public._lower(request)
            coordinates = (
                NumericalCoordinates.model_validate(task["coordinates"])
                if task["coordinates"]
                else None
            )
            try:
                backend = fit_collocation_forward_sensitivity(
                    model,
                    public.unpack_split(train),
                    public.unpack_split(val),
                    config.fitter,
                    directory / "fit",
                    initial_parameters=start,
                    numerical_coordinates=coordinates,
                )
            except (ValueError, RuntimeError, ArithmeticError) as error:
                backend = {
                    "status": "execution_failed",
                    "parameters": None,
                    "error": str(error)[-3000:],
                }
            seal(backend_path, backend)
        backend = read_seal(backend_path)
        replay_path = directory / "replay.json"
        parameters = backend.get("parameters")
        if parameters and not replay_path.exists():
            seal(
                replay_path,
                replay(request, parameters, train, val, config.replay_seconds),
            )
        checked = read_seal(replay_path) if replay_path.exists() else None
        initializer, refinement = (
            backend.get("initializer", {}),
            backend.get("refinement", {}),
        )
        result = {
            "task_id": task["task_id"],
            "status": "complete"
            if checked and checked["complete"]
            else backend["status"]
            if not parameters
            else "replay_failed",
            "backend_sha256": public.content_sha256(backend),
            "replay_sha256": public.content_sha256(checked) if checked else None,
            "training_nmse": checked["metrics"]["train"] if checked else None,
            "validation_nmse": checked["metrics"]["val"] if checked else None,
            "accuracy_passed": bool(
                checked
                and checked["complete"]
                and max(checked["metrics"].values()) <= config.fit_nmse_target
                and checked["maximum_solver_difference"]
                <= config.solver_agreement_limit
            ),
            "initializer_success": initializer.get("success"),
            "initializer_message": initializer.get("message"),
            "budget_exhausted": refinement.get("budget_exhausted"),
            "residual_calls": refinement.get("actual_residual_calls"),
            "fit_seconds": initializer.get("seconds", 0)
            + refinement.get("fit_seconds", 0),
            "parameters": parameters,
            "parameter_absolute_error": {
                n: abs(parameters[n] - v)
                for n, v in task["reference_parameters"].items()
            }
            if parameters
            else {},
            "parameter_recovery_certified": False,
            "validation_initials_fitted": False,
            "test_data_opened": False,
        }
        seal(completed, result)
    return result


def report(root: Path) -> dict:
    """Include missing/failed attempts and paired comparisons, never best-of-N."""
    with _report_lock(root):
        plan, _ = verify(root, runtime=False)
        rows, groups, paired = [], defaultdict(list), defaultdict(dict)
        for task in plan["tasks"]:
            path = root / "results" / task["task_id"] / "result.json"
            row = {k: task[k] for k in ("task_id", "case", "scope", "seed", "arm")}
            row.update(read_seal(path) if path.exists() else {"status": "missing"})
            if path.exists() and row.get("backend_sha256"):
                if (
                    public.content_sha256(read_seal(path.parent / "backend.json"))
                    != row["backend_sha256"]
                ):
                    raise ValueError("backend evidence differs")
                if (
                    row.get("replay_sha256")
                    and public.content_sha256(read_seal(path.parent / "replay.json"))
                    != row["replay_sha256"]
                ):
                    raise ValueError("replay evidence differs")
            rows.append(row)
            groups[(task["case"], task["scope"], task["arm"])].append(row)
            paired[(task["case"], task["scope"], task["seed"])][task["arm"]] = row
        comparisons = []
        for (case, scope, seed), arms in paired.items():
            for left, right in (
                ("control", "scaled"),
                ("control", "domain"),
                ("domain", "scaled_domain"),
                ("scaled", "scaled_domain"),
            ):
                if left not in arms or right not in arms:
                    continue
                comparisons.append(
                    {
                        "case": case,
                        "scope": scope,
                        "seed": seed,
                        "left": left,
                        "right": right,
                        "training_nmse_delta": arms[right]["training_nmse"]
                        - arms[left]["training_nmse"]
                        if arms[right].get("training_nmse") is not None
                        and arms[left].get("training_nmse") is not None
                        else None,
                    }
                )
        counts = Counter(r["status"] for r in rows)
        result = {
            "protocol": PROTOCOL,
            "plan_sha256": public.content_sha256(plan),
            "status": "complete" if not counts["missing"] else "incomplete",
            "expected": len(rows),
            "recorded": len(rows) - counts["missing"],
            "status_counts": dict(counts),
            "rows": rows,
            "groups": [
                {
                    "case": c,
                    "scope": s,
                    "arm": a,
                    "expected": len(rs),
                    "accuracy_passed": sum(bool(r.get("accuracy_passed")) for r in rs),
                    "status_counts": dict(Counter(r["status"] for r in rs)),
                }
                for (c, s, a), rs in groups.items()
            ],
            "paired_comparisons": comparisons,
            "limitation": (
                "Assisted known-equation fitting; numerical starts are not independent "
                "scientific tasks. CSTR domains constrain shared initial concentration "
                "only. Scaling changes optimizer coordinates/constraint units, not "
                "equations, physical starts, parameter domains, target weights or "
                "input interpolation. No automatic production promotion or unique "
                "latent/parameter recovery claim."
            ),
            "test_data_opened": False,
            "live_llm_calls": 0,
        }
        public._write(root / "summary.json", result)
        return result
