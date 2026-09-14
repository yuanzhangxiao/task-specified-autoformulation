"""One final, finite comparison of four approaches to the difficult system."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import least_squares

from autoformalism.data import TrainingScaler
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.fitting.feasibility import (
    EvaluationBudget,
    GuardedOracle,
    SensitivityUnavailable,
)
from autoformalism.fitting.initialization import LatentInitializationPlan
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.fitting.stagnation import instrumented_fit
from autoformalism.rebuttal.attainability_campaign import (
    _checkpoint,
    _identity,
    checked_replay,
)
from autoformalism.rebuttal.attainability_controls import system_for
from autoformalism.rebuttal.attainability_reference import shared_boundary_lower_bound
from autoformalism.rebuttal.final_fitter_models import (
    alternative_starts,
    smaller_problem,
    training_prefix,
)
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    _write_bytes,
    read_json,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_methods import identity_runtime
from autoformalism.rebuttal.piecewise_campaign import safe_path, unpack_split
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash


class FinalFitterPlan(StrictSchema):
    """One allocation per strategy; no automatic extension after this campaign."""

    protocol: Literal["final-fitter-alternatives-1"] = "final-fitter-alternatives-1"
    total_fit_seconds: float = Field(default=4800, ge=10, le=4800)
    selection_seconds: float = Field(default=240, gt=0, le=240)
    selection_calls: int = Field(default=12, ge=4, le=12)
    total_residual_calls: int = Field(default=120, ge=18, le=120)
    replay_seconds: float = Field(default=240, gt=0, le=300)
    supervisor_seconds: int = Field(default=6600, ge=60, le=6900)
    seed: int = 20260915
    fit: CollocationSensitivityConfig

    @model_validator(mode="after")
    def budgets(self):
        if self.selection_seconds >= self.total_fit_seconds:
            raise ValueError("selection must leave a fitting budget")
        if self.fit.initializer_seconds >= self.search_seconds:
            raise ValueError("collocation must leave a refinement budget")
        if (
            self.supervisor_seconds
            < self.total_fit_seconds + 4 * self.replay_seconds + 300
        ):
            raise ValueError("supervisor must include replay and startup grace")
        if not (
            self.fit.budget_profile == "extended_diagnostic"
            and self.fit.defer_production_replay
            and self.fit.recovery_handoff == "best_valid"
            and self.fit.sensitivity_invalid_trials == "reject"
            and self.fit.sensitivity_jacobian_format == "sparse"
            and self.fit.recovery_policy == "feasible"
            and self.fit.collocation_assembly == "mapped"
        ):
            raise ValueError("requires bounded sparse fitting and preserved incumbents")
        return self

    @property
    def search_seconds(self) -> float:
        return self.total_fit_seconds - self.selection_seconds

    @property
    def search_calls(self) -> int:
        return self.total_residual_calls - self.selection_calls


def code_identity() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = [
        *sorted((root / "src/autoformalism").rglob("*.py")),
        *[
            root / p
            for p in (
                "scripts/run_final_fitter.py",
                "scripts/final_fitter_report.py",
                "scripts/hpc/final_fitter_delta.slurm",
                "scripts/hpc/submit_final_fitter_delta.sh",
            )
        ],
    ]
    return content_hash({str(p.relative_to(root)): sha256(p) for p in paths})


def prepare(source: Path, output: Path, plan: FinalFitterPlan) -> dict:
    """Reuse frozen noiseless observations; previous estimates never seed fits."""
    if source.resolve().is_relative_to(
        output.resolve()
    ) or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("source/output must be separate")
    old = read_json(source / "freeze.json")
    _identity(old)
    if (
        old["plan"]["protocol"] != "fitter-parameter-freedom-1"
        or len(old["tasks"]) != 6
    ):
        raise ValueError("requires the parameter-freedom campaign")
    if (
        old.get("test_data_opened") is not False
        or old.get("proposer_access") is not False
    ):
        raise ValueError("source isolation differs")
    if not _checkpoint(source / "gate/result.json", old["identity"]).get("pass"):
        raise ValueError("source gate did not pass")
    runtime = identity_runtime()
    if any(runtime[k] != old["runtime"][k] for k in ("packages", "casadi")) or (
        runtime["python"].split(".")[:2] != old["runtime"]["python"].split(".")[:2]
    ):
        raise ValueError("source numerical runtime differs")
    for name, digest in old["assets"].items():
        if sha256(safe_path(source, name)) != digest:
            raise ValueError("source asset changed: " + name)
    problems = [read_json(source / f"problems/{i:03d}.json") for i in (0, 2)]
    if any(old["tasks"][i]["start"] != "ordinary" for i in (0, 2)):
        raise ValueError("requires ordinary parameter starts")
    synthetic = read_json(source / "inputs/synthetic.json")
    if any(p["splits"] != synthetic["splits"] for p in problems):
        raise ValueError("reference observations differ")
    clean = read_json(source / "inputs/clean.json")
    for split in ("train", "val"):
        rows = synthetic["splits"][split]["rows"]
        if set(clean.get(split, {})) != {r["trajectory_id"] for r in rows}:
            raise ValueError("clean reference trajectory coverage differs")
        for row in rows:
            values = np.asarray(clean[split][row["trajectory_id"]], dtype=float)
            if values.shape != (len(row["time"]),) or not np.isfinite(values).all():
                raise ValueError("clean reference samples differ")
    small = smaller_problem(synthetic["splits"])
    tasks, assets = [], {}
    for family, problem in zip(
        ("fixed_shapes", "free_shapes", "smaller_model"),
        [*problems, small],
        strict=True,
    ):
        strategies = (
            ("collocation_exact",)
            if family == "smaller_model"
            else (
                "collocation_exact",
                "direct_multistart",
                "horizon_continuation",
                "collocation_limited",
            )
        )
        starts = alternative_starts(problem, plan.seed)
        for strategy in strategies:
            task = {
                "index": len(tasks),
                "family": family,
                "strategy": strategy,
                "parameters": len(starts[0]),
                "states": system_for(problem)[0].state_count,
                "oracle_shapes": family == "fixed_shapes",
                "oracle_initials": family != "smaller_model",
                "oracle_weight_start": False,
                "scientific_status": "not judged",
                "starts": starts,
            }
            tasks.append(task)
            name = f"problems/{task['index']:03d}.json"
            write_json(output / name, problem, immutable=True)
            assets[name] = sha256(output / name)
    for name, destination in (
        ("freeze.json", "provenance/source_freeze.json"),
        ("gate/result.json", "provenance/source_gate.json"),
        ("inputs/clean.json", "inputs/clean.json"),
    ):
        _write_bytes(output / destination, (source / name).read_bytes(), immutable=True)
        assets[destination] = sha256(output / destination)
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "tasks": tasks,
        "assets": assets,
        "code": code_identity(),
        "runtime": runtime,
        "source_identity": old["identity"],
        "test_data_opened": False,
        "proposer_access": False,
        "llm_calls": 0,
        "no_automatic_followup": True,
        "historical_estimates_used": False,
        "smaller_model_initialization_limit": shared_boundary_lower_bound(small),
        "bands": {"strict": 1e-4, "good": 0.01, "practical": 0.1},
    }
    frozen["identity"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return verify(output)


def verify(output: Path) -> dict:
    frozen = read_json(output / "freeze.json")
    _identity(frozen)
    if frozen["code"] != code_identity() or frozen["runtime"] != identity_runtime():
        raise ValueError("code/runtime differs; use the pinned checkout")
    for name, digest in frozen["assets"].items():
        if sha256(safe_path(output, name)) != digest:
            raise ValueError("frozen asset changed: " + name)
    return frozen


def checkpoint_stage(root: Path, identity: str, operation: Callable[[], dict]) -> dict:
    """An interrupted stage is charged in full and never silently restarted."""
    result, intent = root / "result.json", root / "started.json"
    if result.exists():
        return _checkpoint(result, identity)
    if intent.exists():
        _checkpoint(intent, identity)
        saved = root / "calls/best_evaluated.json"
        best = read_json(saved) if saved.exists() else None
        value = {
            "status": "interrupted",
            "message": "No fresh stage budget",
            "best": best,
        }
    else:
        write_json(intent, {"identity": identity}, immutable=True)
        started = monotonic()
        try:
            value = operation()
        except (ValueError, RuntimeError, TimeoutError, ArithmeticError) as error:
            value = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
        value["seconds"] = monotonic() - started
    value = _finite_payload({**value, "identity": identity})
    write_json(result, value, immutable=True)
    return value


def sensitivity_stage(
    problem, payload, start, scale, config, seconds, calls, directory
) -> dict:
    """One derivative-based attempt, with no penalty Jacobian or extra restart."""
    system, _ = system_for(problem)
    system = SymbolicODE(
        system.model, allow_piecewise=True, solver_jacobian_format="sparse"
    )
    training = unpack_split(payload)
    budget = EvaluationBudget(monotonic() + seconds, calls)
    settings = config.fit_config().model_copy(
        update={"maximum_wall_time_seconds": seconds}
    )
    oracle = GuardedOracle(
        system,
        training,
        scale,
        settings,
        directory / "calls",
        budget.deadline,
        sensitivities=True,
        budget=budget,
        point_seconds=config.recovery_sensitivity_seconds,
        reject_invalid_trials=True,
    )

    def optimizer(fun, x, **kwargs):
        kwargs.update(jac=oracle.jacobian, ftol=config.least_squares_ftol)
        return least_squares(fun, x, **kwargs)

    try:
        result = instrumented_fit(
            oracle,
            start,
            diff_step=None,
            max_nfev=calls,
            settings=settings,
            optimizer=optimizer,
        )
    except (SensitivityUnavailable, ValueError) as error:
        result = {"message": str(error), "optimizer_native_success": False}
    return {
        "status": "finished",
        "fit": result,
        "best": oracle.best,
        "actual_residual_calls": budget.calls,
        "budget_seconds": seconds,
        "budget_calls": calls,
        "training_fingerprint": training.fingerprint,
        "training_only": True,
    }


def select_full_training(problem, points, scale, plan, directory) -> dict:
    """Evaluate a bounded pool once on full training, never compare prefix costs."""
    system, _ = system_for(problem)
    system = SymbolicODE(system.model, solver_jacobian_format="sparse")
    budget = EvaluationBudget(
        monotonic() + plan.selection_seconds, plan.selection_calls
    )
    oracle = GuardedOracle(
        system,
        unpack_split(problem["splits"]["train"]),
        scale,
        plan.fit.fit_config(),
        directory / "calls",
        budget.deadline,
        sensitivities=False,
        budget=budget,
        point_seconds=plan.fit.recovery_probe_seconds,
    )
    unique, seen = [], set()
    for point in points:
        key = tuple(sorted(point.items()))
        if key not in seen:
            unique.append(point)
            seen.add(key)
    for point in unique[: plan.selection_calls]:
        try:
            oracle(oracle.vector(point))
        except (TimeoutError, ValueError):
            if monotonic() >= budget.deadline:
                break
    return {
        "status": "finished",
        "best": oracle.best,
        "actual_residual_calls": budget.calls,
    }


def fit_task(
    problem: dict, task: dict, plan: FinalFitterPlan, root: Path, identity: str
) -> dict:
    """Execute immutable stage allocations, retaining every full-training best."""
    train = unpack_split(problem["splits"]["train"])
    scale = TrainingScaler().fit(train).scales["target:v01"].standard_deviation
    stages, incumbents, prefix_points = [], [], []
    strategy = task["strategy"]
    if strategy.startswith("collocation"):
        config = plan.fit.model_copy(
            update={
                "refinement_seconds": plan.search_seconds
                - plan.fit.initializer_seconds,
                "maximum_function_evaluations": plan.search_calls,
                "collocation_hessian": "limited-memory"
                if strategy == "collocation_limited"
                else "exact",
                "collocation_solver_log": True,
                "collocation_target_variables": 12000
                if task["family"] == "smaller_model"
                else 24000,
            }
        )

        def coupled():
            fit = fit_collocation_forward_sensitivity(
                compile_candidate(
                    CandidateModel.model_validate(problem["candidate"]),
                    ValidationContext.model_validate(problem["context"]),
                ),
                train,
                unpack_split(problem["splits"]["val"]),
                config,
                root / "coupled/fit",
                initial_parameters=task["starts"][0],
                initialization_plan=LatentInitializationPlan.model_validate(
                    problem["initialization_plan"]
                ),
            )
            refinement = fit.get("refinement", {})
            best = (
                {"parameters": fit["parameters"], "cost": refinement["cost"]}
                if fit.get("parameters")
                else None
            )
            return {
                "status": "finished",
                "fit": fit,
                "best": best,
                "actual_residual_calls": refinement.get("actual_residual_calls", 0),
            }

        row = checkpoint_stage(
            root / "coupled", content_hash([identity, "coupled"]), coupled
        )
        # The adapter journals its best point separately; retain it if the whole
        # coupled stage was interrupted after useful refinement had begun.
        if not row.get("best"):
            candidates = []
            for path in sorted(
                (root / "coupled/fit/recovery").glob("*/best_evaluated.json")
            ):
                value = read_json(path)
                if value.get("parameters") and np.isfinite(value.get("cost", np.inf)):
                    candidates.append(value)
            if candidates:
                row = {**row, "best": min(candidates, key=lambda x: x["cost"])}
        stages.append({"name": "coupled", **row})
        if row.get("best"):
            incumbents.append(row["best"])
    else:
        points = task["starts"]
        fractions = (
            (1.0, 1.0, 1.0) if strategy == "direct_multistart" else (0.25, 0.5, 1.0)
        )
        shares = (
            (1 / 3, 1 / 3, 1 / 3)
            if strategy == "direct_multistart"
            else (0.2, 0.2, 0.6)
        )
        next_point = points[0]
        for index, (fraction, share) in enumerate(zip(fractions, shares, strict=True)):
            payload = (
                problem["splits"]["train"]
                if fraction == 1
                else training_prefix(problem, fraction)
            )
            start = points[index] if strategy == "direct_multistart" else next_point
            directory = root / f"stage_{index}"
            key = content_hash([identity, index, fraction, start])
            calls = plan.search_calls // 3 + (
                plan.search_calls % 3 if index == 2 else 0
            )
            row = checkpoint_stage(
                directory,
                key,
                lambda payload=payload,
                start=start,
                share=share,
                calls=calls,
                directory=directory: sensitivity_stage(
                    problem,
                    payload,
                    start,
                    scale,
                    plan.fit,
                    plan.search_seconds * share,
                    calls,
                    directory,
                ),
            )
            stages.append({"name": f"stage_{index}", "fraction": fraction, **row})
            if row.get("best"):
                next_point = row["best"]["parameters"]
                if fraction == 1:
                    incumbents.append(row["best"])
                else:
                    prefix_points.append(next_point)
    # Prefix fits and the supplied start get a common full-training comparison.
    # Already evaluated full-training incumbents survive even if this times out.
    selection_points = [*reversed(prefix_points), task["starts"][0]]
    selection = checkpoint_stage(
        root / "selection",
        content_hash([identity, "selection", selection_points]),
        lambda: select_full_training(
            problem, selection_points, scale, plan, root / "selection"
        ),
    )
    if selection.get("best"):
        incumbents.append(selection["best"])
    finite = [
        p
        for p in incumbents
        if p.get("parameters") and np.isfinite(p.get("cost", np.inf))
    ]
    best = min(finite, key=lambda p: p["cost"]) if finite else None
    return {
        "parameters": best["parameters"] if best else None,
        "training_cost": best["cost"] if best else None,
        "training_nmse": 2
        * best["cost"]
        / sum(r.number_of_rows for r in train.trajectories)
        if best
        else None,
        "stages": stages,
        "selection": selection,
        "status": "parameters_saved" if best else "fit_failed",
        "selection_rule": "lowest finite full-training rollout cost",
        "validation_used_for_fitting": False,
        "maximum_search_calls": plan.search_calls,
        "maximum_selection_calls": plan.selection_calls,
        "budget_seconds": plan.total_fit_seconds,
    }


def execute(output: Path, index: int) -> dict:
    frozen = verify(output)
    if not 0 <= index < len(frozen["tasks"]):
        raise ValueError("task index outside frozen matrix")
    if not _checkpoint(output / "gate/result.json", frozen["identity"]).get("pass"):
        raise ValueError("numerical smoke gate did not pass")
    task = frozen["tasks"][index]
    identity = content_hash([frozen["identity"], task])
    root = output / f"results/task_{index:03d}"
    if (root / "result.json").exists():
        return _checkpoint(root / "result.json", identity)
    problem = read_json(output / f"problems/{index:03d}.json")
    plan = FinalFitterPlan.model_validate(frozen["plan"])
    record = {
        "identity": identity,
        "task": task,
        "test_data_opened": False,
        "proposer_access": False,
    }
    fit_file = root / "fit.json"
    try:
        if fit_file.exists():
            fit = _checkpoint(fit_file, identity)["fit"]
        else:
            fit = fit_task(problem, task, plan, root, identity)
            write_json(fit_file, {"identity": identity, "fit": fit}, immutable=True)
        record["fit"] = fit
        record["status"] = "fit_failed"
        if fit.get("parameters"):
            system, _ = system_for(problem)
            train = unpack_split(problem["splits"]["train"])
            scale = TrainingScaler().fit(train).scales["target:v01"].standard_deviation
            clean = read_json(output / "inputs/clean.json")
            checks = {}
            for name in ("train", "val"):
                row = checkpoint_stage(
                    root / f"replay_{name}",
                    content_hash([identity, "replay", name]),
                    lambda name=name: {
                        "status": "finished",
                        "check": checked_replay(
                            system.model,
                            unpack_split(problem["splits"][name]),
                            fit["parameters"],
                            scale,
                            clean[name],
                            seconds=plan.replay_seconds,
                        ),
                    },
                )
                checks[name] = row.get(
                    "check", {"pass": False, "status": row["status"]}
                )
            record["replays"] = checks
            record["numerical_replay_pass"] = all(
                c.get("pass") for c in checks.values()
            )
            record["status"] = (
                "complete" if record["numerical_replay_pass"] else "replay_unverified"
            )
            scores = [
                c.get("scores", {}).get("Radau", {}).get("clean_nmse")
                for c in checks.values()
            ]
            record["fit_bands"] = {
                name: bool(
                    record["numerical_replay_pass"]
                    and all(s is not None and s <= limit for s in scores)
                )
                for name, limit in frozen["bands"].items()
            }
    except (RuntimeError, ValueError, ArithmeticError, TimeoutError) as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    record = _finite_payload(record)
    write_json(root / "result.json", record, immutable=True)
    return record
