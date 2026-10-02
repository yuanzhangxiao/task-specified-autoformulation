"""Training-only fixed-mesh factorial for formulation reuse and primal starts.

The graph is reusable within one process only. No previous IPOPT multipliers or
quasi-Newton state are imported. A completed rollout is the only endpoint score.
"""

from __future__ import annotations

from collections import Counter
from contextlib import suppress
from pathlib import Path
from time import monotonic

import numpy as np
from pydantic import Field

from autoformalism.fitting import adaptive_mesh as mesh
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.feasibility import EvaluationBudget, GuardedOracle
from autoformalism.fitting.identifiable_campaign import SETTINGS, scale_for
from autoformalism.fitting.transcription_solver import (
    TranscriptionProblem,
    build_problem,
)
from autoformalism.schemas.base import StrictSchema

ARMS = ("rebuild_cold", "cache_cold", "cache_last_primal", "cache_screened_primal")


class ReuseDiagnosticPolicy(StrictSchema):
    """Two equal native attempts; screening and construction are also charged."""

    native_seconds: float = Field(default=200, ge=1, le=450)
    iterations: int = Field(default=200, ge=1, le=1000)
    point_seconds: float = Field(default=10, ge=0.01, le=30)
    maximum_scaled_defect: float = Field(default=1e-6, gt=0)
    minimum_relative_improvement: float = Field(default=1e-3, ge=0, lt=1)


def screening_reasons(
    record: dict, ordinary_nmse: float | None, policy: ReuseDiagnosticPolicy
) -> list[str]:
    """Explain a veto without treating optimizer success as a quality certificate."""
    reasons = []
    if not record["within_parameter_bounds"]:
        reasons.append("parameter_bounds")
    cost = record.get("training_nmse")
    if cost is None or not np.isfinite(cost):
        reasons.append("training_rollout_unavailable")
    elif ordinary_nmse is not None and cost >= (
        ordinary_nmse * (1 - policy.minimum_relative_improvement)
    ):
        reasons.append("insufficient_training_improvement")
    if record["maximum_scaled_defect"] > policy.maximum_scaled_defect:
        reasons.append("dynamical_inconsistency")
    return reasons


def warm_start_decision(
    arm: str,
    records: list[dict],
    ordinary_nmse: float | None,
    policy: ReuseDiagnosticPolicy,
) -> dict:
    """Decide using finite primal vectors, feasibility and training evidence only."""
    if arm not in ARMS:
        raise ValueError("unknown reuse diagnostic arm")
    eligible = [r for r in records if r["within_parameter_bounds"]]
    if arm == "cache_last_primal" and eligible:
        return {"source": "last_finite_primal", "record": eligible[-1]}
    if arm == "cache_screened_primal":
        qualified = [
            r for r in eligible if not screening_reasons(r, ordinary_nmse, policy)
        ]
        if qualified:
            return {
                "source": "screened_primal",
                "record": min(qualified, key=lambda r: r["training_nmse"]),
            }
    return {"source": "ordinary", "record": None}


def _configure(problem: TranscriptionProblem, policy: ReuseDiagnosticPolicy) -> None:
    """Identical options for every arm, including the first solve and cold repeats."""
    problem.opti.solver(
        "ipopt",
        {"print_time": False},
        {
            "print_level": 0,
            "max_iter": policy.iterations,
            "max_cpu_time": policy.native_seconds,
            "tol": 1e-8,
            "warm_start_init_point": "no",
            "hessian_approximation": "limited-memory"
            if problem.system.has_piecewise
            else "exact",
        },
    )


def _snapshot(problem: TranscriptionProblem, iteration: int, value) -> dict:
    """Preserve the whole primal point; nodal fit is explicitly not a rollout."""
    primal = np.asarray(value(problem.opti.x)).ravel()
    vector = np.asarray(value(problem.theta)).ravel()
    loss = float(value(problem.objective))
    defect = float(np.max(abs(np.asarray(value(problem.defects)))))
    if (
        not np.isfinite(primal).all()
        or not np.isfinite(vector).all()
        or not np.isfinite([loss, defect]).all()
    ):
        raise ValueError("nonfinite_primal_or_objective")
    return {
        "iteration": iteration,
        "primal": primal.tolist(),
        "primal_sha256": public.content_sha256(primal.tolist()),
        "parameters": dict(zip(problem.system.names, vector.tolist(), strict=True)),
        "within_parameter_bounds": bool(
            np.all(vector >= problem.layout.lower)
            and np.all(vector <= problem.layout.upper)
        ),
        "collocation_nmse": loss,
        "maximum_scaled_defect": defect,
    }


def _pool(records: list[dict]) -> list[dict]:
    """Keep latest, least-defective and best feasible nodal fit, at most three."""
    if not records:
        return []
    feasible = [r for r in records if r["maximum_scaled_defect"] <= 1e-6]
    choices = [
        min(records, key=lambda r: r["maximum_scaled_defect"]),
        min(feasible or records, key=lambda r: r["collocation_nmse"]),
        records[-1],
    ]
    return sorted(
        {r["primal_sha256"]: r for r in choices}.values(),
        key=lambda r: r["iteration"],
    )


def native_attempt(
    problem: TranscriptionProblem,
    start: list[float],
    policy: ReuseDiagnosticPolicy,
    directory: Path,
    global_deadline: float,
) -> dict:
    """Run one uninterrupted native solve; persist checkpoints before any screening."""
    directory.mkdir(parents=True, exist_ok=True)
    opti = problem.opti
    opti.set_initial(opti.x, start)
    opti.set_initial(opti.lam_g, np.zeros(opti.ng))
    begun = monotonic()
    deadline = min(global_deadline, begun + policy.native_seconds)
    pool, rejected = [], Counter()
    latest_iteration = 0

    def capture(iteration, value):
        nonlocal pool
        try:
            record = _snapshot(problem, iteration, value)
            if record["within_parameter_bounds"]:
                pool = _pool([*pool, record])
            else:
                rejected["outside_parameter_bounds"] += 1
            public._write(directory / "checkpoints.json", {"pool": pool})
        except (RuntimeError, ValueError, ArithmeticError) as error:
            rejected[str(error)[-200:]] += 1
        public._write(directory / "checkpoint_rejections.json", dict(rejected))

    def callback(iteration):
        nonlocal latest_iteration
        latest_iteration = int(iteration)
        if iteration <= 3 or iteration % 5 == 0:
            capture(int(iteration), opti.debug.value)
        if monotonic() >= deadline:
            raise TimeoutError("diagnostic native allowance exhausted")

    opti.callback(callback)
    public._write(
        directory / "started.json",
        {
            "initial_primal_sha256": public.content_sha256(start),
            "dual_start": "explicit_zero",
            "maximum_iterations": policy.iterations,
            "maximum_native_seconds": policy.native_seconds,
            "warm_start_init_point": "no",
        },
    )
    error = None
    try:
        solved = opti.solve()
        capture(int(solved.stats()["iter_count"]), solved.value)
    except (RuntimeError, ValueError, TimeoutError) as exc:
        error = str(exc)[-1200:]
        capture(latest_iteration, opti.debug.value)
    stats = {}
    with suppress(RuntimeError):
        stats = opti.stats()
    result = {
        "native_success": bool(stats.get("success", False)),
        "return_status": stats.get("return_status"),
        "iterations": stats.get("iter_count"),
        "seconds": monotonic() - begun,
        "native_allowance_reached": monotonic() >= deadline,
        "error": error,
        "checkpoint_rejections": dict(rejected),
        "pool": pool,
    }
    public._write(directory / "native.json", result)
    return result


def fit(payload: dict, directory: Path) -> dict:
    """Two matched solves; no mesh adaptation, dual transfer or truth-based choice."""
    from autoformalism.fitting.transcription_fit import StrategyPolicy

    begun = monotonic()
    (directory / "screens").mkdir(parents=True, exist_ok=True)
    budget_policy = StrategyPolicy.model_validate(payload["policy"])
    policy = ReuseDiagnosticPolicy.model_validate(payload["reuse_diagnostic"])
    deadline = begun + max(0.01, budget_policy.seconds - 1)
    arm = payload["arm"]
    if arm not in ARMS:
        raise ValueError("unknown reuse diagnostic arm")
    native_payload = {
        k: payload[k] for k in ("request", "training", "coordinates", "nodes")
    }
    from autoformalism.schemas.public_fitting import PublicSplit

    training = public.unpack_split(PublicSplit.model_validate(payload["training"]))
    native_payload.update(
        method="collocation",
        seconds=max(0.01, deadline - monotonic()),
        tolerance=1e-8,
        meshes={
            r.trajectory_id: mesh.initial_mesh(r, "collocation")
            for r in training.trajectories
        },
    )
    # The candidate's initial parameter vector comes from the frozen request.
    from autoformalism.schemas.public_fitting import PublicFitRequest

    native_payload["start"] = public._lower(
        PublicFitRequest.model_validate(payload["request"])
    )[1]
    graph_begun = monotonic()
    problem = build_problem(native_payload, directory / "graph-0")
    _configure(problem, policy)
    graph_seconds = monotonic() - graph_begun
    ordinary = (
        np.asarray(problem.opti.debug.value(problem.opti.x, problem.opti.initial()))
        .ravel()
        .tolist()
    )
    budget = EvaluationBudget(deadline, budget_policy.maximum_rollout_calls)
    oracle = GuardedOracle(
        problem.system,
        training,
        scale_for(training),
        SETTINGS,
        directory / "rollouts",
        deadline,
        sensitivities=False,
        budget=budget,
        point_seconds=policy.point_seconds,
    )
    best, screens = None, []

    def screen(record: dict, source: str) -> dict:
        nonlocal best
        entry = {
            "source": source,
            "parameters": record["parameters"],
            "training_nmse": None,
            "status": "started",
        }
        index = len(screens)
        screens.append(entry)
        public._write(directory / "screens" / f"{index:03d}.json", entry)
        started = monotonic()
        try:
            before = oracle.valid_calls
            residual = oracle(oracle.vector(record["parameters"]))
            if oracle.valid_calls > before and np.isfinite(residual).all():
                cost = float(np.mean(residual**2))
                entry.update(status="complete", training_nmse=cost)
                if best is None or cost < best["training_nmse"]:
                    best = {
                        "parameters": record["parameters"],
                        "training_nmse": cost,
                        "source": source,
                    }
                    public._write(directory / "best.json", best)
            else:
                entry["status"] = "rollout_unavailable"
        except (ValueError, RuntimeError, TimeoutError, ArithmeticError) as error:
            entry.update(status="rollout_unavailable", error=str(error)[-500:])
        entry["seconds"] = monotonic() - started
        public._write(directory / "screens" / f"{index:03d}.json", entry)
        return {**record, "training_nmse": entry["training_nmse"]}

    baseline = screen({"parameters": native_payload["start"]}, "ordinary")
    attempts, records, graph_builds = [], [], 1
    for attempt in range(2):
        if monotonic() >= deadline:
            break
        decision = warm_start_decision(arm, records, baseline["training_nmse"], policy)
        if attempt == 1 and arm == "rebuild_cold":
            graph_begun = monotonic()
            native_payload["seconds"] = max(0.01, deadline - monotonic())
            problem = build_problem(native_payload, directory / "graph-1")
            _configure(problem, policy)
            graph_seconds = monotonic() - graph_begun
            graph_builds += 1
        elif attempt == 1:
            graph_seconds = 0.0
        point = decision["record"]
        initial = point["primal"] if point else ordinary
        folder = directory / f"attempt-{attempt}"
        folder.mkdir(parents=True, exist_ok=True)
        public._write(
            folder / "formulation.json",
            {
                "graph_builds_so_far": graph_builds,
                "graph_reused": attempt > 0 and arm != "rebuild_cold",
                "graph_seconds": graph_seconds,
                "start_source": decision["source"],
                "start_checkpoint_sha256": point["primal_sha256"] if point else None,
                "decision_variables": int(problem.opti.nx),
                "constraints": int(problem.opti.ng),
                "observation_residuals": problem.observation_count,
            },
        )
        public._write(
            folder / "warm_start_decision.json",
            {
                "source": decision["source"],
                "ordinary_training_nmse": baseline["training_nmse"],
                "checkpoints": [
                    {
                        "primal_sha256": r["primal_sha256"],
                        "training_nmse": r.get("training_nmse"),
                        "maximum_scaled_defect": r["maximum_scaled_defect"],
                        "screened_policy_vetoes": screening_reasons(
                            r, baseline["training_nmse"], policy
                        ),
                    }
                    for r in records
                ],
            },
        )
        result = native_attempt(problem, initial, policy, folder, deadline)
        records = []
        for item in result.pop("pool"):
            if item["within_parameter_bounds"]:
                item = screen(item, f"attempt-{attempt}:{item['primal_sha256']}")
            records.append(item)
        attempts.append(
            {
                **result,
                "attempt": attempt,
                "graph_seconds": graph_seconds,
                "start_source": decision["source"],
                "initial_primal_sha256": public.content_sha256(initial),
            }
        )
        public._write(directory / "attempts.json", attempts)
    result = {
        "arm": arm,
        "parameters": best["parameters"] if best else None,
        "training_nmse": best["training_nmse"] if best else None,
        "stop_reason": "two_attempts_complete"
        if len(attempts) == 2
        else "budget_exhausted",
        "budget_exhausted": monotonic() >= deadline or budget.calls >= budget.maximum,
        "actual_residual_calls": budget.calls,
        "seconds": monotonic() - begun,
        "graph_builds": graph_builds,
        "screen_seconds": sum(r["seconds"] for r in screens),
        "attempts": attempts,
        "selection": "best_complete_training_rollout_including_ordinary_start",
        "validation_used_for_fitting": False,
        "reference_values_used": False,
        "scope": "fixed mesh, primal starts only; no production promotion",
    }
    public._write(directory / "result.json", result)
    return result
