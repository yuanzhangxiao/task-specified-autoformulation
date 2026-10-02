"""Budgeted training-only strategy comparison; no reference or validation input.

Native subproblems run in killable subprocesses. Retained solutions are selected
only by complete training rollouts, including those from nonconverged NLP iterates.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from time import monotonic

import numpy as np
from pydantic import Field

from autoformalism.fitting import adaptive_mesh as mesh
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.coordinates import NumericalCoordinates
from autoformalism.fitting.feasibility import (
    EvaluationBudget,
    GuardedOracle,
    restart_points,
)
from autoformalism.fitting.identifiable_campaign import SETTINGS, scale_for
from autoformalism.fitting.identifiable_refinement import RefinementPolicy, refine
from autoformalism.fitting.matching_probe import bounded_latent_start
from autoformalism.fitting.sensitivity_probe import SymbolicODE, SymbolicOracle
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

ARMS = ("rollout_only", "collocation_only", "adaptive_shooting", "collocation_rollout")
REUSE_ARM = "collocation_reuse"


class StrategyPolicy(StrictSchema):
    """Identical total wall-time ceiling; backend-specific counts are descriptive."""

    seconds: float = Field(default=180, ge=5, le=1200)
    maximum_rollout_calls: int = Field(default=180, ge=10, le=2000)
    mesh_passes: int = Field(default=3, ge=1, le=5)
    training_nmse: float = Field(default=1e-10, gt=0)
    trajectory_nmse: float = Field(default=1e-9, gt=0)
    collocation_indicator: float = Field(default=1e-6, gt=0)
    shooting_indicator: float = Field(default=0.05, gt=0)


def invoke(
    payload: dict, directory: Path, seconds: float, *, native: bool = False
) -> dict:
    """Bound process startup, numerical work and checkpoint recovery separately.

    The outer fitter owns a process group so killing it also kills an active NLP
    descendant. Logs and atomically written checkpoints survive a wall timeout.
    """
    begun = monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    public._write(directory / "payload.json", payload)
    command = [
        sys.executable,
        "-m",
        "autoformalism.fitting.transcription_fit",
        "native" if native else "fit",
        str(directory),
    ]
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=log, start_new_session=not native
        )

        def terminate():
            with suppress(ProcessLookupError):
                if native:
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    "native process did not exit after SIGKILL; stop campaign"
                ) from error

        timed_out = False
        try:
            process.wait(timeout=max(0.001, seconds - (monotonic() - begun)))
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate()
        except BaseException:
            terminate()
            public._write(
                directory / "process.json",
                {
                    "interrupted": True,
                    "returncode": process.returncode,
                    "elapsed_seconds": monotonic() - begun,
                },
            )
            raise
    status = {
        "wall_timeout": timed_out,
        "returncode": process.returncode,
        "allowance_seconds": seconds,
        "elapsed_seconds": monotonic() - begun,
        "cleanup_grace_seconds": 2,
    }
    public._write(directory / "process.json", status)
    return status


def fit(payload: dict, directory: Path) -> dict:
    """Compare four methods from identical physical starts and training nodes."""
    if payload.get("reuse_diagnostic") is not None:
        from autoformalism.fitting.reuse_diagnostic import fit as diagnostic_fit

        return diagnostic_fit(payload, directory)
    begun = monotonic()
    policy = StrategyPolicy.model_validate(payload["policy"])
    # Parent includes process startup in its wall ceiling. This slightly shorter
    # internal deadline leaves time to flush final metadata on ordinary exits.
    deadline = begun + max(0.01, policy.seconds - 1)
    arm = payload["arm"]
    if arm not in (*ARMS, REUSE_ARM):
        raise ValueError("unknown strategy")
    public_train = PublicSplit.model_validate(payload["training"])
    if public_train.name != "train":
        raise ValueError("fitting requires training data only")
    request = PublicFitRequest.model_validate(payload["request"])
    model, start, _ = public._lower(request)
    train = public.unpack_split(public_train)
    system = SymbolicODE(model, allow_piecewise=True)
    coords = NumericalCoordinates.model_validate(payload["coordinates"])
    scales = scale_for(train)
    layout = SymbolicOracle(
        system, train, scales, SETTINGS, directory / "layout", None, sensitivities=False
    )
    budget = EvaluationBudget(deadline, policy.maximum_rollout_calls)
    stages, best = [], None
    sizes = [len(r.time) * len(system.channels) for r in train.trajectories]
    stop = "no_feasible_training_point"

    def remaining():
        return max(0.001, deadline - monotonic())

    def screen(points, seconds):
        nonlocal best
        end = min(deadline, monotonic() + seconds)
        oracle = GuardedOracle(
            system,
            train,
            scales,
            SETTINGS,
            directory / f"screens-{len(stages)}",
            deadline,
            sensitivities=False,
            budget=budget,
            point_seconds=seconds,
        )
        for point in points:
            if monotonic() >= end or budget.calls >= budget.maximum:
                break
            oracle.point_seconds = max(0.001, end - monotonic())
            try:
                valid_before = oracle.valid_calls
                residual = oracle(oracle.vector(point["parameters"]))
                if oracle.valid_calls == valid_before:
                    continue
                cost = float(np.mean(residual**2))
                if best is None or cost < best["training_nmse"]:
                    best = {
                        "parameters": point["parameters"],
                        "training_nmse": cost,
                        "maximum_trajectory_nmse": max(
                            float(np.mean(v**2))
                            for v in np.split(residual, np.cumsum(sizes)[:-1])
                        ),
                        "source": point.get("source", "NLP_checkpoint"),
                    }
                    public._write(directory / "best.json", best)
            except (ValueError, RuntimeError, TimeoutError, ArithmeticError):
                continue
        public._write(
            directory / "screen_budget.json", {"actual_residual_calls": budget.calls}
        )

    def accurate():
        return (
            best is not None
            and best["training_nmse"] <= policy.training_nmse
            and best["maximum_trajectory_nmse"] <= policy.trajectory_nmse
        )

    def rollout(points):
        nonlocal best, stop
        result = refine(
            system,
            train,
            scales,
            SETTINGS,
            coords,
            points,
            RefinementPolicy(
                seconds=min(1200, remaining()),
                maximum_calls=max(5, budget.maximum - budget.calls),
                training_nmse=policy.training_nmse,
                trajectory_nmse=policy.trajectory_nmse,
            ),
            "joint_stopping",
            directory / "refinement",
        )
        budget.calls += result["actual_residual_calls"]
        stages.append({"method": "rollout", **result})
        stop = result["stop_reason"]
        if result["parameters"] is not None and (
            best is None or result["training_nmse"] < best["training_nmse"]
        ):
            best = {
                "parameters": result["parameters"],
                "training_nmse": result["training_nmse"],
                "source": "rollout_refinement",
            }
            public._write(directory / "best.json", best)

    if arm == "rollout_only":
        rollout([{"source": "ordinary", "parameters": start}])
    elif arm == "collocation_rollout":
        # Matched-budget version of M2's existing pipeline, with the same frozen
        # physical node guesses. No shared initializer is given free extra time.
        initialized = bounded_latent_start(
            system,
            training=train,
            lower=layout.lower,
            upper=layout.upper,
            start=layout.vector(start),
            scale=scales,
            settings=SETTINGS,
            method="collocation_init",
            seconds=min(policy.seconds / 3, remaining()),
            directory=directory / "collocation",
            node_start="rollout_or_observed",
            record_progress=True,
            numerical_coordinates=coords,
            frozen_nodes=payload["nodes"],
            checkpoint_pool_capacity=8,
        )
        points = restart_points(initialized, start, [])
        for index, point in enumerate(
            (initialized.get("progress") or {}).get("checkpoint_pool", [])
        ):
            points.append(
                {"source": f"pool_{index}", "parameters": point["parameters"]}
            )
        stages.append({"method": "fixed_collocation", "result": initialized})
        if remaining() > 1:
            rollout(points)
    else:
        method = "shooting" if arm == "adaptive_shooting" else "collocation"
        reuse = arm == REUSE_ARM
        grids = {
            r.trajectory_id: mesh.initial_mesh(r, method) for r in train.trajectories
        }
        guesses = {
            r.trajectory_id: mesh.interpolate_nodes(
                r.time, payload["nodes"][r.trajectory_id], grids[r.trajectory_id]
            ).tolist()
            for r in train.trajectories
        }
        vector = dict(start)
        # Preserve the ordinary physical point before a potentially inconsistent NLP.
        screen(
            [{"parameters": start, "source": "ordinary"}], min(15, policy.seconds * 0.1)
        )
        for level in range(policy.mesh_passes):
            if remaining() <= 2 or accurate():
                break
            # The opt-in reuse arm solves the current mesh before spending time
            # on a larger one. Reserve a quarter of the remaining budget for
            # actual training screens; unused native time stays available.
            slice_seconds = (
                remaining() if reuse else remaining() / (policy.mesh_passes - level)
            )
            allowance = 0.75 * slice_seconds
            tolerance = (1e-5, 1e-7, 1e-9)[min(level, 2)]
            stage_dir = directory / f"mesh-{level}"
            native_payload = {
                "request": payload["request"],
                "training": payload["training"],
                "start": vector,
                "coordinates": payload["coordinates"],
                "meshes": grids,
                "nodes": guesses,
                "method": method,
                "tolerance": tolerance,
                "seconds": allowance,
            }
            if reuse:
                native_payload["reuse_chunks"] = 8
            process = invoke(native_payload, stage_dir, allowance, native=True)
            record = {
                "method": method,
                "level": level,
                "process": process,
                "tolerance": tolerance,
                "intervals": {k: len(v) - 1 for k, v in grids.items()},
            }
            checkpoint_file = stage_dir / "checkpoints.json"
            if not checkpoint_file.exists():
                stages.append({**record, "checkpoint_available": False})
                stop = "native_stage_unavailable"
                break
            checkpoints = public._read(checkpoint_file)
            pool = checkpoints["pool"]
            # Rank feasible discretized points first but still screen recent
            # infeasible points. Only the independent training rollout can select.
            pool = sorted(
                pool,
                key=lambda p: (
                    p["maximum_scaled_defect"] > 1e-6,
                    p["collocation_nmse"],
                ),
            )
            screen(pool, max(0.001, slice_seconds - process["elapsed_seconds"]))
            latest = checkpoints["latest"]
            vector = latest["parameters"]
            record.update(
                checkpoint_available=True,
                checkpoint_count=len(pool),
                maximum_scaled_defect=latest["maximum_scaled_defect"],
                discretized_training_nmse=latest["collocation_nmse"],
            )
            stages.append(record)
            public._write(directory / "stages.json", stages)
            native_result = (
                public._read(stage_dir / "native.json")
                if (stage_dir / "native.json").exists()
                else {}
            )
            if reuse and not native_result.get("native_success"):
                stop = "coarse_solve_incomplete_pending_replay"
                break
            threshold = (
                policy.collocation_indicator
                if method == "collocation"
                else policy.shooting_indicator
            )
            changed = False
            for row in train.trajectories:
                key = row.trajectory_id
                old = latest["trajectories"][key]
                new_grid = mesh.refine_mesh(
                    grids[key],
                    old["indicators"],
                    threshold=threshold,
                    maximum_intervals=4 * (len(row.time) - 1),
                )
                changed |= new_grid != grids[key]
                guesses[key] = mesh.interpolate_nodes(
                    old["mesh"], old["nodes"], new_grid
                )
                guesses[key][0] = system.initial_for(row, layout.vector(vector))
                guesses[key] = guesses[key].tolist()
                grids[key] = new_grid
            stop = (
                "training_accuracy_reached_pending_replay"
                if accurate()
                else "mesh_passes_exhausted"
            )
            # A converged mesh still gets tighter integration in the next shooting
            # pass. Collocation repeats for refinement or native solver failure.
            if (
                not changed
                and method == "collocation"
                and native_result.get("native_success")
            ):
                stop = "mesh_indicator_satisfied_pending_replay"
                break
    result = {
        "arm": arm,
        "parameters": best["parameters"] if best else None,
        "training_nmse": best["training_nmse"] if best else None,
        "stop_reason": stop,
        "actual_residual_calls": budget.calls,
        "seconds": monotonic() - begun,
        "budget_exhausted": monotonic() >= deadline or budget.calls >= budget.maximum,
        "mesh_stage_timeouts": sum(
            bool(s.get("process", {}).get("wall_timeout")) for s in stages
        ),
        "stages": stages,
        "selection": "best_complete_training_rollout",
        "validation_used_for_fitting": False,
        "reference_values_used": False,
    }
    public._write(directory / "result.json", result)
    return result


def run(payload: dict, directory: Path) -> dict:
    """Enforce the arm's total budget, recovering only verified rollout parameters."""
    process = invoke(
        payload, directory, StrategyPolicy.model_validate(payload["policy"]).seconds
    )
    if (directory / "result.json").exists():
        result = public._read(directory / "result.json")
    else:
        # Both screening and rollout refinement atomically persist their verified
        # incumbent. Compare them if interruption preceded the outer final write.
        candidates = []
        for path in (directory / "best.json", directory / "refinement/best.json"):
            if path.exists():
                candidate = public._read(path)
                if np.isfinite(candidate.get("training_nmse", float("nan"))):
                    candidates.append(candidate)
        best = min(candidates, key=lambda x: x["training_nmse"]) if candidates else {}
        result = {
            "arm": payload["arm"],
            "parameters": best.get("parameters"),
            "training_nmse": best.get("training_nmse"),
            "stop_reason": "wall_budget_exhausted"
            if process["wall_timeout"]
            else "worker_failed",
            "budget_exhausted": process["wall_timeout"],
            "partial_metadata": True,
        }
        if payload.get("reuse_diagnostic") is not None:
            # Recover metadata without retrying a consumed native/screen budget.
            attempts = []
            for attempt in range(2):
                folder = directory / f"attempt-{attempt}"
                if not (folder / "formulation.json").exists():
                    continue
                saved = public._read(folder / "formulation.json")
                if (folder / "native.json").exists():
                    native = public._read(folder / "native.json")
                    native.pop("pool", None)
                    saved.update(native)
                else:
                    saved["status"] = "interrupted"
                attempts.append({"attempt": attempt, **saved})
            result["attempts"] = attempts
            result["graph_builds"] = (
                max((a["graph_builds_so_far"] for a in attempts), default=0) or None
            )
    result["worker_payload_sha256"] = public.content_sha256(payload)
    result["process"] = process
    result["total_seconds"] = process["elapsed_seconds"]
    return result


if __name__ == "__main__":
    mode, folder = sys.argv[1:]
    directory = Path(folder)
    payload = public._read(directory / "payload.json")
    if mode == "native":
        from autoformalism.fitting.transcription_solver import solve

        solve(payload, directory)
    elif mode == "fit":
        fit(payload, directory)
    else:
        raise ValueError("unknown worker mode")
