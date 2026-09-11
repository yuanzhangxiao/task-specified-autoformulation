"""Budgeted derivative-free polling of exact rollout costs.

This is an experimental direct search, not an implementation of MADS and not a
stationarity certificate. Coordinate and seeded rotating directions, plus an
initial wider poll, reduce dependence on a single branch or flat starting point.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.fitting.stagnation import RolloutOracle
from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json


def poll_fit(
    oracle: RolloutOracle,
    starts: list[Mapping[str, float]],
    *,
    scales: np.ndarray,
    max_calls: int,
    seconds: float,
    checkpoint: Path,
    identity: str,
) -> dict:
    """Resume exact point order and cumulative budgets from atomic checkpoints.

    Only feasible full-training evaluations compete. Failure penalties never
    become incumbents. A radius stop describes searched resolution, not proven
    optimality. The caller performs fresh independent production replay.
    """
    vectors = [oracle.vector(start).tolist() for start in starts]
    scales = np.asarray(scales, dtype=float)
    if (
        scales.shape != oracle.lower.shape
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise ValueError("poll scales must be finite and positive")
    if not vectors or max_calls < 1 or seconds <= 0:
        raise ValueError("poll requires starts and positive budgets")
    state = (
        read_json(checkpoint)
        if checkpoint.exists()
        else {
            "identity": identity,
            "history": [],
            "best": None,
            "elapsed": 0.0,
            "queue": [list(point) for point in vectors],
            "iteration": 0,
            "radius": 1.0,
            "anchor_cost": None,
            "finished": False,
            "message": None,
        }
    )
    if state["identity"] != identity:
        raise ValueError("poll checkpoint identity differs")
    started, previous_seconds = monotonic(), state["elapsed"]
    if hasattr(oracle, "deadline"):
        oracle.deadline = min(
            oracle.deadline, started + max(0, seconds - previous_seconds)
        )

    def save():
        state["elapsed"] = previous_seconds + monotonic() - started
        write_json(checkpoint, state)

    while not state["finished"]:
        if (
            len(state["history"]) >= max_calls
            or previous_seconds + monotonic() - started >= seconds
        ):
            state.update(finished=True, message="poll fitting budget reached")
            break
        if not state["queue"]:
            best = state["best"]
            anchor = state["anchor_cost"]
            if state["iteration"]:
                improved = best is not None and (
                    anchor is None
                    or best["cost"] < anchor - 1e-12 * max(1.0, abs(anchor))
                )
                state["radius"] *= 1.5 if improved else 0.5
                state["radius"] = min(state["radius"], 16.0)
            if state["radius"] < 1e-5:
                state.update(
                    finished=True, message="poll radius limit; no stationarity claim"
                )
                break
            center = np.asarray(best["x"] if best else vectors[0])
            state["anchor_cost"] = best["cost"] if best else None
            n = len(center)
            directions = list(np.eye(n))
            if n > 1:
                # A fresh reproducible orthogonal basis also probes coupled moves.
                rng = np.random.default_rng(72019 + state["iteration"])
                directions.extend(np.linalg.qr(rng.normal(size=(n, n)))[0].T)
            radii = (1.0, 4.0, 16.0) if state["iteration"] == 0 else (state["radius"],)
            seen = {tuple(row["x"]) for row in state["history"]}
            queue = []
            for radius in radii:
                for direction in directions:
                    for sign in (1.0, -1.0):
                        point = np.clip(
                            center + sign * radius * scales * direction,
                            oracle.lower,
                            oracle.upper,
                        )
                        key = tuple(point)
                        if np.isfinite(point).all() and key not in seen:
                            queue.append(point.tolist())
                            seen.add(key)
            state["queue"] = queue
            state["iteration"] += 1
            save()
            if not queue:
                continue
        point = np.asarray(state["queue"][0])
        before = oracle.failures.count
        try:
            residual = oracle(point)
            cost = float(0.5 * residual @ residual)
            valid = oracle.failures.count == before and np.isfinite(cost)
        except TimeoutError:
            state["history"].append(
                {
                    "x": point.tolist(),
                    "cost": None,
                    "valid": False,
                    "timeout": True,
                }
            )
            state.update(finished=True, message="poll fitting wall-clock limit reached")
            break
        row = {
            "x": point.tolist(),
            "cost": cost if valid else None,
            "valid": bool(valid),
        }
        state["history"].append(row)
        state["queue"].pop(0)
        if valid and (state["best"] is None or cost < state["best"]["cost"]):
            state["best"] = row
        save()
    save()
    best = state["best"]
    return {
        "method": "directional_poll",
        "parameters": (
            dict(zip(oracle.names, best["x"], strict=True)) if best else None
        ),
        "cost": best["cost"] if best else None,
        "optimizer_success": False,
        "optimizer_native_success": False,
        "optimizer_status": 0,
        "optimality": None,
        "message": state["message"],
        "stationarity_claimed": False,
        "selection": "best_feasible_training_poll" if best else "no_feasible_rollout",
        "selected_training_rollout_verified": False,
        "numerical_status": "production_replay_pending"
        if best
        else "no_feasible_rollout",
        "actual_residual_calls": len(state["history"]),
        "valid_residual_evaluations": sum(row["valid"] for row in state["history"]),
        "integration_failures": sum(not row["valid"] for row in state["history"]),
        "fit_seconds": state["elapsed"],
        "poll_radius": state["radius"],
        "scales": scales.tolist(),
        "initial_parameters": dict(starts[0]),
    }
