"""Infer discharge laws from public basin balances and predict held training events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.research.observed_process_laws import (
    fit_law_menu,
    interval_mean_law,
    predict_law,
)
from autoformalism.search.topology_compatibility import CompleteTrajectory


def _public_event(raw: CompleteTrajectory | dict) -> CompleteTrajectory:
    """Accept only the training channels in the public coupled basin contract."""
    event = CompleteTrajectory.model_validate(raw)
    if not event.trajectory_id.startswith("train_"):
        raise ValueError("Expected a public training event")
    if (
        set(event.targets) != {"h_down"}
        or event.auxiliaries
        or set(event.inputs) != {"inflow_up", "inflow_down"}
        or len(event.time) < 2
    ):
        raise ValueError(
            "Expected downstream depth and the two public runoff histories"
        )
    required = {"area_up", "area_down", "initial_up", "crest_up", "crest_down"}
    c = event.covariates
    if not required <= c.keys() or min(c["area_up"], c["area_down"]) <= 0:
        raise ValueError("Need both positive areas, crests and initial upstream depth")
    if min(c["initial_up"], c["crest_up"], c["crest_down"]) < 0 or any(
        min(channel) < 0 for channel in event.inputs.values()
    ):
        raise ValueError("Isolation assumes nonnegative surveyed depths and runoff")
    return event


@dataclass(frozen=True)
class _Intervals:
    event: CompleteTrajectory
    dt: np.ndarray
    inlet_up: np.ndarray
    inlet_down: np.ndarray
    head_down: np.ndarray
    storage_change_down: np.ndarray
    isolated: np.ndarray

    @classmethod
    def collect(cls, event: CompleteTrajectory) -> _Intervals:
        """Return volume changes and a conservative no-transfer certificate."""
        dt = np.diff(event.time)
        up = np.asarray(event.inputs["inflow_up"])
        down = np.asarray(event.inputs["inflow_down"])
        depth = np.asarray(event.targets["h_down"])
        inlet_up = dt * (up[1:] + up[:-1]) / 2
        c = event.covariates
        room = c["area_up"] * (c["crest_up"] - c["initial_up"])
        return cls(
            event,
            dt,
            inlet_up,
            dt * (down[1:] + down[:-1]) / 2,
            depth - c["crest_down"],
            c["area_down"] * np.diff(depth),
            np.cumsum(inlet_up) <= room,
        )


def _sample(block: _Intervals, head: np.ndarray, volume: np.ndarray, keep) -> dict:
    """Form paired endpoint displacements and interval-average volumetric rates."""
    return {
        "trajectory_id": block.event.trajectory_id,
        "left": head[:-1][keep].tolist(),
        "right": head[1:][keep].tolist(),
        "mean_rate": (volume[keep] / block.dt[keep]).tolist(),
    }


def _outlet_evidence(blocks: list[_Intervals]) -> tuple[dict, list[dict]]:
    samples = [
        _sample(b, b.head_down, b.inlet_down - b.storage_change_down, b.isolated)
        for b in blocks
        if np.any(b.isolated)
    ]
    heads = [v for row in samples for side in ("left", "right") for v in row[side]]
    return {
        "isolated_intervals": {
            b.event.trajectory_id: int(b.isolated.sum()) for b in blocks
        },
        "events_with_isolation": len(samples),
        "head_range_m": [min(heads), max(heads)] if heads else None,
    }, samples


def _transfer_evidence(blocks: list[_Intervals], outlet: dict) -> tuple[dict, list]:
    """Recover upstream head conditionally on the fitted downstream discharge law."""
    rows, samples = {}, []
    for b in blocks:
        c = b.event.covariates
        outlet_volume = b.dt * interval_mean_law(
            outlet, b.head_down[:-1], b.head_down[1:]
        )
        transfer = b.storage_change_down - b.inlet_down + outlet_volume
        # Available upstream storage and cumulative runoff prove zero transfer here.
        # Record the outlet's discrepancy there before imposing that known identity.
        isolated_residual = transfer[b.isolated] / b.dt[b.isolated]
        transfer = np.where(b.isolated, 0.0, transfer)
        depth = (
            c["initial_up"] + np.r_[0, np.cumsum(b.inlet_up - transfer)] / c["area_up"]
        )
        keep = ~b.isolated
        rate = transfer / b.dt
        rows[b.event.trajectory_id] = {
            "conditional_intervals": int(keep.sum()),
            "min_inferred_transfer_m3_per_min": float(rate.min()),
            "min_inferred_upstream_depth_m": float(depth.min()),
            "isolated_max_abs_residual_m3_per_min": (
                float(np.max(abs(isolated_residual)))
                if isolated_residual.size
                else None
            ),
        }
        if np.any(keep):
            samples.append(_sample(b, depth - c["crest_up"], transfer, keep))
    return rows, samples


def _choose(menu: dict, strategy: str) -> dict | None:
    """Use the pre-scoring menu order, or require the registered linear control."""
    if strategy == "nominated":
        return next(iter(menu["candidates"]), None)
    return next((c for c in menu["candidates"] if c["family"] == "linear"), None)


def _fit_event_menu(samples: list[dict]) -> dict:
    if samples:
        return fit_law_menu(samples)
    return {"status": "no_intervals", "candidates": []}


def fit_basin_laws(training_rows: Sequence) -> dict:
    """Fit two full laws and a paired linear control outside the held event.

    Only the downstream discharge on certified no-transfer intervals has direct
    supervision. Both upstream transfer rates and upstream heads are reconstructed
    under the selected outlet and the piecewise-linear observed depth approximation.
    """
    events = sorted(
        (_public_event(r) for r in training_rows), key=lambda r: r.trajectory_id
    )
    ids = [r.trajectory_id for r in events]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Need distinct nonempty training events")
    blocks = [_Intervals.collect(e) for e in events]
    evidence, samples = _outlet_evidence(blocks)
    down_menu = _fit_event_menu(samples)
    result = {
        "training_events": ids,
        "training_depth_std_m": max(
            1e-8, float(np.std(np.concatenate([e.targets["h_down"] for e in events])))
        ),
        "outlet_evidence": evidence,
        "outlet_menu": down_menu,
        "models": [],
    }
    for strategy in ("nominated", "linear_control"):
        down = _choose(down_menu, strategy)
        model = {"strategy": strategy, "downstream": down, "upstream": None}
        if down is None:
            model["status"] = "outlet_unavailable"
        else:
            diagnostic, samples = _transfer_evidence(blocks, down)
            up_menu = _fit_event_menu(samples)
            up = _choose(up_menu, strategy)
            model.update(
                upstream=up,
                upstream_evidence="conditional_on_fitted_downstream_law",
                upstream_menu=up_menu,
                reconstruction=diagnostic,
                status="fitted" if up is not None else "transfer_unavailable",
            )
        result["models"].append(model)
    return result


def _depth_forecast(model: dict, event: CompleteTrajectory) -> np.ndarray:
    """Roll out both physical balances using only initial depths and supplied inlets."""
    t = np.asarray(event.time)
    c = event.covariates
    up, down = model["upstream"], model["downstream"]
    in_up = np.asarray(event.inputs["inflow_up"])
    in_down = np.asarray(event.inputs["inflow_down"])

    def rhs(when, depths):
        transfer = predict_law(up, depths[0] - c["crest_up"])
        outlet = predict_law(down, depths[1] - c["crest_down"])
        return [
            (np.interp(when, t, in_up) - transfer) / c["area_up"],
            (np.interp(when, t, in_down) + transfer - outlet) / c["area_down"],
        ]

    solution = solve_ivp(
        rhs,
        (t[0], t[-1]),
        [c["initial_up"], event.targets["h_down"][0]],
        t_eval=t,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
        max_step=float(np.min(np.diff(t))) / 2,
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise ValueError(f"Downstream rollout failed: {solution.message}")
    return solution.y[1]


def run_fold(training_rows: Sequence, held_row: CompleteTrajectory | dict) -> dict:
    """Fit on other events and score the entire post-initial downstream curve."""
    held = _public_event(held_row)
    events = [_public_event(r) for r in training_rows]
    if held.trajectory_id in {r.trajectory_id for r in events}:
        raise ValueError("Held event also appears in training")
    fit = fit_basin_laws(events)
    for model in fit["models"]:
        model.update(
            held_nmse=None,
            held_rmse_m=None,
            held_samples=0,
            held_predicted_depth_m=None,
        )
        if model["status"] != "fitted":
            continue
        try:
            predicted = _depth_forecast(model, held)
            error = predicted[1:] - np.asarray(held.targets["h_down"])[1:]
            rmse = float(np.sqrt(np.mean(error**2)))
            nmse = (rmse / fit["training_depth_std_m"]) ** 2
            if not np.isfinite(nmse):
                raise ValueError("Downstream error overflowed")
        except (ValueError, OverflowError, FloatingPointError) as exc:
            model.update(status="rollout_failed", detail=str(exc))
            continue
        model.update(
            status="scored",
            held_rmse_m=rmse,
            held_nmse=nmse,
            held_samples=len(error),
            held_predicted_depth_m=predicted.tolist(),
        )
    head = np.asarray(held.targets["h_down"]) - held.covariates["crest_down"]
    support = fit["outlet_evidence"]["head_range_m"]
    return {
        "held_event": held.trajectory_id,
        "held_time_min": list(held.time),
        "held_observed_depth_m": list(held.targets["h_down"]),
        "held_outlet_head_range_m": [float(head.min()), float(head.max())],
        "held_points_outside_outlet_fit_range": (
            int(np.sum((head < support[0]) | (head > support[1]))) if support else None
        ),
        **fit,
    }
