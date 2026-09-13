"""Scale-aware reference checks isolated from fitting and proposer feedback."""

from __future__ import annotations

from itertools import pairwise
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field
from scipy.integrate import solve_ivp

from autoformalism.benchmarks.phase_b_generation import (
    PhaseBProtocol,
    _scalar_input,
    phase_b_protocols,
)
from autoformalism.data import TrainingScaler
from autoformalism.data.models import Trajectory
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.rebuttal.attainability_controls import system_for
from autoformalism.rebuttal.attainability_reference import native_replay_audit
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.schemas.base import StrictSchema


class ReferenceAuditConfig(StrictSchema):
    """Engineering tolerances, frozen separately from the recovery threshold."""

    protocol: Literal["scaled-reference-audit-2"] = "scaled-reference-audit-2"
    saved_absolute_tolerance: float = Field(default=1e-8, gt=0)
    saved_scale_tolerance: float = Field(default=1e-4, gt=0)
    solver_absolute_tolerance: float = Field(default=1e-10, gt=0)
    solver_scale_tolerance: float = Field(default=1e-6, gt=0)
    integration_rtol: float = Field(default=1e-10, gt=0)
    integration_atol: float = Field(default=1e-12, gt=0)
    maximum_seconds: float = Field(default=600, gt=0, le=600)


def agreement(left, right, scale: float, absolute: float, relative: float) -> dict:
    """Reject nonfinite/misaligned results; report error without unit dependence."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (
        left.shape != right.shape
        or left.size == 0
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(
            "reference comparison requires aligned finite arrays and scale"
        )
    delta = left - right
    maximum = float(np.max(abs(delta)))
    limit = absolute + relative * scale
    return {
        "maximum_absolute_difference": maximum,
        "maximum_scaled_difference": maximum / scale,
        "normalized_mse": float(np.mean((delta / scale) ** 2)),
        "absolute_limit": limit,
        "pass": maximum <= limit,
    }


def forcing_edges(protocol: PhaseBProtocol) -> list[float]:
    """Preserve continuous forcing laws and split exactly at their jump times."""
    spec = protocol.specification
    edges = [0.0, protocol.duration]
    if spec["kind"] in {"step", "pulse"}:
        edges += [spec["start"], spec["end"]]
    elif spec["kind"] == "pulses":
        edges += [t for pulse in spec["pulses"] for t in pulse[:2]]
    return sorted({float(t) for t in edges if 0 <= t <= protocol.duration})


def continuous_rollout(
    system: SymbolicODE,
    trajectory: Trajectory,
    protocol: PhaseBProtocol,
    truth: dict[str, float],
    method: str,
    config: ReferenceAuditConfig,
    deadline: float,
) -> np.ndarray:
    """Replay compiled equations with native continuous forcing, without fitting."""
    theta = np.array([truth[n] for n in system.names])
    state = system.initial_for(trajectory, theta)
    times = np.asarray(trajectory.time)
    states = np.empty((len(times), system.state_count))
    states[0] = state
    edges = forcing_edges(protocol)
    for left, right in pairwise(edges):
        indices = np.flatnonzero((times > left) & (times <= right))
        requested = np.unique(np.r_[times[indices], right])

        def forcing(t, left=left, right=right):
            # Evaluate the left limit at this segment's right endpoint.
            value = _scalar_input(
                min(float(t), np.nextafter(right, left)), protocol.specification
            )
            return [
                value if name == "u01" else trajectory.fixed_covariates[name]
                for name in system.inputs
            ]

        def rhs(t, x):
            if monotonic() >= deadline:
                raise TimeoutError("reference audit wall-clock limit reached")
            return np.asarray(system.rhs(t, x, theta, forcing(t))).ravel()

        def jac(t, x):
            return np.asarray(system.state_jacobian(t, x, theta, forcing(t)))

        solved = solve_ivp(
            rhs,
            (left, right),
            state,
            method=method,
            jac=jac,
            t_eval=requested,
            rtol=config.integration_rtol,
            atol=config.integration_atol,
        )
        if not solved.success or not np.isfinite(solved.y).all():
            raise ValueError(
                f"tight reference {method} replay failed: {solved.message}"
            )
        states[indices] = solved.y[:, np.searchsorted(requested, times[indices])].T
        state = solved.y[:, -1]
    return states[:, system.model.state_names.index("v01")]


def scaled_reference_audit(
    problem: dict, bundle: dict, config: ReferenceAuditConfig
) -> dict:
    """Require native/data agreement plus independent tight compiled rollouts.

    Only the training output scale defines all train/validation error budgets.
    The legacy native generator is independent of the restricted compiler. Its
    original numerical settings are retained as provenance, not made stricter.
    """
    deadline = monotonic() + config.maximum_seconds
    splits = {k: unpack_split(problem["splits"][k]) for k in ("train", "val")}
    scale = (
        TrainingScaler().fit(splits["train"]).scales["target:v01"].standard_deviation
    )
    native = native_replay_audit(problem, bundle["spec"])
    system, _ = system_for(problem)
    records = {}
    for key, split_name in (("train", "train"), ("val", "validation")):
        protocols = [
            p for p in phase_b_protocols("alien_device") if p.split == split_name
        ]
        rows = {r.trajectory_id: r for r in splits[key].trajectories}
        records[key] = []
        for i, protocol in enumerate(protocols):
            row = rows[f"{split_name}_{i:03d}"]
            predictions = {
                method: continuous_rollout(
                    system, row, protocol, bundle["truth"], method, config, deadline
                )
                for method in ("Radau", "BDF")
            }
            saved = {
                method: agreement(
                    y,
                    row.targets["v01"],
                    scale,
                    config.saved_absolute_tolerance,
                    config.saved_scale_tolerance,
                )
                for method, y in predictions.items()
            }
            independent = agreement(
                predictions["Radau"],
                predictions["BDF"],
                scale,
                config.solver_absolute_tolerance,
                config.solver_scale_tolerance,
            )
            native_row = dict(native[key][i])
            limit = (
                config.saved_absolute_tolerance + config.saved_scale_tolerance * scale
            )
            native_row.update(
                absolute_limit=limit,
                maximum_scaled_difference=native_row["maximum_absolute_difference"]
                / scale,
                pass_check=native_row["maximum_absolute_difference"] <= limit,
            )
            records[key].append(
                {
                    "trajectory": row.trajectory_id,
                    "native": native_row,
                    "tight_vs_saved": saved,
                    "tight_solver_agreement": independent,
                    "pass": native_row["pass_check"]
                    and independent["pass"]
                    and all(item["pass"] for item in saved.values()),
                }
            )
    return {
        "configuration": config.model_dump(mode="json"),
        "training_output_scale": scale,
        "native_generator_audit": native,
        "checks": records,
        "pass": all(row["pass"] for rows in records.values() for row in rows),
        "test_data_opened": False,
        "proposer_access": False,
    }
