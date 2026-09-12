"""Mapped Radau constraints and full-observation residuals on a chosen mesh."""

from __future__ import annotations

from time import monotonic

import casadi as ca
import numpy as np

from autoformalism.fitting.collocation_mesh import basis, forcing_values, plan_meshes
from autoformalism.rebuttal.fitter_diagnostic import write_json


def interval_function(system):
    """The same two-stage Radau IIA equations as the original interval loop."""
    n, p, nu = system.state_count, len(system.names), len(system.inputs)
    left, inner, end = [ca.MX.sym(k, n) for k in ("left", "inner", "end")]
    theta = ca.MX.sym("theta", p)
    t, dt = ca.MX.sym("time"), ca.MX.sym("dt")
    u0, u1 = ca.MX.sym("u0", nu), ca.MX.sym("u1", nu)
    first = system.rhs(t + dt / 3, inner, theta, u0 + (u1 - u0) / 3)
    last = system.rhs(t + dt, end, theta, u1)
    constraints = ca.vertcat(
        inner - left - dt * (5 * first - last) / 12,
        end - left - dt * (3 * first + last) / 4,
    )
    return ca.Function(
        "radau_interval", [left, inner, end, theta, t, dt, u0, u1], [constraints]
    )


def validate_matrix(system, time, states, theta, inputs):
    """Check guesses in bulk without requiring a successful ODE rollout."""
    count = len(time)
    args = [
        np.asarray(time).reshape(1, -1),
        states,
        np.tile(theta[:, None], (1, count)),
        inputs,
    ]
    outputs = [
        system.rhs.map(count)(*args),
        system.observe.map(count)(*args),
        *system.local.map(count)(*args),
    ]
    if any(not np.isfinite(np.asarray(x)).all() for x in outputs):
        raise ValueError(
            "mapped collocation guess has nonfinite equations or derivatives"
        )


def mapped_collocation(
    system,
    opti,
    theta,
    training,
    start,
    scale,
    directory,
    deadline,
    seed,
    target_variables=None,
    substeps=1,
):
    """Keep every training observation and every forcing interpolation corner.

    The mesh changes only the state polynomial representation. All original
    observation times evaluate that polynomial, including nonlinear mappings.
    """
    meshes, audit = plan_meshes(system, training, target_variables, substeps)
    write_json(directory / "mesh.json", audit)
    interval = interval_function(system)
    objective, nodes, guesses = 0, 0, []
    n = system.state_count
    for data, mesh in zip(training.trajectories, meshes, strict=True):
        if monotonic() >= deadline:
            raise TimeoutError("mapped collocation construction deadline reached")
        guess, record = seed(data)
        guesses.append(record)
        write_json(
            directory / "node_initialization.json",
            {
                "policy": record["policy"],
                "trajectories": guesses,
                "optimizer_started": False,
            },
        )
        boundaries = mesh.time
        count = len(boundaries) - 1
        u = forcing_values(system, data, boundaries)
        initial = system.initial_symbolic(data, theta)
        values = opti.variable(2 * n, count)
        end, inner = values[:n, :], values[n:, :]
        left = ca.horzcat(initial, end[:, :-1])
        guess_boundaries = np.array(
            [np.interp(boundaries, data.time, x) for x in guess.T]
        )
        guess_inner = (2 * guess_boundaries[:, :-1] + guess_boundaries[:, 1:]) / 3
        opti.set_initial(values, np.vstack([guess_boundaries[:, 1:], guess_inner]))
        nodes += 2 * n * count
        validate_matrix(system, boundaries, guess_boundaries, start, u)
        validate_matrix(
            system,
            boundaries[:-1] + np.diff(boundaries) / 3,
            guess_inner,
            start,
            u[:, :-1] + (u[:, 1:] - u[:, :-1]) / 3,
        )
        constraints = interval.map(count)(
            left,
            inner,
            end,
            ca.repmat(theta, 1, count),
            boundaries[:-1].reshape(1, -1),
            np.diff(boundaries).reshape(1, -1),
            u[:, :-1],
            u[:, 1:],
        )
        opti.subject_to(ca.vec(constraints) == 0)
        b0, b1, b2 = [x.reshape(1, -1) for x in basis(mesh.observation_fractions)]
        ix = mesh.observation_intervals.tolist()
        interpolated = (
            ca.times(left[:, ix], ca.repmat(b0, n, 1))
            + ca.times(inner[:, ix], ca.repmat(b1, n, 1))
            + ca.times(end[:, ix], ca.repmat(b2, n, 1))
        )
        predicted = system.observe.map(len(data.time))(
            data.time.reshape(1, -1),
            interpolated,
            ca.repmat(theta, 1, len(data.time)),
            forcing_values(system, data, data.time),
        )
        objective += ca.sumsqr(
            (predicted[0, :] - data.targets["v01"].reshape(1, -1)) / scale
        )
    audit["construction_finished_seconds_remaining"] = max(0.0, deadline - monotonic())
    write_json(directory / "mesh.json", audit)
    return objective, nodes, guesses
