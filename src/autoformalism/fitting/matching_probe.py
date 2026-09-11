"""Training-only matching and latent trajectory initializers for a frozen probe."""

from __future__ import annotations

import multiprocessing
from itertools import pairwise
from pathlib import Path
from time import monotonic
from typing import Literal

import casadi as ca
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import compile_candidate
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    read_json,
    write_json,
)
from autoformalism.schemas import CandidateModel

NodeStartPolicy = Literal["rollout_required", "rollout_or_observed"]


def observed_node_guess(system: SymbolicODE, data: Trajectory) -> np.ndarray:
    """Seed observed states from training data and latent states from their initials.

    These are optimization guesses, not estimates used in final simulation.
    Prescribed initial conditions override the first observed sample.
    """
    initial = system.initial_for(data)
    guess = np.tile(initial, (data.number_of_rows, 1))
    direct = system.model.direct_state_observation_channels
    for column, state in enumerate(system.model.state_names):
        channel = direct.get(state)
        if channel in data.targets:
            guess[:, column] = data.targets[channel]
        elif channel in data.auxiliaries:
            guess[:, column] = data.auxiliaries[channel]
    guess[0] = initial
    if not np.isfinite(guess).all():
        raise ValueError("collocation node guess is nonfinite")
    return guess


def collocation_node_guess(
    system: SymbolicODE,
    data: Trajectory,
    start: np.ndarray,
    settings: FitConfig,
    policy: NodeStartPolicy,
    warmup_deadline: float,
) -> tuple[np.ndarray, dict]:
    """Use a bounded rollout when available; otherwise construct finite node guesses."""
    record = {"trajectory_id": data.trajectory_id, "policy": policy}
    started = monotonic()
    try:
        if monotonic() >= warmup_deadline:
            raise TimeoutError("shared node warm-up budget exhausted")
        _, _, guess, _ = symbolic_rollout(
            system, data, start, settings, warmup_deadline
        )
        record["source"] = "rollout"
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
        if policy == "rollout_required":
            raise
        record.update(
            source="observed_and_fixed_initials",
            rollout_error=str(error)[-1600:],
            rollout_diagnostic=getattr(error, "diagnostic", None),
        )
        guess = observed_node_guess(system, data)
    record["seconds"] = monotonic() - started
    return guess, record


def check_node_guess(
    system: SymbolicODE,
    t: float,
    states: np.ndarray,
    theta: np.ndarray,
    inputs,
) -> None:
    """Reject undefined local equations/derivatives without integrating an ODE."""
    parts = (
        system.rhs(t, states, theta, inputs),
        system.observe(t, states, theta, inputs),
        *system.local(t, states, theta, inputs),
    )
    if any(not np.isfinite(np.asarray(part)).all() for part in parts):
        raise ValueError(
            f"collocation node equations or derivatives are nonfinite at t={t:g}"
        )


def _weak_window(system, forcing, time: np.ndarray, smooth: np.ndarray):
    """Integrate one weak equation on the piecewise-linear observed interpolant.

    Five-point Gauss quadrature is applied separately on every sample interval.
    In particular, both sides integrate constant drift exactly: the quartic test
    function and its derivative times a linear state are polynomials of degree
    at most four. Nonlinear RHS features are evaluated at interpolated states,
    not interpolated from a precomputed feature matrix.
    """
    nodes, weights = np.polynomial.legendre.leggauss(5)
    width = time[-1] - time[0]
    matrix = np.zeros(len(system.names))
    response = 0.0
    zero = np.zeros(len(system.names))
    for i, (left, right) in enumerate(pairwise(time)):
        alpha = (nodes + 1) / 2
        times = left + (right - left) * alpha
        states = smooth[i] + (smooth[i + 1] - smooth[i]) * alpha
        q = (times - time[0]) / width
        phi = q**2 * (1 - q) ** 2
        derivative = (2 * q - 6 * q**2 + 4 * q**3) / width
        quadrature = weights * (right - left) / 2
        for t, x, w, p, dp in zip(
            times, states, quadrature, phi, derivative, strict=True
        ):
            inputs = [forcing.value(name, t) for name in system.inputs]
            design = np.asarray(system.local(t, [x], zero, inputs)[1]).ravel()
            offset = float(system.rhs(t, [x], zero, inputs))
            matrix += w * p * design
            response += w * (-dp * x - p * offset)
    # Integral_0^1 q^2(1-q)^2 dq = 1/30.
    return matrix / (width / 30), response / (width / 30)


def _latent_worker(candidate, context, result_path, arguments) -> None:
    """Rebuild the symbolic graph in an isolated, killable initializer process."""
    try:
        system = SymbolicODE(
            compile_candidate(CandidateModel.model_validate(candidate), context)
        )
        rows, fingerprint = arguments["training"]
        arguments["training"] = DatasetSplit(
            SplitName.TRAIN, tuple(Trajectory(**row) for row in rows), fingerprint
        )
        result = latent_start(system, **arguments)
    except (RuntimeError, ValueError, ArithmeticError) as error:
        result = {"success": False, "parameters": None, "message": str(error)[-1600:]}
    write_json(result_path, _finite_payload(result))


def bounded_latent_start(system: SymbolicODE, **arguments) -> dict:
    """Enforce an initializer wall limit, including native solver calls and startup."""
    started = monotonic()
    training = arguments["training"]
    if training.name is not SplitName.TRAIN:
        raise ValueError("initializer requires training split")
    rows = [
        {
            "trajectory_id": t.trajectory_id,
            "time": t.time,
            "targets": dict(t.targets),
            "auxiliaries": dict(t.auxiliaries),
            "external_inputs": dict(t.external_inputs),
            "fixed_covariates": dict(t.fixed_covariates),
            "derivatives": {},
        }
        for t in training.trajectories
    ]
    worker_arguments = {**arguments, "training": (rows, training.fingerprint)}
    directory = arguments["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    journal = directory / "node_initialization.json"
    write_json(
        journal,
        {
            "policy": arguments.get("node_start", "rollout_required"),
            "trajectories": [],
            "optimizer_started": False,
        },
    )
    result_path = directory / "child_result.json"
    result_path.unlink(missing_ok=True)
    worker = multiprocessing.get_context("spawn").Process(
        target=_latent_worker,
        args=(
            system.model.validated.candidate.model_dump(mode="json"),
            system.model.validated.context,
            result_path,
            worker_arguments,
        ),
    )
    worker.start()
    try:
        worker.join(timeout=max(0, arguments["seconds"] - (monotonic() - started)))
        if worker.is_alive():
            result = {
                "success": False,
                "parameters": None,
                "message": "initializer wall-clock limit reached",
            }
        elif worker.exitcode == 0 and result_path.exists():
            result = read_json(result_path)
        else:
            result = {
                "success": False,
                "parameters": None,
                "message": f"initializer process exited {worker.exitcode}",
            }
    finally:
        if worker.is_alive():
            worker.kill()
        worker.join()
        worker.close()
    node_record = read_json(journal)
    return {
        **result,
        "node_start": node_record["policy"],
        "node_initialization": node_record["trajectories"],
        "collocation_optimizer_started": (
            node_record["optimizer_started"]
            and arguments["method"] == "collocation_init"
        ),
        "method": arguments["method"],
        "seconds": monotonic() - started,
        "training_only": True,
        "hidden_labels_used": False,
        "initial_conditions_optimized": False,
    }


def matching_start(
    system: SymbolicODE,
    training: DatasetSplit,
    lower: np.ndarray,
    upper: np.ndarray,
    method: str,
    *,
    smoothing_points: int = 7,
    window_intervals: int = 8,
    weak_quadrature: Literal["trapezoid-v1", "gauss5-linear-v2"] = "gauss5-linear-v2",
) -> dict:
    """Solve bounded linear equation errors only for a certified observed RHS.

    Derivative, short-window integral and compact-test-function weak matching use
    exactly the same smoothed training observations. No ODE solution or hidden
    labels are used in this stage. Final rollout refinement is separate.
    """
    started = monotonic()
    if training.name is not SplitName.TRAIN:
        raise ValueError("initializer requires training split")
    mapping = system.model.direct_state_observation_channels
    if (
        not system.rhs_affine
        or system.state_count != 1
        or mapping != {system.model.state_names[0]: "v01"}
    ):
        raise ValueError(
            "direct matching requires a certified affine, fully observed scalar system"
        )
    if method not in {"derivative_init", "integral_init", "weak_init"}:
        raise ValueError("unknown matching method")
    if weak_quadrature not in {"trapezoid-v1", "gauss5-linear-v2"}:
        raise ValueError("unknown weak quadrature policy")
    matrices, responses = [], []
    zero = np.zeros(len(system.names))
    for data in training.trajectories:
        t = data.time
        dt = np.diff(t)
        if len(t) < max(smoothing_points, window_intervals + 1) or not np.allclose(
            dt, dt[0]
        ):
            raise ValueError("matching probe requires a sufficiently long uniform grid")
        observed = data.targets["v01"]
        smooth = savgol_filter(observed, smoothing_points, 3, mode="interp")
        derivative = savgol_filter(
            observed, smoothing_points, 3, deriv=1, delta=dt[0], mode="interp"
        )
        forcing = trajectory_forcing(system.model, data)
        design, offset = [], []
        for i, ti in enumerate(t):
            inputs = [forcing.value(name, ti) for name in system.inputs]
            design.append(
                np.asarray(system.local(ti, [smooth[i]], zero, inputs)[1]).ravel()
            )
            offset.append(float(system.rhs(ti, [smooth[i]], zero, inputs)))
        design, offset = np.asarray(design), np.asarray(offset)
        if method == "derivative_init":
            # Endpoints have one-sided smoothing uncertainty; exclude a fixed margin.
            margin = smoothing_points // 2
            matrices.extend(design[margin:-margin])
            responses.extend((derivative - offset)[margin:-margin])
            continue
        stride = max(1, window_intervals // 2)
        for left in range(0, len(t) - window_intervals, stride):
            right = left + window_intervals
            times = t[left : right + 1]
            width = times[-1] - times[0]
            if method == "integral_init":
                matrices.append(
                    trapezoid(design[left : right + 1], times, axis=0) / width
                )
                responses.append(
                    (
                        smooth[right]
                        - smooth[left]
                        - trapezoid(offset[left : right + 1], times)
                    )
                    / width
                )
            elif weak_quadrature == "gauss5-linear-v2":
                matrix, response = _weak_window(
                    system, forcing, times, smooth[left : right + 1]
                )
                matrices.append(matrix)
                responses.append(response)
            else:
                q = (times - times[0]) / width
                weight = q**2 * (1 - q) ** 2
                weight_prime = (2 * q - 6 * q * q + 4 * q**3) / width
                area = trapezoid(weight, times)
                matrices.append(
                    trapezoid(weight[:, None] * design[left : right + 1], times, axis=0)
                    / area
                )
                responses.append(
                    (
                        -trapezoid(weight_prime * smooth[left : right + 1], times)
                        - trapezoid(weight * offset[left : right + 1], times)
                    )
                    / area
                )
    matrix, response = np.asarray(matrices), np.asarray(responses)
    solved = lsq_linear(
        matrix, response, bounds=(lower, upper), tol=1e-10, max_iter=200
    )
    return {
        "method": method,
        "success": bool(solved.success),
        "message": solved.message,
        "parameters": dict(zip(system.names, map(float, solved.x), strict=True)),
        "seconds": monotonic() - started,
        "rows": len(response),
        "design_rank": int(np.linalg.matrix_rank(matrix)),
        "design_condition": float(np.linalg.cond(matrix)),
        "equation_error_cost": float(solved.cost),
        "smoothing_points": smoothing_points,
        "window_intervals": window_intervals,
        "weak_quadrature": weak_quadrature if method == "weak_init" else None,
        "training_only": True,
        "hidden_labels_used": False,
    }


def latent_start(
    system: SymbolicODE,
    training: DatasetSplit,
    lower: np.ndarray,
    upper: np.ndarray,
    start: np.ndarray,
    scale: float,
    settings: FitConfig,
    method: str,
    seconds: float,
    directory: Path,
    node_start: NodeStartPolicy = "rollout_required",
    warmup_seconds: float = 10.0,
) -> dict:
    """Estimate latent trajectories with continuity and unchanged fixed initials.

    Multiple shooting uses an adaptive CVODES step on every input sample interval.
    Integral collocation uses two-stage Radau IIA (order three, stiffly accurate).
    Both return only global parameters; fitted node states never enter final scoring.
    """
    if method not in {"shooting_init", "collocation_init"}:
        raise ValueError("unknown latent initialization method")
    if training.name is not SplitName.TRAIN:
        raise ValueError("initializer requires training split")
    if node_start not in {"rollout_required", "rollout_or_observed"}:
        raise ValueError("unknown collocation node initialization policy")
    if node_start != "rollout_required" and method != "collocation_init":
        raise ValueError("optional rollout seeding is only supported for collocation")
    if not np.isfinite(warmup_seconds) or warmup_seconds <= 0:
        raise ValueError("node warm-up budget must be positive and finite")
    started = monotonic()
    deadline = started + seconds
    warmup_deadline = (
        min(deadline, started + min(warmup_seconds, seconds / 4))
        if node_start == "rollout_or_observed"
        else deadline
    )
    directory.mkdir(parents=True, exist_ok=True)
    opti = ca.Opti()
    theta = opti.variable(len(start))
    opti.set_initial(theta, start)
    for i in range(len(start)):
        if np.isfinite(lower[i]):
            opti.subject_to(theta[i] >= lower[i])
        if np.isfinite(upper[i]):
            opti.subject_to(theta[i] <= upper[i])
    n = system.state_count
    integrator = None
    if method == "shooting_init":
        x = ca.MX.sym("x", n)
        clock = ca.MX.sym("clock")
        packed = ca.MX.sym("packed", len(start) + 2 + 2 * len(system.inputs))
        p = packed[: len(start)]
        t0, dt = packed[len(start)], packed[len(start) + 1]
        u0, u1 = (
            packed[len(start) + 2 : len(start) + 2 + len(system.inputs)],
            packed[len(start) + 2 + len(system.inputs) :],
        )
        rhs = dt * system.rhs(t0 + dt * clock, x, p, u0 + (u1 - u0) * clock)
        integrator = ca.integrator(
            "interval",
            "cvodes",
            {"x": x, "t": clock, "p": packed, "ode": rhs},
            0,
            1,
            {
                "reltol": 1e-7,
                "abstol": 1e-9,
                "max_num_steps": 3000,
                "disable_internal_warnings": True,
            },
        )
    objective = 0
    nodes = 0
    guesses = []
    optimizer_started = False
    try:
        for data in training.trajectories:
            forcing = trajectory_forcing(system.model, data)
            guess, guess_record = collocation_node_guess(
                system, data, start, settings, node_start, warmup_deadline
            )
            guesses.append(guess_record)
            write_json(
                directory / "node_initialization.json",
                {
                    "policy": node_start,
                    "trajectories": guesses,
                    "optimizer_started": False,
                },
            )
            current = ca.DM(system.initial_for(data))
            inputs = [
                [forcing.value(name, t) for name in system.inputs] for t in data.time
            ]
            if node_start == "rollout_or_observed":
                check_node_guess(system, data.time[0], guess[0], start, inputs[0])
            predicted = system.observe(data.time[0], current, theta, inputs[0])[0]
            objective += ((predicted - data.targets["v01"][0]) / scale) ** 2
            for i in range(len(data.time) - 1):
                if monotonic() >= deadline:
                    raise TimeoutError("initializer construction deadline reached")
                t0, t1 = float(data.time[i]), float(data.time[i + 1])
                dt = t1 - t0
                u0, u1 = np.asarray(inputs[i]), np.asarray(inputs[i + 1])
                if node_start == "rollout_or_observed":
                    check_node_guess(system, t1, guess[i + 1], start, u1)
                    check_node_guess(
                        system,
                        t0 + dt / 3,
                        (2 * guess[i] + guess[i + 1]) / 3,
                        start,
                        u0 + (u1 - u0) / 3,
                    )
                end = opti.variable(n)
                opti.set_initial(end, guess[i + 1])
                nodes += n
                if method == "shooting_init":
                    propagated = integrator(
                        x0=current, p=ca.vertcat(theta, t0, dt, u0, u1)
                    )["xf"]
                    opti.subject_to(end == propagated)
                else:
                    inner = opti.variable(n)
                    opti.set_initial(inner, (2 * guess[i] + guess[i + 1]) / 3)
                    nodes += n
                    first = system.rhs(t0 + dt / 3, inner, theta, u0 + (u1 - u0) / 3)
                    last = system.rhs(t1, end, theta, u1)
                    opti.subject_to(inner == current + dt * (5 * first - last) / 12)
                    opti.subject_to(end == current + dt * (3 * first + last) / 4)
                current = end
                predicted = system.observe(t1, current, theta, u1)[0]
                objective += ((predicted - data.targets["v01"][i + 1]) / scale) ** 2
        opti.minimize(objective)
        events = []

        def callback(iteration):
            if monotonic() >= deadline:
                raise RuntimeError("initializer iteration deadline reached")
            events.append({"iteration": iteration, "seconds": monotonic() - started})
            write_json(directory / "iterations.json", events)

        opti.callback(callback)
        remaining = max(0.001, deadline - monotonic())
        opti.solver(
            "ipopt",
            {"print_time": False},
            {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 150,
                "max_cpu_time": remaining,
                "tol": 1e-7,
            },
        )
        optimizer_started = True
        write_json(
            directory / "node_initialization.json",
            {
                "policy": node_start,
                "trajectories": guesses,
                "optimizer_started": True,
            },
        )
        solved = opti.solve()
        values = np.asarray(solved.value(theta)).reshape(-1)
        if (
            not np.isfinite(values).all()
            or np.any(values < lower - 1e-10)
            or np.any(values > upper + 1e-10)
        ):
            raise ValueError("initializer returned invalid parameters")
        values = np.clip(values, lower, upper)
        result = {
            "success": True,
            "parameters": dict(zip(system.names, map(float, values), strict=True)),
            "initializer_objective": float(solved.value(objective)),
            "iterations": solved.stats().get("iter_count"),
            "message": "converged",
        }
    except (RuntimeError, ValueError, TimeoutError, ArithmeticError) as error:
        result = {"success": False, "parameters": None, "message": str(error)[-1600:]}
    return _finite_payload(
        {
            **result,
            "method": method,
            "seconds": monotonic() - started,
            "latent_decision_variables": nodes,
            "training_only": True,
            "hidden_labels_used": False,
            "initial_conditions_optimized": False,
            "node_start": node_start,
            "node_initialization": guesses,
            "collocation_optimizer_started": optimizer_started
            and method == "collocation_init",
        }
    )
