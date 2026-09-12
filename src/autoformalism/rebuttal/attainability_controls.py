"""Attainable observations and fixed-parameter collocation diagnostics."""

from __future__ import annotations

from copy import deepcopy
from time import monotonic

import casadi as ca
import numpy as np

from autoformalism.data import TrainingScaler
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.collocation_assembly import mapped_collocation
from autoformalism.fitting.collocation_mesh import plan_meshes
from autoformalism.fitting.initialization import (
    LatentInitializationPlan,
    apply_initialization_plan,
)
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.rebuttal.fitter_diagnostic import _finite_payload, write_json
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash


def system_for(problem):
    """Compile the exact candidate and its explicit physical boundary rules."""
    model = compile_candidate(
        CandidateModel.model_validate(problem["candidate"]),
        ValidationContext.model_validate(problem["context"]),
    )
    model, guesses, _ = apply_initialization_plan(
        model, LatentInitializationPlan.model_validate(problem["initialization_plan"])
    )
    return SymbolicODE(model, allow_piecewise=True), guesses


def ordinary_start(problem, replicate=0):
    """Use existing guesses or role defaults; never consult generating parameters."""
    system, guesses = system_for(problem)
    start = {
        p.name: 1.0 if p.role.value == "rate" else 0.1
        for p in system.model.validated.candidate.parameters
    }
    start.update(guesses)
    start.update(problem.get("start", {}))
    if replicate:
        rng = np.random.default_rng(20260913 + replicate)
        for name in sorted(start):
            start[name] = (
                start[name] * np.exp(rng.normal(0, 0.6))
                if start[name]
                else float(rng.normal(0, 0.2))
            )
    return start


def simulate_split(problem, parameters, key, seconds=120, zero_input=False):
    """Generate only with the restricted production evaluator, on exact input grids."""
    system, _ = system_for(problem)
    split = unpack_split(problem["splits"][key])
    settings = FitConfig(
        integration_method="Radau", relative_tolerance=1e-9, absolute_tolerance=1e-11
    )
    deadline = monotonic() + seconds
    values, states = {}, {}
    for row in split.trajectories:
        if zero_input:
            from dataclasses import replace

            row = replace(
                row,
                external_inputs={
                    k: np.zeros_like(v) for k, v in row.external_inputs.items()
                },
            )
        result = simulate_trajectory(
            system.model,
            row,
            parameters,
            {},
            settings,
            deadline=deadline,
            reset_observed_states=False,
        )
        if not result.success:
            raise ValueError(f"{key}/{row.trajectory_id}: {result.message}")
        values[row.trajectory_id] = result.predictions["v01"].tolist()
        states[row.trajectory_id] = result.states.T.tolist()
    return values, states


def truth_candidates(problem):
    """Training-only bounded design: damping-weighted and uniformly slowed guesses.

    Damping classification uses the compiled state Jacobian at zero, not parameter
    names or benchmark truth. Every accepted point must pass an activity check.
    """
    system, _ = system_for(problem)
    baseline = ordinary_start(problem)
    zero = np.zeros(system.state_count)
    u = np.zeros(len(system.inputs))
    initial = set(system.initial_parameter_names)
    base = np.zeros(len(system.names))
    # All currently imported candidates are well-defined at zero coefficients.
    # If a future grammar is not, the failed classification is explicit below.
    damping = set()
    try:
        for i, name in enumerate(system.names):
            if name in initial:
                continue
            p = base.copy()
            p[i] = 1.0
            fx = np.asarray(system.local(0, zero, p, u)[0])
            if np.isfinite(fx).all() and np.min(np.diag(fx)) < -0.1:
                damping.add(name)
    except (RuntimeError, ValueError, ArithmeticError):
        damping.clear()
    for mode in ("damping", "ordinary"):
        for factor in (1.0, 0.1, 0.01):
            for initial_value in (0.3, 1.0):
                point = {
                    n: initial_value
                    if n in initial
                    else factor
                    * (
                        (0.3 if n in damping else 0.01)
                        if mode == "damping"
                        else baseline[n]
                    )
                    for n in system.names
                }
                yield f"{mode}/{factor}/initial{initial_value}", point


def choose_truth(problem, directory, seconds=600):
    """Freeze first active finite training point; never resample on validation."""
    deadline, attempts = monotonic() + seconds, []
    for source, point in truth_candidates(problem):
        record = {"source": source, "parameters": point}
        try:
            if monotonic() >= deadline:
                raise TimeoutError("truth-design budget exhausted")
            y, states = simulate_split(
                problem, point, "train", min(40, max(0.01, deadline - monotonic()))
            )
            zero, _ = simulate_split(
                problem,
                point,
                "train",
                min(40, max(0.01, deadline - monotonic())),
                True,
            )
            pooled = np.concatenate(list(y.values()))
            change = np.concatenate([np.asarray(v) - zero[k] for k, v in y.items()])
            state_values = np.concatenate([np.asarray(v) for v in states.values()])
            ranges = np.ptp(state_values, axis=0)
            activity = {
                "output_sd": float(np.std(pooled)),
                "input_effect_rms": float(np.sqrt(np.mean(change**2))),
                "state_ranges": ranges.tolist(),
                "maximum_absolute_state": float(np.max(abs(state_values))),
            }
            accepted = (
                activity["output_sd"] >= 1e-3
                and activity["input_effect_rms"] >= 0.02 * activity["output_sd"]
                and min(ranges) >= 1e-3
                and activity["maximum_absolute_state"] <= 1000
            )
            record.update(activity=activity, accepted=bool(accepted))
            if accepted:
                attempts.append(record)
                write_json(directory / "truth_design.json", attempts)
                return point, y, states, attempts
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
            record.update(accepted=False, error=str(error)[-800:])
        attempts.append(record)
        write_json(directory / "truth_design.json", attempts)
        if monotonic() >= deadline:
            break
    raise ValueError("No active finite candidate truth found within design budget")


def generated_problem(problem, values):
    """Replace target observations only; preserve initial observations and forcing."""
    result = deepcopy(problem)
    for key in ("train", "val"):
        for row in result["splits"][key]["rows"]:
            row["targets"]["v01"] = values[key][row["trajectory_id"]]
        result["splits"][key]["fingerprint"] = content_hash(
            result["splits"][key]["rows"]
        )
    return result


def fixed_state_fit(problem, truth, target, directory, seconds=300):
    """Optimize only node states at fixed generating parameters and physical initials.

    Start from observed output plus constant hidden initial values, not generating
    hidden trajectories. Defects are reported per state/interval; their diagnostic
    scaling does not alter the objective or constraints.
    """
    system, _ = system_for(problem)
    theta = np.array([truth[n] for n in system.names])
    train = unpack_split(problem["splits"]["train"])
    scale = TrainingScaler().fit(train).scales["target:v01"].standard_deviation
    directory.mkdir(parents=True, exist_ok=True)
    opti, started = ca.Opti(), monotonic()
    meshes, _ = plan_meshes(system, train, target)
    scaling = []

    def seed(row):
        initial = system.initial_for(row, theta)
        states = np.tile(initial, (len(row.time), 1))
        for name, channel in system.model.direct_state_observation_channels.items():
            states[:, system.model.state_names.index(name)] = row.targets[channel]
        scaling.append(np.maximum(1.0, np.max(abs(states), axis=0)))
        return states, {
            "policy": "observed_and_constant_initials",
            "trajectory_id": row.trajectory_id,
        }

    objective, nodes, _ = mapped_collocation(
        system,
        opti,
        ca.DM(theta),
        train,
        theta,
        scale,
        directory,
        started + seconds,
        seed,
        target,
    )
    opti.minimize(objective)
    records = []

    def record(iteration):
        row = {"iteration": iteration, "seconds": monotonic() - started}
        try:
            defects = np.asarray(opti.debug.value(opti.g)).ravel()
            offset, worst = 0, []
            for data, mesh, scales in zip(
                train.trajectories, meshes, scaling, strict=True
            ):
                count, n = len(mesh.time) - 1, system.state_count
                block = defects[offset : offset + 2 * n * count].reshape(
                    2 * n, count, order="F"
                )
                for j, name in enumerate(system.model.state_names):
                    absolute = np.max(abs(block[[j, j + n], :]), axis=0)
                    k = int(np.argmax(absolute))
                    worst.append(
                        {
                            "trajectory": data.trajectory_id,
                            "state": name,
                            "interval": [float(mesh.time[k]), float(mesh.time[k + 1])],
                            "absolute_defect": float(absolute[k]),
                            "scaled_defect": float(absolute[k] / scales[j]),
                        }
                    )
                offset += 2 * n * count
            row.update(
                objective=float(opti.debug.value(objective)),
                constraint_maximum=float(np.max(abs(defects), initial=0)),
                worst_defects=sorted(
                    worst, key=lambda r: r["scaled_defect"], reverse=True
                )[:12],
            )
        except (RuntimeError, ValueError) as error:
            row["error"] = str(error)[-500:]
        records.append(row)
        write_json(
            directory / "progress.json", _finite_payload({"iterations": records})
        )
        if monotonic() - started >= seconds:
            raise RuntimeError("fixed-node optimization deadline")

    opti.callback(record)
    opti.solver(
        "ipopt",
        {"print_time": False},
        {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 1000,
            "max_cpu_time": max(0.01, seconds - monotonic() + started),
            "tol": 1e-7,
            "hessian_approximation": "limited-memory"
            if system.requires_first_order_solver
            else "exact",
        },
    )
    success, message = False, ""
    try:
        opti.solve()
        success, message = True, "converged"
    except (RuntimeError, ValueError) as error:
        message = str(error)[-1200:]
    count = sum(len(r.time) for r in train.trajectories)
    return _finite_payload(
        {
            "success": success,
            "message": message,
            "seconds": monotonic() - started,
            "node_variables": nodes,
            "parameters_optimized": False,
            "initials_optimized": False,
            "hidden_trajectory_labels_used": False,
            "collocation_training_nmse": records[-1]["objective"] / count
            if records and "objective" in records[-1]
            else None,
            "last_iteration": records[-1] if records else None,
        }
    )
