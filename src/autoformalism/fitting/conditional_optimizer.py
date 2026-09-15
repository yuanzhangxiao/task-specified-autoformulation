"""Bounded joint or alternating optimization of one conditional collocation graph."""

from __future__ import annotations

import os
from pathlib import Path
from time import monotonic

import casadi as ca
import numpy as np

from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    read_json,
    sha256,
    write_json,
)
from autoformalism.staged_topology import content_hash


def save_point(root: Path, identity: str, z, q, record: dict) -> dict:
    """Publish an array before its atomic pointer; interrupted writes stay unused."""
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f"point-{os.getpid()}.tmp.npz"
    np.savez_compressed(temporary, z=z, q=q)
    digest = sha256(temporary)
    final = root / f"point-{digest}.npz"
    temporary.replace(final)
    value = _finite_payload(
        {**record, "identity": identity, "array": final.name, "array_sha256": digest}
    )
    value["metadata_sha256"] = content_hash(value)
    write_json(root / "checkpoint.json", value)
    return value


def load_point(root: Path, identity: str):
    """Reject stale/corrupt saved points instead of silently changing the start."""
    if not (root / "checkpoint.json").exists():
        return None
    record = read_json(root / "checkpoint.json")
    if record.get("metadata_sha256") != content_hash(
        {k: v for k, v in record.items() if k != "metadata_sha256"}
    ):
        raise ValueError("initializer metadata digest differs")
    name = record["array"]
    if record["identity"] != identity or Path(name).name != name:
        raise ValueError("initializer checkpoint identity/path differs")
    if sha256(root / name) != record["array_sha256"]:
        raise ValueError("initializer array digest differs")
    with np.load(root / name, allow_pickle=False) as arrays:
        z, q = arrays["z"].copy(), arrays["q"].copy()
    if not np.isfinite(z).all() or not np.isfinite(q).all():
        raise ValueError("nonfinite saved initializer")
    return record, z, q


def optimize(
    problem,
    lower,
    upper,
    *,
    arm,
    seconds,
    maximum_iterations,
    cycles,
    block_iterations,
    root,
    identity,
) -> dict:
    """Preserve finite improving iterates; boundary conditions are never variables."""
    if arm not in {"joint", "alternating"}:
        raise ValueError("unknown conditional optimization arm")
    started = monotonic()
    deadline = started + seconds
    low, high = lower / problem.punits, upper / problem.punits
    z, q = problem.z0.copy(), problem.q0.copy()
    if np.any(q < low) or np.any(q > high):
        raise ValueError("ordinary start violates original physical domain")
    root = Path(root)
    progress, last_saved = [], -float("inf")
    best_loss = problem.loss(z, q)
    initial_loss = best_loss

    def persist(force=False):
        nonlocal last_saved
        if not force and monotonic() - last_saved < 1:
            return
        save_point(
            root,
            identity,
            z,
            q,
            {
                "objective": best_loss,
                "initial_objective": initial_loss,
                "components": np.asarray(problem.components(z, q)).ravel().tolist(),
                "parameters": problem.parameters(q),
                "seconds": monotonic() - started,
                "progress": progress,
                "rollout_verified": False,
                "known_initials_fixed": True,
                "common_identity": problem.common["identity"],
            },
        )
        last_saved = monotonic()

    persist(True)
    opti = ca.Opti()
    node = opti.variable(len(z))
    if arm == "joint":
        variable = opti.variable(len(q))
        parameters = variable
        free_indices = list(range(len(q)))
        weights = None
    else:
        variable = opti.variable(len(problem.si))
        weights = opti.parameter(len(problem.wi))
        parameters = ca.MX.zeros(len(q), 1)
        parameters[problem.si] = variable
        parameters[problem.wi] = weights
        free_indices = problem.si
    for j, i in enumerate(free_indices):
        if np.isfinite(low[i]):
            opti.subject_to(variable[j] >= low[i])
        if np.isfinite(high[i]):
            opti.subject_to(variable[j] <= high[i])
    objective = ca.sumsqr(problem.residual(node, parameters)) / 2
    opti.minimize(objective)
    iteration_count = 0

    def consider(accessor):
        nonlocal z, q, best_loss
        try:
            zz = np.asarray(accessor.value(node)).ravel()
            qq = np.asarray(accessor.value(parameters)).ravel()
            loss = float(accessor.value(objective))
            if (
                np.isfinite(zz).all()
                and np.isfinite(qq).all()
                and np.isfinite(loss)
                and np.all(qq >= low)
                and np.all(qq <= high)
                and loss < best_loss
            ):
                z, q, best_loss = zz, qq, loss
                persist()
        except (RuntimeError, ValueError, ArithmeticError):
            pass

    def callback(i):
        nonlocal iteration_count
        iteration_count = int(i)
        consider(opti.debug)
        if monotonic() >= deadline:
            raise RuntimeError("initializer deadline reached")

    opti.callback(callback)
    opti.solver(
        "ipopt",
        {"print_time": True},
        {
            "print_level": 3,
            "sb": "yes",
            "tol": 1e-7,
            "max_iter": maximum_iterations if arm == "joint" else block_iterations,
            "max_cpu_time": max(0.001, deadline - monotonic()),
            "bound_relax_factor": 0,
            "hessian_approximation": "exact",
        },
    )
    native_success = False
    message = "cycle limit"
    for cycle in range(1 if arm == "joint" else cycles):
        if monotonic() >= deadline:
            message = "initializer wall-clock limit reached"
            break
        before = best_loss
        if arm == "alternating":
            q, report = problem.linear_step(z, q, low, high)
            best_loss = problem.loss(z, q)
            progress.append({"cycle": cycle, "block": "linear_weights", **report})
            persist(True)
            opti.set_value(weights, q[problem.wi])
        opti.set_initial(node, z)
        opti.set_initial(variable, q[free_indices])
        try:
            solution = opti.solve()
            consider(solution)
            native_success = True
            message = str(solution.stats().get("return_status"))
        except (RuntimeError, ValueError, ArithmeticError) as error:
            consider(opti.debug)
            native_success = False
            message = str(error)[-1000:]
        progress.append(
            {
                "cycle": cycle,
                "block": arm if arm == "joint" else "nodes_and_shapes",
                "native_success": native_success,
                "iterations": iteration_count,
                "objective": best_loss,
                "seconds": monotonic() - started,
                "message": message,
            }
        )
        persist(True)
        if arm == "alternating" and before - best_loss <= 1e-10 * max(1, abs(before)):
            message = "alternating objective progress small"
            break
    persist(True)
    result = {
        "identity": identity,
        "arm": arm,
        "native_success": native_success,
        "message": message,
        "initial_objective": initial_loss,
        "objective": best_loss,
        "parameters": problem.parameters(q),
        "seconds": monotonic() - started,
        "progress": progress,
        "common_identity": problem.common["identity"],
        "rollout_verified": False,
    }
    write_json(root / "result.json", _finite_payload(result), immutable=True)
    return result
