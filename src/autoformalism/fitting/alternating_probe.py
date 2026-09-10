"""Budgeted joint and alternating solvers for one penalized collocation objective."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import casadi as ca
import numpy as np

from autoformalism.fitting.matching_probe import bounded_latent_start
from autoformalism.fitting.runtime_probe import read_array, save_array
from autoformalism.fitting.separable_collocation import (
    CollocationProblem,
    linear_update,
)
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    write_json,
)
from autoformalism.rebuttal.fitter_stagnation import checkpoint
from autoformalism.staged_topology import content_hash

METHODS = ("joint_collocation", "alternating_collocation")


class NodeOptimizer:
    """Reuse a sparse IPOPT graph and retain its best finite, in-domain iterate."""

    def __init__(self, problem, lower, upper, *, joint, iterations, deadline):
        self.problem, self.lower, self.upper = problem, lower, upper
        self.joint, self.deadline = joint, deadline
        self.opti = opti = ca.Opti()
        self.z = opti.variable(len(problem.z0))
        self.units = np.maximum(1, np.abs(problem.beta0))
        if joint:
            self.v = opti.variable(len(problem.beta0))
            self.beta = self.v * ca.DM(self.units)
            for i in range(len(lower)):
                if np.isfinite(lower[i]):
                    opti.subject_to(self.beta[i] >= lower[i])
                if np.isfinite(upper[i]):
                    opti.subject_to(self.beta[i] <= upper[i])
        else:
            self.beta = opti.parameter(len(problem.beta0))
        self.objective = ca.sumsqr(problem.residual(self.z, self.beta)) / 2
        opti.minimize(self.objective)
        self.best = None
        self.on_improvement = None
        self.iterations = 0
        opti.callback(self.callback)
        opti.solver(
            "ipopt",
            {"print_time": False},
            {
                "print_level": 0,
                "sb": "yes",
                "tol": 1e-7,
                "max_iter": iterations,
                "max_cpu_time": max(0.001, deadline - monotonic()),
                "bound_relax_factor": 0,
            },
        )

    def callback(self, iteration):
        self.iterations = int(iteration)
        self.consider(self.opti.debug)
        if monotonic() >= self.deadline:
            raise RuntimeError("collocation optimizer deadline reached")

    def consider(self, accessor):
        try:
            z = np.asarray(accessor.value(self.z)).ravel()
            beta = np.asarray(accessor.value(self.beta)).ravel()
            loss = float(accessor.value(self.objective))
            if (
                np.isfinite(z).all()
                and np.isfinite(beta).all()
                and np.isfinite(loss)
                and np.all(beta >= self.lower)
                and np.all(beta <= self.upper)
                and loss < self.best[0]
            ):
                self.best = loss, z, beta
                if self.on_improvement is not None:
                    self.on_improvement(self.best, self.iterations)
        except (RuntimeError, ValueError, ArithmeticError):
            pass

    def solve(self, z, beta):
        started = monotonic()
        self.opti.set_initial(self.z, z)
        if self.joint:
            self.opti.set_initial(self.v, beta / self.units)
        else:
            self.opti.set_value(self.beta, beta)
        self.best = self.problem.loss(z, beta), z.copy(), beta.copy()
        self.iterations = 0
        try:
            solved = self.opti.solve()
            self.consider(solved)
            success, message = True, str(solved.stats().get("return_status"))
        except (RuntimeError, ValueError, ArithmeticError) as error:
            self.consider(self.opti.debug)
            success, message = False, str(error)[-1200:]
        return (
            self.best[1],
            self.best[2],
            {
                "optimizer_success": success,
                "message": message,
                "iterations": self.iterations,
                "seconds": monotonic() - started,
            },
        )


def _snapshot(root: Path, identity: str) -> tuple[dict | None, np.ndarray | None]:
    saved = checkpoint(root / "checkpoint.json", identity)
    if saved is None:
        return None, None
    return saved, read_array(root / saved["array"], saved["array_sha256"])


def separable_start(
    system,
    training,
    lower,
    upper,
    start,
    scale,
    settings,
    method,
    seconds,
    directory,
    *,
    penalty,
    cycles,
    state_iterations,
    joint_iterations,
    block_identity,
    elapsed_before=0.0,
) -> dict:
    """Alternate coefficient and node updates on a fixed objective."""
    if method not in METHODS:
        raise ValueError("unknown separable initializer")
    started = monotonic()
    deadline = started + seconds
    problem = CollocationProblem(
        system, training, start, scale, settings, penalty, deadline
    )
    low, high = problem.lifted.bounds(lower, upper)
    beta, z = problem.beta0.copy(), problem.z0.copy()
    saved, values = _snapshot(directory, block_identity)
    progress = [] if saved is None else list(saved["progress"])
    completed = 0 if saved is None else saved["completed_cycles"]
    sequence = 0 if saved is None else saved["sequence"] + 1
    if saved is not None:
        if saved["initial_identity"] != problem.initial_identity:
            raise ValueError("collocation starting nodes or objective changed")
        beta, z = values[: len(beta)], values[len(beta) :]
    updated = bool(saved and saved["updated"])
    initial_loss = problem.loss(problem.z0, problem.beta0)

    def persist(completed_cycles, update):
        nonlocal sequence, updated
        updated = updated or update
        parameters = problem.lifted.parameters(beta)
        physical = np.array([parameters[n] for n in system.names])
        if np.any(physical < lower) or np.any(physical > upper):
            raise ValueError(
                "converted collocation parameters violate physical domains"
            )
        filename = f"point-{sequence:04d}.npz"
        digest = save_array(directory / filename, np.concatenate((beta, z)))
        result = {
            "identity": block_identity,
            "array": filename,
            "array_sha256": digest,
            "sequence": sequence,
            "completed_cycles": completed_cycles,
            "initial_identity": problem.initial_identity,
            "initial_objective": initial_loss,
            "objective": problem.loss(z, beta),
            "components": np.asarray(problem.components(z, beta)).ravel().tolist(),
            "parameters": parameters,
            "progress": progress,
            "seconds": elapsed_before + monotonic() - started,
            "node_variables": len(z),
            "coefficient_variables": len(beta),
            "state_scales": problem.state_scale.tolist(),
            "process_scales": problem.process_scale.tolist(),
            "updated": updated,
            "penalty": penalty,
            "training_only": True,
            "hidden_labels_used": False,
            "initial_conditions_optimized": False,
            "rollout_verified": False,
        }
        write_json(directory / "checkpoint.json", _finite_payload(result))
        sequence += 1
        return result

    current = persist(completed, False)
    joint = method == "joint_collocation"
    optimizer = NodeOptimizer(
        problem,
        low,
        high,
        joint=joint,
        iterations=joint_iterations if joint else state_iterations,
        deadline=deadline,
    )
    stop = "native_return" if joint else "cycle_limit"
    for cycle in range(completed, 1 if joint else cycles):
        if monotonic() >= deadline:
            stop = "wall_limit"
            break
        before = problem.loss(z, beta)
        if not joint:
            then = monotonic()
            proposal, report = linear_update(problem, z, low, high)
            after = problem.loss(z, proposal)
            accepted = report["success"] and after <= before
            if accepted:
                beta = proposal
            progress.append(
                {
                    "cycle": cycle,
                    "block": "coefficients",
                    "before": before,
                    "after": after,
                    "accepted": accepted,
                    **report,
                    "seconds": monotonic() - then,
                }
            )
            current = persist(cycle, accepted)
            if monotonic() >= deadline:
                stop = "wall_limit"
                break
        previous = problem.loss(z, beta)

        def retain_iterate(point, iteration, cycle=cycle):
            nonlocal z, beta, current
            old_loss = problem.loss(z, beta)
            z, beta = point[1], point[2]
            progress.append(
                {
                    "cycle": cycle,
                    "block": "joint_iterate" if joint else "node_iterate",
                    "iteration": iteration,
                    "before": old_loss,
                    "after": point[0],
                    "accepted": True,
                }
            )
            current = persist(cycle, True)

        optimizer.on_improvement = retain_iterate
        proposal_z, proposal_beta, report = optimizer.solve(z, beta)
        after = problem.loss(proposal_z, proposal_beta)
        accepted = bool(after <= previous and np.isfinite(after))
        if accepted:
            z, beta = proposal_z, proposal_beta
        progress.append(
            {
                "cycle": cycle,
                "block": "joint" if joint else "nodes",
                "before": previous,
                "after": after,
                "accepted": accepted,
                **report,
            }
        )
        current = persist(cycle + 1, accepted)
        if not joint and before - current["objective"] <= 1e-10 * max(1, before):
            stop = "block_progress_small"
            break
    return {
        "success": current["updated"],
        "parameters": current["parameters"],
        "message": stop,
        "last_nonlinear_block_success": progress[-1].get("optimizer_success")
        if progress
        else False,
        "acceptance": "finite_in_domain_nonincreasing_penalized_objective",
        "initializer_objective": current["objective"],
        "initial_objective": initial_loss,
        "initial_nodes_identity": problem.initial_identity,
        "components": current["components"],
        "node_variables": len(z),
        "progress": progress,
        "completed_cycles": current["completed_cycles"],
        "penalty": penalty,
        "checkpoint_array": current["array"],
        "checkpoint_sha256": current["array_sha256"],
        "coefficient_order": list(system.names),
        "rate_parameters": [
            n
            for n, rate in zip(system.names, problem.lifted.rates, strict=True)
            if rate
        ],
        "training_only": True,
        "hidden_labels_used": False,
        "initial_conditions_optimized": False,
        "rollout_verified": False,
    }


def bounded_separable_start(system, **arguments) -> dict:
    """Bound native work and recover identity-checked accepted iterates."""
    root = arguments["directory"]
    original_seconds = arguments["seconds"]
    key = content_hash(
        {
            "candidate": system.model.validated.candidate.model_dump(mode="json"),
            "training": arguments["training"].fingerprint,
            "settings": arguments["settings"].model_dump(mode="json"),
            **{
                k: np.asarray(arguments[k]).tolist()
                for k in ("lower", "upper", "start")
            },
            **{
                k: arguments[k]
                for k in (
                    "scale",
                    "method",
                    "penalty",
                    "cycles",
                    "state_iterations",
                    "joint_iterations",
                    "identity",
                )
            },
            "budget": original_seconds,
        }
    )
    saved, _ = _snapshot(root, key)
    charged = 0 if saved is None else saved["seconds"]
    worker_arguments = {k: v for k, v in arguments.items() if k != "identity"}
    result = (
        bounded_latent_start(
            system,
            **{
                **worker_arguments,
                "seconds": max(0.001, original_seconds - charged),
                "block_identity": key,
                "elapsed_before": charged,
            },
        )
        if charged < original_seconds
        else {
            "success": False,
            "seconds": 0,
            "message": "initializer budget already consumed",
        }
    )
    result["seconds"] += charged
    current, _ = _snapshot(root, key)
    if current is not None:
        # Charge native startup and any killed partial block as well as accepted work.
        current["seconds"] = result["seconds"]
        write_json(root / "checkpoint.json", current)
        if not result["success"] and current["updated"]:
            result.update(
                success=True,
                parameters=current["parameters"],
                acceptance="last_accepted_point_after_native_limit",
                initializer_objective=current["objective"],
                initial_objective=current["initial_objective"],
                initial_nodes_identity=current["initial_identity"],
                components=current["components"],
                progress=current["progress"],
                completed_cycles=current["completed_cycles"],
                node_variables=current["node_variables"],
                penalty=current["penalty"],
                checkpoint_array=current["array"],
                checkpoint_sha256=current["array_sha256"],
                last_nonlinear_block_success=False,
            )
    if not result["success"]:
        result["parameters"] = None
    result.update(
        method=arguments["method"],
        training_only=True,
        hidden_labels_used=False,
        initial_conditions_optimized=False,
        rollout_verified=False,
    )
    return result
