"""Bounded alternating node/shape and coefficient solves for M22 warm starts."""

from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.fitting.conditional_optimizer import save_point
from autoformalism.fitting.trajectory_profile import TrajectoryProblem


def optimize(
    problem,
    start,
    penalty,
    deadline,
    checkpoint,
    directory,
    identity,
    *,
    previous=None,
    cycles=4,
    block_iterations=25,
):
    """Reuse the graph, reset starts, and preserve finite improving primal iterates."""
    z, q = problem.guesses(start, previous)
    op = problem.opti
    loss = float(np.sum(np.asarray(problem.residual(z, q, penalty)) ** 2) / 2)
    history, iterations = [], 0
    lo, hi = problem.qbounds()

    def persist():
        parameters = problem.parameters(q)
        record = {
            "parameters": parameters,
            "objective": loss,
            "components": np.asarray(problem.components(z, q)).ravel().tolist(),
            "rollout_verified": False,
            "penalty": penalty,
        }
        save_point(directory, identity, z, q, record)
        checkpoint({"conditional_checkpoint": record, "iterations": iterations})

    def consider(accessor):
        nonlocal z, q, loss
        try:
            zz = np.asarray(accessor.value(problem.nz)).ravel()
            qq = np.asarray(accessor.value(problem.full_q)).ravel()
            objective = float(accessor.value(problem.objective))
            problem.parameters(qq)  # Strict physical bounds, apart from roundoff.
            if np.isfinite(zz).all() and np.isfinite(objective) and objective < loss:
                z, q, loss = zz, qq, objective
        except (RuntimeError, ValueError, ArithmeticError):
            pass

    def callback(i):
        nonlocal iterations
        iterations += 1
        consider(op.debug)
        if i % 10 == 0:
            persist()
        if monotonic() >= deadline:
            raise RuntimeError("conditional node solve deadline reached")

    op.callback(callback)
    op.set_value(problem.penalty, penalty)
    persist()
    for cycle in range(cycles):
        if monotonic() >= deadline:
            break
        q, linear = problem.linear_step(z, q, penalty)
        loss = float(np.sum(np.asarray(problem.residual(z, q, penalty)) ** 2) / 2)
        history.append({"cycle": cycle, "linear_solve": linear})
        persist()
        if monotonic() >= deadline:
            break
        op.set_value(problem.weights, q[problem.linear])
        op.set_initial(problem.nz, z)
        if problem.outer:
            op.set_initial(problem.free, q[problem.outer])
        # Never warm-start duals/optimizer state from a preceding basin.
        op.set_initial(op.lam_g, np.zeros(op.ng))
        op.solver(
            "ipopt",
            {"print_time": False},
            {
                "print_level": 0,
                "sb": "yes",
                "max_iter": block_iterations,
                "tol": 1e-8,
                "bound_relax_factor": 0,
                "max_cpu_time": max(0.001, deadline - monotonic()),
                "hessian_approximation": "limited-memory",
            },
        )
        try:
            solved = op.solve()
            consider(solved)
            reason = str(solved.stats().get("return_status"))
        except (RuntimeError, ValueError, ArithmeticError) as error:
            consider(op.debug)
            reason = str(error)[-700:]
        history[-1]["node_solve"] = reason
        persist()
    # The final updated trajectories also receive their optimal linear block.
    if monotonic() < deadline:
        q, linear = problem.linear_step(z, q, penalty)
        loss = float(np.sum(np.asarray(problem.residual(z, q, penalty)) ** 2) / 2)
        history.append({"final_linear_solve": linear})
        persist()
    if not np.all(q >= lo - 1e-12) or not np.all(q <= hi + 1e-12):
        raise ValueError("conditional final optimizer point violates bounds")
    parameters = problem.parameters(q)
    return {
        "parameters": parameters,
        "objective": loss,
        "history": history,
        "iterations": iterations,
        "trajectories": problem.trajectories(z, q),
        "rollout_verified": False,
        "mesh": problem.audit,
    }


class ConditionalEngine:
    """Cache formulation, never previous-trial solutions or reference trajectories."""

    def __init__(self, oracle, problem, policy):
        self.oracle, self.problem, self.policy = oracle, problem, policy
        self.cache = {}

    def propose(self, start, deadline, checkpoint, directory: Path):
        """Select mesh checkpoints using full training rollouts within this budget."""
        begun, previous, records, proposals, calls = monotonic(), None, [], [], 0
        allowance = deadline - begun
        # Reserve part of the same conditional allowance for physical rollouts.
        node_end = begun + 0.75 * allowance
        current = start
        for level, (target, penalty) in enumerate(
            zip(self.policy.node_targets, self.policy.penalties, strict=True)
        ):
            if monotonic() >= node_end:
                break
            level_end = monotonic() + (node_end - monotonic()) / (
                len(self.policy.node_targets) - level
            )
            reused = level in self.cache
            build_start = monotonic()
            try:
                if not reused:
                    self.cache[level] = TrajectoryProblem(
                        self.oracle,
                        self.problem,
                        target,
                        self.policy.minimum_intervals,
                        self.policy.observation_anchors,
                        level_end,
                    )
                native = optimize(
                    self.cache[level],
                    current,
                    penalty,
                    level_end,
                    checkpoint,
                    directory / f"mesh-{level}",
                    f"{directory.name}-mesh-{level}",
                    previous=previous,
                    cycles=self.policy.cycles,
                    block_iterations=self.policy.block_iterations,
                )
                previous = native.pop("trajectories")
                current = native["parameters"]
                proposals.append(current)
                records.append(
                    {
                        "level": level,
                        "graph_reused": reused,
                        "native": native,
                        "seconds": monotonic() - build_start,
                    }
                )
            except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
                records.append(
                    {
                        "level": level,
                        "graph_reused": reused,
                        "error": str(error)[-1000:],
                        "seconds": monotonic() - build_start,
                    }
                )
                break
            checkpoint({"conditional_levels": records})
        # Record the original pair as a control. Explore the best finite new
        # conditional basin even if its raw rollout is worse: the common fitter
        # still protects the independently verified incumbent from replacement.
        pool, seen = [], set()
        for p in [start, *proposals]:
            key = tuple(p[n] for n in self.oracle.names)
            if key not in seen:
                pool.append(p)
                seen.add(key)
        candidates = []
        for i, p in enumerate(pool):
            if monotonic() >= deadline:
                break
            point_end = monotonic() + (deadline - monotonic()) / (len(pool) - i)
            calls += 1
            checkpoint({"calls": calls})
            try:
                residual, _ = self.oracle.evaluate(
                    self.oracle.vector(p), point_end, jacobian=False
                )
                loss = float(np.mean(residual**2))
                if not np.isfinite(loss):
                    raise ValueError("nonfinite conditional checkpoint rollout")
                candidates.append({"parameters": p, "training_nmse": loss, "origin": i})
            except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
                records.append({"rollout_point": i, "error": str(error)[-700:]})
            checkpoint({"rollout_candidates": candidates})
        alternatives = [c for c in candidates if c["origin"] > 0]
        eligible = alternatives or candidates
        best = min(eligible, key=lambda c: c["training_nmse"]) if eligible else None
        return {
            "parameters": best["parameters"] if best else start,
            "selected": best,
            "candidates": candidates,
            "levels": records,
            "calls": calls,
            "seconds": monotonic() - begun,
            "selection": "best_finite_conditional_basin_else_original",
            "original_parameters": start,
            "incumbent_replacement_authorized": False,
            "independent_verification_pending": True,
        }
