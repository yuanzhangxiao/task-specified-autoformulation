"""Lift algebraic processes to expose conditionally affine collocation weights."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, replace
from time import monotonic

import casadi as ca
import numpy as np
from scipy.optimize import lsq_linear

from autoformalism.data import DatasetSplit, SplitName
from autoformalism.fitting.casadi_initializer import _DIVISION_EPSILON, _translate
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.staged_topology import content_hash


@dataclass
class LiftedSystem:
    """Keep process values independent until their algebraic equations are enforced."""

    system: SymbolicODE

    def __post_init__(self) -> None:
        model = self.system.model
        self.names = model.parameter_names
        self.processes = model.validated.process_order
        roles = {p.name: p.role.value for p in model.validated.candidate.parameters}
        self.rates = np.array([roles[n] == "time_constant" for n in self.names])
        self.n, self.q = len(model.state_names), len(self.processes)
        x, q = ca.SX.sym("x", self.n), ca.SX.sym("q", self.q)
        beta = ca.SX.sym("beta", len(self.names))
        u, t = ca.SX.sym("u", len(self.system.inputs)), ca.SX.sym("t")
        env = {
            **dict(zip(model.state_names, ca.vertsplit(x), strict=True)),
            **dict(zip(self.system.inputs, ca.vertsplit(u), strict=True)),
            **{
                n: 1 / beta[i] if self.rates[i] else beta[i]
                for i, n in enumerate(self.names)
            },
            model.validated.context.time_symbol: t,
        }
        rate_names = {n for n, rate in zip(self.names, self.rates, strict=True) if rate}
        aliases = {
            n: f"__collocation_rate_{i}" for i, n in enumerate(sorted(rate_names))
        }
        if any(n in env or n in self.processes for n in aliases.values()):
            raise ValueError("internal rate symbol collision")
        env.update({aliases[n]: beta[self.names.index(n)] for n in rate_names})

        def translate(expression, environment):
            # The original interpreter clamps positive divisors below epsilon.
            # Only simple time-constant divisors admit this exact effective-rate map.
            for parent in ast.walk(expression.tree):
                for child in ast.iter_child_nodes(parent):
                    if (
                        isinstance(child, ast.Name)
                        and child.id in rate_names
                        and not (
                            isinstance(parent, ast.BinOp)
                            and isinstance(parent.op, ast.Div)
                            and parent.right is child
                        )
                    ):
                        raise ValueError("time constant outside a simple denominator")

            class Rewrite(ast.NodeTransformer):
                def visit_BinOp(self, node):
                    node = self.generic_visit(node)
                    if (
                        isinstance(node.op, ast.Div)
                        and isinstance(node.right, ast.Name)
                        and node.right.id in aliases
                    ):
                        return ast.BinOp(
                            left=node.left,
                            op=ast.Mult(),
                            right=ast.Name(id=aliases[node.right.id], ctx=ast.Load()),
                        )
                    return node

            tree = Rewrite().visit(copy.deepcopy(expression.tree))
            return _translate(ca, replace(expression, tree=tree), environment)

        expanded = dict(env)
        for name in self.processes:
            expanded[name] = translate(
                model.validated.process_expressions[name], expanded
            )
        self.guess_processes = ca.Function(
            "process_guess",
            [t, x, beta, u],
            [ca.vertcat(*[expanded[n] for n in self.processes])],
        )
        env.update(dict(zip(self.processes, ca.vertsplit(q), strict=True)))
        rhs = ca.vertcat(
            *[
                translate(model.validated.equation_expressions[n], env)
                for n in model.state_names
            ]
        )
        algebraic = ca.vertcat(
            *[
                q[i] - translate(model.validated.process_expressions[n], env)
                for i, n in enumerate(self.processes)
            ]
        )
        observed = ca.vertcat(
            *[
                translate(model.validated.observation_expressions[n], env)
                for n in self.system.channels
            ]
        )
        joined = ca.vertcat(rhs, algebraic, observed)
        if ca.depends_on(ca.jacobian(joined, beta), beta):
            raise ValueError("lifted equations are not affine in all coefficients")
        self.point = ca.Function(
            "lifted_point", [t, x, q, beta, u], [rhs, algebraic, observed]
        )

    def coefficients(self, physical: np.ndarray) -> np.ndarray:
        """Map declared time constants to rates; preserve other identities."""
        values = np.asarray(physical, dtype=float).copy()
        if not np.isfinite(values).all() or np.any(values[self.rates] <= 0):
            raise ValueError("invalid physical parameters")
        values[self.rates] = 1 / np.maximum(values[self.rates], _DIVISION_EPSILON)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite rate conversion")
        return values

    def parameters(self, beta: np.ndarray) -> dict[str, float]:
        """Convert back without allowing zero rates or infinite time constants."""
        values = np.asarray(beta, dtype=float).copy()
        if not np.isfinite(values).all() or np.any(values[self.rates] <= 0):
            raise ValueError("invalid physical parameters from rates")
        values[self.rates] = 1 / values[self.rates]
        if not np.isfinite(values).all():
            raise ValueError("nonfinite physical time constants")
        return dict(zip(self.names, map(float, values), strict=True))

    def bounds(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map domains and exclude reciprocals of unrepresentable time constants."""
        low, high = lower.copy(), upper.copy()
        floor = np.nextafter(1 / np.finfo(float).max, np.inf)
        for i in np.flatnonzero(self.rates):
            if lower[i] <= 0:
                raise ValueError(
                    "time constants require positive physical lower bounds"
                )
            low[i] = max(floor, 1 / max(upper[i], _DIVISION_EPSILON))
            high[i] = 1 / max(lower[i], _DIVISION_EPSILON)
        if np.any(low >= high):
            raise ValueError("coefficient domains must have nonzero width")
        return low, high


class CollocationProblem:
    """One fixed Radau mesh, scale vector and penalized objective for both solvers."""

    def __init__(
        self,
        system: SymbolicODE,
        training: DatasetSplit,
        start: np.ndarray,
        scale: float,
        settings: FitConfig,
        penalty: float,
        deadline: float,
    ) -> None:
        if training.name is not SplitName.TRAIN:
            raise ValueError("collocation requires training split")
        if (
            not np.isfinite(penalty)
            or penalty <= 0
            or not np.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("positive finite collocation scales required")
        self.lifted = lifted = LiftedSystem(system)
        self.beta0 = lifted.coefficients(start)
        guesses = []
        for data in training.trajectories:
            _, _, states, _ = symbolic_rollout(system, data, start, settings, deadline)
            forcing = trajectory_forcing(system.model, data)
            inputs = np.array(
                [[forcing.value(n, t) for n in system.inputs] for t in data.time]
            )
            guesses.append((data, states, inputs))
        # Model-generated guesses, never hidden labels, define fixed state units.
        self.state_scale = np.maximum(
            1, np.sqrt(np.mean(np.concatenate([s for _, s, _ in guesses]) ** 2, axis=0))
        )
        all_q = [
            np.asarray(lifted.guess_processes(t, x, self.beta0, u)).ravel()
            for data, states, inputs in guesses
            for t, x, u in zip(data.time, states, inputs, strict=True)
        ]
        self.process_scale = np.maximum(
            scale, np.sqrt(np.mean(np.array(all_q) ** 2, axis=0))
        )
        unit = np.concatenate((self.state_scale, self.process_scale))
        beta = ca.SX.sym("beta", len(start))
        variables, initial, observations, dynamics, algebraics = [], [], [], [], []

        def node(x, u, t, *, first=False):
            if monotonic() >= deadline:
                raise TimeoutError("collocation construction deadline reached")
            q = np.asarray(lifted.guess_processes(t, x, self.beta0, u)).ravel()
            guess = q if first else np.concatenate((x, q))
            units = self.process_scale if first else unit
            v = ca.SX.sym(f"node{len(variables)}", len(guess))
            variables.append(v)
            initial.extend(guess / units)
            physical = v * ca.DM(units)
            xx = ca.DM(system.initial) if first else physical[: lifted.n]
            qq = physical if first else physical[lifted.n :]
            rhs, residual, observed = lifted.point(t, xx, qq, beta, u)
            algebraics.append(residual / ca.DM(self.process_scale))
            return xx, rhs, observed

        for data, guess, inputs in guesses:
            current, _, obs = node(guess[0], inputs[0], data.time[0], first=True)
            observations.append((obs[0] - data.targets["v01"][0]) / scale)
            for i in range(len(data.time) - 1):
                t0, t1 = data.time[i : i + 2]
                dt = t1 - t0
                inner, first, _ = node(
                    (2 * guess[i] + guess[i + 1]) / 3,
                    (2 * inputs[i] + inputs[i + 1]) / 3,
                    t0 + dt / 3,
                )
                end, last, obs = node(guess[i + 1], inputs[i + 1], t1)
                dynamics.extend(
                    [
                        (inner - current - dt * (5 * first - last) / 12)
                        / ca.DM(self.state_scale),
                        (end - current - dt * (3 * first + last) / 4)
                        / ca.DM(self.state_scale),
                    ]
                )
                observations.append((obs[0] - data.targets["v01"][i + 1]) / scale)
                current = end
        self.z0 = np.asarray(initial)
        z = ca.vertcat(*variables)
        observed = ca.vertcat(*observations)
        dynamic = ca.vertcat(*dynamics)
        algebraic = ca.vertcat(*algebraics)
        components = [
            observed / np.sqrt(observed.numel()),
            dynamic / np.sqrt(dynamic.numel()),
            algebraic / np.sqrt(max(1, algebraic.numel())),
        ]
        residual = ca.vertcat(
            components[0],
            np.sqrt(penalty) * components[1],
            np.sqrt(penalty) * components[2],
        )
        matrix = ca.jacobian(residual, beta)
        if ca.depends_on(matrix, beta):
            raise ValueError("collocation residuals lost coefficient linearity")
        offset = ca.substitute(residual, beta, ca.SX.zeros(len(start)))
        self.residual = ca.Function("collocation_residual", [z, beta], [residual])
        self.design = ca.Function("linear_design", [z], [matrix, -offset])
        self.components = ca.Function(
            "loss_components",
            [z, beta],
            [ca.vertcat(*[ca.sumsqr(c) for c in components])],
        )
        self.initial_identity = content_hash(
            {
                "nodes": self.z0.tolist(),
                "coefficients": self.beta0.tolist(),
                "state_scale": self.state_scale.tolist(),
                "process_scale": self.process_scale.tolist(),
                "penalty": penalty,
                "training": training.fingerprint,
            }
        )

    def loss(self, z: np.ndarray, beta: np.ndarray) -> float:
        """Evaluate exactly the same normalized objective in both block solvers."""
        residual = np.asarray(self.residual(z, beta)).ravel()
        return float(0.5 * residual @ residual)


def linear_update(
    problem: CollocationProblem,
    z: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Solve the shared coefficient block with scaling and domain bounds."""
    design, target = problem.design(z)
    a, b = np.asarray(design), np.asarray(target).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite coefficient subproblem")
    scales = np.linalg.norm(a, axis=0)
    scales[scales == 0] = 1
    scaled = a / scales
    result = lsq_linear(
        scaled,
        b,
        bounds=(lower * scales, upper * scales),
        method="bvls",
        lsq_solver="exact",
        tol=1e-10,
        max_iter=100,
    )
    beta = np.clip(result.x / scales, lower, upper)
    # Projected gradient is a KKT check in the documented column-scaled coordinates.
    grad = scaled.T @ (a @ beta - b)
    active = result.active_mask
    violation = np.where(
        active == -1,
        np.minimum(grad, 0),
        np.where(active == 1, np.maximum(grad, 0), grad),
    )
    kkt = float(np.max(np.abs(violation), initial=0))
    singular = np.linalg.svd(scaled, compute_uv=False)
    rank = int(np.linalg.matrix_rank(scaled))
    verified = bool(np.isfinite(beta).all() and kkt <= 1e-7 * max(1, np.linalg.norm(b)))
    return beta, {
        "success": verified,
        "solver_success": bool(result.success),
        "message": result.message,
        "iterations": int(result.nit),
        "scaled_kkt": kkt,
        "rank": rank,
        "columns": a.shape[1],
        "scaled_condition": float(singular[0] / singular[-1])
        if singular[-1] > 0
        else None,
        "active_mask": active.tolist(),
        "column_scale": scales.tolist(),
    }


def lifted_equivalence(system, trajectory, physical, states, predicted) -> float:
    """Audit rate/lifted equations against the existing expanded evaluator."""
    lifted = LiftedSystem(system)
    beta = lifted.coefficients(physical)
    forcing = trajectory_forcing(system.model, trajectory)
    error = 0.0
    for i, t in enumerate(trajectory.time):
        u = np.array([forcing.value(n, t) for n in system.inputs])
        q = lifted.guess_processes(t, states[i], beta, u)
        rhs, algebraic, output = lifted.point(t, states[i], q, beta, u)
        original = np.asarray(system.rhs(t, states[i], physical, u)).ravel()
        error = max(
            error,
            float(
                np.max(abs(np.asarray(rhs).ravel() - original), initial=0)
                / max(1, np.max(abs(original), initial=0))
            ),
            float(
                np.max(abs(np.asarray(output).ravel() - predicted[i]), initial=0)
                / max(1, np.max(abs(predicted[i]), initial=0))
            ),
            float(np.max(abs(np.asarray(algebraic)), initial=0)),
        )
    return error
