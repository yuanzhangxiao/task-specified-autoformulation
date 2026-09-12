"""Restricted-expression features and reproducible branch-aware starting points."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import casadi as ca
import numpy as np

from autoformalism.expressions.parser import ParsedExpression
from autoformalism.fitting.casadi_initializer import _translate
from autoformalism.fitting.sensitivity_contract import certify_expressions
from autoformalism.fitting.simulation import _known_initial_values, trajectory_forcing


@dataclass
class Branch:
    """Sign of a branch margin; zero denotes a tie, not two covered branches."""

    label: str
    function: object
    initial_only: bool = False


class BranchCatalog:
    """Extract at most 32 branch comparisons from already validated ASTs."""

    def __init__(self, system):
        self.system = system
        equations, observations, audit = certify_expressions(
            system.model, allow_piecewise=True
        )
        n, p = system.state_count, len(system.names)
        t, x, theta, u = (
            ca.SX.sym("t"),
            ca.SX.sym("x", n),
            ca.SX.sym("p", p),
            ca.SX.sym("u", len(system.inputs)),
        )
        known = ca.SX.sym("known", len(system.initial_symbols))
        env = {
            **dict(zip(system.names, ca.vertsplit(theta), strict=True)),
            **dict(zip(system.model.state_names, ca.vertsplit(x), strict=True)),
            **dict(zip(system.inputs, ca.vertsplit(u), strict=True)),
            system.model.validated.context.time_symbol: t,
        }
        initial_env = {
            **dict(zip(system.names, ca.vertsplit(theta), strict=True)),
            **dict(zip(system.initial_symbols, ca.vertsplit(known), strict=True)),
        }
        self.branches: list[Branch] = []
        self.truncated = False
        self.expressions = {
            **{f"rhs:{k}": v for k, v in equations.items()},
            **{f"observation:{k}": v for k, v in observations.items()},
        }
        expressions = {
            **self.expressions,
            **{f"initial:{k}": v for k, v in audit.initial_expressions.items()},
        }
        for location, expression in expressions.items():
            for node in ast.walk(expression.tree):
                if not isinstance(node, ast.Call) or node.func.id not in {
                    "abs",
                    "min",
                    "max",
                }:
                    continue
                # Compare each argument to the actual competing branches, not
                # merely the first argument of an n-ary min/max.
                margins = (
                    [node.args[0]]
                    if node.func.id == "abs"
                    else [
                        ast.BinOp(
                            arg,
                            ast.Sub(),
                            ast.Call(
                                node.func,
                                [other for j, other in enumerate(node.args) if j != i],
                                [],
                            ),
                        )
                        if len(node.args) > 2
                        else ast.BinOp(arg, ast.Sub(), node.args[1 - i])
                        for i, arg in enumerate(
                            node.args[:1] if len(node.args) == 2 else node.args
                        )
                    ]
                )
                for margin in margins:
                    if len(self.branches) >= 32:
                        self.truncated = True
                        continue
                    parsed = ParsedExpression(
                        ast.unparse(margin), ast.Expression(margin), expression.symbols
                    )
                    initial = location.startswith("initial:")
                    value = _translate(ca, parsed, initial_env if initial else env)
                    function = ca.Function(
                        f"branch_{len(self.branches)}",
                        [theta, known] if initial else [t, x, theta, u],
                        [value],
                    )
                    self.branches.append(
                        Branch(f"{location}:{parsed.source}", function, initial)
                    )

    def values(self, data, t, x, theta) -> np.ndarray:
        """Read branch margins without modifying equations or observations."""
        forcing = trajectory_forcing(self.system.model, data)
        u = [forcing.value(name, t) for name in self.system.inputs]
        known = _known_initial_values(self.system.model, data)
        return np.array(
            [
                float(
                    b.function(theta, [known[k] for k in self.system.initial_symbols])
                )
                if b.initial_only
                else float(b.function(t, x, theta, u))
                for b in self.branches
            ]
        )

    def initial_values(self, data, theta):
        return self.values(
            data, data.time[0], self.system.initial_for(data, theta), theta
        )

    def coverage(self, data, theta, states) -> list[dict]:
        """Report sampled coverage, never a continuous-time reachability proof."""
        if not self.branches:
            return []
        values = np.array(
            [
                self.values(data, t, x, theta)
                for t, x in zip(data.time, states, strict=True)
            ]
        )
        return [
            {
                "branch": b.label,
                "minimum": float(np.min(values[:, i])),
                "maximum": float(np.max(values[:, i])),
                "negative": bool(np.any(values[:, i] < -1e-8)),
                "positive": bool(np.any(values[:, i] > 1e-8)),
            }
            for i, b in enumerate(self.branches)
        ]


def designed_starts(
    system, training, start, lower, upper, *, branches: bool, seed: int = 20260912
) -> tuple[list[dict], dict]:
    """Target feasible parameter domains, not guaranteed trajectory feasibility."""
    catalog = BranchCatalog(system)
    start = np.asarray(start, dtype=float)
    scales = np.maximum(np.abs(start), 1.0)
    points: list[dict] = []
    requests: list[dict] = []
    seen = {tuple(start)}

    def add(point, source):
        point = np.clip(point, lower, upper)
        if np.isfinite(point).all() and tuple(point) not in seen:
            seen.add(tuple(point))
            points.append(
                {
                    "source": source,
                    "parameters": dict(zip(system.names, point, strict=True)),
                }
            )

    if branches:
        data = training.trajectories[0]
        for index, branch in enumerate(catalog.branches):
            for sign in (-1, 1):
                point = start.copy()
                reached = False
                try:
                    original = catalog.initial_values(data, point)[index]
                    target = sign * max(1.0, abs(original))
                    for _ in range(5):
                        margin = catalog.initial_values(data, point)[index]
                        if sign * margin > 1e-6:
                            reached = True
                            break
                        gradient = np.zeros(len(point))
                        for j in range(len(point)):
                            step = 1e-4 * scales[j]
                            plus, minus = point.copy(), point.copy()
                            plus[j] = min(upper[j], point[j] + step)
                            minus[j] = max(lower[j], point[j] - step)
                            if plus[j] != minus[j]:
                                gradient[j] = (
                                    catalog.initial_values(data, plus)[index]
                                    - catalog.initial_values(data, minus)[index]
                                ) / (plus[j] - minus[j])
                        direction = gradient * scales**2
                        denominator = float(gradient @ direction)
                        if denominator < 1e-20:
                            break
                        point = np.clip(
                            point + (target - margin) * direction / denominator,
                            lower,
                            upper,
                        )
                    if reached:
                        add(point, f"branch:{index}:{sign}")
                    requests.append(
                        {
                            "branch": branch.label,
                            "sign": sign,
                            "initial_side_reached": reached,
                        }
                    )
                except (ValueError, RuntimeError, ArithmeticError) as error:
                    requests.append(
                        {
                            "branch": branch.label,
                            "sign": sign,
                            "initial_side_reached": False,
                            "error": str(error)[-300:],
                        }
                    )
    # If the physical boundary cannot move, an independent latent-node guess
    # can still explore a different branch while preserving that boundary.
    for index, branch in enumerate(catalog.branches):
        for sign in (-1, 1):
            if branches and any(
                r["branch"] == branch.label
                and r["sign"] == sign
                and not r["initial_side_reached"]
                for r in requests
            ):
                points.append(
                    {
                        "source": f"branch_nodes:{index}:{sign}",
                        "parameters": dict(zip(system.names, start, strict=True)),
                        "node_target": (index, sign),
                    }
                )
    initial_names = set(system.initial_parameter_names)
    for factor in (0.1, 0.01, 0.001):
        point = start.copy()
        for j, name in enumerate(system.names):
            if name not in initial_names:
                point[j] *= factor
        add(point, f"slower_equations:{factor}")
    rng = np.random.default_rng(seed)
    for index in range(3):
        point = np.where(
            lower >= 0,
            np.maximum(start, 0.01 * scales) * 10 ** rng.uniform(-2, 1, len(start)),
            start + scales * rng.uniform(-3, 3, len(start)),
        )
        add(point, f"seeded_scaled:{index}")
    return points, {
        "branch_requests": requests,
        "catalog_truncated": catalog.truncated,
        "seed": seed,
        "selection_uses_training_only": True,
        "unreached_is_not_proven_unreachable": True,
    }


def target_node_branch(system, data, theta, guess, target) -> tuple[np.ndarray, dict]:
    """Change latent node guesses only; preserve the physical first boundary.

    These guesses need not satisfy the ODE. Local undefined expressions are still
    rejected by the collocation builder. No target data are changed.
    """
    catalog = BranchCatalog(system)
    index, sign = target
    if index >= len(catalog.branches) or sign not in {-1, 1}:
        raise ValueError("invalid branch node target")
    mutable = [
        i
        for i, state in enumerate(system.model.state_names)
        if state not in system.model.direct_state_observation_channels
    ]
    result = guess.copy()
    for row in range(1, len(result)):
        x = result[row]
        for _ in range(3):
            margin = catalog.values(data, data.time[row], x, theta)[index]
            if sign * margin > 1e-6:
                break
            gradient = np.zeros(len(x))
            scales = np.maximum(np.abs(x), 1.0)
            for j in mutable:
                step = 1e-4 * scales[j]
                plus, minus = x.copy(), x.copy()
                plus[j] += step
                minus[j] -= step
                gradient[j] = (
                    catalog.values(data, data.time[row], plus, theta)[index]
                    - catalog.values(data, data.time[row], minus, theta)[index]
                ) / (2 * step)
            direction = gradient * scales**2
            denominator = float(gradient @ direction)
            if denominator < 1e-20:
                break
            x += (sign * max(1.0, abs(margin)) - margin) * direction / denominator
    return result, {
        "target": target,
        "physical_initial_preserved": True,
        "coverage": catalog.coverage(data, theta, result),
    }


def model_features(system, training, theta) -> dict:
    """Describe actual structure and initial-point conditioning, without labels."""
    catalog = BranchCatalog(system)
    state_set = set(system.model.state_names)
    dependencies = {
        name: {
            n.id
            for n in ast.walk(expression.tree)
            if isinstance(n, ast.Name) and n.id in state_set
        }
        for location, expression in catalog.expressions.items()
        if location.startswith("rhs:")
        for name in [location.removeprefix("rhs:")]
    }
    reach = {n: set(dependencies[n]) | {n} for n in dependencies}
    for _ in state_set:
        reach = {
            n: set().union(*(reach[k] for k in reachable))
            for n, reachable in reach.items()
        }
    components, seen = [], set()
    for name in sorted(state_set):
        if name not in seen:
            component = sorted(
                n for n in state_set if n in reach[name] and name in reach[n]
            )
            components.append(component)
            seen.update(component)
    powers = []
    for location, expression in catalog.expressions.items():
        for node in ast.walk(expression.tree):
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Pow)
                and any(
                    isinstance(k, ast.Name) and k.id in state_set
                    for k in ast.walk(node.left)
                )
            ):
                powers.append({"location": location, "expression": ast.unparse(node)})
    records = []
    for data in training.trajectories:
        row = {
            "trajectory_id": data.trajectory_id,
            "samples": data.number_of_rows,
            "horizon": float(data.time[-1] - data.time[0]),
        }
        try:
            x = system.initial_for(data, theta)
            forcing = trajectory_forcing(system.model, data)
            u = [forcing.value(k, data.time[0]) for k in system.inputs]
            jac = np.asarray(system.state_jacobian(data.time[0], x, theta, u))
            row.update(
                initial_states=dict(zip(system.model.state_names, x, strict=True)),
                initial_rhs_maximum=float(
                    np.max(np.abs(system.rhs(data.time[0], x, theta, u)))
                ),
                initial_jacobian_maximum=float(np.max(np.abs(jac))),
                initial_spectral_abscissa=float(np.max(np.linalg.eigvals(jac).real)),
            )
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as error:
            row["error"] = str(error)[-400:]
        records.append(row)
    return {
        "states": system.state_count,
        "rhs_expressions": {
            k: v.source for k, v in catalog.expressions.items() if k.startswith("rhs:")
        },
        "coupled_components": components,
        "has_nonsmooth_composite": system.has_piecewise,
        "parameters": len(system.names),
        "initial_parameters": list(system.initial_parameter_names),
        "training_trajectories": len(training.trajectories),
        "state_powers": powers,
        "branches": [b.label for b in catalog.branches],
        "initial_point_diagnostics": records,
        "local_eigenvalues_are_not_a_global_stability_certificate": True,
    }
