"""Restricted symbolic derivatives and forward sensitivities for frozen probes.

This opt-in diagnostic supports global parameters, fixed numeric initial states,
open rollouts, smooth expressions and no state constraints. It never evaluates
proposer text as Python and does not change the production fitter defaults.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from time import monotonic

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.data import DatasetSplit, Trajectory
from autoformalism.expressions import CompiledModel
from autoformalism.fitting.casadi_initializer import _translate
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import forcing_segment_indices, trajectory_forcing
from autoformalism.fitting.stagnation import RolloutOracle
from autoformalism.rebuttal.fitter_diagnostic import _finite_payload, write_json


@dataclass
class SymbolicODE:
    """Expand processes before differentiating both dynamics and observations."""

    model: CompiledModel

    def __post_init__(self) -> None:
        candidate = self.model.validated.candidate
        if self.model.validated.context.lagged_targets or candidate.constraints:
            raise ValueError(
                "sensitivity probe requires open rollouts without constraints"
            )
        if any(p.scope.value != "global" for p in candidate.parameters):
            raise ValueError("sensitivity probe requires global parameters")
        initial = {i.state: i.fixed_value for i in candidate.initial_conditions}
        if set(initial) != set(self.model.state_names) or any(
            v is None for v in initial.values()
        ):
            raise ValueError("sensitivity probe requires fixed numeric initial states")
        expressions = [
            *self.model.validated.process_expressions.values(),
            *self.model.validated.equation_expressions.values(),
            *self.model.validated.observation_expressions.values(),
        ]
        for expression in expressions:
            for node in ast.walk(expression.tree):
                if (
                    isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.Pow)
                    and (
                        not isinstance(node.right, ast.Constant) or node.right.value < 0
                    )
                ):
                    raise ValueError("probe declines negative or variable powers")
                if isinstance(node, ast.Call) and node.func.id in {
                    "abs",
                    "min",
                    "max",
                    "sqrt",
                    "log",
                }:
                    raise ValueError(
                        "probe declines nonsmooth or domain-restricted functions"
                    )
        self.names = self.model.parameter_names
        self.channels = tuple(self.model.validated.context.targets)
        self.inputs = tuple(sorted(self.model.validated.forcing_symbols))
        self.initial = np.array(
            [initial[n] for n in self.model.state_names], dtype=float
        )
        n, p = len(self.initial), len(self.names)
        x, theta = ca.SX.sym("x", n), ca.SX.sym("theta", p)
        u, time = ca.SX.sym("u", len(self.inputs)), ca.SX.sym("time")
        env = {
            **dict(zip(self.model.state_names, ca.vertsplit(x), strict=True)),
            **dict(zip(self.names, ca.vertsplit(theta), strict=True)),
            **dict(zip(self.inputs, ca.vertsplit(u), strict=True)),
            self.model.validated.context.time_symbol: time,
        }
        for name in self.model.validated.process_order:
            env[name] = _translate(
                ca, self.model.validated.process_expressions[name], env
            )
        rhs = ca.vertcat(
            *[
                _translate(ca, self.model.validated.equation_expressions[n], env)
                for n in self.model.state_names
            ]
        )
        obs = ca.vertcat(
            *[
                _translate(ca, self.model.validated.observation_expressions[n], env)
                for n in self.channels
            ]
        )
        fx, ft = ca.jacobian(rhs, x), ca.jacobian(rhs, theta)
        hx, ht = ca.jacobian(obs, x), ca.jacobian(obs, theta)
        args = [time, x, theta, u]
        self.rhs = ca.Function("rhs", args, [rhs])
        self.observe = ca.Function("observe", args, [obs])
        self.local = ca.Function("partials", args, [fx, ft, hx, ht])
        self.state_jacobian = ca.Function("state_jacobian", args, [fx])
        self.affine = not ca.depends_on(ca.jacobian(ca.vertcat(rhs, obs), theta), theta)
        self.rhs_affine = not ca.depends_on(ft, theta)
        sensitivity = ca.SX.sym("S", n, p)
        augmented = ca.vertcat(x, ca.reshape(sensitivity, n * p, 1))
        augmented_rhs = ca.vertcat(rhs, ca.reshape(fx @ sensitivity + ft, n * p, 1))
        self.augmented_rhs = ca.Function(
            "augmented_rhs", [time, augmented, theta, u], [augmented_rhs]
        )
        self.augmented_jacobian = ca.Function(
            "augmented_jacobian",
            [time, augmented, theta, u],
            [ca.jacobian(augmented_rhs, augmented)],
        )
        self.audit = {
            "parameter_order": self.names,
            "expanded_rhs": str(rhs),
            "expanded_observation": str(obs),
            "rhs_parameter_affine_certified": self.rhs_affine,
            "joint_rhs_observation_affine_certified": self.affine,
            "local_parameter_partials": str(ft),
            "rollout_linearity_claimed": False,
        }


def symbolic_rollout(
    system: SymbolicODE,
    trajectory: Trajectory,
    theta: np.ndarray,
    settings: FitConfig,
    deadline: float | None,
    *,
    sensitivities: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict]:
    """Integrate states and optionally S'=F_x S+F_theta with exact local AD."""
    n, p = len(system.initial), len(theta)
    time = trajectory.time
    forcing = trajectory_forcing(system.model, trajectory)
    states = np.empty((len(time), n))
    values = np.empty((len(time), len(system.channels)))
    jacobian = np.empty((len(time), len(system.channels), p)) if sensitivities else None
    current = (
        np.concatenate((system.initial, np.zeros(n * p)))
        if sensitivities
        else system.initial.copy()
    )
    all_values = np.empty((len(time), len(current)))
    all_values[0] = current
    counters = {"nfev": 0, "njev": 0, "nlu": 0, "segments": 0}

    def inputs(t):
        if deadline is not None and monotonic() >= deadline:
            raise TimeoutError("sensitivity rollout wall-clock limit reached")
        return np.array([forcing.value(name, t) for name in system.inputs])

    rhs_function = system.augmented_rhs if sensitivities else system.rhs
    jac_function = system.augmented_jacobian if sensitivities else system.state_jacobian

    def rhs(t, z):
        result = np.asarray(rhs_function(t, z, theta, inputs(t))).ravel()
        if not np.isfinite(result).all():
            raise ValueError("symbolic RHS is nonfinite")
        return result

    def jac(t, z):
        return np.asarray(jac_function(t, z, theta, inputs(t)))

    for left, right in pairwise(forcing_segment_indices(system.model, trajectory)):
        solved = solve_ivp(
            rhs,
            (time[left], time[right]),
            current,
            t_eval=time[left : right + 1],
            method=settings.integration_method,
            rtol=settings.relative_tolerance,
            atol=settings.absolute_tolerance,
            jac=jac,
        )
        if not solved.success or not np.isfinite(solved.y).all():
            raise ValueError(f"symbolic integration failed: {solved.message}")
        all_values[left : right + 1] = solved.y.T
        current = solved.y[:, -1]
        for name in ("nfev", "njev", "nlu"):
            counters[name] += int(getattr(solved, name, 0))
        counters["segments"] += 1
    states[:] = all_values[:, :n]
    for i, t in enumerate(time):
        u = inputs(t)
        values[i] = np.asarray(system.observe(t, states[i], theta, u)).ravel()
        if sensitivities:
            _, _, hx, ht = system.local(t, states[i], theta, u)
            s = all_values[i, n:].reshape((n, p), order="F")
            jacobian[i] = np.asarray(hx) @ s + np.asarray(ht)
    if not np.isfinite(values).all() or (
        jacobian is not None and not np.isfinite(jacobian).all()
    ):
        raise ValueError("symbolic output or sensitivity is nonfinite")
    return values, jacobian, states, counters


class SymbolicOracle(RolloutOracle):
    """Instrument comparable compiled finite-difference and sensitivity arms."""

    def __init__(
        self,
        system: SymbolicODE,
        training: DatasetSplit,
        scale: float,
        settings: FitConfig,
        directory: Path,
        deadline: float,
        *,
        sensitivities: bool,
    ):
        super().__init__(
            system.model, training, {"v01": scale}, settings, directory, deadline
        )
        if self.names != system.names or system.channels != ("v01",):
            raise ValueError(
                "probe requires the production parameter order and one target"
            )
        self.system, self.training, self.scale = system, training, scale
        self.settings, self.deadline = settings, deadline
        self.with_sensitivities = sensitivities
        self.last_x = self.last_jac = None
        self.solver_counts = {"nfev": 0, "njev": 0, "nlu": 0, "segments": 0}

    def __call__(self, values: np.ndarray) -> np.ndarray:
        self.calls += 1
        started = monotonic()
        record = {
            "call": self.calls,
            "parameters": dict(zip(self.names, map(float, values), strict=True)),
        }
        try:
            self.vector(record["parameters"])
            residuals, matrices = [], []
            for trajectory in self.training.trajectories:
                predictions, jac, _, counts = symbolic_rollout(
                    self.system,
                    trajectory,
                    values,
                    self.settings,
                    self.deadline,
                    sensitivities=self.with_sensitivities,
                )
                residuals.append(
                    (predictions[:, 0] - trajectory.targets["v01"]) / self.scale
                )
                if jac is not None:
                    matrices.append(jac[:, 0, :] / self.scale)
                for key, value in counts.items():
                    self.solver_counts[key] += value
            residual = np.concatenate(residuals)
            if np.max(np.abs(residual)) >= self.settings.failure_penalty:
                raise ValueError("probe residual reaches production clipping threshold")
            self.last_x = values.copy()
            self.last_jac = np.concatenate(matrices) if matrices else None
            cost = float(0.5 * residual @ residual)
            record.update(status="evaluated", cost=cost)
            if self.best is None or cost < self.best["cost"]:
                self.best = {
                    "parameters": record["parameters"],
                    "cost": cost,
                    "call": self.calls,
                }
                write_json(self.directory / "best_evaluated.json", self.best)
            return residual
        except TimeoutError:
            record["status"] = "timeout"
            raise
        except (ValueError, RuntimeError, ArithmeticError) as error:
            self.failures.record(str(error))
            record.update(status="failed", error=str(error))
            size = sum(t.number_of_rows for t in self.training.trajectories)
            self.last_x = values.copy()
            self.last_jac = np.zeros((size, len(values)))
            return np.full(size, self.settings.failure_penalty)
        finally:
            seconds = monotonic() - started
            self.seconds += seconds
            record["seconds"] = seconds
            write_json(
                self.directory / f"{self.calls:06d}.json", _finite_payload(record)
            )

    def jacobian(self, values: np.ndarray) -> np.ndarray:
        """Reuse the augmented solve from the optimizer's matching residual call."""
        if not self.with_sensitivities:
            raise ValueError(
                "sensitivity Jacobian requested from finite-difference arm"
            )
        if self.last_x is None or not np.array_equal(self.last_x, values):
            self(values)
        return self.last_jac
