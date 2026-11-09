"""Training rollouts for the five audited T2 models, including a value-only clip.

The fast evaluator uses exact affine propagation when the restricted equations
admit it. Nonlinear trials use a substepped RK4 program built from those same
expressions. Independent adaptive original-equation replay guards retention.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from time import monotonic

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from autoformalism.expressions import compile_candidate
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import remaining_portfolio_search as existing
from autoformalism.fitting.casadi_initializer import _translate
from autoformalism.fitting.fitter import _parameter_variable
from autoformalism.fitting.initialization_graph import apply_initialization_graph
from autoformalism.fitting.public_fitting import unpack_split
from autoformalism.fitting.simulation import (
    forcing_segment_indices,
    trajectory_forcing,
    trajectory_initial_state,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit


class BreadthRollout:
    """Preserve the original graph, physical bounds and per-trajectory root slots."""

    def __init__(
        self, branch: str, problem: PrefixFitProblem, training: PublicSplit, warm: dict
    ):
        if training.name != "train":
            raise ValueError("T2 fitting requires public training data")
        self.branch = branch
        self.problem = problem
        self.training = training
        self.rows = unpack_split(training).trajectories
        base = compile_candidate(problem.base_candidate, problem.context)
        self.equations = base
        self.model, _, graph = apply_initialization_graph(base, problem.graph)
        self.roots = graph["trajectory_root_parameters"]
        self.shared = tuple(base.parameter_names)
        self.states = self.model.state_names
        self.channels = tuple(problem.context.targets)
        self.inputs = tuple(sorted(self.model.validated.forcing_symbols))
        if set(self.channels) != {"Gp", "I", "U"} or not set(self.inputs) <= set(
            problem.context.external_inputs
        ):
            raise ValueError("unexpected targets or external forcing for T2 breadth")
        self.time = np.asarray(self.rows[0].time)
        if any(not np.array_equal(row.time, self.time) for row in self.rows):
            raise ValueError("batched fitting requires a common public time grid")
        if not np.allclose(np.diff(self.time), 1, rtol=0, atol=1e-12):
            raise ValueError("T2 search step requires the supplied one-minute knots")
        self.observed = np.stack(
            [
                np.column_stack([row.targets[c] for c in self.channels])
                for row in self.rows
            ]
        )
        self.scales = np.maximum(np.std(self.observed, axis=(0, 1)), 1e-12)
        self.forcing = np.stack(
            [
                np.column_stack([r.external_inputs[k] for k in self.inputs])
                for r in self.rows
            ]
        )
        self.original_names = self.shared + tuple(
            f"trajectory:{row.trajectory_id}:{name}"
            for row in self.rows
            for name in self.roots.values()
        )
        if set(warm) != set(self.original_names):
            raise ValueError(
                "incomplete original shared coefficients or training roots"
            )
        self.warm = dict(warm)
        self._compile_values()
        self.chart = None
        if branch != "c15-0":
            route = "affine" if self.affine else "nonlinear"
            _, self.chart, self.start = existing.setup(
                problem, training, kernel.NumericalPolicy(), warm, route
            )
            self.names = tuple(self.chart.names)
            self.lower, self.upper = self.chart.lower.copy(), self.chart.upper.copy()
        else:
            self.names = self.original_names
            self.start = self.warm.copy()
            variables = {
                s.name: _parameter_variable(self.model, s, kernel.FitConfig())
                for s in self.model.validated.candidate.parameters
            }
            layout = [variables[n.split(":")[-1]] for n in self.names]
            self.lower = np.array([x.lower for x in layout])
            self.upper = np.array([x.upper for x in layout])
        self._grid_cache: dict = {}
        self.vector(self.start)

    def _compile_values(self):
        """Lower validated expressions for numerical values, including kinks."""
        model = self.model
        self.n = len(self.states)
        t = ca.SX.sym("t")
        x = ca.SX.sym("x", self.n)
        p = ca.SX.sym("p", len(self.shared))
        u = ca.SX.sym("u", len(self.inputs))
        env = dict(zip(self.states, ca.vertsplit(x), strict=True))
        env.update(zip(self.shared, ca.vertsplit(p), strict=True))
        env.update(zip(self.inputs, ca.vertsplit(u), strict=True))
        env[model.validated.context.time_symbol] = t
        for name in model.validated.process_order:
            env[name] = _translate(ca, model.validated.process_expressions[name], env)
        rhs = ca.vertcat(
            *[
                _translate(ca, model.validated.equation_expressions[n], env)
                for n in self.states
            ]
        )
        obs = ca.vertcat(
            *[
                _translate(ca, model.validated.observation_expressions[n], env)
                for n in self.channels
            ]
        )
        if ca.depends_on(ca.vertcat(rhs, obs), t) or ca.depends_on(
            obs, ca.vertcat(p, u)
        ):
            raise ValueError(
                "T2 fast rollout requires autonomous equations and fixed readouts"
            )
        # The published readouts in these packets are exact homogeneous linear maps.
        self.H = np.asarray(ca.DM(ca.jacobian(obs, x)))
        if not ca.is_equal(
            ca.substitute(obs, x, ca.SX.zeros(self.n)),
            ca.SX.zeros(len(self.channels)),
            5,
        ):
            raise ValueError("T2 readout has nonzero offset")
        self.rhs = ca.Function("original_rhs_values", [x, p, u], [rhs])
        self.used_shared = {
            n for j, n in enumerate(self.shared) if ca.depends_on(rhs, p[j])
        }
        xu = ca.vertcat(x, u)
        # The clipped branch always takes the numerical value route.
        self.affine = self.branch != "c15-0" and not ca.depends_on(
            ca.jacobian(rhs, xu), xu
        )
        if self.affine:
            self.coefficients = ca.Function(
                "affine_coefficients",
                [p],
                [
                    ca.jacobian(rhs, x),
                    ca.jacobian(rhs, u),
                    ca.substitute(rhs, xu, ca.SX.zeros(xu.numel())),
                ],
            )
        self._rk = {}

    def vector(self, point: Mapping[str, float]) -> np.ndarray:
        if set(point) != set(self.names):
            raise ValueError("incomplete search coordinates")
        v = np.asarray([point[n] for n in self.names], float)
        if not np.isfinite(v).all() or np.any(v < self.lower) or np.any(v > self.upper):
            raise ValueError("search coordinates outside declared domain")
        return v

    def lift(self, v: np.ndarray) -> dict:
        if self.chart is None:
            point = dict(zip(self.names, v.tolist(), strict=True))
            self.vector(point)
            return point
        return self.chart.original.parameters(self.chart.reduction.lift(v))

    def reduce(self, point: dict) -> np.ndarray:
        if self.chart is None:
            return self.vector(point)
        return self.chart.reduction.reduce(self.chart.original.vector(point))

    def initial(self, point: dict, row) -> np.ndarray:
        params = {k: point[k] for k in self.shared}
        params.update(
            {
                n: point[f"trajectory:{row.trajectory_id}:{n}"]
                for n in self.roots.values()
            }
        )
        return trajectory_initial_state(self.model, row, {}, parameters=params)

    def grid(self, subdivisions: int):
        if subdivisions not in self._grid_cache:
            t = np.r_[
                np.concatenate(
                    [
                        np.linspace(a, b, subdivisions, endpoint=False)
                        for a, b in pairwise(self.time)
                    ]
                ),
                self.time[-1],
            ]
            forcing = np.stack(
                [
                    np.stack(
                        [
                            np.interp(t, self.time, arr[:, j])
                            for j in range(len(self.inputs))
                        ]
                    )
                    for arr in self.forcing
                ]
            )
            self._grid_cache[subdivisions] = (t, forcing)
        return self._grid_cache[subdivisions]

    def _rk_flow(self, steps: int, dt: float):
        """Evaluate the public input ramp at every Runge-Kutta stage."""
        key = steps, dt
        if key in self._rk:
            return self._rk[key]
        npar, m = len(self.shared), len(self.inputs)
        x = ca.SX.sym("x", self.n)
        p = ca.SX.sym("p", npar)
        u = ca.SX.sym("u", m)
        v = ca.SX.sym("v", m)
        sub = max(1, round(dt * 8))
        h = dt / sub
        y = x
        for k in range(sub):
            u0 = u + (v - u) * (k / sub)
            um = u + (v - u) * ((k + 0.5) / sub)
            u1 = u + (v - u) * ((k + 1) / sub)
            k1 = self.rhs(y, p, u0)
            k2 = self.rhs(y + h * k1 / 2, p, um)
            k3 = self.rhs(y + h * k2 / 2, p, um)
            k4 = self.rhs(y + h * k3, p, u1)
            y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        step = ca.Function("breadth_rk_step", [x, p, u, v], [y])
        X = ca.MX.sym("X", self.n)
        P = ca.MX.sym("P", npar)
        U = ca.MX.sym("U", m, steps + 1)
        flow = step.mapaccum(steps)(X, ca.repmat(P, 1, steps), U[:, :-1], U[:, 1:])
        self._rk[key] = ca.Function(
            "breadth_trajectory", [X, P, U], [ca.horzcat(X, flow)]
        ).map(len(self.rows))
        return self._rk[key]

    def fast(self, point: dict, subdivisions: int = 2):
        """Return predictions arranged by training trajectory, time and channel."""
        ts, u = self.grid(subdivisions)
        p = np.array([point[n] for n in self.shared])
        initial = np.column_stack([self.initial(point, row) for row in self.rows])
        dt = ts[1] - ts[0]
        if self.affine:
            M, B, b = map(np.asarray, self.coefficients(p))
            n, m = self.n, len(self.inputs)
            aug = np.zeros((n + 2 * m + 1, n + 2 * m + 1))
            aug[:n, :n] = M
            aug[:n, n : n + m] = B
            aug[:n, -1:] = b
            aug[n : n + m, n + m : n + 2 * m] = np.eye(m)
            with np.errstate(over="ignore", invalid="ignore"):
                transition = expm(aug * dt)[:n]
                left = transition[:, n : n + m]
                slopes = transition[:, n + m : n + 2 * m] / dt
                forcing = np.einsum("ij,rjt->tir", left, u[:, :, :-1]) + np.einsum(
                    "ij,rjt->tir", slopes, np.diff(u, axis=2)
                )
                forcing += transition[:, -1, None]
                x = initial
                all_x = np.empty((len(ts), n, len(self.rows)))
                all_x[0] = x
                for pos, add in enumerate(forcing, 1):
                    x = transition[:, :n] @ x + add
                    all_x[pos] = x
                values = np.einsum("ci,tir->rtc", self.H, all_x)
        else:
            packed = np.asarray(
                self._rk_flow(len(ts) - 1, dt)(
                    initial,
                    np.tile(p[:, None], (1, len(self.rows))),
                    np.concatenate(u, axis=1),
                )
            )
            xs = packed.reshape(self.n, len(self.rows), len(ts))
            values = np.einsum("ci,irt->rtc", self.H, xs)
        if not np.isfinite(values).all() or np.max(np.abs(values)) > 1e18:
            raise ValueError("nonfinite or unresolved search rollout")
        return values

    def original(
        self, point: dict, subdivisions: int = 1, *, method="Radau", seconds=240.0
    ):
        """Replay original expressions with adaptive integration at input corners."""
        ts, _ = self.grid(subdivisions)
        p = {n: point[n] for n in self.shared}
        states = []
        deadline = monotonic() + seconds
        # The clipped term changes slope when I crosses zero. Cap internal steps
        # for this branch even in long intervals of constant input.
        tolerances = (
            {"rtol": 1e-12, "atol": 1e-14, "max_step": 2.0}
            if self.branch == "c15-0"
            else {"rtol": 1e-9, "atol": 1e-11}
        )
        for row in self.rows:
            x = self.initial(point, row)
            blocks = [x[None, :]]
            forcing = trajectory_forcing(self.model, row)

            def rhs(t, y, forcing=forcing):
                if monotonic() >= deadline:
                    raise TimeoutError("original-equation replay deadline")
                return self.equations.rhs(t, y, p, forcing)

            for a, b in pairwise(
                np.asarray(row.time)[forcing_segment_indices(self.model, row)]
            ):
                grid = ts[(ts > a) & (ts <= b)]
                solution = solve_ivp(
                    rhs, (a, b), x, method=method, t_eval=grid, **tolerances
                )
                if not solution.success or solution.y.shape != (self.n, len(grid)):
                    raise ValueError("original ODE integration failed")
                x = solution.y[:, -1]
                blocks.append(solution.y.T)
            trajectory_states = np.concatenate(blocks)
            for j in (0, len(ts) // 2, len(ts) - 1):
                output = self.equations.observe(ts[j], trajectory_states[j], p, forcing)
                if not np.allclose(
                    self.H @ trajectory_states[j],
                    [output[c] for c in self.channels],
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "original readout and compiled observation map differ"
                    )
            states.append(trajectory_states)
        states = np.stack(states)
        values = states @ self.H.T
        if not np.isfinite(values).all():
            raise ValueError("nonfinite original ODE outputs")
        return values, states

    def nmse(self, values: np.ndarray) -> float:
        return float(np.mean(((values - self.observed) / self.scales) ** 2))
