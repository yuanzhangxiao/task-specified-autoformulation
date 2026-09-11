"""Restricted symbolic derivatives and forward sensitivities for frozen probes.

This opt-in diagnostic supports global parameters, parameter-independent causal
initializers, open rollouts, certified smooth composites and no state constraints.
It never evaluates proposer text as Python or changes production fitter defaults.
"""

from __future__ import annotations

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
from autoformalism.fitting.sensitivity_contract import (
    SensitivityContractError,
    certify_expressions,
)
from autoformalism.fitting.simulation import (
    forcing_segment_indices,
    trajectory_forcing,
    trajectory_initial_state,
)
from autoformalism.fitting.stagnation import RolloutOracle
from autoformalism.rebuttal.fitter_diagnostic import _finite_payload, write_json


@dataclass
class SymbolicODE:
    """Expand processes before differentiating both dynamics and observations."""

    model: CompiledModel

    def __post_init__(self) -> None:
        candidate = self.model.validated.candidate
        if self.model.validated.context.lagged_targets or candidate.constraints:
            raise SensitivityContractError(
                "sensitivity probe requires open rollouts without constraints"
            )
        if any(p.scope.value != "global" for p in candidate.parameters):
            raise SensitivityContractError(
                "sensitivity probe requires global parameters"
            )
        initials = {item.state: item for item in candidate.initial_conditions}
        if set(initials) != set(self.model.state_names):
            raise SensitivityContractError(
                "sensitivity probe requires one initializer per state"
            )
        fitted_initials = sorted(
            name
            for name, item in initials.items()
            if item.initialization_range is not None
        )
        if fitted_initials:
            raise SensitivityContractError(
                "sensitivity probe does not yet optimize fitted initial states: "
                f"{fitted_initials}"
            )
        equations, observations, smoothness = certify_expressions(self.model)
        self.requires_first_order_solver = smoothness.c1_only
        self.names = self.model.parameter_names
        self.channels = tuple(self.model.validated.context.targets)
        self.inputs = tuple(sorted(self.model.validated.forcing_symbols))
        self.state_count = len(self.model.state_names)
        self._fixed_initial = (
            np.asarray(
                [initials[name].fixed_value for name in self.model.state_names],
                dtype=float,
            )
            if all(item.fixed_value is not None for item in initials.values())
            else None
        )
        n, p = self.state_count, len(self.names)
        x, theta = ca.SX.sym("x", n), ca.SX.sym("theta", p)
        u, time = ca.SX.sym("u", len(self.inputs)), ca.SX.sym("time")
        env = {
            **dict(zip(self.model.state_names, ca.vertsplit(x), strict=True)),
            **dict(zip(self.names, ca.vertsplit(theta), strict=True)),
            **dict(zip(self.inputs, ca.vertsplit(u), strict=True)),
            self.model.validated.context.time_symbol: time,
        }
        rhs = ca.vertcat(
            *[
                _translate(ca, equations[n], env)
                for n in self.model.state_names
            ]
        )
        obs = ca.vertcat(
            *[
                _translate(ca, observations[n], env)
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
        self.augmented_jacobian = (
            None
            if smoothness.rhs_c1_only
            else ca.Function(
                "augmented_jacobian",
                [time, augmented, theta, u],
                [ca.jacobian(augmented_rhs, augmented)],
            )
        )
        self.audit = {
            "parameter_order": self.names,
            "expanded_rhs": str(rhs),
            "expanded_observation": str(obs),
            "rhs_parameter_affine_certified": self.rhs_affine,
            "joint_rhs_observation_affine_certified": self.affine,
            "local_parameter_partials": str(ft),
            "trajectory_specific_initialization": self._fixed_initial is None,
            "initial_parameter_sensitivity_zero_certified": True,
            "rollout_linearity_claimed": False,
            "smoothness_contract": "certified-composites-1",
            "smooth_composite_certificates": smoothness.certificates,
            "uncertified_crossing_policy": "reject_before_optimization",
            "c1_composites_require_first_order_solver": (
                self.requires_first_order_solver
            ),
            "augmented_solver_jacobian": (
                "numerical_newton_approximation"
                if smoothness.rhs_c1_only else "automatic_differentiation"
            ),
            "collocation_hessian": (
                "limited-memory" if self.requires_first_order_solver else "exact"
            ),
            "division_policy": "existing_production_guard_unchanged",
        }

    @property
    def initial(self) -> np.ndarray:
        """Return the common fixed initial vector for legacy scalar matching probes."""
        if self._fixed_initial is None:
            raise SensitivityContractError(
                "candidate has trajectory-specific analytic initial states"
            )
        return self._fixed_initial.copy()

    def initial_for(self, trajectory: Trajectory) -> np.ndarray:
        """Resolve the parameter-independent causal initializer for one trajectory."""
        return trajectory_initial_state(self.model, trajectory, {})


class SymbolicRolloutFailure(ValueError):
    """Numerical failure with measured context, without inferring scientific cause."""

    def __init__(self, message: str, diagnostic: dict):
        super().__init__(message)
        self.diagnostic = diagnostic


class SymbolicRolloutTimeout(TimeoutError):
    """Preserve deadline semantics while retaining the last attempted RHS state."""

    def __init__(self, message: str, diagnostic: dict):
        super().__init__(message)
        self.diagnostic = diagnostic


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
    initial = system.initial_for(trajectory)
    n, p = system.state_count, len(theta)
    time = trajectory.time
    forcing = trajectory_forcing(system.model, trajectory)
    states = np.empty((len(time), n))
    values = np.empty((len(time), len(system.channels)))
    jacobian = np.empty((len(time), len(system.channels), p)) if sensitivities else None
    current = (
        np.concatenate((initial, np.zeros(n * p))) if sensitivities else initial.copy()
    )
    all_values = np.empty((len(time), len(current)))
    all_values[0] = current
    counters = {"nfev": 0, "njev": 0, "nlu": 0, "segments": 0}
    last_attempt: dict = {}

    def diagnostic(left, right, solved=None):
        evidence = {
            "trajectory_id": trajectory.trajectory_id,
            "stage": "integration",
            "integration_interval": [float(time[left]), float(time[right])],
            "last_rhs_attempt": last_attempt,
            "attempt_is_accepted_solver_state": False,
            "sensitivities": sensitivities,
        }
        if solved is not None and len(solved.t):
            evidence["last_returned_sample"] = {
                "time": float(solved.t[-1]),
                "states": dict(
                    zip(
                        system.model.state_names, solved.y[:n, -1].tolist(), strict=True
                    )
                ),
            }
        return _finite_payload(evidence)

    def inputs(t):
        if deadline is not None and monotonic() >= deadline:
            raise TimeoutError("sensitivity rollout wall-clock limit reached")
        return np.array([forcing.value(name, t) for name in system.inputs])

    rhs_function = system.augmented_rhs if sensitivities else system.rhs
    jac_function = system.augmented_jacobian if sensitivities else system.state_jacobian

    def rhs(t, z):
        last_attempt.clear()
        last_attempt.update(
            time=float(t),
            states=dict(zip(system.model.state_names, z[:n].tolist(), strict=True)),
        )
        result = np.asarray(rhs_function(t, z, theta, inputs(t))).ravel()
        last_attempt["state_rhs"] = dict(
            zip(system.model.state_names, result[:n].tolist(), strict=True)
        )
        last_attempt["nonfinite_state_rhs"] = [
            name
            for name, value in zip(system.model.state_names, result[:n], strict=True)
            if not np.isfinite(value)
        ]
        last_attempt["nonfinite_sensitivity_rhs"] = bool(
            not np.isfinite(result[n:]).all()
        )
        if not np.isfinite(result).all():
            raise ValueError("symbolic RHS is nonfinite")
        return result

    def jac(t, z):
        return np.asarray(jac_function(t, z, theta, inputs(t)))

    for left, right in pairwise(forcing_segment_indices(system.model, trajectory)):
        try:
            solved = solve_ivp(
                rhs,
                (time[left], time[right]),
                current,
                t_eval=time[left : right + 1],
                method=settings.integration_method,
                rtol=settings.relative_tolerance,
                atol=settings.absolute_tolerance,
                # First parameter sensitivities remain analytic. Only the
                # integrator's Newton matrix is approximated for C1-only laws.
                jac=jac if jac_function is not None else None,
            )
        except TimeoutError as error:
            raise SymbolicRolloutTimeout(str(error), diagnostic(left, right)) from error
        except (ValueError, RuntimeError, ArithmeticError) as error:
            raise SymbolicRolloutFailure(str(error), diagnostic(left, right)) from error
        if not solved.success or not np.isfinite(solved.y).all():
            raise SymbolicRolloutFailure(
                f"symbolic integration failed: {solved.message}",
                diagnostic(left, right, solved),
            )
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
        self.failure_evidence: list[dict] = []

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
                record["trajectory_id"] = trajectory.trajectory_id
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
            if not np.isfinite(cost):
                raise ValueError("full training residual cost is nonfinite")
            self.valid_calls += 1
            record.update(
                status="evaluated", cost=cost, valid_full_training_evaluation=True
            )
            if self.best is None or cost < self.best["cost"]:
                self.best = {
                    "parameters": record["parameters"],
                    "cost": cost,
                    "call": self.calls,
                }
                write_json(self.directory / "best_evaluated.json", self.best)
            return residual
        except TimeoutError as error:
            record.update(
                status="timeout",
                error=str(error),
                diagnostic=getattr(error, "diagnostic", None),
            )
            raise
        except (ValueError, RuntimeError, ArithmeticError) as error:
            self.failures.record(str(error))
            record.update(
                status="failed",
                error=str(error),
                diagnostic=getattr(error, "diagnostic", None),
            )
            size = sum(t.number_of_rows for t in self.training.trajectories)
            self.last_x = values.copy()
            self.last_jac = np.zeros((size, len(values)))
            return np.full(size, self.settings.failure_penalty)
        finally:
            seconds = monotonic() - started
            self.seconds += seconds
            record["seconds"] = seconds
            if (
                record.get("status") in {"failed", "timeout"}
                and len(self.failure_evidence) < 5
            ):
                self.failure_evidence.append(_finite_payload(dict(record)))
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
