"""Opt-in exact affine propagation for scoped training and prefix oracles.

Only autonomous, jointly affine state/input dynamics are admitted. Parameter
dependence and causal initial maps may be nonlinear. The inherited parameter
domains, training normalization, output mappings and trajectory-root layout stay
unchanged. Explicit ODE methods delegate to the original compiled-model oracle;
matrix-exponential agreement is not an optimization convergence certificate.
"""

from __future__ import annotations

from collections.abc import Mapping
from time import monotonic

import casadi as ca
import numpy as np
from scipy.linalg import expm, expm_frechet

from autoformalism.data import Trajectory
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit


def _check_deadline(deadline: float | None) -> None:
    """Reject overruns both before and after noninterruptible matrix operations."""
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("scoped affine rollout wall-clock limit reached")


def _finite(values: np.ndarray, label: str) -> np.ndarray:
    """Fail closed on an unusable state, derivative or transition matrix."""
    if not np.isfinite(values).all():
        raise NumericalInvalidTrial(f"nonfinite scoped affine {label}")
    return values


class _Propagation:
    """Certify one compiled system and selected local derivative coordinates."""

    def __init__(self, system: SymbolicODE, columns: tuple[int, ...]) -> None:
        self.system, self.columns = system, columns
        n, m, p = system.state_count, len(system.inputs), len(system.names)
        t, x = ca.SX.sym("t"), ca.SX.sym("x", n)
        u, theta = ca.SX.sym("u", m), ca.SX.sym("theta", p)
        rhs = system.rhs(t, x, theta, u)
        a, b = ca.jacobian(rhs, x), ca.jacobian(rhs, u)
        zero = ca.substitute(rhs, ca.vertcat(x, u), ca.SX.zeros(n + m))
        if ca.depends_on(ca.vertcat(ca.vec(a), ca.vec(b), zero), ca.vertcat(t, x, u)):
            raise ValueError("requires autonomous jointly affine state/input dynamics")
        # [state, forcing, forcing slope, 1] preserves every supplied linear piece.
        generator = ca.SX.zeros(n + 2 * m + 1, n + 2 * m + 1)
        generator[:n, :n], generator[:n, n : n + m] = a, b
        generator[:n, -1] = zero
        generator[n : n + m, n + m : n + 2 * m] = ca.SX.eye(m)
        self.generator = ca.Function("scoped_affine_generator", [theta], [generator])
        self.derivatives = ca.Function(
            "scoped_affine_derivatives",
            [theta],
            [ca.jacobian(ca.vec(generator), theta)[:, list(columns)]],
        )
        self.dynamic = tuple(j for j in range(p) if ca.depends_on(generator, theta[j]))
        self.audit = {
            "protocol": "scoped-prefix-affine-rollout-1",
            "eligible": True,
            "certificate": "autonomous-affine-state-input-1",
            "state_count": n,
            "channel_order": list(system.channels),
            "local_parameter_order": list(system.names),
            "sensitivity_parameters": [system.names[j] for j in columns],
            "generator_parameters": [system.names[j] for j in self.dynamic],
            "forcing": "public piecewise-linear interpolant; every interval retained",
            "parameter_derivatives": "matrix-exponential Frechet derivatives",
            "initialization": "original causal graph evaluated separately for each row",
            "observations": "original compiled output mappings and their derivatives",
            "cache_scope": "one oracle evaluation; exact dynamics coordinates and dt",
            "reference_coefficients_used": False,
        }

    def rollout(
        self,
        row: Trajectory,
        vector: np.ndarray,
        deadline: float | None,
        *,
        jacobian: bool,
        cache: dict,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Propagate one physical boundary without mixing roots between rows."""
        _check_deadline(deadline)
        system, columns = self.system, list(self.columns)
        n, k = system.state_count, len(columns)
        generator = _finite(np.asarray(self.generator(vector)), "generator")
        derivative = (
            _finite(
                np.asarray(self.derivatives(vector)).reshape(
                    (*generator.shape, k), order="F"
                ),
                "generator derivative",
            )
            if jacobian
            else None
        )
        forcing = trajectory_forcing(system.model, row)
        inputs = np.asarray(
            [[forcing.value(name, t) for name in system.inputs] for t in row.time]
        ).reshape(len(row.time), len(system.inputs))
        _finite(inputs, "forcing")
        state = _finite(system.initial_for(row, vector), "initial state")
        sensitivity = (
            _finite(
                system.initial_sensitivity(row, vector)[:, columns],
                "initial derivative",
            )
            if jacobian
            else None
        )
        dynamic_key = tuple(vector[j] for j in self.dynamic)
        values, gradients = [], []
        for index, time in enumerate(row.time):
            _check_deadline(deadline)
            if index:
                dt = float(time - row.time[index - 1])
                key = dynamic_key, dt
                if key not in cache:
                    matrix = _finite(expm(generator * dt), "transition")
                    _check_deadline(deadline)
                    derivatives = []
                    if jacobian:
                        for j in range(k):
                            _check_deadline(deadline)
                            # Initial-only parameters have exactly zero generator
                            # derivatives; their initial sensitivities still evolve.
                            direction = derivative[:, :, j]
                            derivatives.append(
                                _finite(
                                    expm_frechet(
                                        generator * dt,
                                        direction * dt,
                                        compute_expm=False,
                                    ),
                                    "transition derivative",
                                )
                                if np.any(direction)
                                else np.zeros_like(generator)
                            )
                            _check_deadline(deadline)
                    cache[key] = matrix, derivatives
                matrix, derivatives = cache[key]
                extended = np.r_[
                    state,
                    inputs[index - 1],
                    (inputs[index] - inputs[index - 1]) / dt,
                    1,
                ]
                if jacobian:
                    sensitivity = matrix[:n, :n] @ sensitivity
                    if k:
                        sensitivity += np.column_stack(
                            [(item @ extended)[:n] for item in derivatives]
                        )
                    _finite(sensitivity, "state derivative")
                state = _finite((matrix @ extended)[:n], "state")
            values.append(
                np.asarray(system.observe(time, state, vector, inputs[index])).ravel()
            )
            if jacobian:
                _, _, hx, ht = system.local(time, state, vector, inputs[index])
                gradients.append(
                    np.asarray(hx) @ sensitivity + np.asarray(ht)[:, columns]
                )
        _check_deadline(deadline)
        return _finite(np.asarray(values), "outputs"), (
            _finite(np.asarray(gradients), "output derivatives") if jacobian else None
        )


class _AffineEvaluation:
    """Share propagation while retaining the two inherited coordinate layouts."""

    def _initialize_affine(self, columns: tuple[int, ...], indices: dict) -> None:
        self._propagation = _Propagation(self.system, columns)
        self._affine_indices = indices
        self.affine_audit = self._propagation.audit | {
            "problem_sha256": content_sha256(self.problem),
            "joint_parameter_order": list(self.names),
        }

    def evaluate(
        self,
        vector: np.ndarray,
        deadline: float | None,
        *,
        jacobian: bool = True,
        method: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Use affine propagation unless an original ODE method is requested."""
        if deadline is not None and not np.isfinite(deadline):
            raise ValueError("affine deadline must be finite or None")
        _check_deadline(deadline)
        if method is not None:
            return super().evaluate(vector, deadline, jacobian=jacobian, method=method)
        self.vector(self.parameters(vector))
        scales = np.asarray([self.scales[name] for name in self.system.channels])
        residuals, derivatives, cache = [], [], {}
        for row in self.rows:
            values, local = self._propagation.rollout(
                row,
                self.local_vector(vector, row),
                deadline,
                jacobian=jacobian,
                cache=cache,
            )
            observed = np.column_stack(
                [row.targets[name] for name in self.system.channels]
            )
            residuals.append(((values - observed) / scales).ravel())
            if jacobian:
                local = (local / scales[None, :, None]).reshape(values.size, -1)
                joint = np.zeros((values.size, len(self.names)))
                joint[:, self._affine_indices[row.trajectory_id]] = local
                derivatives.append(joint)
        _check_deadline(deadline)
        return _finite(np.concatenate(residuals), "residuals"), (
            _finite(np.vstack(derivatives), "residual derivatives")
            if jacobian
            else None
        )


class TrainingOracle(_AffineEvaluation, kernel.TrainingOracle):
    """Affine alternative with unchanged shared/map and per-trajectory roots."""

    def __init__(
        self,
        problem: PrefixFitProblem,
        training: PublicSplit,
        policy: kernel.NumericalPolicy,
    ) -> None:
        super().__init__(problem, training, policy)
        self.training, self.numerical_policy = training, policy
        self._initialize_affine(tuple(range(len(self.system.names))), self.indices)

    def with_numerical_policy(self, policy: kernel.NumericalPolicy) -> TrainingOracle:
        """Recompile an independent replay oracle with explicitly changed tolerances."""
        if not isinstance(policy, kernel.NumericalPolicy):
            raise TypeError("typed numerical policy required")
        return type(self)(self.problem, self.training, policy)


class PrefixOracle(_AffineEvaluation, kernel.PrefixOracle):
    """Affine prefix alternative differentiating only unfrozen trajectory roots."""

    def __init__(
        self,
        problem: PrefixFitProblem,
        fit_result: dict,
        prefix: kernel.PrefixView,
        policy: kernel.NumericalPolicy,
        *,
        root_guesses: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(problem, fit_result, prefix, policy, root_guesses=root_guesses)
        self._initialize_affine(
            tuple(self.system.names.index(name) for name in self.names),
            {row.trajectory_id: np.arange(len(self.names)) for row in self.rows},
        )
