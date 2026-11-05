"""Exact joint BVLS for shared meal/drift forcing and trajectory initial roots.

Conditional on the outer dynamics, all supplied training trajectories share one
meal forcing gain and one auxiliary drift. Their initial roots remain separate.
This module certifies affine dependence before constructing one global design;
it never treats the two shared coefficients as trajectory-specific variables.
"""

from __future__ import annotations

from collections.abc import Mapping

import casadi as ca
import numpy as np

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_meal_coordinates as meal
from autoformalism.fitting import prefix_multioutput_profile as projection
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.fitting.simulation import trajectory_forcing

PROTOCOL = "shared-meal-drift-root-profile-1"
DEFAULT_DRIFT = "k_Uii_term_12_0_244d72574a"


class JointMealProfile(roots.RootProfile):
    """Profile g=k*q, drift and all roots in one domain-preserving linear solve."""

    def __init__(
        self,
        oracle,
        *,
        failure_diagnostics: bool = False,
        drift: str = DEFAULT_DRIFT,
        meal_options: Mapping[str, str] | None = None,
    ):
        super().__init__(oracle, failure_diagnostics=failure_diagnostics)
        self.meal = meal.certify(oracle, oracle.start, **dict(meal_options or {}))
        if drift not in oracle.names or drift in {
            self.meal.rate,
            self.meal.input_gain,
            *self.root_names,
        }:
            raise ValueError("joint drift must be a distinct shared coordinate")
        self.drift = drift
        self.shared_names = (self.meal.input_gain, drift)
        roots._root_chart(oracle, self.shared_names)
        self.inner = tuple(
            oracle.names.index(n) for n in (*self.shared_names, *self.root_names)
        )
        self.outer = tuple(i for i in range(len(oracle.names)) if i not in self.inner)
        self.inner_names = (self.meal.flux_gain, drift, *self.root_names)
        for name in self.shared_names:
            old = self.original.names.index(name)
            new = oracle.names.index(name)
            if (
                self.original.lower[old] != oracle.lower[new]
                or self.original.upper[old] != oracle.upper[new]
            ):
                raise ValueError("joint shared coefficient domains differ")
        system = self.original.system
        n, m, p = system.state_count, len(system.inputs), len(system.names)
        theta = ca.SX.sym("joint_parameters", p)
        x, u, t = ca.SX.sym("joint_x", n), ca.SX.sym("joint_u", m), ca.SX.sym("joint_t")
        shared = theta[[system.names.index(name) for name in self.shared_names]]
        rhs = system.rhs(t, x, theta, u)
        matrix = ca.jacobian(rhs, x)
        forcing = ca.substitute(rhs, x, ca.SX.zeros(n))
        columns = ca.jacobian(forcing, shared)
        if ca.depends_on(matrix, shared):
            raise ValueError("joint coefficients must not alter the dynamics matrix")
        if ca.depends_on(columns, shared):
            raise ValueError(
                "joint forcing must be jointly affine in shared coefficients"
            )
        readout = system.observe(t, x, theta, u)
        if ca.depends_on(readout, shared):
            raise ValueError("joint coefficients must not occur in readouts")
        for row in self.original.rows:
            if ca.depends_on(system.initial_symbolic(row, theta), shared):
                raise ValueError("joint coefficients must not occur in initializers")
        baseline = ca.substitute(forcing, shared, ca.SX.zeros(2))
        self.forcing_basis = ca.Function(
            "joint_forcing_basis", [t, theta, u], [ca.horzcat(baseline, columns)]
        )
        # Universal affine forcing propagator. Each column receives its own
        # forcing and forcing slope, avoiding differencing two large rollouts.
        generator = ca.SX.zeros(3 * n, 3 * n)
        generator[:n, :n] = matrix
        generator[:n, n : 2 * n] = ca.SX.eye(n)
        generator[n : 2 * n, 2 * n :] = ca.SX.eye(n)
        self.joint_generator = ca.Function("joint_generator", [theta], [generator])
        record = {
            **self.audit,
            "protocol": PROTOCOL,
            "root_profile_certificate_sha256": self.audit["sha256"],
            "positive_meal_certificate": self.meal.audit,
            "shared_original_names": list(self.shared_names),
            "inner_names": list(self.inner_names),
            "outer_names": [oracle.names[i] for i in self.outer],
            "shared_count": 2,
            "trajectory_root_count": len(self.root_names),
            "scope": "one shared gain and drift across all training trajectories",
            "forcing_jointly_affine": True,
            "dynamics_matrix_independent_of_profiled_shared_coefficients": True,
            "readout_and_initializers_independent_of_shared_coefficients": True,
            "shared_domain_and_finite_original_lift_preserved": True,
            "basis_construction": "symbolic forcing columns; no rollout differencing",
            "parameter_confidence_certified": False,
        }
        record.pop("sha256")
        self.audit = record | {"sha256": content_sha256(record)}

    def design(self, parameters: Mapping[str, float], deadline: float) -> dict:
        """Assemble a weighted global residual affine in two shared gains and roots."""
        roots._check(deadline)
        encoded = self.meal.encode(parameters)
        k = float(parameters[self.meal.rate])
        vector = self.oracle.reduction.lift(self.oracle.vector(parameters))
        self.original.vector(self.original.parameters(vector))
        original, system = self.original, self.original.system
        n, nr, m = system.state_count, len(original.roots), len(system.inputs)
        columns = 3 + nr
        scales = np.asarray([original.scales[c] for c in system.channels])
        design_rows, offsets, cache = [], [], {}
        for row_index, (row, initializer) in enumerate(
            zip(original.rows, self.initializers, strict=True)
        ):
            roots._check(deadline)
            local = original.local_vector(vector, row)
            generator = roots._finite(self.joint_generator(local), "joint generator")
            initial = roots._finite(initializer(local), "joint initial basis")
            state = np.zeros((n, columns))
            state[:, 0], state[:, 3:] = initial[:, 0], initial[:, 1:]
            forcing = trajectory_forcing(system.model, row)
            inputs = np.asarray(
                [[forcing.value(name, t) for name in system.inputs] for t in row.time]
            ).reshape(len(row.time), m)
            force = []
            for time, values in zip(row.time, inputs, strict=True):
                roots._check(deadline)
                basis = roots._finite(
                    self.forcing_basis(time, local, values), "forcing"
                )
                # Original q=g/k, so the derivative with respect to g is F_q/k.
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    basis[:, 1] /= k
                force.append(roots._finite(basis, "joint gain forcing"))
            outputs = []
            for i, time in enumerate(row.time):
                roots._check(deadline)
                if i:
                    dt = float(time - row.time[i - 1])
                    key = generator.tobytes(), dt
                    if key not in cache:
                        cache[key] = self.transitions(generator * dt, [], deadline)[0]
                    extended = np.zeros((3 * n, columns))
                    extended[:n] = state
                    extended[n : 2 * n, :3] = force[i - 1]
                    extended[2 * n :, :3] = (force[i] - force[i - 1]) / dt
                    state = roots._finite((cache[key] @ extended)[:n], "joint basis")
                h, offset = self.readout(time, local, inputs[i])
                values = roots._finite(np.asarray(h) @ state, "joint readout basis")
                values[:, 0] += np.asarray(offset).ravel()
                outputs.append(values)
            values = roots._finite(outputs, "joint weighted outputs")
            observed = np.column_stack([row.targets[c] for c in system.channels])
            local_design = (values[:, :, 1:] / scales[None, :, None]).reshape(
                -1, 2 + nr
            )
            global_design = np.zeros((len(local_design), len(self.inner)))
            global_design[:, :2] = local_design[:, :2]
            j = 2 + row_index * nr
            global_design[:, j : j + nr] = local_design[:, 2:]
            design_rows.append(global_design)
            offsets.append(((values[:, :, 0] - observed) / scales).ravel())
        roots._check(deadline)
        lower, upper = (
            self.oracle.lower[list(self.inner)].copy(),
            self.oracle.upper[list(self.inner)].copy(),
        )
        # The certified nonnegative unbounded q domain is identical for g at k>0.
        if lower[0] != 0 or upper[0] != np.inf:
            raise ValueError("certified meal gain domain changed")
        return {
            "design": roots._finite(np.concatenate(design_rows), "global design"),
            "offset": roots._finite(np.concatenate(offsets), "global offset"),
            "lower": lower,
            "upper": upper,
            "source_inner": encoded[list(self.inner)],
        }

    def _profile(self, parameters: Mapping[str, float], deadline: float) -> dict:
        # Ephemeral paired residual for bounded local diagnostics. Never include
        # this potentially large array in endpoint receipts or checkpoints.
        self.last_residual = None
        block = self._call("basis", lambda: self.design(parameters, deadline))
        roots._check(deadline)
        a, b, lower, upper = (
            block[name] for name in ("design", "offset", "lower", "upper")
        )
        fixed = lower == upper
        fitted = lower.copy()
        try:
            fitted[~fixed], predicted, _, audit = projection._project(
                a[:, ~fixed],
                b + a[:, fixed] @ lower[fixed],
                None,
                None,
                lower[~fixed],
                upper[~fixed],
                deadline,
            )
        except Exception as error:
            if self.failure_diagnostics:
                error.root_profile_diagnostic = {"stage": "joint_inner_solve"}
            raise
        chart_point = self.meal.encode(parameters)
        chart_point[list(self.inner)] = fitted
        vector = self.meal.decode(chart_point)
        fitted_parameters = self.oracle.parameters(vector)
        original_parameters = continuation.to_original(
            self.original, self.oracle, fitted_parameters
        )
        actual, _ = self._call(
            "pairing", lambda: self.oracle.evaluate(vector, deadline, jacobian=False)
        )
        roots._check(deadline)
        discrepancy = float(np.max(np.abs(predicted - actual), initial=0))
        predicted_loss, actual_loss = (
            float(np.mean(predicted**2)),
            float(np.mean(actual**2)),
        )
        if (
            not np.isfinite([discrepancy, predicted_loss, actual_loss]).all()
            or discrepancy > portfolio.PAIRING_TOLERANCE
            or abs(predicted_loss - actual_loss) > max(1e-12, 1e-8 * actual_loss)
        ):
            error = NumericalInvalidTrial(
                "joint profile basis/full-rollout pairing differs"
            )
            if self.failure_diagnostics:
                error.root_profile_diagnostic = {
                    "stage": "pairing",
                    "maximum_pairing_discrepancy": discrepancy,
                    "projected_training_nmse": predicted_loss,
                    "training_nmse": actual_loss,
                }
            raise error
        self.last_residual = np.asarray(actual, dtype=float).copy()
        return {
            "parameters": fitted_parameters,
            "original_parameters": original_parameters,
            "training_nmse": actual_loss,
            "projected_training_nmse": predicted_loss,
            "maximum_pairing_discrepancy": discrepancy,
            "joint_solve": audit
            | {
                "scope": "all supplied trajectories; shared coefficients appear once",
                "coordinate_names": list(self.inner_names),
                "fixed_coordinates": [
                    self.inner_names[i] for i in np.flatnonzero(fixed)
                ],
                "free_coordinate_names": [
                    self.inner_names[i] for i in np.flatnonzero(~fixed)
                ],
            },
            "root_solve": [
                audit | {"trajectory_id": "joint_all_training_trajectories"}
            ],
            "certificate_sha256": self.audit["sha256"],
            "conditional_root_optimality_checked": True,
            "joint_coefficient_optimality_checked": True,
            "outer_convergence_certified": False,
            "independent_original_verification_required": True,
        }
