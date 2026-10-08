"""Certified constant-coefficient affine rollouts derived from supplied equations.

This diagnostic contains no reference coefficients or benchmark-specific matrices.
It rejects nonlinear/time-varying dynamics instead of approximating them as linear.
"""

from time import monotonic

import casadi as ca
import numpy as np
from scipy.linalg import expm, expm_frechet

from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting.simulation import trajectory_forcing


class AffineRollouts(checks.Rollouts):
    """Exact affine propagation up to floating-point matrix-exponential error."""

    def __init__(self, problem, policy):
        super().__init__(problem, policy)
        s = self.system
        n, m, p = s.state_count, len(s.inputs), len(self.names)
        t, x = ca.SX.sym("t"), ca.SX.sym("x", n)
        u, theta = ca.SX.sym("u", m), ca.SX.sym("theta", p)
        f = s.rhs(t, x, theta, u)
        a, b = ca.jacobian(f, x), ca.jacobian(f, u)
        zero = ca.substitute(f, ca.vertcat(x, u), ca.SX.zeros(n + m))
        if ca.depends_on(ca.vertcat(ca.vec(a), ca.vec(b), zero), ca.vertcat(t, x, u)):
            raise ValueError("requires autonomous jointly affine state/input dynamics")
        # Augmented state [x, input, input slope, 1] covers each public interval.
        g = ca.SX.zeros(n + 2 * m + 1, n + 2 * m + 1)
        g[:n, :n], g[:n, n : n + m], g[:n, -1] = a, b, zero
        g[n : n + m, n + m : n + 2 * m] = ca.SX.eye(m)
        self.generator = ca.Function("affine_generator", [theta], [g])
        self.derivatives = ca.Function(
            "affine_generator_derivatives",
            [theta],
            [ca.jacobian(ca.vec(g), theta)],
        )
        self.audit = {
            "certificate": "autonomous-affine-state-input-1",
            "state_count": n,
            "parameter_count": p,
            "forcing": "public piecewise-linear interpolant; every interval retained",
            "parameter_derivatives": "matrix-exponential Frechet derivatives",
            "reference_coefficients_used": False,
        }

    def evaluate(self, x, deadline, *, jacobian=True, method=None):
        """A requested ODE method uses the independent, original equation evaluator."""
        if method:
            return super().evaluate(x, deadline, jacobian=jacobian, method=method)
        self.vector(self.parameters(x))

        def check_time():
            if monotonic() >= deadline:
                raise TimeoutError("affine rollout wall-clock limit reached")

        check_time()
        s, p = self.system, len(x)
        n = s.state_count
        g = np.asarray(self.generator(x))
        dg = (
            np.asarray(self.derivatives(x)).reshape((*g.shape, p), order="F")
            if jacobian
            else None
        )
        transitions, residuals, jacobians = {}, [], []
        scales = np.array([self.scales[c] for c in s.channels])
        for row in self.train.trajectories:
            forcing = trajectory_forcing(s.model, row)
            inputs = np.array(
                [[forcing.value(k, t) for k in s.inputs] for t in row.time]
            )
            state = s.initial_for(row, x)
            derivative = s.initial_sensitivity(row, x) if jacobian else None
            outputs, gradients = [], []
            for i, t in enumerate(row.time):
                check_time()
                if i:
                    dt = float(t - row.time[i - 1])
                    # Exact float keys: never round a forcing interval or miss a pulse.
                    if dt not in transitions:
                        matrix = expm(g * dt)
                        derivatives = []
                        if jacobian:
                            for j in range(p):
                                check_time()
                                derivatives.append(
                                    expm_frechet(
                                        g * dt, dg[:, :, j] * dt, compute_expm=False
                                    )
                                )
                        transitions[dt] = matrix, derivatives
                    matrix, derivatives = transitions[dt]
                    extended = np.r_[
                        state, inputs[i - 1], (inputs[i] - inputs[i - 1]) / dt, 1
                    ]
                    if jacobian:
                        derivative = matrix[:n, :n] @ derivative + np.column_stack(
                            [(d @ extended)[:n] for d in derivatives]
                        )
                    state = (matrix @ extended)[:n]
                outputs.append(np.asarray(s.observe(t, state, x, inputs[i])).ravel())
                if jacobian:
                    _, _, hx, ht = s.local(t, state, x, inputs[i])
                    gradients.append(np.asarray(hx) @ derivative + np.asarray(ht))
            observed = np.column_stack([row.targets[c] for c in s.channels])
            residuals.append(((np.array(outputs) - observed) / scales).ravel())
            if jacobian:
                jacobians.append(
                    (np.array(gradients) / scales[None, :, None]).reshape(-1, p)
                )
        r = np.concatenate(residuals)
        j = np.vstack(jacobians) if jacobian else None
        if not np.isfinite(r).all() or (j is not None and not np.isfinite(j).all()):
            raise ValueError("nonfinite affine rollout or derivative")
        check_time()
        return r, j


def verify_outputs(oracle, parameters, deadline, checkpoint):
    """Check outputs only; no augmented sensitivity integration at each profile."""
    vector = oracle.vector(parameters)
    checkpoint({"calls": 1})
    r, _ = oracle.evaluate(vector, deadline, jacobian=False)
    checkpoint({"calls": 2})
    other, _ = oracle.evaluate(vector, deadline, jacobian=False, method="DOP853")
    return {
        "parameters": parameters,
        "training_nmse": float(np.mean(r**2)),
        "alternate_training_nmse": float(np.mean(other**2)),
        "solver_loss_discrepancy": abs(float(np.mean(r**2) - np.mean(other**2))),
        "solver_prediction_discrepancy": float(np.max(np.abs(r - other))),
        "verification": "matrix_exponential_vs_DOP853_outputs_only",
    }


def numerically_verified(point, tolerance):
    """Require agreement in predictions as well as scalar loss."""
    return bool(
        point
        and point["solver_loss_discrepancy"] <= tolerance / 10
        and point["solver_prediction_discrepancy"] <= np.sqrt(tolerance) / 10
    )
