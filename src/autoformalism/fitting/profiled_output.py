"""Certified terminal linear output dynamics and bounded variable projection.

An opt-in fitting experiment, not a change to production fitting. Certification
uses the restricted symbolic model, never parameter names or sampled probes.
"""

from __future__ import annotations

from itertools import pairwise
from time import monotonic

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import lsq_linear

from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.fitting.simulation import forcing_segment_indices, trajectory_forcing


def _zero(expression) -> bool:
    return bool(ca.simplify(expression).is_zero())


class ProfiledOutput:
    """Certify y'=a(z,u,beta)y+c(z,u,beta)+Phi(z,u,beta)g.

    The other states and all initializers must be independent of g, and the
    other equations must be independent of y. One directly observed state is
    supported. Bounds on g remain exactly those of the original fitter.
    """

    def __init__(self, system: SymbolicODE, gains: tuple[str, ...] | None = None):
        self.system = system
        n, p = system.state_count, len(system.names)
        t, x = ca.SX.sym("t"), ca.SX.sym("x", n)
        theta, u = ca.SX.sym("theta", p), ca.SX.sym("u", len(system.inputs))
        f, h = system.rhs(t, x, theta, u), system.observe(t, x, theta, u)
        observed = [i for i in range(n) if h.numel() == 1 and _zero(h - x[i])]
        if len(system.channels) != 1 or len(observed) != 1:
            raise ValueError("profiling requires one identity-observed state")
        self.output = observed[0]
        self.latent = [i for i in range(n) if i != self.output]
        y, latent_rhs = x[self.output], f[self.latent]
        if ca.depends_on(latent_rhs, y):
            raise ValueError("output feeds back into other state equations")
        a = ca.jacobian(f[self.output], y)
        if ca.depends_on(a, y):
            raise ValueError("output dynamics are not linear in the output state")
        remainder = ca.substitute(f[self.output], y, ca.SX(0))
        initial = ca.SX.zeros(n)
        if system.initial_function is not None:
            known = ca.SX.sym("known", system.initial_function.size1_in(1))
            initial = system.initial_function(theta, known)
        eligible = []
        for j, name in enumerate(system.names):
            column = ca.jacobian(remainder, theta[j])
            if (
                not _zero(column)
                and not ca.depends_on(column, theta[j])
                and not ca.depends_on(ca.vertcat(latent_rhs, a, initial), theta[j])
            ):
                eligible.append(name)
        chosen = tuple(eligible) if gains is None else tuple(gains)
        if (
            not chosen
            or len(set(chosen)) != len(chosen)
            or not set(chosen) <= set(eligible)
        ):
            raise ValueError(
                "requested gain block is not conditionally linear/separable"
            )
        self.gain_indices = [system.names.index(name) for name in chosen]
        self.outer_indices = [i for i in range(p) if i not in self.gain_indices]
        self.gain_names = chosen
        self.outer_names = tuple(system.names[i] for i in self.outer_indices)
        g = theta[self.gain_indices]
        phi = ca.jacobian(remainder, g)
        if ca.depends_on(phi, g):
            raise ValueError("gain block is not jointly affine")
        c = ca.substitute(remainder, g, ca.SX.zeros(len(chosen)))
        if not self.outer_indices:
            raise ValueError("experiment requires at least one outer unknown")
        # Integrate latent states, the output at g=0, and one filtered feature/gain.
        m = len(self.latent)
        w = ca.SX.sym("filtered", m + 1 + len(chosen))
        substitution = ca.vertcat(
            *[w[m] if i == self.output else w[self.latent.index(i)] for i in range(n)]
        )
        field = ca.vertcat(latent_rhs, a * w[m] + c, a * w[m + 1 :] + phi.T)
        field = ca.substitute(field, x, substitution)
        beta = ca.SX.sym("beta", len(self.outer_indices))
        values = ca.SX.zeros(p)
        values[self.outer_indices] = beta
        field = ca.substitute(field, theta, values)
        s = ca.SX.sym("sensitivity", w.numel(), beta.numel())
        ds = ca.jacobian(field, w) @ s + ca.jacobian(field, beta)
        self.rhs = ca.Function("filtered_rhs", [t, w, beta, u], [field])
        self.augmented_rhs = ca.Function(
            "filtered_sensitivities",
            [t, ca.vertcat(w, ca.vec(s)), beta, u],
            [ca.vertcat(field, ca.vec(ds))],
        )
        self.audit = {
            "certificate": "symbolic-terminal-affine-output-1",
            "output_state": system.model.state_names[self.output],
            "gain_names": list(chosen),
            "outer_names": list(self.outer_names),
            "estimated_unknowns": p,
            "outer_unknowns": len(self.outer_indices),
            "gain_independent_initialization": True,
            "output_feedback_absent": True,
            "signed_features_preserved": True,
        }

    def trajectory(self, trajectory, vector, settings, deadline, *, sensitivities=True):
        """Integrate the exact filtered representation at all public input corners."""
        system = self.system
        m, k, p = len(self.latent), len(self.gain_indices), len(self.outer_indices)
        initial = system.initial_for(trajectory, vector)
        w = np.r_[initial[self.latent], initial[self.output], np.zeros(k)]
        nw = len(w)
        if sensitivities:
            ds = np.zeros((nw, p))
            initial_jac = system.initial_sensitivity(trajectory, vector)
            ds[: m + 1] = initial_jac[[*self.latent, self.output]][
                :, self.outer_indices
            ]
            w = np.r_[w, ds.ravel(order="F")]
        times = trajectory.time
        output = np.empty((len(times), len(w)))
        output[0] = w
        forcing = trajectory_forcing(system.model, trajectory)
        function = self.augmented_rhs if sensitivities else self.rhs
        beta = vector[self.outer_indices]
        counts = {"nfev": 0, "segments": 0}

        def rhs(t, state):
            if monotonic() >= deadline:
                raise TimeoutError("profiled rollout point deadline")
            inputs = [forcing.value(name, t) for name in system.inputs]
            result = np.asarray(function(t, state, beta, inputs)).ravel()
            if not np.isfinite(result).all():
                raise ValueError("nonfinite filtered dynamics")
            return result

        for left, right in pairwise(forcing_segment_indices(system.model, trajectory)):
            solved = solve_ivp(
                rhs,
                (times[left], times[right]),
                w,
                t_eval=times[left : right + 1],
                method=settings.integration_method,
                rtol=settings.relative_tolerance,
                atol=settings.absolute_tolerance,
            )
            if not solved.success or solved.y.shape[1] != right - left + 1:
                raise ValueError("incomplete filtered rollout")
            output[left : right + 1] = solved.y.T
            w = solved.y[:, -1]
            counts["nfev"] += solved.nfev
            counts["segments"] += 1
        if not np.isfinite(output).all():
            raise ValueError("nonfinite filtered rollout")
        derivative = (
            output[:, nw:].reshape(len(times), p, nw).transpose(0, 2, 1)
            if sensitivities
            else None
        )
        return (
            output[:, m + 1 : nw],
            output[:, m],
            derivative[:, m + 1 :] if sensitivities else None,
            derivative[:, m] if sensitivities else None,
            counts,
        )


def project(design, offset, d_design, d_offset, lower, upper) -> dict:
    """Bounded linear LS and exact residual Jacobian on a fixed active set.

    offset is the already normalized g=0 prediction minus observations. An SVD
    solves the differentiated free-column normal equations without forming an
    inverse/normal matrix. Rank-deficient designs are explicitly unavailable.
    At active-set boundaries this is a piecewise Jacobian, not a smoothness claim.
    """
    a, b = np.asarray(design, float), np.asarray(offset, float)
    da, db = np.asarray(d_design, float), np.asarray(d_offset, float)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if (
        a.ndim != 2
        or b.shape != (len(a),)
        or da.shape[:2] != a.shape
        or da.ndim != 3
        or db.shape != (len(a), da.shape[2])
        or lo.shape != (a.shape[1],)
        or hi.shape != lo.shape
        or np.any(lo >= hi)
        or any(not np.isfinite(v).all() for v in (a, b, da, db))
        or np.isnan(lo).any()
        or np.isnan(hi).any()
    ):
        raise ValueError("invalid bounded projection arrays")
    units = np.linalg.norm(a, axis=0)
    if np.any(units == 0):
        raise ValueError("rank-deficient output design")
    normalized = a / units
    singular = np.linalg.svd(normalized, compute_uv=False)
    if len(singular) < a.shape[1] or singular[-1] <= 1e-12 * singular[0]:
        raise ValueError("rank-deficient output design")
    solved = lsq_linear(
        normalized,
        -b,
        bounds=(lo * units, hi * units),
        method="bvls",
        tol=1e-12,
        max_iter=100,
    )
    if not solved.success or not np.isfinite(solved.x).all():
        raise ValueError("bounded output solve unavailable")
    gains = np.clip(solved.x / units, lo, hi)
    residual = a @ gains + b
    free = solved.active_mask == 0
    jacobian = np.einsum("nkp,k->np", da, gains) + db
    if np.any(free):
        h = normalized[:, free]
        u, s, vt = np.linalg.svd(h, full_matrices=False)
        term = np.einsum(
            "nkp,n->kp", da[:, free] / units[free][None, :, None], residual
        )
        dh = -vt.T @ ((u.T @ jacobian) / s[:, None] + (vt @ term) / s[:, None] ** 2)
        jacobian += h @ dh
    if not np.isfinite(jacobian).all():
        raise ValueError("nonfinite profiled Jacobian")
    return {
        "gains": gains,
        "residual": residual,
        "jacobian": jacobian,
        "audit": {
            "active_mask": solved.active_mask.tolist(),
            "column_normalized_singular_values": singular.tolist(),
            "optimality": float(solved.optimality),
            "derivative": "fixed-active-set exact residual derivative",
        },
    }
