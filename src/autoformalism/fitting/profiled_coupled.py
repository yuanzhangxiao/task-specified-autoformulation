"""Exact variable projection through a certified, coupled linear state system."""

from itertools import pairwise
from time import monotonic

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.fitting.profiled_output import _zero
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.fitting.simulation import forcing_segment_indices, trajectory_forcing


class ProfiledCoupled:
    """Certify x'=A(t,u,beta)x+B(t,u,beta)g+d(t,u,beta).

    Affine shared initializers may also depend on g. The entire dynamical system
    must be linear in states; this first experiment supports one identity output.
    Feedback is allowed, but any parameter in A stays in the nonlinear outer
    problem. The existing gain attribute names implement the M14 worker interface.
    """

    def __init__(self, system: SymbolicODE, gains: tuple[str, ...] | None = None):
        self.system = system
        n, p = system.state_count, len(system.names)
        t, x = ca.SX.sym("t"), ca.SX.sym("x", n)
        theta, u = ca.SX.sym("theta", p), ca.SX.sym("u", len(system.inputs))
        f, h = system.rhs(t, x, theta, u), system.observe(t, x, theta, u)
        observed = [i for i in range(n) if h.numel() == 1 and _zero(h - x[i])]
        if len(system.channels) != 1 or len(observed) != 1:
            raise ValueError("coupled profiling requires one identity-observed state")
        self.output = observed[0]
        a = ca.jacobian(f, x)
        if ca.depends_on(a, x):
            raise ValueError("coupled dynamics are not jointly affine in states")
        forcing = ca.substitute(f, x, ca.SX.zeros(n))
        initial = ca.SX.zeros(n)
        if system.initial_function is not None:
            initial = system.initial_function(
                theta, ca.SX.sym("known", len(system.initial_symbols))
            )
        source = ca.vertcat(forcing, initial)
        eligible = []
        for j, name in enumerate(system.names):
            column = ca.jacobian(source, theta[j])
            if (
                not _zero(column)
                and not ca.depends_on(column, theta[j])
                and not ca.depends_on(a, theta[j])
            ):
                eligible.append(name)
        chosen = tuple(eligible) if gains is None else tuple(gains)
        if (
            not chosen
            or len(set(chosen)) != len(chosen)
            or not set(chosen) <= set(eligible)
        ):
            raise ValueError(
                "requested inner block is not conditionally affine/separable"
            )
        self.gain_indices = [system.names.index(name) for name in chosen]
        self.outer_indices = [j for j in range(p) if j not in self.gain_indices]
        self.gain_names = chosen
        self.outer_names = tuple(system.names[j] for j in self.outer_indices)
        g = theta[self.gain_indices]
        if ca.depends_on(ca.jacobian(source, g), g):
            raise ValueError("inner block is not jointly affine")
        if not self.outer_indices:
            raise ValueError("experiment requires at least one outer unknown")
        k, nb = len(chosen), len(self.outer_indices)
        b = ca.jacobian(forcing, g)
        d = ca.substitute(forcing, g, ca.SX.zeros(k))
        # Each feature is a full state response; off-diagonal A propagates coupling.
        w = ca.SX.sym("filtered", n * (k + 1))
        v = ca.reshape(w[n:], n, k)
        field = ca.vertcat(a @ w[:n] + d, ca.vec(a @ v + b))
        beta = ca.SX.sym("beta", nb)
        values = ca.SX.zeros(p)
        values[self.outer_indices] = beta
        field = ca.substitute(field, theta, values)
        s = ca.SX.sym("sensitivity", w.numel(), nb)
        ds = ca.jacobian(field, w) @ s + ca.jacobian(field, beta)
        self.rhs = ca.Function("coupled_rhs", [t, w, beta, u], [field])
        self.augmented_rhs = ca.Function(
            "coupled_sensitivities",
            [t, ca.vertcat(w, ca.vec(s)), beta, u],
            [ca.vertcat(field, ca.vec(ds))],
        )
        self.theta = theta
        self.audit = {
            "certificate": "symbolic-coupled-affine-state-1",
            "state_names": list(system.model.state_names),
            "output_state": system.model.state_names[self.output],
            "inner_names": list(chosen),
            "gain_names": list(chosen),
            "outer_names": list(self.outer_names),
            "profiled_initial_parameters": [
                name
                for name in chosen
                if not _zero(ca.jacobian(initial, theta[system.names.index(name)]))
            ],
            "estimated_unknowns": p,
            "outer_unknowns": nb,
            "filtered_states": w.numel(),
            "augmented_states": w.numel() * (1 + nb),
            "state_matrix_independent_of_inner_block": True,
            "initialization_jointly_affine_in_inner_block": True,
            "coupling_preserved": True,
            "signed_features_preserved": True,
            "exactness": "algebraic separability; numerical tolerance error remains",
        }

    def trajectory(self, trajectory, vector, settings, deadline, *, sensitivities=True):
        """Integrate affine basis responses and their exact outer sensitivities."""
        system = self.system
        n, k, p = system.state_count, len(self.gain_indices), len(self.outer_indices)
        theta = self.theta
        g = theta[self.gain_indices]
        initial = system.initial_symbolic(trajectory, theta)
        basis = ca.jacobian(initial, g)
        w0 = ca.vertcat(ca.substitute(initial, g, ca.SX.zeros(k)), ca.vec(basis))
        # Includes mixed beta/inner dependence in analytic initializers.
        initial_function = ca.Function(
            "coupled_initial",
            [theta],
            [w0, ca.jacobian(w0, theta)[:, self.outer_indices]],
        )
        physical, derivative = initial_function(vector)
        w = np.asarray(physical).ravel()
        nw = len(w)
        if sensitivities:
            w = np.r_[w, np.asarray(derivative).ravel(order="F")]
        times = trajectory.time
        output = np.empty((len(times), len(w)))
        output[0] = w
        forcing = trajectory_forcing(system.model, trajectory)
        function = self.augmented_rhs if sensitivities else self.rhs
        beta = vector[self.outer_indices]
        counts = {"nfev": 0, "segments": 0}

        def rhs(t, state):
            if monotonic() >= deadline:
                raise TimeoutError("coupled profiled point deadline")
            inputs = [forcing.value(name, t) for name in system.inputs]
            result = np.asarray(function(t, state, beta, inputs)).ravel()
            if not np.isfinite(result).all():
                raise ValueError("nonfinite coupled filtered dynamics")
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
                raise ValueError("incomplete coupled filtered rollout")
            output[left : right + 1] = solved.y.T
            w = solved.y[:, -1]
            counts["nfev"] += solved.nfev
            counts["segments"] += 1
        if not np.isfinite(output).all():
            raise ValueError("nonfinite coupled filtered rollout")
        derivative = (
            output[:, nw:].reshape(len(times), p, nw).transpose(0, 2, 1)
            if sensitivities
            else None
        )
        columns = [n * (j + 1) + self.output for j in range(k)]
        return (
            output[:, columns],
            output[:, self.output],
            derivative[:, columns] if sensitivities else None,
            derivative[:, self.output] if sensitivities else None,
            counts,
        )
