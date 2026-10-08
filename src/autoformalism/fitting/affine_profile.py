"""Exact affine-state basis propagation for bounded gain/initial profiling."""

from time import monotonic

import casadi as ca
import numpy as np
from scipy.linalg import expm, expm_frechet

from autoformalism.fitting.profiled_coupled import ProfiledCoupled
from autoformalism.fitting.profiled_output import project
from autoformalism.fitting.simulation import trajectory_forcing


class AffineProfile:
    """Reuse symbolic separability; propagate all basis columns with one matrix.

    Requires the autonomous affine certificate of the supplied AffineRollouts.
    A 3n generator advances state, affine source, and source slope, simultaneously
    for the offset and every inner feature. Derivatives include mixed initializer
    and forcing dependence on the nonlinear outer coefficients.
    """

    def __init__(self, oracle):
        if oracle.audit.get("certificate") != "autonomous-affine-state-input-1":
            raise ValueError("requires certified affine state/input dynamics")
        self.oracle = oracle
        system = oracle.system
        self.structure = ProfiledCoupled(system)
        self.inner = self.structure.gain_indices
        self.outer = self.structure.outer_indices
        n, p, m = system.state_count, len(system.names), len(system.inputs)
        self.n, self.k, self.nb = n, len(self.inner) + 1, len(self.outer)
        theta = ca.SX.sym("theta", p)
        u, x = ca.SX.sym("u", m), ca.SX.sym("x", n)
        f = system.rhs(0, x, theta, u)
        a = ca.jacobian(f, x)
        forcing = ca.substitute(f, x, ca.SX.zeros(n))
        gamma = theta[self.inner]
        sources = ca.horzcat(
            ca.substitute(forcing, gamma, ca.SX.zeros(len(self.inner))),
            ca.jacobian(forcing, gamma),
        )
        generator = ca.SX.zeros(3 * n, 3 * n)
        generator[:n, :n] = a
        generator[:n, n : 2 * n] = ca.SX.eye(n)
        generator[n : 2 * n, 2 * n :] = ca.SX.eye(n)
        self.transition = ca.Function(
            "basis_transition",
            [theta],
            [generator, ca.jacobian(ca.vec(generator), theta)[:, self.outer]],
        )
        self.sources = ca.Function(
            "basis_sources",
            [theta, u],
            [sources, ca.jacobian(ca.vec(sources), theta)[:, self.outer]],
        )
        self.initializers = []
        for row in oracle.train.trajectories:
            initial = system.initial_symbolic(row, theta)
            basis = ca.horzcat(
                ca.substitute(initial, gamma, ca.SX.zeros(len(self.inner))),
                ca.jacobian(initial, gamma),
            )
            self.initializers.append(
                ca.Function(
                    "basis_initial",
                    [theta],
                    [basis, ca.jacobian(ca.vec(basis), theta)[:, self.outer]],
                )
            )
        self.audit = self.structure.audit | {
            "propagation": "affine-source-matrix-exponential-1",
            "transition_dimension": 3 * n,
        }

    def evaluate(self, vector, deadline):
        """Return the bounded inner optimum, residual, and outer Jacobian."""
        oracle = self.oracle
        oracle.vector(oracle.parameters(vector))
        n, k, nb = self.n, self.k, self.nb

        def time_check():
            if monotonic() >= deadline:
                raise TimeoutError("profiled affine rollout allowance exhausted")

        def unpack(function, *args):
            value, derivative = function(*args)
            return np.asarray(value), np.asarray(derivative).reshape(
                n, k, nb, order="F"
            )

        time_check()
        g, dg = self.transition(vector)
        g = np.asarray(g)
        dg = np.asarray(dg).reshape(3 * n, 3 * n, nb, order="F")
        transitions = {}
        designs, offsets, derivatives, offset_derivatives = [], [], [], []
        output = self.structure.output
        channel = oracle.system.channels[0]
        scale = oracle.scales[channel]
        for row, initializer in zip(
            oracle.train.trajectories, self.initializers, strict=True
        ):
            state, ds = unpack(initializer, vector)
            forcing = trajectory_forcing(oracle.system.model, row)
            inputs = np.array(
                [
                    [forcing.value(name, t) for name in oracle.system.inputs]
                    for t in row.time
                ]
            )
            zero, dzero = unpack(
                self.sources, vector, np.zeros(len(oracle.system.inputs))
            )
            ys, js = [], []
            for i, t in enumerate(row.time):
                time_check()
                if i:
                    dt = float(t - row.time[i - 1])
                    if dt not in transitions:
                        matrix = expm(g * dt)
                        dm = []
                        for j in range(nb):
                            time_check()
                            dm.append(
                                expm_frechet(
                                    g * dt, dg[:, :, j] * dt, compute_expm=False
                                )
                            )
                        transitions[dt] = matrix, dm
                    matrix, dm = transitions[dt]
                    source, dsource = unpack(self.sources, vector, inputs[i - 1])
                    slope, dslope = unpack(
                        self.sources, vector, (inputs[i] - inputs[i - 1]) / dt
                    )
                    extended = np.vstack([state, source, slope - zero])
                    dextended = np.concatenate([ds, dsource, dslope - dzero], axis=0)
                    ds = np.einsum("ab,bkj->akj", matrix[:n], dextended) + np.stack(
                        [(d @ extended)[:n] for d in dm], axis=-1
                    )
                    state = (matrix @ extended)[:n]
                ys.append(state[output].copy())
                js.append(ds[output].copy())
            ys, js = np.asarray(ys) / scale, np.asarray(js) / scale
            designs.append(ys[:, 1:])
            offsets.append(ys[:, 0] - np.asarray(row.targets[channel]) / scale)
            derivatives.append(js[:, 1:])
            offset_derivatives.append(js[:, 0])
        time_check()
        result = project(
            np.vstack(designs),
            np.concatenate(offsets),
            np.concatenate(derivatives),
            np.vstack(offset_derivatives),
            oracle.lower[self.inner],
            oracle.upper[self.inner],
        )
        time_check()
        fitted = vector.copy()
        fitted[self.inner] = result["gains"]
        oracle.vector(oracle.parameters(fitted))
        return fitted, result["residual"], result["jacobian"], result["audit"]
