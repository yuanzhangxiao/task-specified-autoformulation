"""Training-only Radau residuals with a certified conditional coefficient block.

This is alternating minimization, not exact profiling through nonlinear rollout.
Free node trajectories are optimization auxiliaries and are never predictions.
"""

from time import monotonic

import casadi as ca
import numpy as np
from scipy.optimize import lsq_linear

from autoformalism.fitting.collocation_mesh import basis, forcing_values, plan_meshes
from autoformalism.fitting.recovery_mesh import observation_anchors


def coefficient_block(system) -> dict:
    """Deterministically certify a jointly affine RHS block, excluding initials."""
    t, x = ca.SX.sym("t"), ca.SX.sym("x", system.state_count)
    p, u = ca.SX.sym("p", len(system.names)), ca.SX.sym("u", len(system.inputs))
    f, h = system.rhs(t, x, p, u), system.observe(t, x, p, u)
    if ca.depends_on(h, p):
        raise ValueError("conditional transcription requires parameter-free readout")
    selected, excluded = [], {}
    for i, name in enumerate(system.names):
        if name in system.initial_parameter_names:
            excluded[name] = "initializer parameter remains in node/shape optimization"
            continue
        column = ca.jacobian(f, p[i])
        trial = [*selected, i]
        if ca.simplify(column).is_zero():
            excluded[name] = "absent from the state equations"
        elif ca.depends_on(ca.jacobian(f, p[trial]), p[trial]):
            excluded[name] = "not jointly affine with the selected block"
        else:
            selected.append(i)
    if not selected:
        raise ValueError("no certified conditional coefficient block")
    return {
        "indices": selected,
        "names": [system.names[i] for i in selected],
        "outer_indices": [i for i in range(len(system.names)) if i not in selected],
        "excluded": excluded,
        "certificate": "symbolic-joint-RHS-affinity-fixed-states-1",
        "rollout_linearity_claimed": False,
    }


class TrajectoryProblem:
    """Reusable sparse node/shape graph; all initial values remain causal globals."""

    def __init__(self, oracle, problem, target, minimum, anchors, deadline):
        self.oracle, self.system = oracle, oracle.system
        s = self.system
        self.partition = coefficient_block(s)
        self.linear = self.partition["indices"]
        self.outer = self.partition["outer_indices"]
        self.center, self.units = problem.coordinates.arrays(
            "states", s.model.state_names
        )
        self.pcenter = oracle.vector(problem.start)
        self.punits = oracle.units
        meshes, audit = plan_meshes(s, oracle.train, target, minimum_intervals=minimum)
        self.grids = []
        for row, mesh, detail in zip(
            oracle.train.trajectories, meshes, audit["trajectories"], strict=True
        ):
            at = observation_anchors(row.time, row.targets, anchors)
            time = np.array(sorted(set(mesh.time).union(row.time[at])))
            self.grids.append(time)
            detail.update(
                boundaries=time.tolist(),
                intervals=len(time) - 1,
                observation_anchor_indices=at,
                maximum_interval=float(np.max(np.diff(time))),
            )
        self.node_count = 2 * s.state_count * sum(len(g) - 1 for g in self.grids)
        audit["actual_variables"] = self.node_count + len(s.names)
        audit["observation_anchor_centres"] = anchors
        self.audit = audit | {
            "partition": self.partition,
            "shared_initials_fitted": True,
        }
        z, q = ca.MX.sym("nodes", self.node_count), ca.MX.sym("q", len(s.names))
        theta = self.pcenter + self.punits * q
        self.z, self.q = z, q
        n, p = s.state_count, len(s.names)
        left, inner, end = [ca.MX.sym(k, n) for k in ("left", "inner", "end")]
        pp, t, dt = ca.MX.sym("pp", p), ca.MX.sym("t"), ca.MX.sym("dt")
        u0, u1 = [ca.MX.sym(k, len(s.inputs)) for k in ("u0", "u1")]
        f1 = s.rhs(t + dt / 3, inner, pp, u0 + (u1 - u0) / 3)
        f2 = s.rhs(t + dt, end, pp, u1)
        interval = ca.Function(
            "conditional_radau",
            [left, inner, end, pp, t, dt, u0, u1],
            [
                ca.vertcat(
                    (inner - left - dt * (5 * f1 - f2) / 12) / self.units,
                    (end - left - dt * (3 * f1 + f2) / 4) / self.units,
                )
            ],
        )
        observations, defects, cursor = [], [], 0
        for row, time in zip(oracle.train.trajectories, self.grids, strict=True):
            if monotonic() >= deadline:
                raise TimeoutError("conditional graph construction allowance exhausted")
            count, size = len(time) - 1, 2 * n * (len(time) - 1)
            nodes = ca.reshape(z[cursor : cursor + size], 2 * n, count)
            cursor += size
            values = ca.repmat(ca.DM(np.tile(self.center, 2)), 1, count) + ca.times(
                nodes, ca.repmat(ca.DM(np.tile(self.units, 2)), 1, count)
            )
            middle, ends = values[:n, :], values[n:, :]
            initial = s.initial_symbolic(row, theta)
            lefts = ca.horzcat(initial, ends[:, :-1])
            u = forcing_values(s, row, time)
            defects.append(
                ca.vec(
                    interval.map(count)(
                        lefts,
                        middle,
                        ends,
                        ca.repmat(theta, 1, count),
                        time[:-1].reshape(1, -1),
                        np.diff(time).reshape(1, -1),
                        u[:, :-1],
                        u[:, 1:],
                    )
                )
            )
            ix = np.clip(np.searchsorted(time, row.time, side="left") - 1, 0, count - 1)
            frac = (row.time - time[ix]) / (time[ix + 1] - time[ix])
            factors = [ca.repmat(ca.DM(v.reshape(1, -1)), n, 1) for v in basis(frac)]
            states = sum(
                ca.times(v[:, ix.tolist()], w)
                for v, w in zip((lefts, middle, ends), factors, strict=True)
            )
            prediction = s.observe.map(len(row.time))(
                row.time.reshape(1, -1),
                states,
                ca.repmat(theta, 1, len(row.time)),
                forcing_values(s, row, row.time),
            )
            actual = np.array([row.targets[c] for c in s.channels])
            scales = ca.repmat(
                ca.DM([oracle.scales[c] for c in s.channels]), 1, len(row.time)
            )
            observations.append(ca.vec((prediction - actual) / scales))
        observed, defect = ca.vertcat(*observations), ca.vertcat(*defects)
        penalty = ca.MX.sym("penalty")
        residual = ca.vertcat(
            observed / np.sqrt(observed.numel()),
            ca.sqrt(penalty / defect.numel()) * defect,
        )
        self.residual = ca.Function("conditional_residual", [z, q, penalty], [residual])
        self.design = ca.Function(
            "conditional_design",
            [z, q, penalty],
            [ca.jacobian(residual, q)[:, self.linear], residual],
        )
        self.components = ca.Function(
            "conditional_metrics",
            [z, q],
            [
                ca.vertcat(
                    ca.sumsqr(observed) / observed.numel(),
                    ca.sumsqr(defect) / defect.numel(),
                    ca.mmax(ca.fabs(defect)),
                )
            ],
        )
        self.build_optimizer()

    def build_optimizer(self):
        """Reuse formulation only; reset every primal start and do not reuse duals."""
        self.opti = ca.Opti()
        self.nz = self.opti.variable(self.node_count)
        self.free = self.opti.variable(len(self.outer)) if self.outer else None
        self.weights = self.opti.parameter(len(self.linear))
        self.penalty = self.opti.parameter()
        q = ca.MX.zeros(len(self.system.names))
        q[self.linear] = self.weights
        if self.outer:
            q[self.outer] = self.free
            lo, hi = self.qbounds()
            for j, i in enumerate(self.outer):
                if np.isfinite(lo[i]):
                    self.opti.subject_to(self.free[j] >= lo[i])
                if np.isfinite(hi[i]):
                    self.opti.subject_to(self.free[j] <= hi[i])
        self.full_q = q
        self.objective = ca.sumsqr(self.residual(self.nz, q, self.penalty)) / 2
        self.opti.minimize(self.objective)

    def qbounds(self):
        return (
            (self.oracle.lower - self.pcenter) / self.punits,
            (self.oracle.upper - self.pcenter) / self.punits,
        )

    def parameters(self, q):
        from autoformalism.fitting.recovery_numerics import restore

        vector = restore(
            self.pcenter,
            self.punits,
            q,
            self.oracle.lower,
            self.oracle.upper,
            self.system.names,
            [],
        )
        return self.oracle.parameters(vector)

    def guesses(self, parameters, previous=None):
        """Observed nodes plus causal constant hidden starts; optional mesh transfer."""
        vector = self.oracle.vector(parameters)
        chunks = []
        direct = self.system.model.direct_state_observation_channels
        for row, time in zip(self.oracle.train.trajectories, self.grids, strict=True):
            at = np.ravel(np.column_stack((time[:-1] + np.diff(time) / 3, time[1:])))
            if previous is None:
                x = np.tile(self.system.initial_for(row, vector), (len(at), 1))
                for j, name in enumerate(self.system.model.state_names):
                    if name in direct:
                        x[:, j] = np.interp(at, row.time, row.targets[direct[name]])
            else:
                old = previous[row.trajectory_id]
                oldt = np.asarray(old["time"])
                ix = np.clip(
                    np.searchsorted(oldt, at, side="right") - 1, 0, len(oldt) - 2
                )
                a, b, c = [
                    v[:, None] for v in basis((at - oldt[ix]) / np.diff(oldt)[ix])
                ]
                edges, inner = np.asarray(old["edges"]), np.asarray(old["inner"])
                x = a * edges[ix] + b * inner[ix] + c * edges[ix + 1]
            chunks.append(((x - self.center) / self.units).ravel())
        return np.concatenate(chunks), (vector - self.pcenter) / self.punits

    def trajectories(self, z, q):
        result, cursor = {}, 0
        vector = self.oracle.vector(self.parameters(q))
        for row, time in zip(self.oracle.train.trajectories, self.grids, strict=True):
            size = 2 * self.system.state_count * (len(time) - 1)
            values = (
                z[cursor : cursor + size].reshape(-1, self.system.state_count)
                * self.units
                + self.center
            )
            cursor += size
            result[row.trajectory_id] = {
                "time": time.tolist(),
                "edges": np.vstack(
                    (self.system.initial_for(row, vector), values[1::2])
                ).tolist(),
                "inner": values[::2].tolist(),
            }
        return result

    def linear_step(self, z, q, penalty):
        """Bounded convex LS; zero columns retain their old values, rank is reported."""
        matrix, residual = self.design(z, q, penalty)
        a, r = np.asarray(matrix), np.asarray(residual).ravel()
        if not np.isfinite(a).all() or not np.isfinite(r).all():
            raise ValueError("nonfinite conditional linear problem")
        lo, hi = self.qbounds()
        norm = np.linalg.norm(a, axis=0)
        active = norm > 1e-14
        changed = q.copy()
        if not np.any(active):
            return changed, {"accepted": False, "reason": "all_zero_columns", "rank": 0}
        selected = np.asarray(self.linear)[active]
        scaled = a[:, active] / norm[active]
        solved = lsq_linear(
            scaled,
            a[:, active] @ q[selected] - r,
            bounds=(lo[selected] * norm[active], hi[selected] * norm[active]),
            method="bvls",
            tol=1e-12,
            max_iter=200,
        )
        if not solved.success or not np.isfinite(solved.x).all():
            raise ValueError("conditional bounded least squares failed")
        changed[selected] = np.clip(solved.x / norm[active], lo[selected], hi[selected])
        candidate = np.asarray(self.residual(z, changed, penalty)).ravel()
        accepted = float(candidate @ candidate) <= float(r @ r)
        return changed if accepted else q.copy(), {
            "accepted": accepted,
            "rank": int(np.linalg.matrix_rank(scaled)),
            "columns": len(self.linear),
            "active_mask": solved.active_mask.tolist(),
            "zero_columns_retained": [
                self.system.names[i] for i in np.asarray(self.linear)[~active]
            ],
            "before_loss": float(r @ r / 2),
            "after_loss": float(candidate @ candidate / 2),
        }
