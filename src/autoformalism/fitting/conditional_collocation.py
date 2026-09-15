"""One dimensionless penalized Radau problem shared by joint and block solvers."""

from __future__ import annotations

from time import monotonic

import casadi as ca
import numpy as np
from scipy.optimize import lsq_linear

from autoformalism.data import SplitName, TrainingScaler
from autoformalism.fitting.collocation_mesh import basis, forcing_values, plan_meshes
from autoformalism.fitting.matching_probe import collocation_node_guess
from autoformalism.staged_topology import content_hash


def common_initialization(
    system,
    training,
    start,
    settings,
    partition,
    *,
    target_variables,
    minimum_intervals,
    warmup_seconds=5,
):
    """Freeze guesses once for both arms; no reference trajectory labels enter."""
    if training.name is not SplitName.TRAIN:
        raise ValueError("common initialization requires training data")
    meshes, mesh_audit = plan_meshes(
        system, training, target_variables, 1, minimum_intervals
    )
    guesses, records = [], []
    deadline = monotonic() + warmup_seconds
    for row in training.trajectories:
        guess, record = collocation_node_guess(
            system, row, start, settings, "rollout_or_observed", deadline
        )
        guesses.append(guess)
        records.append(record)
    latent = np.asarray(partition["state_exponents"], dtype=bool)
    latent_unit = max(
        1.0,
        float(
            np.sqrt(
                np.mean(np.concatenate([g[:, latent].ravel() for g in guesses]) ** 2)
            )
        ),
    )
    output_unit = TrainingScaler().fit(training).scales["target:v01"].standard_deviation
    units = np.where(latent, latent_unit, output_unit)
    nodes = []
    for row, mesh, guess in zip(training.trajectories, meshes, guesses, strict=True):
        edge = np.array([np.interp(mesh.time, row.time, col) for col in guess.T])
        inner = (2 * edge[:, :-1] + edge[:, 1:]) / 3
        nodes.extend(
            (np.vstack((inner, edge[:, 1:])) / np.tile(units, 2)[:, None]).ravel(
                order="F"
            )
        )
    payload = {
        "state_units": units.tolist(),
        "parameter_units": np.maximum(1, abs(start)).tolist(),
        "ordinary_parameters": start.tolist(),
        "output_scale": output_unit,
        "partition": partition,
        "mesh": mesh_audit,
        "node_initialization": records,
        "training_fingerprint": training.fingerprint,
        "known_initials_fixed": True,
        "hidden_trajectory_labels_used": False,
        "scaling_policy": "frozen-guess-common-latent-unit-1",
    }
    z0 = np.asarray(nodes, dtype=float)
    if not np.isfinite(z0).all():
        raise ValueError("nonfinite collocation node guess")
    payload["nodes_identity"] = content_hash(z0.tolist())
    payload["identity"] = content_hash(payload)
    return payload, z0


class ConditionalCollocation:
    """Known physical boundaries; all observations; no gauge constraint added.

    ``unit_factor`` is only for the equivalence audit: it transforms equations'
    parameters, state units and boundary coordinates together. Fits use one.
    """

    def __init__(self, system, training, common, z0, penalty, *, unit_factor=1.0):
        if (
            training.name is not SplitName.TRAIN
            or training.fingerprint != common["training_fingerprint"]
        ):
            raise ValueError("collocation training fingerprint differs")
        if (
            not np.isfinite(penalty)
            or penalty <= 0
            or not np.isfinite(unit_factor)
            or unit_factor <= 0
        ):
            raise ValueError("positive finite scaling and penalty required")
        self.system, self.common, self.z0 = system, common, np.asarray(z0)
        self.partition = common["partition"]
        self.wi, self.si = (
            self.partition[k] for k in ("weight_indices", "shape_indices")
        )
        transform = unit_factor ** np.asarray(self.partition["state_exponents"])
        ptransform = unit_factor ** np.asarray(self.partition["parameter_exponents"])
        self.units = np.asarray(common["state_units"]) * transform
        self.punits = np.asarray(common["parameter_units"]) * ptransform
        self.q0 = np.asarray(common["ordinary_parameters"]) / common["parameter_units"]
        self.penalty = penalty
        n, p = system.state_count, len(system.names)
        z, q = ca.MX.sym("nodes", len(z0)), ca.MX.sym("parameters", p)
        theta = q * ca.DM(self.punits)
        left, inner, end = [ca.MX.sym(k, n) for k in ("left", "inner", "end")]
        t, dt = ca.MX.sym("t"), ca.MX.sym("dt")
        pp = ca.MX.sym("pp", p)
        u0, u1 = [ca.MX.sym(k, len(system.inputs)) for k in ("u0", "u1")]
        f1 = system.rhs(t + dt / 3, inner, pp, u0 + (u1 - u0) / 3)
        f2 = system.rhs(t + dt, end, pp, u1)
        defect = ca.vertcat(
            (inner - left - dt * (5 * f1 - f2) / 12) / ca.DM(self.units),
            (end - left - dt * (3 * f1 + f2) / 4) / ca.DM(self.units),
        )
        interval = ca.Function(
            "scaled_radau", [left, inner, end, pp, t, dt, u0, u1], [defect]
        )
        observation, dynamics, cursor = [], [], 0
        for row, mesh in zip(
            training.trajectories, common["mesh"]["trajectories"], strict=True
        ):
            if row.trajectory_id != mesh["trajectory_id"]:
                raise ValueError("mesh trajectory ordering differs")
            times = np.asarray(mesh["boundaries"])
            count = len(times) - 1
            size = 2 * n * count
            values = ca.reshape(z[cursor : cursor + size], 2 * n, count)
            cursor += size
            physical = ca.times(
                values, ca.repmat(ca.DM(np.tile(self.units, 2)), 1, count)
            )
            inner, end = physical[:n, :], physical[n:, :]
            initial = (
                system.initial_for(row, np.asarray(common["ordinary_parameters"]))
                * transform
            )
            left = ca.horzcat(ca.DM(initial), end[:, :-1])
            u = forcing_values(system, row, times)
            dynamics.append(
                ca.vec(
                    interval.map(count)(
                        left,
                        inner,
                        end,
                        ca.repmat(theta, 1, count),
                        times[:-1].reshape(1, -1),
                        np.diff(times).reshape(1, -1),
                        u[:, :-1],
                        u[:, 1:],
                    )
                )
            )
            ix = np.clip(
                np.searchsorted(times, row.time, side="left") - 1, 0, count - 1
            )
            fractions = (row.time - times[ix]) / (times[ix + 1] - times[ix])
            b = [ca.repmat(ca.DM(v.reshape(1, -1)), n, 1) for v in basis(fractions)]
            states = (
                ca.times(left[:, ix.tolist()], b[0])
                + ca.times(inner[:, ix.tolist()], b[1])
                + ca.times(end[:, ix.tolist()], b[2])
            )
            predicted = system.observe.map(len(row.time))(
                row.time.reshape(1, -1),
                states,
                ca.repmat(theta, 1, len(row.time)),
                forcing_values(system, row, row.time),
            )
            observation.append(
                ca.vec(
                    (predicted[0, :] - row.targets["v01"].reshape(1, -1))
                    / common["output_scale"]
                )
            )
        if cursor != len(z0):
            raise ValueError("node vector size differs from frozen mesh")
        observed, defects = ca.vertcat(*observation), ca.vertcat(*dynamics)
        residual = ca.vertcat(
            observed / np.sqrt(observed.numel()),
            np.sqrt(penalty / defects.numel()) * defects,
        )
        self.residual = ca.Function("conditional_residual", [z, q], [residual])
        self.components = ca.Function(
            "conditional_components",
            [z, q],
            [
                ca.vertcat(
                    ca.sumsqr(observed) / observed.numel(),
                    ca.sumsqr(defects) / defects.numel(),
                    ca.mmax(ca.fabs(defects)),
                )
            ],
        )
        # Affinity is certified on the local RHS before building this graph.
        design = ca.jacobian(residual, q)[:, self.wi]
        self.linear_design = ca.Function(
            "conditional_design", [z, q], [design, residual]
        )

    def parameters(self, q):
        """Return original physical parameter identities without changing bounds."""
        return dict(
            zip(self.system.names, map(float, np.asarray(q) * self.punits), strict=True)
        )

    def loss(self, z, q):
        r = np.asarray(self.residual(z, q)).ravel()
        return float(0.5 * r @ r)

    def linear_step(self, z, q, lower, upper):
        """Solve the full shared 26-column block; keep nonlinear shapes fixed."""
        design, residual = self.linear_design(z, q)
        a, residual = np.asarray(design), np.asarray(residual).ravel()
        b = a @ q[self.wi] - residual
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite linear subproblem")
        units = np.linalg.norm(a, axis=0)
        units[units == 0] = 1
        scaled = a / units
        solved = lsq_linear(
            scaled,
            b,
            bounds=(lower[self.wi] * units, upper[self.wi] * units),
            method="bvls",
            tol=1e-10,
            max_iter=200,
        )
        proposal = q.copy()
        proposal[self.wi] = solved.x / units
        gradient = scaled.T @ (a @ proposal[self.wi] - b)
        mask = solved.active_mask
        projected = np.where(
            mask == -1,
            np.minimum(gradient, 0),
            np.where(mask == 1, np.maximum(gradient, 0), gradient),
        )
        kkt = float(np.max(abs(projected), initial=0))
        accepted = bool(
            np.isfinite(proposal).all()
            and np.all(proposal >= lower)
            and np.all(proposal <= upper)
            and kkt <= 1e-7 * max(1, np.linalg.norm(b))
            and self.loss(z, proposal) <= self.loss(z, q)
        )
        return (proposal if accepted else q.copy()), {
            "accepted": accepted,
            "solver_success": bool(solved.success),
            "message": str(solved.message),
            "scaled_projected_gradient": kkt,
            "rank": int(np.linalg.matrix_rank(scaled)),
            "columns": len(self.wi),
            "condition": float(np.linalg.cond(scaled)),
            "objective_before": self.loss(z, q),
            "objective_after": self.loss(z, proposal),
        }
