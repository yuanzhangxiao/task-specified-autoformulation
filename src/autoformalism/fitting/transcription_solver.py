"""Fixed-mesh native NLP worker with physical checkpoint trajectories.

Direct Radau collocation and CVODES multiple shooting share observations, bounds,
initialization, and coordinate transforms. Intermediate iterates may violate
continuity. No node trajectory is ever deployed as a prediction.
"""

from __future__ import annotations

from contextlib import suppress
from itertools import pairwise
from pathlib import Path
from time import monotonic

import casadi as ca
import numpy as np

from autoformalism.fitting import adaptive_mesh as mesh_tools
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.coordinates import NumericalCoordinates
from autoformalism.fitting.identifiable_campaign import SETTINGS, scale_for
from autoformalism.fitting.sensitivity_probe import SymbolicODE, SymbolicOracle
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


def interval_integrator(system, tolerance: float):
    """Adaptive BDF integration of one unmodified piecewise-linear input segment."""
    n, p, nu = system.state_count, len(system.names), len(system.inputs)
    state, clock = ca.MX.sym("state", n), ca.MX.sym("clock")
    packed = ca.MX.sym("packed", p + 2 + 2 * nu)
    theta, t0, dt = packed[:p], packed[p], packed[p + 1]
    u0, u1 = packed[p + 2 : p + 2 + nu], packed[p + 2 + nu :]
    integrator = ca.integrator(
        "segment",
        "cvodes",
        {
            "x": state,
            "t": clock,
            "p": packed,
            "ode": dt
            * system.rhs(t0 + dt * clock, state, theta, u0 + clock * (u1 - u0)),
        },
        0,
        1,
        {
            "reltol": tolerance,
            "abstol": tolerance * 0.01,
            "max_num_steps": 5000,
            "disable_internal_warnings": True,
        },
    )
    return integrator


def solve(payload: dict, directory: Path) -> dict:
    """Run one NLP; all input payload fields are training-only and frozen by caller."""
    begun = monotonic()
    deadline = begun + payload["seconds"]
    request = PublicFitRequest.model_validate(payload["request"])
    data = PublicSplit.model_validate(payload["training"])
    if data.name != "train":
        raise ValueError("transcription requires training data")
    model, _, _ = public._lower(request)
    system = SymbolicODE(model, allow_piecewise=True)
    training = public.unpack_split(data)
    scales = scale_for(training)
    coordinates = NumericalCoordinates.model_validate(payload["coordinates"])
    pc, ps = coordinates.arrays("parameters", system.names)
    sc, ss = coordinates.arrays("states", model.state_names)
    layout = SymbolicOracle(
        system,
        training,
        scales,
        SETTINGS,
        directory / "layout",
        None,
        sensitivities=False,
    )
    start = layout.vector(payload["start"])
    if (
        not np.isfinite(start).all()
        or np.any(start < layout.lower)
        or np.any(start > layout.upper)
    ):
        raise ValueError("starting parameters violate bounds")
    method = payload["method"]
    if method not in {"shooting", "collocation"}:
        raise ValueError("unknown transcription")
    chunks = payload.get("reuse_chunks", 1)
    if type(chunks) is not int or not 1 <= chunks <= 8:
        raise ValueError("reuse_chunks must be an integer in 1..8")
    if chunks != 1 and method != "collocation":
        raise ValueError("graph reuse is qualified for collocation only")
    if set(payload["meshes"]) != {r.trajectory_id for r in training.trajectories}:
        raise ValueError("mesh identities differ")
    opti = ca.Opti()
    q = opti.variable(len(start))
    theta = pc + ps * q
    opti.set_initial(q, (start - pc) / ps)
    for i in range(len(start)):
        if np.isfinite(layout.lower[i]):
            opti.subject_to(q[i] >= (layout.lower[i] - pc[i]) / ps[i])
        if np.isfinite(layout.upper[i]):
            opti.subject_to(q[i] <= (layout.upper[i] - pc[i]) / ps[i])

    def state_node(guess):
        node = opti.variable(system.state_count)
        opti.set_initial(node, (guess - sc) / ss)
        return sc + ss * node

    integrate = (
        interval_integrator(system, payload["tolerance"])
        if method == "shooting"
        else None
    )
    objective, count, defects = 0, 0, []
    expressions = {}
    for row in training.trajectories:
        key = row.trajectory_id
        mesh = mesh_tools.validate_mesh(row.time, payload["meshes"][key])
        guess = np.asarray(payload["nodes"][key], float)
        if (
            guess.shape != (len(mesh), system.state_count)
            or not np.isfinite(guess).all()
        ):
            raise ValueError("node guess shape/values differ")
        if not np.allclose(
            guess[0], system.initial_for(row, start), rtol=0, atol=1e-10
        ):
            raise ValueError("node guess violates initial boundary")
        if method == "collocation" and not set(row.time).issubset(set(mesh)):
            raise ValueError(
                "collocation mesh must include all supplied sample/input knots"
            )
        forcing = trajectory_forcing(model, row)

        def inputs(t, forcing=forcing):
            return np.array([forcing.value(n, t) for n in system.inputs])

        current = system.initial_symbolic(row, theta)
        boundaries, inner, samples, scores = (
            [current],
            [],
            {float(row.time[0]): current},
            [],
        )
        for i, (left, right) in enumerate(pairwise(mesh)):
            if monotonic() >= deadline:
                raise TimeoutError("NLP graph construction budget exhausted")
            end = state_node(guess[i + 1])
            dt = float(right - left)
            if method == "collocation":
                middle = state_node((2 * guess[i] + guess[i + 1]) / 3)
                f1 = system.rhs(left + dt / 3, middle, theta, inputs(left + dt / 3))
                f2 = system.rhs(right, end, theta, inputs(right))
                local_defects = [
                    (middle - current - dt * (5 * f1 - f2) / 12) / ss,
                    (end - current - dt * (3 * f1 + f2) / 4) / ss,
                ]
                inner.append(middle)
                # Every original sample occurs at a mesh boundary; added subnodes
                # carry no additional observation weight.
                if right in set(row.time):
                    samples[float(right)] = end
                score = ca.mmax(ca.fabs(ca.vertcat(*local_defects)))
            else:
                local = current
                local_loss = 0
                for a, b in pairwise(
                    mesh_tools.integration_times(row.time, left, right)
                ):
                    local = integrate(
                        x0=local, p=ca.vertcat(theta, a, b - a, inputs(a), inputs(b))
                    )["xf"]
                    if b in set(row.time):
                        # Evaluate from the left interval, before imposing its end
                        # continuity equality. No measurement resets are possible.
                        samples[float(b)] = local
                        pred = system.observe(b, local, theta, inputs(b))
                        index = int(np.searchsorted(row.time, b))
                        for j, channel in enumerate(system.channels):
                            local_loss = ca.fmax(
                                local_loss,
                                ca.fabs(
                                    (pred[j] - row.targets[channel][index])
                                    / scales[channel]
                                ),
                            )
                local_defects = [(end - local) / ss]
                # Observed residual and continuity guide the outer mesh; CVODES
                # adapts internal steps using all state error estimates.
                score = ca.fmax(local_loss, ca.mmax(ca.fabs(local_defects[0])))
            defects.extend(local_defects)
            opti.subject_to(ca.vertcat(*local_defects) == 0)
            scores.append(score)
            boundaries.append(end)
            current = end
        if set(samples) != set(row.time):
            raise ValueError("not all observations are included exactly once")
        for i, t in enumerate(row.time):
            prediction = system.observe(t, samples[float(t)], theta, inputs(t))
            for j, channel in enumerate(system.channels):
                objective += (
                    (prediction[j] - row.targets[channel][i]) / scales[channel]
                ) ** 2
                count += 1
        expressions[key] = {
            "nodes": ca.hcat(boundaries).T,
            "inner": ca.hcat(inner).T if inner else ca.MX.zeros(0, system.state_count),
            "indicators": ca.vertcat(*scores),
        }
    objective /= count
    opti.minimize(objective)
    all_defects = ca.vertcat(*defects)
    pool, seen, last_iteration = [], set(), -1
    last_checkpoint_time = monotonic()
    iteration_offset = 0

    def snapshot(iteration, value):
        nonlocal pool, last_iteration, last_checkpoint_time
        vector = np.asarray(value(theta)).ravel()
        loss = float(value(objective))
        defect = float(np.max(abs(np.asarray(value(all_defects)))))
        if not np.isfinite(vector).all() or not np.isfinite(loss + defect):
            return
        if np.any(vector < layout.lower - 1e-10) or np.any(
            vector > layout.upper + 1e-10
        ):
            return
        vector = np.clip(vector, layout.lower, layout.upper)
        record = {
            "iteration": int(iteration),
            "seconds": monotonic() - begun,
            "parameters": dict(zip(system.names, vector.tolist(), strict=True)),
            "collocation_nmse": loss,
            "maximum_scaled_defect": defect,
            "trajectories": {},
        }
        for row in training.trajectories:
            key = row.trajectory_id
            states = np.asarray(value(expressions[key]["nodes"])).reshape(
                -1, system.state_count
            )
            stages = np.asarray(value(expressions[key]["inner"])).reshape(
                -1, system.state_count
            )
            if not np.isfinite(states).all() or not np.isfinite(stages).all():
                return
            indicators = (
                mesh_tools.collocation_indicators(
                    system, row, vector, states, stages, payload["meshes"][key], ss
                )
                if method == "collocation"
                else np.asarray(value(expressions[key]["indicators"])).ravel().tolist()
            )
            record["trajectories"][key] = {
                "mesh": payload["meshes"][key],
                "nodes": states.tolist(),
                "indicators": [v if np.isfinite(v) else 1e100 for v in indicators],
            }
        digest = public.content_sha256(record["parameters"])
        if digest not in seen:
            seen.add(digest)
            pool.append(record)
        # Keep a diverse recent history and the least-defective / best feasible
        # points. Loss at an inconsistent trajectory is not a rollout score.
        protected = (
            [min(pool, key=lambda p: p["maximum_scaled_defect"])] if pool else []
        )
        feasible = [p for p in pool if p["maximum_scaled_defect"] <= 1e-6]
        if feasible:
            protected.append(min(feasible, key=lambda p: p["collocation_nmse"]))
        pool = list(
            {
                public.content_sha256(p["parameters"]): p
                for p in [*protected, *pool[-6:]]
            }.values()
        )
        public._write(directory / "checkpoints.json", {"pool": pool, "latest": record})
        last_iteration = iteration
        last_checkpoint_time = monotonic()

    def callback(iteration):
        if (
            iteration <= 3
            or iteration % 5 == 0
            or monotonic() - last_checkpoint_time >= 1
        ):
            snapshot(iteration_offset + iteration, opti.debug.value)
        if monotonic() >= deadline:
            raise TimeoutError("native NLP deadline reached")

    opti.callback(callback)
    # Repeated solves retain this Opti graph and its solver instance. Reuse is
    # within one worker/mesh, not a serialized solver or cross-job state resume.
    native_options = {
        "print_level": 0,
        "max_iter": 400 // chunks,
        "tol": 1e-8,
        "max_cpu_time": max(0.01, deadline - monotonic()),
        "hessian_approximation": "limited-memory" if system.has_piecewise else "exact",
    }
    if chunks > 1:
        native_options["warm_start_init_point"] = "yes"
    opti.solver(
        "ipopt",
        {"print_time": False},
        native_options,
    )
    public._write(
        directory / "layout.json",
        {
            "decision_variables": int(opti.nx),
            "constraints": int(opti.ng),
            "observation_residuals": count,
            "graph_seconds": monotonic() - begun,
            "method": method,
            "graph_builds": 1,
            "maximum_solve_chunks": chunks,
            "iterations_per_chunk": 400 // chunks,
        },
    )
    success, message, chunk_records, evaluation_counts = False, "", [], {}
    for chunk in range(chunks):
        if monotonic() >= deadline:
            break
        chunk_start = monotonic()
        try:
            solved = opti.solve()
            success = True
            snapshot(iteration_offset + int(solved.stats()["iter_count"]), solved.value)
            message = solved.stats()["return_status"]
        except (RuntimeError, ValueError, TimeoutError) as error:
            message = str(error)[-1200:]
            with suppress(RuntimeError, ValueError):
                snapshot(last_iteration, opti.debug.value)
        stats = opti.stats()
        for key, value in stats.items():
            if key.startswith("n_call_"):
                evaluation_counts[key] = evaluation_counts.get(key, 0) + value
        chunk_records.append(
            {
                "chunk": chunk,
                "graph_reused": chunk > 0,
                "primal_dual_warm_start": chunk > 0,
                "return_status": stats.get("return_status"),
                "iterations": int(stats.get("iter_count", 0)),
                "seconds": monotonic() - chunk_start,
            }
        )
        public._write(directory / "solve_chunks.json", chunk_records)
        if (
            success
            or stats.get("return_status") != "Maximum_Iterations_Exceeded"
            or chunk + 1 == chunks
            or monotonic() >= deadline
        ):
            break
        # Transfer primal and constraint multipliers explicitly. This is a
        # warm start, not restoration of all IPOPT internal state (e.g. L-BFGS).
        primal = np.asarray(opti.debug.value(opti.x)).ravel()
        dual = np.asarray(opti.debug.value(opti.lam_g)).ravel()
        if not np.isfinite(primal).all() or not np.isfinite(dual).all():
            break
        opti.set_initial(opti.x, primal)
        opti.set_initial(opti.lam_g, dual)
        iteration_offset += int(stats.get("iter_count", 0))
    result = {
        "native_evaluation_counts": evaluation_counts,
        "native_success": success,
        "message": message,
        "seconds": monotonic() - begun,
        "iterations_at_last_checkpoint": last_iteration,
        "graph_builds": 1,
        "solve_chunks": chunk_records,
    }
    public._write(directory / "native.json", result)
    return result
