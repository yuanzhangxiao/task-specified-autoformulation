"""Evaluator-only M23 diagnostics; never imported by the fitting optimizer.

Reference trajectories/derivatives isolate the conditional coefficient solve
from discretization and estimated-node errors. They are not fitting starts.
"""

from dataclasses import replace
from pathlib import Path
from time import monotonic

import casadi as ca
import numpy as np
from scipy.optimize import lsq_linear

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import nonlinear_rollout
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.collocation_mesh import forcing_values
from autoformalism.fitting.conditional_optimizer import load_point
from autoformalism.fitting.sensitivity_probe import symbolic_rollout
from autoformalism.fitting.trajectory_profile import TrajectoryProblem

PROTOCOL = "conditional-coefficient-diagnostics-1"
SOURCE_PLAN = "89d3225646b010783bc8baa14b19b40279c7cb120da5f9bd239f3298fdb5d0db"


def export(source: Path, output: Path) -> dict:
    """Export only six hash-checked M22 warm-node endpoints, with source identity."""
    plan = read_seal(source / "plan.json")
    if public.content_sha256(plan) != SOURCE_PLAN:
        raise ValueError("expected the reviewed M22 plan")
    rows = []
    for task in plan["tasks"]:
        if task["method"] != "conditional_then_best_rollout":
            continue
        folder = source / "results" / task["task_id"]
        backend = read_seal(folder / "backend.json")
        result = read_seal(folder / "result.json")
        if public.content_sha256(backend) != result["backend_sha256"]:
            raise ValueError("source backend digest differs")
        for level in range(2):
            loaded = load_point(
                folder / "fit/warm/nodes" / f"mesh-{level}", f"nodes-mesh-{level}"
            )
            if loaded is None:
                raise ValueError("missing warm checkpoint")
            record, z, q = loaded
            rows.append(
                {
                    "common": task["common"],
                    "level": level,
                    "source_backend_sha256": result["backend_sha256"],
                    "checkpoint": record,
                    "z": z.tolist(),
                    "q": q.tolist(),
                }
            )
    value = {
        "protocol": PROTOCOL,
        "source_plan_sha256": SOURCE_PLAN,
        "inputs_sha256": plan["inputs_sha256"],
        "rows": rows,
        "test_data_opened": False,
    }
    seal(output, value)
    return {"identity": public.content_sha256(value), "checkpoints": len(rows)}


def coefficient_error(names: list[str], values: dict, truth: dict) -> float:
    """Maximum relative error for the nonzero dynamic block in this diagnostic."""
    return max(abs(values[n] - truth[n]) / max(abs(truth[n]), 1e-12) for n in names)


def bounded_diagnostic(a, target, lower, upper) -> dict:
    """Solve a scaled linear problem, reporting rank before claiming recovery."""
    a, target = np.asarray(a), np.asarray(target).ravel()
    if not np.isfinite(a).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite derivative diagnostic")
    norms = np.linalg.norm(a, axis=0)
    zero = norms <= 1e-14
    norms[zero] = 1.0
    scaled = a / norms
    sv = np.linalg.svd(scaled, compute_uv=False)
    rank = int(np.linalg.matrix_rank(scaled))
    solved = lsq_linear(
        scaled,
        target,
        bounds=(lower * norms, upper * norms),
        method="bvls",
        tol=1e-12,
        max_iter=200,
    )
    if not solved.success or not np.isfinite(solved.x).all():
        raise ValueError("bounded derivative diagnostic failed")
    return {
        "status": "complete",
        "rank": rank,
        "columns": a.shape[1],
        "full_rank": rank == a.shape[1],
        "zero_columns": np.flatnonzero(zero).tolist(),
        "singular_values": sv.tolist(),
        "values": (solved.x / norms).tolist(),
        "residual_relative_norm": float(
            np.linalg.norm(scaled @ solved.x - target)
            / max(np.linalg.norm(target), 1e-12)
        ),
    }


def reference_nodes(graph, truth: dict, deadline: float):
    """Integrate at actual stage nodes, retaining every original input knot."""
    oracle, system = graph.oracle, graph.system
    theta = oracle.vector(truth)
    chunks, samples = [], []
    for row, grid in zip(oracle.train.trajectories, graph.grids, strict=True):
        at = np.ravel(np.column_stack((grid[:-1] + np.diff(grid) / 3, grid[1:])))
        time = np.unique(np.r_[row.time, at])
        fine = replace(
            row,
            time=time,
            derivatives={},
            targets={k: np.interp(time, row.time, v) for k, v in row.targets.items()},
            auxiliaries={
                k: np.interp(time, row.time, v) for k, v in row.auxiliaries.items()
            },
            external_inputs={
                k: np.interp(time, row.time, v) for k, v in row.external_inputs.items()
            },
        )
        _, _, states, _ = symbolic_rollout(
            system,
            fine,
            theta,
            oracle.settings.model_copy(
                update={"relative_tolerance": 1e-11, "absolute_tolerance": 1e-13}
            ),
            deadline,
            sensitivities=False,
        )
        stage = states[np.searchsorted(time, at)]
        chunks.append(((stage - graph.center) / graph.units).ravel())
        samples.append((at, stage, forcing_values(system, row, at)))
    return np.concatenate(chunks), samples


def derivative_diagnostic(graph, truth: dict, samples) -> dict:
    """Exact RHS derivatives at reference states remove collocation truncation."""
    s = graph.system
    t, x = ca.SX.sym("t"), ca.SX.sym("x", s.state_count)
    p, u = ca.SX.sym("p", len(s.names)), ca.SX.sym("u", len(s.inputs))
    f = s.rhs(t, x, p, u)
    linear = graph.linear
    design = ca.Function(
        "derivative_design", [t, x, p, u], [ca.jacobian(f, p)[:, linear], f]
    )
    theta = graph.oracle.vector(truth)
    matrices, targets = [], []
    for time, states, inputs in samples:
        count = len(time)
        a, rhs = design.map(count)(
            time.reshape(1, -1), states.T, np.tile(theta[:, None], (1, count)), inputs
        )
        a = np.asarray(a).reshape(s.state_count, count, len(linear)).transpose(1, 0, 2)
        rhs = np.asarray(rhs).T
        # Reconstruct the affine offset while fixing all non-profiled shapes.
        offset = rhs - a @ theta[linear]
        matrices.append((a / graph.units[None, :, None]).reshape(-1, len(linear)))
        targets.append(((rhs - offset) / graph.units).ravel())
    solved = bounded_diagnostic(
        np.vstack(matrices),
        np.concatenate(targets),
        graph.oracle.lower[linear],
        graph.oracle.upper[linear],
    )
    error = None
    if solved["values"] is not None:
        names = graph.partition["names"]
        values = dict(zip(names, solved["values"], strict=True))
        error = coefficient_error(names, values, truth)
    return solved | {
        "maximum_relative_error": error,
        "correctness_passed": bool(
            solved["status"] == "complete"
            and solved["residual_relative_norm"] < 1e-8
            and (not solved["full_rank"] or error < 1e-5)
        ),
        "coefficient_recovery_certified": bool(
            solved["full_rank"] and error is not None and error < 1e-5
        ),
        "scope": "reference-derived exact RHS; no trajectory error or discretization",
    }


def diagnose_checkpoint(
    problem, truth: dict, saved: dict, policy, deadline, save
) -> dict:
    """Compare exact derivatives, discretized reference nodes and saved nodes."""
    oracle, _, _ = nonlinear_rollout.build(problem)
    level = saved["level"]
    graph = TrajectoryProblem(
        oracle,
        problem,
        policy.node_targets[level],
        policy.minimum_intervals,
        policy.observation_anchors,
        deadline,
    )
    z, q = np.asarray(saved["z"]), np.asarray(saved["q"])
    if z.shape != (graph.node_count,) or q.shape != (len(oracle.names),):
        raise ValueError("saved node/parameter layout differs")
    if not np.isfinite(z).all() or not np.isfinite(q).all():
        raise ValueError("nonfinite saved node checkpoint")
    values = graph.parameters(q)
    if not np.allclose(
        oracle.vector(values),
        oracle.vector(saved["checkpoint"]["parameters"]),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("saved coordinates differ from checkpoint parameters")
    reference, samples = reference_nodes(graph, truth, deadline)
    exact = derivative_diagnostic(graph, truth, samples)
    save({"exact_derivative": exact})
    qref = (oracle.vector(truth) - graph.pcenter) / graph.punits
    rows = []
    for label, nodes, shapes in (
        ("reference_nodes_reference_shapes", reference, qref),
        ("estimated_nodes_reference_shapes", z, qref),
        ("estimated_nodes_estimated_shapes", z, q),
    ):
        if monotonic() >= deadline:
            raise TimeoutError("conditional diagnostic allowance exhausted")
        start = shapes.copy()
        # Begin every coefficient solve from the saved generic coefficients.
        start[graph.linear] = q[graph.linear]
        fitted, audit = graph.linear_step(nodes, start, policy.penalties[level])
        rows.append(
            {
                "case": label,
                "audit": audit,
                "maximum_relative_error": coefficient_error(
                    graph.partition["names"], graph.parameters(fitted), truth
                ),
                "components_before": np.asarray(graph.components(nodes, start))
                .ravel()
                .tolist(),
                "components_after": np.asarray(graph.components(nodes, fitted))
                .ravel()
                .tolist(),
            }
        )
        save({"cases": rows})
    return {
        "exact_derivative": exact,
        "cases": rows,
        "mesh": graph.audit,
        "reference_available_to_fitter": False,
        "validation_opened": False,
        "test_data_opened": False,
    }


def run(root: Path) -> dict:
    """Checkpoint evaluator-only operations; interrupted budgets never restart."""
    from autoformalism.fitting import nonlinear_comparison_campaign as campaign

    plan, data = campaign.verify(root)
    source = read_seal(root / "diagnostic-inputs.json")
    policy = campaign.ComparisonPolicy.model_validate(plan["policy"])
    problems = campaign.bases(data)
    rows = []
    for item in source["rows"]:
        name = f"{item['common']}_mesh{item['level']}"
        record = checks.operation(
            root / "diagnostics" / name,
            {
                "plan_sha256": public.content_sha256(plan),
                "checkpoint_sha256": public.content_sha256(item),
            },
            180,
            lambda end, save, item=item: diagnose_checkpoint(
                problems[item["common"]],
                data["case"]["reference_parameters"],
                item,
                policy,
                end,
                save,
            ),
        )
        rows.append({"task": name, **record})
    complete = all(r["status"] == "complete" for r in rows)
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if complete else "incomplete",
        "rows": rows,
        "correctness_passed": complete
        and all(r["value"]["exact_derivative"]["correctness_passed"] for r in rows),
        "reference_available_to_fitter": False,
        "test_data_opened": False,
        "wall_seconds": sum(r["wall_seconds"] or 0 for r in rows),
        "cpu_seconds": sum(r["cpu_seconds"] or 0 for r in rows),
        "budget_charge_seconds": sum(r["budget_charge_seconds"] for r in rows),
        "cost_complete": all(r["accounting_complete"] for r in rows),
    }
    seal(root / "diagnostics" / "result.json", result)
    return result
