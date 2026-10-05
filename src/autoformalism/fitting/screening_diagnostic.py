"""M9: bounded screening and explicitly assisted, fixed-then-free collocation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import monotonic

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import adaptive_mesh as mesh
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.bounded_screening import TrainingOnlySplit, screen
from autoformalism.fitting.collocation_mesh import plan_meshes
from autoformalism.fitting.identifiable_campaign import SETTINGS
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest

PROTOCOL = "phase-c-fitting-screening-assistance-1"
ARMS = {
    f"{base}_bounded_{method.lower()}": (base, method, False)
    for base in (
        "fixed_dense_collocation",
        "fixed_reduced_collocation",
        "fixed_shooting_lbfgs",
    )
    for method in ("RK45", "Radau")
} | {
    f"{base}_assisted": (base, "Radau", True)
    for base in ("fixed_dense_collocation", "fixed_reduced_collocation")
}


class DiagnosticPolicy(StrictSchema):
    """Explicit training-only eligibility and nested wall ceilings."""

    point_seconds: float = Field(default=20, ge=1, le=300)
    node_seconds: float = Field(default=120, ge=1, le=180)
    fixed_seconds: float = Field(default=250, ge=1, le=400)
    eligible_training_nmse: float = Field(default=1e-8, gt=0, le=1e-3)


def export_starts(source: Path, output: Path) -> dict:
    """Export every declared M7 continuation seed, without reading test/val metrics.

    Only complete training-evaluated endpoints supply vectors. Full source payload
    identity, unmodified training bytes, source costs and ineligible starts remain
    auditable. Reference parameters are never used to build assisted starts.
    """
    from autoformalism.fitting import transcription_campaign as campaign

    plan, inputs = campaign.verify(source, runtime=False)
    if (
        plan["protocol"] != campaign.numerical.PROTOCOL
        or plan.get("test_data_opened") is not False
    ):
        raise ValueError("assistance requires the frozen M7 development campaign")
    result = deepcopy(inputs)
    result["assisted_starts"] = {}
    result["frozen_start_commons"] = deepcopy(plan["commons"])
    result["assisted_source_inputs_sha256"] = public.content_sha256(inputs)
    result["assisted_source_plan_sha256"] = public.content_sha256(plan)
    for task in plan["tasks"]:
        if task["arm"] != "rollout_continue":
            continue
        common = plan["commons"][task["common"]]
        backend = read_seal(source / "results" / task["task_id"] / "backend.json")
        if backend.get("worker_payload_sha256") != public.content_sha256(
            campaign.worker_payload(plan, inputs, task)
        ):
            raise ValueError("assisted backend payload differs")
        good_provenance = (
            backend.get("selection")
            == "best_complete_training_rollout_across_both_phases"
            and backend.get("reference_values_used") is False
            and backend.get("validation_used_for_fitting") is False
        )
        result["assisted_starts"][task["common"]] = {
            "parameters": backend.get("parameters") if good_provenance else None,
            "training_nmse": backend.get("training_nmse"),
            "training_verified": good_provenance,
            "source_plan_sha256": public.content_sha256(plan),
            "source_backend_sha256": public.content_sha256(backend),
            "request_sha256": public.content_sha256(common["request"]),
            "source_task": task["task_id"],
            "source_seconds": backend["total_seconds"],
            "source_residual_calls": backend.get("actual_residual_calls"),
            "selection_rule": (
                "all_M7_rollout_continue_seeds; training-only eligibility applied later"
            ),
        }
    seal(output, result)
    return {
        "inputs_sha256": public.content_sha256(result),
        "source_starts": len(result["assisted_starts"]),
    }


def bind_start(common: dict, inputs: dict, policy: DiagnosticPolicy) -> dict:
    """Validate assistance against this exact request; never consult truth or val."""
    key = f"{common['case']}_s{common['seed']}"
    item = deepcopy(inputs.get("assisted_starts", {}).get(key))
    if item is None or item.get("request_sha256") != public.content_sha256(
        common["request"]
    ):
        raise ValueError("missing or mismatched assisted source request")
    score = item.get("training_nmse")
    item["eligible"] = bool(
        item.get("training_verified")
        and item.get("parameters")
        and score is not None
        and np.isfinite(score)
        and score <= policy.eligible_training_nmse
    )
    if not item["eligible"]:
        item["parameters"] = None
    return item


def node_worker(payload: dict, directory: Path) -> dict:
    """Integrate a fitted training model at boundaries and Radau internal nodes.

    Extra evaluation times refine the same piecewise-linear forcing. Interpolated
    observations are container scaffolding only, never used for fitting or scoring.
    """
    begun = monotonic()
    data = TrainingOnlySplit.model_validate(payload["training"])
    system = SymbolicODE(
        public._lower(PublicFitRequest.model_validate(payload["request"]))[0],
        allow_piecewise=True,
    )
    vector = np.asarray([payload["start"][n] for n in system.names])
    expanded = data.model_dump(mode="json")
    for row in expanded["rows"]:
        original = np.asarray(row["time"])
        grid = np.asarray(payload["meshes"][row["trajectory_id"]])
        internal = grid[:-1] + np.diff(grid) / 3
        times = np.unique(np.r_[original, grid, internal])
        for kind in ("targets", "external_inputs", "auxiliaries"):
            row[kind] = {
                n: np.interp(times, original, values).tolist()
                for n, values in row.get(kind, {}).items()
            }
        row["time"] = times.tolist()
    expanded["fingerprint"] = public.content_sha256(expanded["rows"])
    training = public.unpack_split(TrainingOnlySplit.model_validate(expanded))
    nodes, inner = {}, {}
    for row in training.trajectories:
        _, _, states, _ = symbolic_rollout(
            system,
            row,
            vector,
            SETTINGS.model_copy(update={"integration_method": "Radau"}),
            begun + payload["seconds"],
        )
        grid = np.asarray(payload["meshes"][row.trajectory_id])
        nodes[row.trajectory_id] = states[np.searchsorted(row.time, grid)].tolist()
        inner[row.trajectory_id] = states[
            np.searchsorted(row.time, grid[:-1] + np.diff(grid) / 3)
        ].tolist()
    result = {
        "nodes": nodes,
        "inner_nodes": inner,
        "seconds": monotonic() - begun,
        "source": "training_fitted_model_rollout",
        "payload_sha256": public.content_sha256(payload),
    }
    public._write(directory / "nodes.json", result)
    return result


def assisted_fit(payload: dict, directory: Path) -> dict:
    """Two new graphs, shared outer budget; no hidden trajectories or truth starts."""
    from autoformalism.fitting.transcription_fit import StrategyPolicy, invoke

    begun = monotonic()
    policy = DiagnosticPolicy.model_validate(payload["screening_diagnostic"])
    fit_policy = StrategyPolicy.model_validate(payload["policy"])
    deadline = begun + fit_policy.seconds - 1
    source = payload["assisted_start"]
    if not source["eligible"]:
        raise ValueError("ineligible assisted source must not launch a worker")
    parameters = source["parameters"]
    best, calls = screen(
        payload,
        directory,
        "assisted-input",
        [{"parameters": parameters, "source": "assisted_input"}],
        deadline=min(deadline, monotonic() + policy.point_seconds),
        maximum=fit_policy.maximum_rollout_calls,
        best=None,
    )
    record = {
        "source": source,
        "phases": [],
        "graph_builds_planned": 2,
        "node_origin": "integration_of_training_fitted_endpoint",
        "hidden_reference_trajectories_used": False,
    }
    public._write(directory / "assisted.json", record)

    def finish(reason):
        result = {
            "arm": payload["arm"],
            "parameters": best["parameters"] if best else None,
            "training_nmse": best["training_nmse"] if best else None,
            "selected_source": best.get("source") if best else None,
            "stop_reason": reason,
            "seconds": monotonic() - begun,
            "budget_exhausted": monotonic() >= deadline,
            "actual_residual_calls": calls,
            "assisted": record,
            "selection": "best_complete_training_rollout_including_assisted_input",
            "validation_used_for_fitting": False,
            "reference_values_used": False,
        }
        public._write(directory / "result.json", result)
        return result

    if best is None or best["training_nmse"] > policy.eligible_training_nmse:
        return finish("assisted_input_verification_failed")
    training = public.unpack_split(
        TrainingOnlySplit.model_validate(payload["training"])
    )
    system = SymbolicODE(
        public._lower(PublicFitRequest.model_validate(payload["request"]))[0],
        allow_piecewise=True,
    )
    reduced = payload["arm"] == "fixed_reduced_collocation"
    grids = {
        r.trajectory_id: mesh.initial_mesh(r, "collocation")
        for r in training.trajectories
    }
    if reduced:
        planned, audit = plan_meshes(
            system,
            training,
            payload["numerical_diagnostic"]["target_variables"],
            minimum_intervals=payload["numerical_diagnostic"]["minimum_intervals"],
        )
        grids = {
            r.trajectory_id: p.time.tolist()
            for r, p in zip(training.trajectories, planned, strict=True)
        }
        public._write(directory / "mesh_audit.json", audit)
    native = {k: payload[k] for k in ("request", "training", "coordinates")} | {
        "start": parameters,
        "meshes": grids,
        "method": "collocation",
        "tolerance": 1e-7,
        "hessian_approximation": "exact",
        "collocation_dense_output": reduced,
        "checkpoint_mode": "compact",
        "retain_inner_nodes": True,
    }
    allowance = min(policy.node_seconds, deadline - monotonic())
    node_payload = native | {"seconds": allowance}
    node_dir = directory / "rollout-nodes"
    record["node_process"] = invoke(
        node_payload, node_dir, allowance, native=True, worker_mode="nodes"
    )
    public._write(directory / "assisted.json", record)
    if not (node_dir / "nodes.json").exists():
        return finish("rollout_node_generation_unavailable")
    generated = public._read(node_dir / "nodes.json")
    if generated["payload_sha256"] != public.content_sha256(node_payload):
        raise ValueError("node generation identity differs")
    native.update({k: generated[k] for k in ("nodes", "inner_nodes")})
    for fixed, phase in ((True, "fixed"), (False, "released")):
        allowance = (
            min(policy.fixed_seconds, deadline - monotonic() - 5)
            if fixed
            else min(
                0.75 * (deadline - monotonic()),
                deadline - monotonic() - payload.get("final_screen_reserve_seconds", 0),
            )
        )
        if allowance <= 1:
            return finish("assisted_phase_budget_exhausted")
        folder = directory / phase
        process = invoke(
            native | {"seconds": allowance, "fixed_parameters": fixed},
            folder,
            allowance,
            native=True,
        )
        detail = (
            public._read(folder / "final_checkpoint_diagnostics.json")
            if (folder / "final_checkpoint_diagnostics.json").exists()
            else None
        )
        stage = {
            "phase": phase,
            "initial_nodes": public._read(folder / "initial_node_diagnostics.json")
            if (folder / "initial_node_diagnostics.json").exists()
            else None,
            "process": process,
            "native": public._read(folder / "native.json")
            if (folder / "native.json").exists()
            else None,
            "layout": public._read(folder / "layout.json")
            if (folder / "layout.json").exists()
            else None,
            "final": {
                k: detail[k]
                for k in (
                    "parameters",
                    "collocation_nmse",
                    "maximum_scaled_defect",
                    "iteration",
                )
            }
            if detail
            else None,
        }
        record["phases"].append(stage)
        public._write(directory / "assisted.json", record)
        if fixed:
            if detail is None:
                return finish("fixed_stage_no_final_nodes")
            # Exact fitted coefficients and initials are still fixed; transfer only
            # primal node values, including Radau internal points. No dual transfer.
            native.update(
                {
                    kind: {k: v[kind] for k, v in detail["trajectories"].items()}
                    for kind in ("nodes", "inner_nodes")
                }
            )
        else:
            pool_path = folder / "checkpoints.json"
            pool = public._read(pool_path)["pool"] if pool_path.exists() else []
            points = (
                [detail | {"source": "released_final"}] if detail else []
            ) + sorted(
                pool,
                key=lambda p: (
                    p["maximum_scaled_defect"] > 1e-6,
                    p["collocation_nmse"],
                ),
            )
            unique, seen = [], set()
            for point in points:
                digest = public.content_sha256(point["parameters"])
                if digest not in seen:
                    unique.append(point)
                    seen.add(digest)
            best, calls = screen(
                payload,
                directory,
                "released-pool",
                unique,
                deadline=deadline,
                maximum=fit_policy.maximum_rollout_calls,
                best=best,
            )
    record["retained_assisted_input"] = bool(
        best and best.get("source") == "assisted_input"
    )
    record["parameter_vector_changed"] = bool(best and best["parameters"] != parameters)
    public._write(directory / "assisted.json", record)
    return finish("assisted_diagnostic_complete")
