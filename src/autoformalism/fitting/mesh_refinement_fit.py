"""Training-only fixed-resolution and coarse-to-fine M11 experiments."""

from __future__ import annotations

from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import mesh_refinement_process as process
from autoformalism.fitting import mesh_refinement_worker as worker
from autoformalism.fitting import public_fitting as public


def point(base: dict, parameters: dict, folder: Path, seconds: float) -> dict:
    """Only a complete, matching full-training evaluation is eligible to rank."""
    payload = {k: base[k] for k in ("request", "training")} | {
        "parameters": parameters,
        "seconds": seconds,
        "screening": {"method": "Radau", "point_seconds": seconds},
    }
    outcome = process.invoke("point", payload, folder, seconds)
    path = folder / "result.json"
    record = public._read(path) if path.exists() else {}
    if record and record.get("payload_sha256") != public.content_sha256(payload):
        raise ValueError("training screen identity differs")
    complete = (
        outcome["termination_confirmed"]
        and outcome["status"] == "complete"
        and record.get("status") == "complete"
    )
    if complete and record["parameters"] != parameters:
        raise ValueError("training screen vector differs")
    return {
        "status": "complete"
        if complete
        else outcome["status"]
        if outcome["status"] != "complete"
        else "unavailable",
        "training_nmse": record["training_nmse"] if complete else None,
        "process": outcome,
    }


def _native(payload: dict, folder: Path) -> dict:
    """Retain final nodes and all native checkpoints; convergence is separate."""
    outcome = process.invoke("native", payload, folder, payload["seconds"])

    def read(name):
        path = folder / name
        return public._read(path) if path.exists() else None

    detail = read("final_checkpoint_diagnostics.json")
    native = read("native.json")
    result = {
        "process": outcome,
        "native": native,
        "layout": read("layout.json"),
        "initial_nodes": read("initial_node_diagnostics.json"),
        "detail_sha256": public.content_sha256(detail) if detail else None,
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
        "maximum_off_node_defect": max(
            (
                max(t["indicators"], default=0.0)
                for t in detail["trajectories"].values()
            ),
            default=0.0,
        )
        if detail
        else None,
    }
    result["qualified"] = bool(
        outcome["status"] == "complete"
        and outcome["termination_confirmed"]
        and native
        and native.get("native_success")
        and detail
        and detail["maximum_scaled_defect"] <= 1e-6
    )
    return result


def _read_detail(folder: Path, phase: str, record: dict) -> dict:
    detail = public._read(folder / phase / "final_checkpoint_diagnostics.json")
    if public.content_sha256(detail) != record["detail_sha256"]:
        raise ValueError("native node checkpoint differs")
    return detail


def solve_level(
    base: dict, folder: Path, target: int | None, policy: dict, previous: dict | None
) -> dict:
    """One bounded fixed-then-released solve, with a distinct final rollout score."""
    identity = {
        "base_sha256": public.content_sha256(base),
        "target": target,
        "policy": policy,
        "previous_sha256": public.content_sha256(previous),
    }
    saved = folder / "level.json"
    if saved.exists():
        result = read_seal(saved)
        if result["identity"] != identity:
            raise ValueError("mesh level identity differs")
        for phase in ("fixed", "released"):
            if result.get(phase, {}).get("detail_sha256"):
                _read_detail(folder, phase, result[phase])
        return result
    folder.mkdir(parents=True, exist_ok=True)
    meshes, audit = worker.grids(base, target, policy["minimum_intervals"])
    result = {
        "identity": identity,
        "mesh": audit,
        "status": "started",
        "transfer": "old_radau_polynomial" if previous else "fitted_model_rollout",
        "advance_qualified": False,
    }

    def finish(status):
        result["status"] = status
        seal(saved, result)
        return result

    parameters = previous["parameters"] if previous else base["source"]["parameters"]
    if previous:
        guesses = worker.transfer(previous, meshes)
    else:
        payload = worker.native_payload(base, parameters, meshes, {}) | {
            "seconds": policy["node_seconds"]
        }
        node_dir = folder / "rollout-nodes"
        result["node_process"] = process.invoke(
            "nodes", payload, node_dir, policy["node_seconds"]
        )
        if result["node_process"]["status"] != "complete":
            return finish("node_generation_unavailable")
        generated = public._read(node_dir / "nodes.json")
        if generated["payload_sha256"] != public.content_sha256(payload):
            raise ValueError("rollout node identity differs")
        guesses = {k: generated[k] for k in ("nodes", "inner_nodes")}
    native = worker.native_payload(base, parameters, meshes, guesses)
    public._write(folder / "mesh_audit.json", audit)
    for fixed, phase in ((True, "fixed"), (False, "released")):
        payload = native | {
            "seconds": policy[f"{phase}_seconds"],
            "fixed_parameters": fixed,
        }
        result[phase] = _native(payload, folder / phase)
        if not result[phase]["process"]["termination_confirmed"]:
            return finish("cleanup_or_interruption_stop")
        if not result[phase]["detail_sha256"]:
            return finish(f"{phase}_endpoint_unavailable")
        detail = _read_detail(folder, phase, result[phase])
        if fixed:
            if not result[phase]["qualified"]:
                return finish("fixed_solve_unqualified")
            native.update(
                {
                    kind: {k: t[kind] for k, t in detail["trajectories"].items()}
                    for kind in ("nodes", "inner_nodes")
                }
            )
    result["endpoint_parameters"] = detail["parameters"]
    result["endpoint_screen"] = point(
        base, detail["parameters"], folder / "screen", policy["point_seconds"]
    )
    result["advance_qualified"] = bool(
        result["released"]["qualified"]
        and result["endpoint_screen"]["status"] == "complete"
    )
    return finish(
        "complete" if result["advance_qualified"] else "released_solve_unqualified"
    )


def fit(base: dict, task: dict, policy: dict, folder: Path) -> dict:
    """Complete all training decisions before any validation/reference evaluation.

    The continuation path follows each converged endpoint even if its rollout is
    worse than the incumbent. Otherwise the diagnostic would silently restart at
    the assisted source and would not test coarse-to-fine transfer. Deployment
    selection is separate and always preserves the best complete training score.
    """
    initial = point(
        base,
        base["source"]["parameters"],
        folder / "source-screen",
        policy["point_seconds"],
    )
    result = {
        "source_screen": initial,
        "levels": [],
        "selected": None,
        "validation_used_for_fitting": False,
        "reference_values_used": False,
        "source_fit_seconds": base["source"]["source_seconds"],
    }
    if initial["status"] != "complete" or initial["training_nmse"] > 1e-8:
        return result | {"stop_reason": "source_recheck_failed"}
    result["selected"] = {
        "parameters": base["source"]["parameters"],
        "training_nmse": initial["training_nmse"],
        "origin": "supplied_M7",
    }
    previous = None
    for level in task["levels"]:
        path = folder / f"level-{level}"
        record = solve_level(base, path, policy["targets"][level], policy, previous)
        result["levels"].append({"level": level, **record})
        score = record.get("endpoint_screen", {})
        if (
            score.get("status") == "complete"
            and score["training_nmse"] < result["selected"]["training_nmse"]
        ):
            result["selected"] = {
                "parameters": record["endpoint_parameters"],
                "training_nmse": score["training_nmse"],
                "origin": f"level-{level}",
            }
        public._write(folder / "best.json", result["selected"])
        public._write(folder / "progress.json", result)
        if not record["advance_qualified"]:
            return result | {"stop_reason": "unqualified_level_no_refinement"}
        previous = _read_detail(path, "released", record["released"])
    return result | {"stop_reason": "planned_levels_complete"}
