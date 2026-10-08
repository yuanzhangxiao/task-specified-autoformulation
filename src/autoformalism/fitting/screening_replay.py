"""M10: calibrate training screens, replay saved pools, and retry assisted fits."""

from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import monotonic

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import bounded_screening as bounded
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import transcription_campaign as campaign
from autoformalism.fitting import transcription_fit
from autoformalism.fitting.screening_replay_export import PROTOCOL as INPUT_PROTOCOL
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "phase-c-screening-replay-1"
METHODS = ("RK45", "Radau")


def _safe_name(value: str) -> bool:
    return bool(value) and all(c.isalnum() or c in "_-" for c in value)


class ReplayPolicy(StrictSchema):
    """Calibrate before pool scoring; all ceilings include setup and I/O."""

    calibration_seconds: float = Field(default=120, ge=1, le=300)
    minimum_point_seconds: float = Field(default=60, ge=1, le=300)
    maximum_point_seconds: float = Field(default=300, ge=1, le=300)
    time_multiplier: float = Field(default=2, ge=1, le=5)
    margin_seconds: float = Field(default=10, ge=0, le=60)
    pool_seconds: float = Field(default=900, ge=5, le=1800)
    assisted_seconds: float = Field(default=900, ge=5, le=1200)
    node_seconds: float = Field(default=180, ge=1, le=180)
    fixed_seconds: float = Field(default=250, ge=1, le=400)
    replay_seconds: float = Field(default=300, ge=1, le=600)

    @model_validator(mode="after")
    def valid_limits(self):
        if self.minimum_point_seconds > self.maximum_point_seconds:
            raise ValueError("point limits inverted")
        return self


def prepare(root: Path, inputs: Path, policy: ReplayPolicy) -> dict:
    """Freeze copies of saved evidence; never generate new benchmark arrays."""
    data = read_seal(inputs)
    if data["protocol"] != INPUT_PROTOCOL or data["test_data_opened"]:
        raise ValueError("requires a development M9 replay export")
    names = [p["id"] for p in data["calibration_points"]] + [
        p["task_id"] for p in data["pools"] + data["assisted"]
    ]
    if any(not _safe_name(n) for n in names):
        raise ValueError("unsafe replay identifier")
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "methods": list(METHODS),
        "tasks": [
            {"task_id": f"{kind}_{item['task_id']}", "kind": kind, "index": i}
            for kind, items in (("assisted", data["assisted"]), ("pool", data["pools"]))
            for i, item in enumerate(items)
        ],
        "source_plan_sha256": data["source_plan_sha256"],
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {
        "identity": public.content_sha256(plan),
        "tasks": len(plan["tasks"]),
        "calibration_methods": len(METHODS),
    }


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    if plan["protocol"] != PROTOCOL or plan["inputs_sha256"] != public.content_sha256(
        data
    ):
        raise ValueError("replay identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("replay source/runtime differs")
    return plan, data


def calibrated_limit(rows: list[dict], policy: ReplayPolicy) -> dict:
    """Runtime-only allowance; both previously qualified starts must recheck."""
    good = [r for r in rows if r["required_good_start"]]
    ready = bool(good) and all(
        r["status"] == "complete" and r["training_nmse"] <= 1e-8 for r in good
    )
    times = [r["seconds"] for r in rows if r["status"] == "complete"]
    requested = max(
        policy.minimum_point_seconds,
        math.ceil(
            policy.time_multiplier * max(times, default=0) + policy.margin_seconds
        ),
    )
    return {
        "ready": ready and requested <= policy.maximum_point_seconds,
        "point_seconds": requested
        if ready and requested <= policy.maximum_point_seconds
        else None,
        "requested_point_seconds": requested,
        "reason": (
            "qualified"
            if ready and requested <= policy.maximum_point_seconds
            else "required_source_recheck_unavailable_or_allowance_exceeds_ceiling"
        ),
    }


def calibrate(root: Path, index: int) -> dict:
    """One bounded timing pilot per method; interruptions do not restart its budget."""
    plan, data = verify(root)
    if not 0 <= index < len(METHODS):
        raise ValueError("invalid calibration index")
    policy = ReplayPolicy.model_validate(plan["policy"])
    method = METHODS[index]
    folder = root / "calibration" / method
    identity = {"plan_sha256": public.content_sha256(plan), "method": method}
    with public._lock(folder):
        if (folder / "result.json").exists():
            return _calibration(root, plan, method)
        if (folder / "started.json").exists():
            if read_seal(folder / "started.json") != identity:
                raise ValueError("calibration identity differs")
            result = {
                **identity,
                "status": "interrupted",
                "ready": False,
                "budget_restarted": False,
                "rows": public._read(folder / "progress.json")
                if (folder / "progress.json").exists()
                else [],
            }
        else:
            seal(folder / "started.json", identity)
            rows = []
            for point in data["calibration_points"]:
                common = data["commons"][point["common"]]
                payload = {
                    "request": common["request"],
                    "training": data["case"]["training"],
                    "screening": {
                        "method": method,
                        "point_seconds": policy.calibration_seconds,
                    },
                }
                target = folder / point["id"]
                best, _ = bounded.screen(
                    payload,
                    target,
                    "calibration",
                    [point],
                    deadline=monotonic() + policy.calibration_seconds,
                    maximum=1,
                    best=None,
                )
                attempts = public._read(target / "screening.json")["attempts"]
                attempt = (
                    attempts[0]
                    if attempts
                    else {
                        "status": "not_started",
                        "process": {"elapsed_seconds": policy.calibration_seconds},
                    }
                )
                row = {
                    "id": point["id"],
                    "required_good_start": point["required_good_start"],
                    "status": attempt["status"],
                    "seconds": attempt["process"]["elapsed_seconds"],
                    "training_nmse": best["training_nmse"] if best else None,
                }
                rows.append(row)
                public._write(folder / "progress.json", rows)
            result = {
                **identity,
                "status": "complete",
                "rows": rows,
                **calibrated_limit(rows, policy),
                "new_seconds": sum(r["seconds"] for r in rows),
            }
        seal(folder / "result.json", result)
    return result


def _calibration(root: Path, plan: dict, method: str) -> dict:
    result = read_seal(root / "calibration" / method / "result.json")
    if (
        result["plan_sha256"] != public.content_sha256(plan)
        or result["method"] != method
    ):
        raise ValueError("calibration identity differs")
    return result


def _pool(
    data: dict, item: dict, point_seconds: float, policy: ReplayPolicy, folder: Path
) -> dict:
    """Rescore only unevaluated vectors; a verified historical incumbent is retained."""
    begun = monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "request": data["commons"][item["common"]]["request"],
        "training": data["case"]["training"],
        "screening": {"method": item["method"], "point_seconds": point_seconds},
    }
    cached, pending = [], []
    for point in item["points"]:
        source = f"saved:{point['parameters_sha256']}"
        if point["cached"] is not None:
            cache = point["cached"]
            if (
                cache["method"] != payload["screening"]["method"]
                or cache["parameters"] != point["parameters"]
                or not math.isfinite(cache["training_nmse"])
                or cache["training_nmse"] < 0
            ):
                raise ValueError("cached screen method or vector differs")
            cached.append({**cache, "source": source})
        else:
            pending.append({"parameters": point["parameters"], "source": source})
    best = min(cached, key=lambda c: c["training_nmse"]) if cached else None
    if best:
        public._write(folder / "best.json", best)
    selected, calls = bounded.screen(
        payload,
        folder,
        "saved-pool",
        pending,
        deadline=begun + policy.pool_seconds,
        maximum=max(1, len(pending)),
        best=best,
    )
    return {
        "parameters": selected["parameters"] if selected else None,
        "training_nmse": selected["training_nmse"] if selected else None,
        "selected_source": selected.get("source") if selected else None,
        "stop_reason": "saved_pool_screened",
        "new_seconds": monotonic() - begun,
        "total_candidates": len(item["points"]),
        "reused_complete_screens": len(cached),
        "new_screen_attempts": calls,
        "unattempted": len(pending) - calls,
        "original_selected_parameters": item["source_selected_parameters"],
        "source_fit_seconds": item["source_fit_seconds"],
        "native_optimization_performed": False,
        "screening": public._read(folder / "screening.json"),
    }


def _assisted_payload(
    data: dict, item: dict, point_seconds: float, policy: ReplayPolicy
) -> dict:
    payload = deepcopy(item["payload"])
    payload.update(
        {
            k: data["commons"][item["common"]][k]
            for k in ("request", "coordinates", "nodes")
        }
    )
    payload["training"] = data["case"]["training"]
    if public.content_sha256(payload) != item["source_worker_payload_sha256"]:
        raise ValueError("assisted source payload differs")
    payload["policy"]["seconds"] = policy.assisted_seconds
    payload["screening"]["point_seconds"] = point_seconds
    payload["screening_diagnostic"].update(
        point_seconds=point_seconds,
        node_seconds=policy.node_seconds,
        fixed_seconds=policy.fixed_seconds,
    )
    payload["final_screen_reserve_seconds"] = point_seconds + 5
    return payload


def _evaluation(
    folder: Path, data: dict, common: dict, parameters: dict | None, seconds: float
) -> dict:
    """Separate validation/reference evaluation, after training selection freezes."""
    if not parameters:
        return {
            "status": "no_selected_vector",
            "training_nmse": None,
            "validation_nmse": None,
        }
    # The replay journal is written before the final sealed score. A fresh
    # evaluation stage therefore needs its directory before launching replay.
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "replay.json"
    if not path.exists():
        checked = campaign.replay(
            PublicFitRequest.model_validate(common["request"]),
            parameters,
            PublicSplit.model_validate(data["case"]["training"]),
            PublicSplit.model_validate(data["case"]["validation"]),
            seconds,
            journal=folder / "replay_progress.json",
        )
        seal(path, checked)
    checked = read_seal(path)
    errors = campaign.coefficient_metrics(data["case"], common["request"], parameters)
    error = errors.get("maximum_coefficient_relative_error")
    initial = errors.get("initial_parameter_absolute_errors", {})
    agreement = bool(
        checked["complete"] and checked["maximum_solver_difference"] <= 1e-4
    )
    gate = data["config"]["cstr_nmse"] if data["case_name"] != "linear" else 1e-6
    return {
        "status": "complete" if checked["complete"] else "no_complete_replay",
        "training_nmse": checked["metrics"]["train"],
        "validation_nmse": checked["metrics"]["val"],
        "accuracy_passed": agreement and max(checked["metrics"].values()) <= gate,
        "coefficient_recovery": errors,
        "coefficients_recovered": bool(
            agreement
            and error is not None
            and error <= data["config"]["parameter_relative"]
        ),
        "initials_recovered": max(initial.values())
        <= data["config"]["initial_absolute"]
        if initial
        else None,
        "evaluation_stop_reason": checked.get("stop_reason"),
    }


def run_task(root: Path, index: int) -> dict:
    """Exact terminal resume; interrupted computation never receives a fresh budget."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("invalid task index")
    policy = ReplayPolicy.model_validate(plan["policy"])
    task = plan["tasks"][index]
    item = data["pools" if task["kind"] == "pool" else "assisted"][task["index"]]
    folder = root / "results" / task["task_id"]
    common = data["commons"][item["common"]]
    method = (
        item["method"]
        if task["kind"] == "pool"
        else item["payload"]["screening"]["method"]
    )
    calibration = _calibration(root, plan, method)
    identity = {
        "plan_sha256": public.content_sha256(plan),
        "calibration_sha256": public.content_sha256(calibration),
        "task": task,
    }
    with public._lock(folder):
        path = folder / "result.json"
        if path.exists():
            saved = read_seal(path)
            if saved["identity"] != identity:
                raise ValueError("task result identity differs")
            return saved
        backend_path = folder / "backend.json"
        started = folder / "started.json"
        if started.exists() and read_seal(started) != identity:
            raise ValueError("task identity differs")
        if backend_path.exists() and not started.exists():
            raise ValueError("backend has no recorded task start")
        if not calibration["ready"]:
            result = {**task, "status": "calibration_blocked", "identity": identity}
        elif started.exists() and not backend_path.exists():
            if read_seal(started) != identity:
                raise ValueError("task identity differs")
            result = {
                **task,
                "status": "interrupted",
                "budget_restarted": False,
                "identity": identity,
            }
        else:
            if not backend_path.exists():
                seal(started, identity)
                if task["kind"] == "pool":
                    backend = _pool(
                        data, item, calibration["point_seconds"], policy, folder / "fit"
                    )
                else:
                    payload = _assisted_payload(
                        data, item, calibration["point_seconds"], policy
                    )
                    backend = transcription_fit.run(payload, folder / "fit")
                    backend["new_seconds"] = backend["total_seconds"]
                seal(backend_path, backend)
            backend = read_seal(backend_path)
            result = {
                **task,
                "identity": identity,
                "common": item["common"],
                "seed": common["seed"],
                "method": method,
                "source_task": item["task_id"],
                "backend_sha256": public.content_sha256(backend),
                "parameters": backend.get("parameters"),
                "new_seconds": backend["new_seconds"],
                "stop_reason": backend["stop_reason"],
                "selected_source": backend.get("selected_source"),
                **_evaluation(
                    folder,
                    data,
                    common,
                    backend.get("parameters"),
                    policy.replay_seconds,
                ),
            }
            if task["kind"] == "pool":
                result.update(
                    {
                        k: backend[k]
                        for k in (
                            "total_candidates",
                            "reused_complete_screens",
                            "new_screen_attempts",
                            "unattempted",
                            "source_fit_seconds",
                        )
                    }
                )
                result["selected_vector_changed"] = (
                    backend.get("parameters") != item["source_selected_parameters"]
                )
            else:
                result["assisted"] = backend.get("assisted")
                phases = (backend.get("assisted") or {}).get("phases", [])
                released = next(
                    (p.get("final") for p in phases if p["phase"] == "released"), None
                )
                if released:
                    result["released_final_coefficient_recovery"] = (
                        campaign.coefficient_metrics(
                            data["case"], common["request"], released["parameters"]
                        )
                    )
                    attempts = (backend.get("screening") or {}).get("attempts", [])
                    final_screen = next(
                        (a for a in attempts if a.get("source") == "released_final"), {}
                    )
                    result["released_final_training_nmse"] = final_screen.get(
                        "training_nmse"
                    )
                    result["released_final_screen_status"] = final_screen.get(
                        "status", "not_started"
                    )
        seal(path, result)
    return result


def report(root: Path) -> dict:
    """Report all planned outcomes; calibration and upstream costs stay separate."""
    plan, data = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        path = root / "results" / task["task_id"] / "result.json"
        row = read_seal(path) if path.exists() else {**task, "status": "missing"}
        if path.exists() and (
            row["identity"]["plan_sha256"] != public.content_sha256(plan)
            or row["identity"]["task"] != task
        ):
            raise ValueError("reported task identity differs")
        if row.get("backend_sha256") and row["backend_sha256"] != public.content_sha256(
            read_seal(path.parent / "backend.json")
        ):
            raise ValueError("backend digest differs")
        rows.append(row)
    calibration = {}
    for method in METHODS:
        path = root / "calibration" / method / "result.json"
        calibration[method] = (
            _calibration(root, plan, method) if path.exists() else {"status": "missing"}
        )
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete"
        if all(r["status"] != "missing" for r in rows)
        and all(r["status"] != "missing" for r in calibration.values())
        else "incomplete",
        "expected": len(rows),
        "recorded": sum(r["status"] != "missing" for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "calibration": calibration,
        "rows": rows,
        "excluded_assisted": data["excluded_assisted"],
        "source_plan_sha256": data["source_plan_sha256"],
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": (
            "Extra training-only screening of fixed M9 pools is not new optimizer "
            "recovery. Assisted runs use training-fitted source vectors; retained "
            "source accuracy is not collocation recovery. Calibration, source fitting "
            "and new work costs are separate. No production promotion."
        ),
    }
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
