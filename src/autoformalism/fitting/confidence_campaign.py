"""M19 portable, identity-bound diagnostics of all twenty-four M18 endpoints."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import monotonic

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import confidence_fit, screening_replay
from autoformalism.fitting import public_fitting as public

PROTOCOL = "phase-c-fitting-confidence-1"
INPUT_PROTOCOL = "phase-c-fitting-confidence-inputs-1"
SOURCE_INPUTS = "4e009057d899bb227303438fc4ba724d1fd25fd7b4b2552bb74d2d6dc29c380f"
ConfidencePolicy = checks.ConfidencePolicy


def export(source: Path, output: Path) -> dict:
    """Read the completed M18 root, retaining every method/start without filtering."""
    plan, data = read_seal(source / "plan.json"), read_seal(source / "inputs.json")
    if (
        plan["protocol"] != "phase-c-larger-coupled-1"
        or public.content_sha256(data) != SOURCE_INPUTS
    ):
        raise ValueError("requires exact M18 source inputs")
    if plan["inputs_sha256"] != SOURCE_INPUTS or len(plan["tasks"]) != 24:
        raise ValueError("M18 roster or input identity differs")
    summary = public._read(source / "summary.json")
    if (
        summary["plan_sha256"] != public.content_sha256(plan)
        or summary["status"] != "complete"
    ):
        raise ValueError("incomplete or mismatched source summary")
    endpoints = []
    for task in plan["tasks"]:
        row = next(r for r in summary["rows"] if r["task_id"] == task["task_id"])
        backend = read_seal(source / "results" / task["task_id"] / "backend.json")
        if (
            row["backend_sha256"] != public.content_sha256(backend)
            or not row["selected"]
        ):
            raise ValueError("missing or changed source endpoint")
        if row["selected"]["parameters"] != backend["selected"]["parameters"]:
            raise ValueError("selected vector differs from backend")
        endpoints.append(
            {
                **task,
                "parameters": row["selected"]["parameters"],
                "source_backend_sha256": row["backend_sha256"],
                "source_fit_seconds": row["fit_seconds"],
                "source_evaluation": row["evaluation"],
            }
        )
    result = {
        "protocol": INPUT_PROTOCOL,
        "source_plan_sha256": public.content_sha256(plan),
        "data": data,
        "endpoints": endpoints,
        "test_data_opened": False,
    }
    seal(output, result)
    return {"inputs_sha256": public.content_sha256(result), "tasks": len(endpoints)}


def training_problem(data: dict, endpoint: dict) -> checks.TrainingProblem:
    """Only approved training fields cross into reliability/profile code."""
    base = data["data"]["commons"][endpoint["common"]]
    return checks.TrainingProblem.model_validate(
        {
            "request": deepcopy(base["request"]),
            "coordinates": deepcopy(base["coordinates"]),
            "start": deepcopy(base["start"]),
            "incumbent": deepcopy(endpoint["parameters"]),
            "training": deepcopy(data["data"]["cases"][base["case"]]["training"]),
        }
    )


def prepare(root: Path, inputs: Path, policy: ConfidencePolicy) -> dict:
    data = read_seal(inputs)
    if data["protocol"] != INPUT_PROTOCOL or data["test_data_opened"] is not False:
        raise ValueError("requires a development confidence export")
    if (
        public.content_sha256(data["data"]) != SOURCE_INPUTS
        or len(data["endpoints"]) != 24
    ):
        raise ValueError("changed inputs or incomplete source roster")
    tasks = [{k: e[k] for k in ("task_id", "common", "arm")} for e in data["endpoints"]]
    expected = {
        f"{key}_{arm}"
        for key in data["data"]["commons"]
        for arm in ("rollout_only", "coupled_profiled_rollout")
    }
    if {t["task_id"] for t in tasks} != expected:
        raise ValueError("source task roster differs")
    for endpoint in data["endpoints"]:
        training_problem(data, endpoint)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": tasks,
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(tasks)}


def verify(root: Path, *, runtime=True):
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    if plan["protocol"] != PROTOCOL or plan["inputs_sha256"] != public.content_sha256(
        data
    ):
        raise ValueError("campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("source/runtime differs")
    return plan, data


def run_task(root: Path, index: int):
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown task")
    task, endpoint = plan["tasks"][index], data["endpoints"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"]
    policy = ConfidencePolicy.model_validate(plan["policy"])
    with public._lock(folder):
        if (folder / "result.json").exists():
            result = read_seal(folder / "result.json")
            if result["identity"] != identity:
                raise ValueError("result identity differs")
            return result
        backend = confidence_fit.run(
            training_problem(data, endpoint), policy, folder / "fit"
        )
        # This seal must exist before the evaluator sees reference/validation data.
        seal(folder / "backend.json", backend)
        base = data["data"]["commons"][task["common"]]
        evaluation_data = {
            "case": data["data"]["cases"][base["case"]],
            "case_name": base["case"],
            "config": data["data"]["config"],
        }
        stages = {}
        begun = monotonic()
        for key in ("after_reliability", "selected"):
            selected = backend.get(key)
            stages[key] = screening_replay._evaluation(
                folder / f"evaluation-{key}",
                evaluation_data,
                base,
                selected["parameters"] if selected else None,
                policy.replay_seconds,
            )
        result = {
            **task,
            "identity": identity,
            "case": base["case"],
            "seed": base["seed"],
            "status": backend["status"],
            "source_fit_seconds": endpoint["source_fit_seconds"],
            "before": endpoint["source_evaluation"],
            "evaluations": stages,
            "backend_sha256": public.content_sha256(backend),
            "confidence": backend.get("confidence"),
            "additional_wall_seconds": backend.get("additional_wall_seconds"),
            "additional_cpu_seconds": backend.get("additional_cpu_seconds"),
            "cost_complete": backend.get("cost_complete", False),
            "evaluation_seconds": monotonic() - begun,
        }
        seal(folder / "result.json", result)
        return result


def report(root: Path):
    plan, _ = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        path = root / "results" / task["task_id"] / "result.json"
        row = read_seal(path) if path.exists() else task | {"status": "missing"}
        if path.exists():
            if row["identity"] != {
                "plan_sha256": public.content_sha256(plan),
                "task": task,
            }:
                raise ValueError("report row identity differs")
            backend = read_seal(path.with_name("backend.json"))
            if row["backend_sha256"] != public.content_sha256(backend):
                raise ValueError("backend changed")
        rows.append(row)
    groups = {}
    for arm in ("rollout_only", "coupled_profiled_rollout"):
        selected_rows = [r for r in rows if r["arm"] == arm]
        group = {"expected": len(selected_rows)}
        for stage in ("before", "after_reliability", "selected"):
            evaluations = [
                r.get("before", {})
                if stage == "before"
                else r.get("evaluations", {}).get(stage, {})
                for r in selected_rows
            ]
            group[stage] = {
                key: sum(bool(e.get(key)) for e in evaluations)
                for key in (
                    "accuracy_passed",
                    "coefficients_recovered",
                    "initials_recovered",
                )
            }
        group["additional_wall_seconds_observed"] = sum(
            r.get("additional_wall_seconds") or 0 for r in selected_rows
        )
        group["additional_cpu_seconds_observed"] = sum(
            r.get("additional_cpu_seconds") or 0 for r in selected_rows
        )
        group["cost_complete"] = all(
            r.get("cost_complete", False) for r in selected_rows
        )
        groups[arm] = group
    counts = dict(Counter(r["status"] for r in rows))
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if "missing" not in counts else "incomplete",
        "expected": len(rows),
        "recorded": sum(r["status"] != "missing" for r in rows),
        "status_counts": counts,
        "rows": rows,
        "groups": groups,
        "confidence_counts": dict(
            Counter(
                (r.get("confidence") or {}).get("level", "unavailable") for r in rows
            )
        ),
        "confidence_calibration": dict(
            Counter(
                str(
                    (
                        (r.get("confidence") or {}).get("level", "unavailable"),
                        r.get("evaluations", {})
                        .get("selected", {})
                        .get("coefficients_recovered"),
                    )
                )
                for r in rows
            )
        ),
        "test_data_opened": False,
        "limitation": (
            "Noiseless correct-skeleton numerical profiles; no coverage probability. "
            "Local profile minima are upper bounds. Additional-budget recovery "
            "is not a matched speed comparison. Nonlinear cases remain later work."
        ),
    }
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
