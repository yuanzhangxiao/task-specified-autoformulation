"""M20 portable export of all M19 endpoints and their saved profile candidates."""

from collections import Counter
from pathlib import Path
from time import monotonic

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_campaign as source
from autoformalism.fitting import profile_recovery as fitting
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import screening_replay

PROTOCOL = "phase-c-profile-recovery-1"
INPUT_PROTOCOL = "phase-c-profile-recovery-inputs-1"
SOURCE_PLAN = "318cdc03bc281a226a76d4f66f08a44166ab62f54b0d7a642eb9fc5a4fce3738"
RecoveryPolicy = fitting.RecoveryPolicy


def export(root: Path, output: Path) -> dict:
    """Require the completed frozen M19 roster; never filter by evaluation scores."""
    plan, data = source.verify(root, runtime=False)
    if public.content_sha256(plan) != SOURCE_PLAN or len(plan["tasks"]) != 24:
        raise ValueError("requires the exact 24-endpoint M19 plan")
    endpoints = []
    for i, task in enumerate(plan["tasks"]):
        folder = root / "results" / task["task_id"]
        result, backend = (
            read_seal(folder / "result.json"),
            read_seal(folder / "backend.json"),
        )
        if (
            result["identity"] != {"plan_sha256": SOURCE_PLAN, "task": task}
            or result["backend_sha256"] != public.content_sha256(backend)
            or backend != read_seal(folder / "fit/finished.json")
            or result["status"] != "complete"
        ):
            raise ValueError("incomplete or changed source endpoint")
        original = source.training_problem(data, data["endpoints"][i])
        if backend["identity"] != {
            "problem_sha256": public.content_sha256(original.model_dump(mode="json")),
            "policy": plan["policy"],
        }:
            raise ValueError("source fitting identity differs")
        grid = read_seal(folder / "fit/profile-grid.json")
        profiles = []
        for j, point in enumerate(grid["points"]):
            if any(backend["profiles"][j][k] != v for k, v in point.items()):
                raise ValueError("profile grid differs from source backend")
            path = folder / "fit/profiles" / f"{j:03d}" / "progress.json"
            progress = public._read(path) if path.exists() else None
            candidate = (progress or {}).get("best")
            if progress is not None:
                started = read_seal(path.with_name("started.json"))
                if started["parameters"] != {"anchor": grid["anchor"], "point": point}:
                    raise ValueError("profile progress operation identity differs")
            profiles.append(
                {
                    **point,
                    "candidate": candidate["parameters"] if candidate else None,
                    "progress_sha256": public.content_sha256(progress)
                    if progress
                    else None,
                }
            )
        alternatives = [
            o["value"]["parameters"]
            for o in backend["operations"]
            if o["operation"].startswith("reliability-check-") and o.get("value")
        ]
        payload = original.model_dump(mode="json") | {
            "incumbent": backend["selected"]["parameters"],
            "anchor": grid["anchor"],
            "profiles": profiles,
            "alternatives": alternatives,
        }
        problem = fitting.RecoveryInput.model_validate(payload)
        endpoints.append(
            {
                **task,
                "problem": problem.model_dump(mode="json"),
                "source_backend_sha256": public.content_sha256(backend),
                "source_result_sha256": public.content_sha256(result),
                "source_evaluation": result["evaluations"]["selected"],
                "source_cost": {
                    k: backend[k]
                    for k in (
                        "additional_wall_seconds",
                        "additional_cpu_seconds",
                        "rollout_calls_observed",
                        "cost_complete",
                    )
                },
            }
        )
    value = {
        "protocol": INPUT_PROTOCOL,
        "source_plan": plan,
        "source_data": data,
        "endpoints": endpoints,
        "test_data_opened": False,
    }
    seal(output, value)
    return {
        "inputs_sha256": public.content_sha256(value),
        "tasks": len(endpoints),
        "saved_profile_candidates": sum(
            p["candidate"] is not None
            for e in endpoints
            for p in e["problem"]["profiles"]
        ),
    }


def validate_inputs(data):
    """Bind the public arrays/roster to M19 and validate the training allowlist."""
    if (
        data["protocol"] != INPUT_PROTOCOL
        or data["test_data_opened"] is not False
        or public.content_sha256(data["source_plan"]) != SOURCE_PLAN
        or public.content_sha256(data["source_data"])
        != data["source_plan"]["inputs_sha256"]
    ):
        raise ValueError("profile recovery source identity differs")
    tasks = [{k: e[k] for k in ("task_id", "common", "arm")} for e in data["endpoints"]]
    if tasks != data["source_plan"]["tasks"] or len(tasks) != 24:
        raise ValueError("profile recovery roster differs")
    for i, e in enumerate(data["endpoints"]):
        p = fitting.RecoveryInput.model_validate(e["problem"])
        original = source.training_problem(
            data["source_data"], data["source_data"]["endpoints"][i]
        )
        for key in ("request", "training", "coordinates", "start"):
            if getattr(p, key) != getattr(original, key):
                raise ValueError("training problem differs from frozen source")
    return tasks


def prepare(root: Path, inputs: Path, policy: RecoveryPolicy):
    data = read_seal(inputs)
    tasks = validate_inputs(data)
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
        raise ValueError("profile recovery identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("profile recovery source/runtime differs")
    return plan, data


def run_task(root: Path, index: int):
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown task")
    task, endpoint = plan["tasks"][index], data["endpoints"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"]
    policy = RecoveryPolicy.model_validate(plan["policy"])
    with public._lock(folder):
        if (folder / "result.json").exists():
            saved = read_seal(folder / "result.json")
            if saved["identity"] != identity:
                raise ValueError("result identity differs")
            return saved
        backend = fitting.run(
            fitting.RecoveryInput.model_validate(endpoint["problem"]),
            policy,
            folder / "fit",
        )
        seal(folder / "backend.json", backend)
        # Reference and validation are visible only after training selection seals.
        raw = data["source_data"]["data"]
        base = raw["commons"][task["common"]]
        evaluation = {
            "case": raw["cases"][base["case"]],
            "case_name": base["case"],
            "config": raw["config"],
        }
        begun = monotonic()
        stages = {}
        for stage in ("after_saved", "selected"):
            point = backend.get(stage)
            stages[stage] = screening_replay._evaluation(
                folder / f"evaluation-{stage}",
                evaluation,
                base,
                point["parameters"] if point else None,
                policy.replay_seconds,
            )
        row = {
            **task,
            "identity": identity,
            "case": base["case"],
            "seed": base["seed"],
            "status": backend["status"],
            "backend_sha256": public.content_sha256(backend),
            "before": endpoint["source_evaluation"],
            "evaluations": stages,
            "evaluation_seconds": monotonic() - begun,
            "confidence": backend["confidence"],
            "profiles": dict(Counter(p["status"] for p in backend["profiles"])),
            "profile_refits": len(backend["profile_refit_indices"]),
            "rescue_starts": backend["rescue_starts"],
            "source_cost": endpoint["source_cost"],
            **{
                k: backend[k]
                for k in (
                    "additional_wall_seconds",
                    "additional_cpu_seconds",
                    "cost_complete",
                    "stage_costs",
                    "rollout_calls_observed",
                )
            },
        }
        seal(folder / "result.json", row)
        return row


def report(root: Path):
    plan, _ = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        path = root / "results" / task["task_id"] / "result.json"
        row = read_seal(path) if path.exists() else task | {"status": "missing"}
        if path.exists() and (
            row["identity"]
            != {"plan_sha256": public.content_sha256(plan), "task": task}
            or row["backend_sha256"]
            != public.content_sha256(read_seal(path.with_name("backend.json")))
        ):
            raise ValueError("reported endpoint identity differs")
        rows.append(row)
    recorded = [r for r in rows if r["status"] != "missing"]
    totals = {}
    for stage in ("before", "after_saved", "selected"):
        values = [
            r.get("before", {})
            if stage == "before"
            else r.get("evaluations", {}).get(stage, {})
            for r in recorded
        ]
        totals[stage] = {
            k: sum(bool(v.get(k)) for v in values)
            for k in ("accuracy_passed", "coefficients_recovered", "initials_recovered")
        }
    profiles = Counter()
    for row in recorded:
        profiles.update(row["profiles"])
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if len(recorded) == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": len(recorded),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "totals": totals,
        "profiles": dict(profiles),
        "confidence_counts": dict(Counter(r["confidence"]["level"] for r in recorded)),
        "cost": {
            k: sum(r[k] for r in recorded)
            for k in (
                "additional_wall_seconds",
                "additional_cpu_seconds",
                "rollout_calls_observed",
                "evaluation_seconds",
            )
        },
        "cost_complete": len(recorded) == len(rows)
        and all(r["cost_complete"] for r in recorded),
        "rows": rows,
        "test_data_opened": False,
        "limitation": (
            "Correct-skeleton linear controls, numerical tolerance witnesses; "
            "no statistical coverage or global exclusion. Bounded subset of "
            "nuisance refits; source budgets retained separately."
        ),
    }
    public._write(root / "summary.json", summary)
    return summary
