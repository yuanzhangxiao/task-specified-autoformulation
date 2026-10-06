"""M11: assisted mesh convergence and coarse-to-fine continuation on frozen data."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import mesh_refinement_fit as fitting
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import screening_replay as replay
from autoformalism.fitting.bounded_screening import TrainingOnlySplit
from autoformalism.fitting.screening_replay_export import PROTOCOL as INPUT_PROTOCOL
from autoformalism.schemas.base import StrictSchema

PROTOCOL = "phase-c-mesh-refinement-1"


class MeshPolicy(StrictSchema):
    """Controlled nested ladder with a soft size target and exact forcing."""

    targets: tuple[int | None, ...] = (6000, 60000, 90000, None)
    minimum_intervals: int = Field(default=24, ge=1, le=200)
    node_seconds: float = Field(default=180, ge=1, le=180)
    fixed_seconds: float = Field(default=250, ge=1, le=400)
    released_seconds: float = Field(default=250, ge=1, le=400)
    point_seconds: float = Field(default=120, ge=1, le=300)
    replay_seconds: float = Field(default=300, ge=1, le=600)

    @model_validator(mode="after")
    def valid_ladder(self):
        if (
            not 2 <= len(self.targets) <= 4
            or self.targets[-1] is not None
            or any(t is None or t < 1 for t in self.targets[:-1])
            or list(self.targets[:-1]) != sorted(set(self.targets[:-1]))
        ):
            raise ValueError("require increasing positive targets followed by dense")
        return self


def bases(data: dict) -> dict:
    """Reconstruct and verify every assisted source before forming worker allowlists."""
    if data["protocol"] != INPUT_PROTOCOL or data["test_data_opened"] is not False:
        raise ValueError("requires sealed development screening export")
    if data["case_name"] not in {"alien_hard", "linear"}:
        raise ValueError("unqualified mesh diagnostic case")
    TrainingOnlySplit.model_validate(data["case"]["training"])
    result = {}
    for item in data["assisted"]:
        if not replay._safe_name(item["common"]):
            raise ValueError("unsafe common identity")
        original = replay._assisted_payload(data, item, 120, replay.ReplayPolicy())
        source = original["assisted_start"]
        if not (
            source["eligible"]
            and source["training_verified"]
            and source["parameters"]
            and source["training_nmse"] <= 1e-8
        ):
            raise ValueError("unqualified fitted assistance")
        value = {k: original[k] for k in ("request", "training", "coordinates")} | {
            "source": source
        }
        if item["common"] in result and result[item["common"]] != value:
            raise ValueError("assisted arms disagree about their source")
        result[item["common"]] = value
    if not result:
        raise ValueError("no eligible sources")
    return result


def prepare(root: Path, inputs: Path, policy: MeshPolicy) -> dict:
    """Freeze a diagnostic using existing arrays; no generation or optimizer work."""
    data = read_seal(inputs)
    starts = bases(data)
    tasks = []
    for common in starts:
        for level in range(len(policy.targets)):
            tasks.append(
                {
                    "task_id": f"{common}_independent_{level}",
                    "common": common,
                    "arm": "independent",
                    "levels": [level],
                }
            )
        tasks.append(
            {
                "task_id": f"{common}_continuation",
                "common": common,
                "arm": "continuation",
                "levels": list(range(len(policy.targets))),
            }
        )
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": tasks,
        "source_plan_sha256": data["source_plan_sha256"],
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {
        "identity": public.content_sha256(plan),
        "tasks": len(tasks),
        "sources": len(starts),
        "levels": len(policy.targets),
    }


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    if plan["protocol"] != PROTOCOL or plan["inputs_sha256"] != public.content_sha256(
        data
    ):
        raise ValueError("mesh plan/input identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("mesh source/runtime differs")
    return plan, data


def _evaluate(
    folder: Path, data: dict, common: str, parameters: dict, seconds: float
) -> dict:
    """Identity-bound independent evaluation only after the entire fit is frozen."""
    with public._lock(folder):
        seal(
            folder / "identity.json",
            {
                "inputs_sha256": public.content_sha256(data),
                "common": common,
                "parameters": parameters,
                "seconds": seconds,
            },
        )
        return replay._evaluation(
            folder, data, data["commons"][common], parameters, seconds
        )


def run_task(root: Path, index: int) -> dict:
    """Resume completed operations; a started operation never gets another allowance."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("invalid mesh task index")
    task = plan["tasks"][index]
    folder = root / "results" / task["task_id"]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    with public._lock(folder):
        output = folder / "result.json"
        if output.exists():
            saved = read_seal(output)
            if saved["identity"] != identity or saved[
                "backend_sha256"
            ] != public.content_sha256(read_seal(folder / "backend.json")):
                raise ValueError("mesh task result identity differs")
            return saved
        seal(folder / "started.json", identity)
        path = folder / "backend.json"
        if not path.exists():
            backend = fitting.fit(
                bases(data)[task["common"]], task, plan["policy"], folder / "fit"
            )
            seal(path, {"identity": identity, **backend})
        backend = read_seal(path)
        if backend["identity"] != identity:
            raise ValueError("mesh backend identity differs")
        processes = [read_seal(p) for p in (folder / "fit").rglob("process.json")]
        safe_to_evaluate = all(p["termination_confirmed"] for p in processes)
        # All resolutions/continuation decisions are now immutable. No evaluation
        # output is passed back to fit(), refinement, transfer or best selection.
        rows = []
        for stage in backend["levels"]:
            row = {
                k: stage[k]
                for k in ("level", "status", "mesh", "transfer", "advance_qualified")
            }
            for phase in ("fixed", "released"):
                if phase in stage:
                    row[phase] = stage[phase]
            row["training_screen"] = stage.get("endpoint_screen")
            params = stage.get("endpoint_parameters")
            row["endpoint_evaluation"] = (
                _evaluate(
                    folder / f"evaluation-level-{stage['level']}",
                    data,
                    task["common"],
                    params,
                    plan["policy"]["replay_seconds"],
                )
                if params and safe_to_evaluate
                else None
            )
            rows.append(row)
        selected = backend.get("selected")
        selected_evaluation = (
            _evaluate(
                folder / "evaluation-selected",
                data,
                task["common"],
                selected["parameters"],
                plan["policy"]["replay_seconds"],
            )
            if selected and safe_to_evaluate
            else None
        )
        result = {
            **task,
            "identity": identity,
            "backend_sha256": public.content_sha256(backend),
            "status": "complete"
            if backend["stop_reason"] == "planned_levels_complete"
            else "incomplete_fit",
            "stop_reason": backend["stop_reason"],
            "levels": rows,
            "selected": selected,
            "selected_evaluation": selected_evaluation,
            "evaluation_blocked_by_cleanup": not safe_to_evaluate,
            "evaluations_complete": bool(
                selected_evaluation
                and selected_evaluation["status"] == "complete"
                and len(rows) == len(task["levels"])
                and all(
                    r["endpoint_evaluation"]
                    and r["endpoint_evaluation"]["status"] == "complete"
                    for r in rows
                )
            ),
            "source_fit_seconds": backend["source_fit_seconds"],
            "new_process_seconds": sum(p.get("elapsed_seconds", 0) for p in processes),
            "cost_complete": all("elapsed_seconds" in p for p in processes),
        }
        seal(output, result)
    return result


def report(root: Path) -> dict:
    """Separate numerical solves, endpoint recovery and retained incumbent accuracy."""
    plan, _ = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        path = root / "results" / task["task_id"] / "result.json"
        row = read_seal(path) if path.exists() else {**task, "status": "missing"}
        if path.exists() and (
            row["identity"]
            != {"plan_sha256": public.content_sha256(plan), "task": task}
            or row["backend_sha256"]
            != public.content_sha256(read_seal(path.parent / "backend.json"))
        ):
            raise ValueError("reported mesh identity differs")
        rows.append(row)
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete"
        if all(r["status"] != "missing" for r in rows)
        else "incomplete",
        "expected": len(rows),
        "recorded": sum(r["status"] != "missing" for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": (
            "Assisted mesh convergence, not generic-start recovery. Continuation "
            "follows converged endpoints; selection preserves the best training "
            "rollout. Reference and validation are post-fit evaluation only. "
            "Independent and continuation costs differ. A fixed ladder is not "
            "yet an automatic error-driven mesh controller."
        ),
    }
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
