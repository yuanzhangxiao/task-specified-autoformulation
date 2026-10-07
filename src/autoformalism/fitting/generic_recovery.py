"""M12: all original generic starts, matched CPU budgets and post-fit evaluation."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_fit as fitting
from autoformalism.fitting import screening_replay as replay
from autoformalism.fitting.bounded_screening import TrainingOnlySplit
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest

PROTOCOL = "phase-c-generic-recovery-1"
INPUT_PROTOCOL = "phase-c-generic-recovery-inputs-1"
ARMS = ("rollout_only", "medium_rollout", "mesh_rollout")


class RecoveryPolicy(StrictSchema):
    """One fitting ceiling per arm; independent final evaluation is separate."""

    seconds: float = Field(default=1200, ge=10, le=1200)
    targets: tuple[int | None, ...] = (6000, 60000, 90000, None)
    medium_target: int = Field(default=60000, ge=1)
    minimum_intervals: int = Field(default=24, ge=1, le=200)
    observation_anchors: int = Field(default=8, ge=0, le=32)
    collocation_fraction: float = Field(default=0.75, gt=0, lt=1)
    native_seconds: float = Field(default=250, ge=1, le=600)
    point_seconds: float = Field(default=90, ge=1, le=300)
    certificate_seconds: float = Field(default=180, ge=1, le=300)
    maximum_rollout_calls: int = Field(default=900, ge=5, le=2000)
    training_nmse: float = Field(default=1e-6, gt=0)
    trajectory_nmse: float = Field(default=1e-5, gt=0)
    solver_agreement: float = Field(default=1e-5, gt=0)
    replay_seconds: float = Field(default=300, ge=1, le=600)

    @model_validator(mode="after")
    def limits(self):
        if (
            not 2 <= len(self.targets) <= 4
            or self.targets[-1] is not None
            or any(t is None or t < 1 for t in self.targets[:-1])
            or list(self.targets[:-1]) != sorted(set(self.targets[:-1]))
        ):
            raise ValueError("require increasing positive meshes then dense")
        if self.certificate_seconds >= self.seconds / 2:
            raise ValueError("reserve must leave a fitting allowance")
        return self


def export_inputs(source: Path, output: Path) -> dict:
    """Discard all fitted assistance/pools; keep every original generic start."""
    data = read_seal(source)
    if (
        data["protocol"] != "phase-c-screening-replay-inputs-1"
        or data["test_data_opened"] is not False
    ):
        raise ValueError("expected a sealed development export")
    result = {
        "protocol": INPUT_PROTOCOL,
        "source_export_sha256": public.content_sha256(data),
        "source_plan_sha256": data["source_plan_sha256"],
        "case_name": data["case_name"],
        "case": deepcopy(data["case"]),
        "config": {
            k: data["config"][k]
            for k in ("cstr_nmse", "parameter_relative", "initial_absolute")
        },
        "commons": {},
        "test_data_opened": False,
    }
    for key, value in data["commons"].items():
        common = {
            k: deepcopy(value[k])
            for k in ("request", "coordinates", "nodes", "case", "seed")
        }
        start = public._lower(PublicFitRequest.model_validate(common["request"]))[1]
        if "start" in value and value["start"] != start:
            raise ValueError("generic starting vector differs from request")
        common["start"] = start
        result["commons"][key] = common
    bases(result)
    seal(output, result)
    return {
        "inputs_sha256": public.content_sha256(result),
        "starts": len(result["commons"]),
    }


def bases(data: dict) -> dict:
    """Training-only allowlist; no reference, validation or fitted-source fields."""
    if data["protocol"] != INPUT_PROTOCOL or data["test_data_opened"] is not False:
        raise ValueError("wrong generic recovery input contract")
    case = data["case_name"]
    if case not in {"alien_hard", "linear"}:
        raise ValueError("unqualified generic recovery case")
    expected = {f"{case}_s{i}" for i in range(3 if case == "alien_hard" else 1)}
    if set(data["commons"]) != expected:
        raise ValueError("all original generic starts are required")
    TrainingOnlySplit.model_validate(data["case"]["training"])
    result = {}
    for key, common in data["commons"].items():
        if common["case"] != case or key != f"{case}_s{common['seed']}":
            raise ValueError("generic case/seed identity differs")
        request = PublicFitRequest.model_validate(common["request"])
        if common["start"] != public._lower(request)[1]:
            raise ValueError("generic request/start mismatch")
        result[key] = {
            k: deepcopy(common[k]) for k in ("request", "coordinates", "nodes", "start")
        }
        result[key]["training"] = deepcopy(data["case"]["training"])
    return result


def prepare(
    root: Path,
    inputs: Path,
    policy: RecoveryPolicy,
    *,
    protocol: str = PROTOCOL,
    arms: tuple[str, ...] = ARMS,
    base_factory=None,
) -> dict:
    """Freeze the complete roster and budgets; no optimizer or regeneration."""
    data = read_seal(inputs)
    tasks = [
        {"task_id": f"{common}_{arm}", "common": common, "arm": arm}
        for common in sorted((base_factory or bases)(data))
        for arm in arms
    ]
    plan = {
        "protocol": protocol,
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


def verify(root: Path, *, runtime=True, protocol=PROTOCOL):
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    if plan["protocol"] != protocol or plan["inputs_sha256"] != public.content_sha256(
        data
    ):
        raise ValueError(f"not a matching generic recovery campaign: {root}")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("generic recovery source/runtime differs")
    return plan, data


def run_task(
    root: Path,
    index: int,
    *,
    protocol=PROTOCOL,
    fitter=None,
    base_factory=None,
    evaluation_data=None,
) -> dict:
    """Seal all training decisions before scoring validation or coefficients."""
    plan, data = verify(root, protocol=protocol)
    fitter = fitter or fitting.fit
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("invalid task index")
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
                raise ValueError("recovery result identity differs")
            return saved
        path = folder / "backend.json"
        if not path.exists():
            backend = fitter(
                (base_factory or bases)(data)[task["common"]],
                task["arm"],
                plan["policy"],
                folder / "fit",
            )
            seal(path, {"identity": identity, **backend})
        backend = read_seal(path)
        if backend["identity"] != identity:
            raise ValueError("recovery backend differs")
        operations = list((folder / "fit").rglob("started.json"))
        processes = [
            read_seal(p.with_name("process.json"))
            for p in operations
            if p.with_name("process.json").exists()
        ]
        safe = len(processes) == len(operations) and all(
            p["termination_confirmed"] for p in processes
        )
        selected = backend.get("selected")
        evaluation = None
        if safe and selected:
            evaluation_folder = folder / "evaluation"
            seal(
                evaluation_folder / "identity.json",
                {"identity": identity, "parameters": selected["parameters"]},
            )
            evaluation = replay._evaluation(
                evaluation_folder,
                evaluation_data(data, task) if evaluation_data else data,
                data["commons"][task["common"]],
                selected["parameters"],
                plan["policy"]["replay_seconds"],
            )
        result = {
            **task,
            "identity": identity,
            "backend_sha256": public.content_sha256(backend),
            "status": "complete"
            if evaluation and evaluation["status"] == "complete"
            else "evaluation_unavailable",
            "stop_reason": backend["stop_reason"],
            "selected": selected,
            "evaluation": evaluation,
            "levels": backend["levels"],
            "training_prediction_certified": backend["stop_reason"]
            == "training_prediction_certified",
            "fit_seconds": backend.get("fit_seconds"),
            "evaluation_blocked_by_cleanup": not safe,
            "new_process_seconds": sum(p.get("elapsed_seconds", 0) for p in processes),
            "cost_complete": all("elapsed_seconds" in p for p in processes)
            and len(processes) == len(operations),
        }
        for key in ("portfolio", "rollout_call_accounting"):
            if key in backend:
                result[key] = backend[key]
        seal(output, result)
    return result


def report(root: Path, *, protocol=PROTOCOL, arms=ARMS) -> dict:
    """All starts remain in denominators, including failed fits/evaluations."""
    plan, _ = verify(root, runtime=False, protocol=protocol)
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
            raise ValueError("reported recovery identity differs")
        rows.append(row)
    groups = {}
    for arm in arms:
        selected = [r for r in rows if r["arm"] == arm]
        groups[arm] = {
            "expected": len(selected),
            **{
                k: sum(bool((r.get("evaluation") or {}).get(k)) for r in selected)
                for k in (
                    "accuracy_passed",
                    "coefficients_recovered",
                    "initials_recovered",
                )
            },
        }
    result = {
        "protocol": protocol,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete"
        if all(r["status"] != "missing" for r in rows)
        else "incomplete",
        "expected": len(rows),
        "recorded": sum(r["status"] != "missing" for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "groups": groups,
        "rows": rows,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": (
            "Three fixed generic starts with assisted correct equations/anchored "
            "coordinates. Training-only empirical prediction certificates do not "
            "certify parameter recovery. No global optimization guarantee or "
            "production promotion."
        ),
    }
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
