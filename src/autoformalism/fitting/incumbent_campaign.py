"""M24: compare continuation policies from identical sealed M23 warm fits."""

from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import nonlinear_comparison as fitting
from autoformalism.fitting import nonlinear_comparison_campaign as source
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting import screening_replay

PROTOCOL = "phase-c-incumbent-continuation-1"
SOURCE_PLAN = "3f9c8d1286131dc3ae57c6eda8c0d0e3d61f9ae4a17e02edd8b26b95845a250c"
METHODS = ("incumbent_first", "restart_first")


class Policy(fitting.ComparisonPolicy):
    """Equal search ceilings; safety guards are not equal actual computation."""

    allocation: Literal["measured"] = "measured"
    portfolio_size: Literal[3] = 3
    continuation_seconds: float = Field(default=1200, ge=1, le=1200)
    continuation_calls: int = Field(default=60, ge=2, le=1200)
    trial_seconds: float = Field(default=400, ge=1, le=600)
    trial_calls: int = Field(default=15, ge=2, le=600)
    check_seconds: float = Field(default=180, ge=1, le=180)
    sensitivity_seconds: float = Field(default=180, ge=1, le=180)


def warm_record(backend, problem):
    """Require the training-only warm stage, never a later/validation-best endpoint."""
    identity = backend["identity"]
    if (
        identity["problem_sha256"]
        != public.content_sha256(problem.model_dump(mode="json"))
        or identity["method"] != "best_rollout"
    ):
        raise ValueError("source warm problem/method differs")
    ops = {o["operation"]: o for o in backend["operations"]}
    fit, check = ops["warm/rollout"], ops["warm/check"]
    point = backend["after_warm"]
    if (
        fit["identity"]["point"]["start"] != problem.incumbent
        or fit["status"] != "complete"
        or check["status"] != "complete"
        or not numerical.usable(point)
        or point["origin"] != "warm/check"
        or fit["value"]["best"]["parameters"] != point["parameters"]
        or check["value"]["parameters"] != point["parameters"]
        or not numerical.usable(check["value"])
    ):
        raise ValueError("source warm checkpoint is unavailable or inconsistent")
    return point


def export_inputs(root: Path, output: Path):
    """Export all three predetermined best_rollout warm fits with source witnesses."""
    plan, data = source.verify(root, runtime=False)
    if public.content_sha256(plan) != SOURCE_PLAN:
        raise ValueError("M24 requires the frozen M23 source plan")
    parents = {}
    for name, problem in source.bases(data).items():
        task = {
            "task_id": name + "__best_rollout",
            "common": name,
            "method": "best_rollout",
        }
        folder = root / "results" / task["task_id"]
        row, backend = (
            read_seal(folder / "result.json"),
            read_seal(folder / "backend.json"),
        )
        if row["identity"] != {"plan_sha256": SOURCE_PLAN, "task": task} or row[
            "backend_sha256"
        ] != public.content_sha256(backend):
            raise ValueError("source result/backend identity differs")
        warm_record(backend, problem)
        parents[name] = {
            "backend": backend,
            "backend_sha256": row["backend_sha256"],
            "source_result_sha256": public.content_sha256(row),
        }
    bundle = {
        "protocol": PROTOCOL,
        "source_plan": plan,
        "data": data,
        "parents": parents,
    }
    validate_inputs(bundle)
    seal(output, bundle)
    return {
        "inputs_sha256": public.content_sha256(bundle),
        "shared_warm_fits": len(parents),
    }


def validate_inputs(bundle):
    """Bind warm vectors to original training problems and frozen source provenance."""
    plan, data = bundle["source_plan"], bundle["data"]
    if (
        bundle["protocol"] != PROTOCOL
        or public.content_sha256(plan) != SOURCE_PLAN
        or plan["inputs_sha256"] != public.content_sha256(data)
    ):
        raise ValueError("continuation source plan/inputs differ")
    problems = source.bases(data)
    if set(bundle["parents"]) != set(problems):
        raise ValueError("all original warm starts required")
    for name, problem in problems.items():
        parent = bundle["parents"][name]
        if parent["backend_sha256"] != public.content_sha256(parent["backend"]):
            raise ValueError("source backend digest differs")
        if parent["backend"]["identity"]["policy"] != plan["policy"]:
            raise ValueError("source policy differs")
        warm_record(parent["backend"], problem)
    return problems


def roster(bundle):
    return [
        {"task_id": f"{name}__{method}", "common": name, "method": method}
        for name in sorted(validate_inputs(bundle))
        for method in METHODS
    ]


def prepare(root: Path, inputs: Path, policy: Policy):
    bundle = read_seal(inputs)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(bundle),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": roster(bundle),
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
    with public._lock(root):
        seal(root / "inputs.json", bundle)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime=True):
    plan, bundle = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["tasks"] != roster(bundle)
        or plan["inputs_sha256"] != public.content_sha256(bundle)
    ):
        raise ValueError("continuation plan/inputs/roster differ")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("continuation source/runtime differs")
    return plan, bundle


def _completed(directory, identity):
    row = read_seal(directory / "result.json")
    backend = read_seal(directory / "backend.json")
    if row["identity"] != identity or row["backend_sha256"] != public.content_sha256(
        backend
    ):
        raise ValueError("completed continuation identity/backend differs")
    return row


def search_progress(backend):
    """Summarize completed trial evaluations without calling them accepted iterates."""
    rows = []
    for op in backend["operations"]:
        if not op["operation"].endswith(("/rollout", "/joint-fallback")):
            continue
        value = op.get("value") or {}
        trace = value.get("evaluations", [])
        best = value.get("best")
        digest = public.content_sha256(best["parameters"]) if best else None
        best_evaluation = next(
            (r for r in reversed(trace) if r["parameters_sha256"] == digest), None
        )
        rows.append(
            {
                "operation": op["operation"],
                "status": op["status"],
                "calls_started": value.get("calls"),
                "calls_completed": value.get("completed_calls"),
                "stop_reason": value.get("stop_reason", op["status"]),
                "message": value.get("message"),
                "first_evaluation": trace[0] if trace else None,
                "best_evaluation": best_evaluation,
                "last_evaluation": trace[-1] if trace else None,
                "last_five_best_losses": [r["best_training_nmse"] for r in trace[-5:]],
            }
        )
    return rows


def run_task(root: Path, index: int):
    plan, bundle = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown continuation task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    directory = root / "results" / task["task_id"]
    with public._lock(directory):
        if (directory / "result.json").exists():
            return _completed(directory, identity)
        problem = validate_inputs(bundle)[task["common"]]
        point = warm_record(bundle["parents"][task["common"]]["backend"], problem)
        policy = Policy.model_validate(plan["policy"])
        backend = fitting.run(
            problem,
            policy,
            "best_rollout",
            directory / "fit",
            shared_warm=point["parameters"],
            continuation_order=task["method"],
        )
        seal(directory / "backend.json", backend)
        # Validation and reference coefficients are visible only after decisions seal.
        begun = monotonic()
        evaluations = {
            stage: screening_replay._evaluation(
                directory / f"evaluation-{stage}",
                bundle["data"],
                bundle["data"]["commons"][task["common"]],
                backend[stage]["parameters"] if backend[stage] else None,
                policy.replay_seconds,
            )
            for stage in ("after_warm", "selected")
        }
        row = task | {
            "identity": identity,
            "backend_sha256": public.content_sha256(backend),
            "shared_parameters_sha256": public.content_sha256(point["parameters"]),
            "evaluations": evaluations,
            "evaluation_seconds": monotonic() - begun,
            "stage_costs": source.stage_costs(backend),
            "search_progress": search_progress(backend),
            **{
                k: backend[k]
                for k in (
                    "status",
                    "assessment",
                    "trajectory_assessment",
                    "routing",
                    "trial_count",
                    "continuation_run",
                    "incumbent_continuation_run",
                    "search_calls_started",
                    "search_calls_completed",
                    "search_call_accounting_complete",
                    "additional_wall_seconds",
                    "additional_cpu_seconds",
                    "additional_budget_charge_seconds",
                    "cost_complete",
                )
            },
        }
        seal(directory / "result.json", row)
        return row


def report(root: Path):
    plan, bundle = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        directory = root / "results" / task["task_id"]
        rows.append(
            _completed(
                directory, {"plan_sha256": public.content_sha256(plan), "task": task}
            )
            if (directory / "result.json").exists()
            else task | {"status": "missing"}
        )
    shared_cost = {}
    for name, parent in bundle["parents"].items():
        backend = parent["backend"]
        operations = backend["operations"]
        warm_end = next(
            i for i, op in enumerate(operations) if op["operation"] == "warm/check"
        )
        sources = operations[: warm_end + 1] + backend["setup"]
        shared_cost[name] = {
            k: sum(r[k] or 0 for r in sources) for k in ("wall_seconds", "cpu_seconds")
        }
    recorded = [r for r in rows if r["status"] != "missing"]
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if len(recorded) == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": len(recorded),
        "rows": rows,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "shared_warm_cost_once_per_seed": shared_cost,
        "cost_complete": len(recorded) == len(rows)
        and all(r["cost_complete"] for r in recorded),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": "Known nonlinear skeleton, three original starts. Warm "
        "parameters are shared, not optimizer state. Equal ceilings do not imply "
        "equal completed "
        "work. Training-only retention; reference/validation scoring is retrospective. "
        "No global identifiability or calibrated confidence claim.",
    }
    public._write(root / "summary.json", result)
    return result
