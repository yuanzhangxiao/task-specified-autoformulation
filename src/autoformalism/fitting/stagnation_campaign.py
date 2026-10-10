"""M25: shared derivative audits and matched scaling from frozen M24 endpoints."""

from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import incumbent_campaign as source
from autoformalism.fitting import nonlinear_comparison as fitting
from autoformalism.fitting import nonlinear_rollout as rollout
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting import screening_replay
from autoformalism.fitting.stagnation_audit import AuditPolicy, audit

PROTOCOL = "phase-c-fitting-stagnation-1"
SOURCE_PLAN = "562879ac15f0614cc69a7110536fe5dbe2d369f75687165cf581e7b2bd525391"
METHODS = ("jacobian", "fixed_coordinates")


class Policy(fitting.ComparisonPolicy):
    """One bounded continuation per scale; no new initializations or node fits."""

    allocation: Literal["measured"] = "measured"
    portfolio_size: Literal[0] = 0
    continuation_calls: int = Field(default=120, ge=2, le=120)
    continuation_seconds: float = Field(default=2400, ge=1, le=2400)
    check_seconds: float = Field(default=180, ge=1, le=180)
    sensitivity_seconds: float = Field(default=180, ge=1, le=180)
    audit: AuditPolicy = Field(default_factory=AuditPolicy)


def endpoint(backend: dict, problem, shared: dict, policy: dict) -> dict:
    """Select the predeclared incumbent-continuation point, never final best N."""
    identity = backend["identity"]
    if (
        identity["problem_sha256"]
        != public.content_sha256(problem.model_dump(mode="json"))
        or identity["method"] != "best_rollout"
        or identity["continuation_order"] != "incumbent_first"
        or identity["policy"] != policy
        or identity["shared_warm"] != shared
    ):
        raise ValueError("source continuation identity differs")
    point = backend["after_incumbent_continuation"]
    ops = {o["operation"]: o for o in backend["operations"]}
    search, check = (
        ops["incumbent-continuation/rollout"],
        ops["incumbent-continuation/check"],
    )
    if (
        point["origin"] != "incumbent-continuation/check"
        or not numerical.usable(point)
        or search["status"] != "complete"
        or check["status"] != "complete"
        or not numerical.usable(check["value"])
        or check["value"]["parameters"] != point["parameters"]
        or search["value"]["best"]["parameters"] != point["parameters"]
        or search["identity"]["point"]["start"] != shared
    ):
        raise ValueError("source continuation checkpoint inconsistent")
    return point


def validate_inputs(bundle: dict):
    plan, inputs = bundle["source_plan"], bundle["source_inputs"]
    if (
        bundle["protocol"] != PROTOCOL
        or public.content_sha256(plan) != SOURCE_PLAN
        or plan["inputs_sha256"] != public.content_sha256(inputs)
    ):
        raise ValueError("stagnation source plan/inputs differ")
    problems = source.validate_inputs(inputs)
    if set(bundle["parents"]) != set(problems):
        raise ValueError("all three original endpoints required")
    for name, problem in problems.items():
        parent = bundle["parents"][name]
        if parent["backend_sha256"] != public.content_sha256(parent["backend"]):
            raise ValueError("source backend digest differs")
        shared = source.warm_record(inputs["parents"][name]["backend"], problem)
        endpoint(parent["backend"], problem, shared["parameters"], plan["policy"])
    return problems


def export_inputs(root: Path, output: Path):
    """Verify three predetermined endpoints without importing evaluation scores."""
    plan, inputs = source.verify(root, runtime=False)
    if public.content_sha256(plan) != SOURCE_PLAN:
        raise ValueError("M25 requires the frozen M24 source plan")
    parents = {}
    for name in source.validate_inputs(inputs):
        task = {
            "common": name,
            "method": "incumbent_first",
            "task_id": name + "__incumbent_first",
        }
        folder = root / "results" / task["task_id"]
        row = source._completed(folder, {"plan_sha256": SOURCE_PLAN, "task": task})
        parents[name] = {
            "backend": read_seal(folder / "backend.json"),
            "backend_sha256": row["backend_sha256"],
            "source_result_sha256": public.content_sha256(row),
        }
    bundle = {
        "protocol": PROTOCOL,
        "source_plan": plan,
        "source_inputs": inputs,
        "parents": parents,
    }
    validate_inputs(bundle)
    seal(output, bundle)
    return {"inputs_sha256": public.content_sha256(bundle), "endpoints": len(parents)}


def roster(bundle: dict) -> list[dict]:
    return [
        {"task_id": name, "common": name} for name in sorted(validate_inputs(bundle))
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
        raise ValueError("stagnation plan/inputs/roster differ")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("stagnation source/runtime differs")
    return plan, bundle


def run_task(root: Path, index: int):
    plan, bundle = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown stagnation task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return source._completed(folder, identity)
        problem = validate_inputs(bundle)[task["common"]]
        parameters = bundle["parents"][task["common"]]["backend"][
            "after_incumbent_continuation"
        ]["parameters"]
        policy = Policy.model_validate(plan["policy"])

        def diagnose(end, save):
            oracle, profile, route = rollout.build(problem)
            return audit(oracle, profile, parameters, policy.audit, end, save) | {
                "routing": route
            }

        diagnostic = checks.operation(
            folder / "derivatives", identity, policy.audit.seconds, diagnose
        )
        gate = (diagnostic.get("value") or {}).get("status", diagnostic["status"])
        arms = {}
        if gate == "passed":
            for method in METHODS:
                arms[method] = fitting.run(
                    problem,
                    policy,
                    "best_rollout",
                    folder / method,
                    shared_warm=parameters,
                    continuation_order="incumbent_first",
                    optimizer_scaling=method,
                )
        backend = {
            "identity": identity,
            "derivative_audit": diagnostic,
            "arms": arms,
            "parameters": parameters,
            "gate": gate,
            "reference_values_used": False,
            "validation_used_for_selection": False,
        }
        seal(folder / "backend.json", backend)
        # Seal BOTH independent strategies before opening evaluation-only data.
        data = bundle["source_inputs"]["data"]
        began = monotonic()
        evaluations = {
            method: screening_replay._evaluation(
                folder / ("evaluation-" + method),
                data,
                data["commons"][task["common"]],
                arm["retained_parameters"],
                policy.replay_seconds,
            )
            for method, arm in arms.items()
        }
        result = task | {
            "identity": identity,
            "backend_sha256": public.content_sha256(backend),
            "status": "complete" if gate == "passed" else "diagnostic_blocked",
            "derivative_audit": diagnostic,
            "evaluation_seconds": monotonic() - began,
            "arms": {
                method: {
                    "evaluation": evaluations[method],
                    "stage_costs": source.source.stage_costs(arm),
                    "search_progress": source.search_progress(arm),
                    **{
                        k: arm[k]
                        for k in (
                            "status",
                            "assessment",
                            "trajectory_assessment",
                            "retained_parameters",
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
                for method, arm in arms.items()
            },
            "cost_complete": diagnostic["accounting_complete"]
            and all(a["cost_complete"] for a in arms.values()),
        }
        seal(folder / "result.json", result)
        return result


def report(root: Path):
    plan, _ = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        folder = root / "results" / task["task_id"]
        rows.append(
            source._completed(
                folder, {"plan_sha256": public.content_sha256(plan), "task": task}
            )
            if (folder / "result.json").exists()
            else task | {"status": "missing"}
        )
    recorded = [r for r in rows if r["status"] != "missing"]
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if len(recorded) == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": len(recorded),
        "rows": rows,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "cost_complete": len(recorded) == len(rows)
        and all(r["cost_complete"] for r in recorded),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": "Four-direction local derivative checks gate two scaling "
        "continuations. Known skeleton, three predetermined points; no new "
        "generic-start success rate, global identifiability or calibrated "
        "confidence claim. Shared audit cost is additional and charged once per "
        "point. Retrospective scoring never selects a fit.",
    }
    public._write(root / "summary.json", result)
    return result
