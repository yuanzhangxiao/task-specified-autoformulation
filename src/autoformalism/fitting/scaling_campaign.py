"""M26: three scaling policies, original generic starts, identical restart lists."""

from collections import Counter
from pathlib import Path
from time import monotonic

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import (
    incumbent_campaign,
    screening_replay,
    stagnation_campaign,
)
from autoformalism.fitting import nonlinear_comparison_campaign as source
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting import scaling_recovery as fitting

PROTOCOL = "phase-c-generic-scaling-1"
Policy = fitting.Policy


def export_inputs(root: Path, output: Path) -> dict:
    """Extract original immutable data from M25; exclude all fitted endpoints."""
    _, bundle = stagnation_campaign.verify(root, runtime=False)
    data = bundle["source_inputs"]["data"]
    problems = source.bases(data)
    seal(output, data)
    return {
        "inputs_sha256": public.content_sha256(data),
        "original_starts": len(problems),
    }


def roster(data: dict) -> list[dict]:
    return [
        {"task_id": f"{name}__{method}", "common": name, "method": method}
        for name in sorted(source.bases(data))
        for method in fitting.METHODS
    ]


def portfolios(data: dict, policy: Policy) -> dict:
    """Freeze shared domain-only candidates before any scaling arm runs."""
    result = {}
    for name, problem in source.bases(data).items():
        oracle = checks.Rollouts(problem, checks.ConfidencePolicy())
        seed = int(public.content_sha256(problem.model_dump(mode="json"))[:8], 16)
        result[name] = [
            problem.start,
            *numerical.diverse_starts(oracle, problem, policy.restart_count, seed),
        ]
    return result


def prepare(root: Path, inputs: Path, policy: Policy) -> dict:
    data = read_seal(inputs)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": roster(data),
        "starts": portfolios(data, policy),
        "historical_fits_used": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    policy = Policy.model_validate(plan["policy"])
    problems = source.bases(data)
    if (
        plan["protocol"] != PROTOCOL
        or plan["tasks"] != roster(data)
        or plan["inputs_sha256"] != public.content_sha256(data)
        or set(plan["starts"]) != set(problems)
    ):
        raise ValueError("scaling plan, inputs or roster differ")
    for name, problem in problems.items():
        if (
            len(plan["starts"][name]) != policy.restart_count + 1
            or plan["starts"][name][0] != problem.start
        ):
            raise ValueError("original start or complete portfolio differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("scaling source/runtime differs")
    return plan, data


def run_task(root: Path, index: int) -> dict:
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown scaling task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return incumbent_campaign._completed(folder, identity)
        policy = Policy.model_validate(plan["policy"])
        backend = fitting.run(
            source.bases(data)[task["common"]],
            plan["starts"][task["common"]],
            policy,
            task["method"],
            folder / "fit",
        )
        seal(folder / "backend.json", backend)
        # Evaluator-only validation/coefficients cannot affect sealed decisions.
        begun = monotonic()
        evaluation = screening_replay._evaluation(
            folder / "evaluation",
            data,
            data["commons"][task["common"]],
            backend["retained_parameters"],
            policy.replay_seconds,
        )
        row = task | {
            "identity": identity,
            "backend_sha256": public.content_sha256(backend),
            "starts_sha256": public.content_sha256(plan["starts"][task["common"]]),
            "evaluation": evaluation,
            "evaluation_seconds": monotonic() - begun,
            "stage_costs": source.stage_costs(backend),
            "search_progress": incumbent_campaign.search_progress(backend),
            **{
                k: backend[k]
                for k in (
                    "status",
                    "assessment",
                    "accuracy_verified",
                    "blocks_used",
                    "decisions",
                    "search_calls_started",
                    "search_calls_completed",
                    "maximum_search_calls",
                    "search_call_accounting_complete",
                    "search_budget_charge_seconds",
                    "additional_wall_seconds",
                    "additional_cpu_seconds",
                    "additional_budget_charge_seconds",
                    "cost_complete",
                )
            },
        }
        seal(folder / "result.json", row)
        return row


def report(root: Path) -> dict:
    plan, _ = verify(root, runtime=False)
    rows = [
        incumbent_campaign._completed(
            root / "results" / t["task_id"],
            {
                "plan_sha256": public.content_sha256(plan),
                "task": t,
            },
        )
        if (root / "results" / t["task_id"] / "result.json").exists()
        else t | {"status": "missing"}
        for t in plan["tasks"]
    ]
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
        "live_llm_calls": 0,
        "test_data_opened": False,
        "limitation": "Three original starts on one known nonlinear skeleton. "
        "Same frozen restart lists and maximum budgets, adaptive actual work. "
        "Validation/coefficient accuracy are retrospective; no global identifiability, "
        "calibrated confidence or universal scaling winner is claimed.",
    }
    public._write(root / "summary.json", result)
    return result
