"""M21 matched joint/profiled recovery from all 24 frozen M19 incumbents."""

from collections import Counter
from pathlib import Path
from time import monotonic

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import assessed_recovery as fitting
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import profile_recovery_campaign as source
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import screening_replay

PROTOCOL = "phase-c-assessed-recovery-1"
RecoveryPolicy = fitting.RecoveryPolicy


def training_problem(endpoint: dict) -> checks.TrainingProblem:
    """Exclude saved profiles, alternatives, source metrics and evaluator data."""
    payload = endpoint["problem"]
    return checks.TrainingProblem.model_validate(
        {
            k: payload[k]
            for k in ("request", "training", "coordinates", "start", "incumbent")
        }
    )


def roster(data: dict) -> list[dict]:
    """Every historical endpoint receives both new methods; no outcome filtering."""
    source.validate_inputs(data)
    return [
        {
            "task_id": e["task_id"] + "__" + method,
            "source_index": i,
            "source_task": e["task_id"],
            "source_arm": e["arm"],
            "common": e["common"],
            "method": method,
        }
        for i, e in enumerate(data["endpoints"])
        for method in fitting.METHODS
    ]


def prepare(root: Path, inputs: Path, policy: RecoveryPolicy) -> dict:
    """Freeze one separate campaign without changing any historical directory."""
    data = read_seal(inputs)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": roster(data),
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime=True):
    """Require the frozen inputs and executable runtime on execution/resume."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != roster(data)
    ):
        raise ValueError("assessed recovery plan or inputs differ")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("assessed recovery source/runtime differs")
    return plan, data


def run_task(root: Path, index: int) -> dict:
    """Select and seal solely with training before retrospective evaluation."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown recovery task")
    task = plan["tasks"][index]
    endpoint = data["endpoints"][task["source_index"]]
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
            training_problem(endpoint), policy, task["method"], folder / "fit"
        )
        seal(folder / "backend.json", backend)
        raw = data["source_data"]["data"]
        base = raw["commons"][task["common"]]
        evaluation = {
            "case": raw["cases"][base["case"]],
            "case_name": base["case"],
            "config": raw["config"],
        }
        begun = monotonic()
        stages = {}
        for stage in ("after_warm", "selected"):
            point = backend[stage]
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
            "assessment": backend["assessment"],
            "evaluation_seconds": monotonic() - begun,
            "source_cost": endpoint["source_cost"],
            **{
                k: backend[k]
                for k in (
                    "additional_wall_seconds",
                    "additional_cpu_seconds",
                    "additional_budget_charge_seconds",
                    "cost_complete",
                    "rollout_calls_observed",
                    "stage_costs",
                    "trial_count",
                    "portfolio_triggered",
                    "continuation_run",
                )
            },
        }
        seal(folder / "result.json", row)
        return row


def report(root: Path) -> dict:
    """Retain every method/source endpoint and separate diagnostics from truth."""
    plan, _ = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        path = root / "results" / task["task_id"] / "result.json"
        if not path.exists():
            rows.append(task | {"status": "missing"})
            continue
        row = read_seal(path)
        if row["identity"] != {
            "plan_sha256": public.content_sha256(plan),
            "task": task,
        } or row["backend_sha256"] != public.content_sha256(
            read_seal(path.with_name("backend.json"))
        ):
            raise ValueError("reported endpoint identity differs")
        rows.append(row)
    recorded = [r for r in rows if r["status"] != "missing"]
    groups = []
    for method in fitting.METHODS:
        group = [r for r in recorded if r["method"] == method]
        totals = {}
        for stage in ("before", "after_warm", "selected"):
            values = [
                r["before"] if stage == "before" else r["evaluations"][stage]
                for r in group
            ]
            totals[stage] = {
                k: sum(bool(v.get(k)) for v in values)
                for k in (
                    "accuracy_passed",
                    "coefficients_recovered",
                    "initials_recovered",
                )
            }
        groups.append(
            {
                "method": method,
                "expected": 24,
                "recorded": len(group),
                "totals": totals,
                "actions": dict(
                    Counter(r["assessment"]["recommended_action"] for r in group)
                ),
                "parameter_information": dict(
                    Counter(r["assessment"]["parameter_information"] for r in group)
                ),
                "additional_wall_seconds": sum(
                    r["additional_wall_seconds"] for r in group
                ),
                "additional_cpu_seconds": sum(
                    r["additional_cpu_seconds"] for r in group
                ),
                "evaluation_seconds": sum(r["evaluation_seconds"] for r in group),
                "additional_budget_charge_seconds": sum(
                    r["additional_budget_charge_seconds"] for r in group
                ),
            }
        )
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if len(recorded) == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": len(recorded),
        "groups": groups,
        "rows": rows,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "cost_complete": len(recorded) == len(rows)
        and all(r["cost_complete"] for r in recorded),
        "test_data_opened": False,
        "limitation": (
            "Noiseless known linear controls; training-only decisions. Local "
            "diagnostics are not coefficient correctness, global identifiability, "
            "or calibrated confidence. Matched ceilings, actual effort may differ."
        ),
    }
    public._write(root / "summary.json", result)
    (root / "assessment.md").write_text(assessment_table(rows))
    return result


def assessment_table(rows: list[dict]) -> str:
    """Human-readable companion with training-only evidence and no truth scores."""
    lines = [
        "# Fitting assessments",
        "",
        "These are conditional numerical diagnoses, not correctness probabilities.",
        "",
        "| Endpoint | Method | Training NMSE | Numerical status | "
        "Parameter information | Next action |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        a = row.get("assessment", {})
        value = a.get("training_nmse")
        columns = [
            row["source_task"],
            row["method"],
            f"{value:.3g}" if value is not None else "unavailable",
            a.get("numerical_status", row["status"]),
            a.get("parameter_information", "unavailable"),
            a.get("recommended_action", "await_result"),
        ]
        lines.append("| " + " | ".join(columns) + " |")
    return "\n".join(lines) + "\n"
