"""M22 frozen nonlinear benchmark comparison; retrospective evaluator stays separate."""

from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import generic_recovery as source
from autoformalism.fitting import nonlinear_comparison as fitting
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import screening_replay
from autoformalism.fitting.confidence_checks import TrainingProblem

PROTOCOL = "phase-c-nonlinear-conditional-comparison-1"
ALLOCATION_PROTOCOL = "phase-c-nonlinear-allocation-1"
INPUT_DIGEST = "ed71b7ee77608f94b66e7c16a91c0a0c29f9c05ec7b25bc594694ba079d42d09"
ComparisonPolicy = fitting.ComparisonPolicy


def bases(data):
    """Reuse all original M12/M15 starts; exclude frozen guesses and evaluator data."""
    if public.content_sha256(data) != INPUT_DIGEST:
        raise ValueError("M22 requires the unchanged frozen nonlinear inputs")
    base = source.bases(data)
    return {
        name: TrainingProblem.model_validate(
            {k: item[k] for k in ("request", "training", "coordinates", "start")}
            | {"incumbent": item["start"]}
        )
        for name, item in base.items()
    }


def roster(data):
    return [
        {"task_id": f"{name}__{method}", "common": name, "method": method}
        for name in sorted(bases(data))
        for method in fitting.METHODS
    ]


def prepare(root: Path, inputs: Path, policy: ComparisonPolicy):
    data = read_seal(inputs)
    diagnostic = None
    if policy.allocation == "measured":
        # Saved nodes have a specific time layout, not just an array shape.
        # Changing observation anchors can preserve the size but change meaning.
        mesh_keys = (
            "node_targets",
            "penalties",
            "minimum_intervals",
            "observation_anchors",
        )
        if any(getattr(policy, k) != getattr(ComparisonPolicy(), k) for k in mesh_keys):
            raise ValueError(
                "M23 saved-node diagnostics require the frozen M22 mesh policy"
            )
        diagnostic = read_seal(inputs.parent / "conditional-diagnostic-inputs.json")
        from autoformalism.fitting.conditional_diagnostics import PROTOCOL as DP
        from autoformalism.fitting.conditional_diagnostics import SOURCE_PLAN

        if (
            diagnostic["protocol"] != DP
            or diagnostic["source_plan_sha256"] != SOURCE_PLAN
            or diagnostic["inputs_sha256"] != public.content_sha256(data)
            or sorted((r["common"], r["level"]) for r in diagnostic["rows"])
            != [(n, k) for n in sorted(bases(data)) for k in range(2)]
        ):
            raise ValueError("conditional diagnostic source identity/roster differs")
    plan = {
        "protocol": ALLOCATION_PROTOCOL if diagnostic is not None else PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": roster(data),
        "test_data_opened": False,
        "live_llm_calls": 0,
        **(
            {"diagnostic_inputs_sha256": public.content_sha256(diagnostic)}
            if diagnostic is not None
            else {}
        ),
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        if diagnostic is not None:
            seal(root / "diagnostic-inputs.json", diagnostic)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime=True):
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    measured = plan["policy"].get("allocation", "legacy") == "measured"
    if (
        plan["protocol"] != (ALLOCATION_PROTOCOL if measured else PROTOCOL)
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != roster(data)
    ):
        raise ValueError("nonlinear comparison plan/inputs/roster differ")
    if (
        measured
        and public.content_sha256(read_seal(root / "diagnostic-inputs.json"))
        != plan["diagnostic_inputs_sha256"]
    ):
        raise ValueError("conditional diagnostic input digest differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("nonlinear comparison source/runtime differs")
    return plan, data


def run_task(root: Path, index: int):
    plan, data = verify(root)
    if plan["protocol"] == ALLOCATION_PROTOCOL:
        diagnostic = read_seal(root / "diagnostics/result.json")
        if (
            diagnostic["plan_sha256"] != public.content_sha256(plan)
            or not diagnostic["correctness_passed"]
        ):
            raise ValueError(
                "complete matching coefficient diagnostic required before fitting"
            )
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown nonlinear task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    directory = root / "results" / task["task_id"]
    with public._lock(directory):
        if (directory / "result.json").exists():
            saved = read_seal(directory / "result.json")
            if saved["identity"] != identity:
                raise ValueError("completed nonlinear task identity differs")
            if saved["backend_sha256"] != public.content_sha256(
                read_seal(directory / "backend.json")
            ):
                raise ValueError("completed nonlinear backend differs")
            return saved
        policy = ComparisonPolicy.model_validate(plan["policy"])
        backend = fitting.run(
            bases(data)[task["common"]], policy, task["method"], directory / "fit"
        )
        seal(directory / "backend.json", backend)
        # Only after sealing all fitting decisions can scoring see validation/truth.
        begun = monotonic()
        evaluations = {}
        for name in ("after_warm", "selected"):
            point = backend[name]
            evaluations[name] = screening_replay._evaluation(
                directory / f"evaluation-{name}",
                data,
                data["commons"][task["common"]],
                point["parameters"] if point else None,
                policy.replay_seconds,
            )
        row = task | {
            "identity": identity,
            "status": backend["status"],
            "backend_sha256": public.content_sha256(backend),
            "evaluations": evaluations,
            "evaluation_seconds": monotonic() - begun,
            **{
                k: backend[k]
                for k in (
                    "assessment",
                    "trajectory_assessment",
                    "verified_trial_count",
                    "routing",
                    "portfolio_triggered",
                    "trial_count",
                    "continuation_run",
                    "additional_wall_seconds",
                    "additional_cpu_seconds",
                    "additional_budget_charge_seconds",
                    "cost_complete",
                    "rollout_calls_observed",
                )
            },
            "stage_costs": stage_costs(backend),
        }
        seal(directory / "result.json", row)
        return row


def stage_costs(backend):
    totals = defaultdict(
        lambda: {
            "wall_seconds": 0.0,
            "cpu_seconds": 0.0,
            "budget_charge_seconds": 0.0,
            "accounting_complete": True,
        }
    )
    for op in backend["operations"]:
        name = op["operation"].split("/")[-1]
        stage = (
            "conditional"
            if name == "conditional"
            else "search"
            if name in {"rollout", "joint-fallback"}
            else "sensitivity"
            if name == "final-sensitivity"
            else "verification"
        )
        for k in ("wall_seconds", "cpu_seconds", "budget_charge_seconds"):
            totals[stage][k] += op[k] or 0
        totals[stage]["accounting_complete"] &= op["accounting_complete"]
    return dict(totals)


def report(root: Path):
    plan, _ = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        directory = root / "results" / task["task_id"]
        if not (directory / "result.json").exists():
            rows.append(task | {"status": "missing"})
            continue
        row = read_seal(directory / "result.json")
        if row["identity"] != {
            "plan_sha256": public.content_sha256(plan),
            "task": task,
        }:
            raise ValueError("reported nonlinear task identity differs")
        backend = read_seal(directory / "backend.json")
        if row["backend_sha256"] != public.content_sha256(backend):
            raise ValueError("reported nonlinear backend differs")
        rows.append(row)
    recorded = [r for r in rows if r["status"] != "missing"]
    groups = []
    for method in fitting.METHODS:
        rr = [r for r in recorded if r["method"] == method]
        groups.append(
            {
                "method": method,
                "recorded": len(rr),
                "totals": {
                    stage: {
                        gate: sum(bool(r["evaluations"][stage].get(gate)) for r in rr)
                        for gate in (
                            "accuracy_passed",
                            "coefficients_recovered",
                            "initials_recovered",
                        )
                    }
                    for stage in ("after_warm", "selected")
                },
                "actions": dict(
                    Counter(r["assessment"]["recommended_action"] for r in rr)
                ),
                **{
                    k: sum(r[k] for r in rr)
                    for k in (
                        "additional_wall_seconds",
                        "additional_cpu_seconds",
                        "additional_budget_charge_seconds",
                        "evaluation_seconds",
                    )
                },
            }
        )
    result = {
        "protocol": plan["protocol"],
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if len(recorded) == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": len(recorded),
        "rows": rows,
        "groups": groups,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "cost_complete": len(recorded) == len(rows)
        and all(r["cost_complete"] for r in recorded),
        "test_data_opened": False,
        "limitation": "Known anchored nonlinear skeleton, noiseless development data, "
        "three seeds. "
        "Conditional coefficient profiling is not exact nonlinear rollout profiling. "
        "No global identifiability or calibrated confidence claim; "
        "production fitter unchanged.",
    }
    if plan["protocol"] == ALLOCATION_PROTOCOL:
        path = root / "diagnostics/result.json"
        diagnostic = read_seal(path) if path.exists() else {"status": "pending"}
        if path.exists() and diagnostic["plan_sha256"] != public.content_sha256(plan):
            raise ValueError("diagnostic report identity differs")
        result["diagnostics"] = diagnostic
    public._write(root / "summary.json", result)
    lines = [
        "# Nonlinear fitting assessments",
        "",
        "Training-only diagnoses; reference errors are separate.",
        "",
        "| Task | Kernel | Training NMSE | Parameter information | Action |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        a = row.get("assessment", {})
        kernel = row.get("routing", {}).get("selected", "pending")
        lines.append(
            f"| {row['task_id']} | {kernel} | "
            f"{a.get('training_nmse')} | {a.get('parameter_information', 'pending')} | "
            f"{a.get('recommended_action', 'pending')} |"
        )
    (root / "assessment.md").write_text("\n".join(lines) + "\n")
    return result
