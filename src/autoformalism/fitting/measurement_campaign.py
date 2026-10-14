"""M34 fixed-sigma qualification with immutable inputs and post-fit evaluation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import monotonic, process_time

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import measurement_fit, measurement_fixtures
from autoformalism.fitting import public_fitting as public

PROTOCOL = "phase-c-measurement-fitting-1"
INPUT_PROTOCOL = "phase-c-measurement-fitting-inputs-1"
STRATEGIES = ("direct", "profiled")
Policy = measurement_fit.Policy


def roster(data: dict) -> list[dict]:
    """Validate the frozen six-case, three-noise-seed qualification roster."""
    expected = {
        f"{name}_s{seed}"
        for name in measurement_fixtures.CASES
        for seed in measurement_fixtures.SEEDS
    }
    if (
        data["protocol"] != INPUT_PROTOCOL
        or data["test_data_opened"] is not False
        or set(data["problems"]) != expected
        or set(data["evaluation"]) != expected
    ):
        raise ValueError("measurement qualification roster differs")
    measurement_fixtures.verify_raw_artifacts(data)
    return [
        {
            "task_id": f"{name}_s{seed}__{strategy}",
            "common": f"{name}_s{seed}",
            "case": name,
            "seed": seed,
            "strategy": strategy,
        }
        for name in measurement_fixtures.CASES
        for seed in measurement_fixtures.SEEDS
        for strategy in STRATEGIES
    ]


def training_problem(data: dict, task: dict):
    """Pass only the explicitly sealed measurement problem to the fitter."""
    from autoformalism.schemas.measurement_fitting import MeasurementProblem

    return MeasurementProblem.model_validate(data["problems"][task["common"]])


def prepare(root: Path, inputs: Path | None, policy: Policy) -> dict:
    """Freeze generated controls once, or import their exact portable export."""
    data = (
        read_seal(inputs) if inputs is not None else measurement_fixtures.make_inputs()
    )
    tasks = roster(data)
    for task in tasks[::2]:
        training_problem(data, task)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": tasks,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "qualification_only": True,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(tasks)}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Reject changed data, objective, policy, roster, source or runtime."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != roster(data)
        or plan["test_data_opened"] is not False
        or plan["live_llm_calls"] != 0
        or plan["qualification_only"] is not True
    ):
        raise ValueError("measurement campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("measurement source/runtime differs")
    return plan, data


def fit_identity(plan: dict, data: dict, task: dict) -> dict:
    """Derive the fitter's exact identity without passing evaluator data to it."""
    problem = training_problem(data, task)
    return {
        "protocol": "measurement-fitting-execution-1",
        "source_sha256": plan["source_sha256"],
        "runtime": plan["runtime"],
        "problem_sha256": public.content_sha256(problem.model_dump(mode="json")),
        "base_request_sha256": public.content_sha256(
            problem.request.model_dump(mode="json")
        ),
        "policy": plan["policy"],
        "strategy": task["strategy"],
    }


def validate_backend(backend: dict, identity: dict, execution: dict) -> None:
    """Reject a copied/stale intermediate backend before scoring or terminal reuse."""
    if (
        backend.get("identity") != execution
        or backend.get("campaign_identity") != identity
    ):
        raise ValueError("measurement backend identity differs")


def evaluation_identity(identity: dict, backend: dict, evaluator: dict) -> dict:
    """Bind post-fit arithmetic to the exact endpoint and independent reference."""
    return {
        "task_identity": identity,
        "backend_sha256": public.content_sha256(backend),
        "evaluator_sha256": public.content_sha256(evaluator),
    }


def validate_evaluation(value: dict, identity: dict) -> None:
    """Refuse an evaluator result from another endpoint, task or fixture."""
    if value.get("identity") != identity:
        raise ValueError("measurement evaluation identity differs")


def completed(folder: Path, identity: dict, execution: dict, evaluator: dict) -> dict:
    """Verify the final fit and evaluator seals before reusing a terminal result."""
    result = read_seal(folder / "result.json")
    if result["identity"] != identity:
        raise ValueError("measurement result identity differs")
    backend = read_seal(folder / "backend.json")
    validate_backend(backend, identity, execution)
    evaluation = read_seal(folder / "evaluation.json")
    expected_evaluation = evaluation_identity(identity, backend, evaluator)
    validate_evaluation(evaluation, expected_evaluation)
    if read_seal(folder / "evaluation-started.json") != expected_evaluation:
        raise ValueError("measurement evaluation journal identity differs")
    if (
        result["backend_sha256"] != public.content_sha256(backend)
        or result["backend"] != backend
        or result["evaluation"] != evaluation
    ):
        raise ValueError("measurement backend/evaluation differs")
    return result


def evaluate_once(folder: Path, data: dict, task: dict, backend: dict, identity: dict):
    """Journal evaluation before starting; interrupted usage is never reset."""
    expected = evaluation_identity(
        identity, backend, data["evaluation"][task["common"]]
    )
    path, started = folder / "evaluation.json", folder / "evaluation-started.json"
    if path.exists():
        result = read_seal(path)
        validate_evaluation(result, expected)
        if read_seal(started) != expected:
            raise ValueError("measurement evaluation journal identity differs")
        return result
    if started.exists():
        if read_seal(started) != expected:
            raise ValueError("measurement evaluation journal identity differs")
        result = {
            "status": "interrupted",
            "identity": expected,
            "wall_seconds": None,
            "cpu_seconds": None,
            "accounting_complete": False,
            "analytic_objective_passed": None,
            "reason": "started evaluation has unknown usage; it was not repeated",
        }
    else:
        seal(started, expected)
        began, cpu = monotonic(), process_time()
        try:
            result = evaluate(data, task, backend)
        except (ValueError, RuntimeError, ArithmeticError) as error:
            result = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
                "analytic_objective_passed": None,
            }
        result = result | {
            "identity": expected,
            "wall_seconds": monotonic() - began,
            "cpu_seconds": process_time() - cpu,
            "accounting_complete": True,
        }
    seal(path, result)
    return result


def evaluate(data: dict, task: dict, backend: dict) -> dict:
    """Compare an already selected endpoint with evaluator-only analytic controls."""
    result = measurement_fixtures.evaluate(
        data["evaluation"][task["common"]], backend.get("parameters")
    )
    reported = backend.get("objective_half_sum_squares")
    independent = result.get("independent_analytic_endpoint_objective")
    difference = (
        abs(reported - independent)
        if reported is not None and independent is not None
        else None
    )
    result["analytic_rollout_objective_difference"] = difference
    result["analytic_rollout_objective_agreement_passed"] = (
        difference <= 1e-5 * (1 + abs(independent)) if difference is not None else None
    )
    return result


def run_task(root: Path, index: int) -> dict:
    """Run one bounded arm, seal its endpoint, then open the analytic evaluator."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown measurement task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    execution = fit_identity(plan, data, task)
    evaluator = data["evaluation"][task["common"]]
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return completed(folder, identity, execution, evaluator)
        if (folder / "backend.json").exists():
            backend = read_seal(folder / "backend.json")
            validate_backend(backend, identity, execution)
        else:
            began, cpu = monotonic(), process_time()
            try:
                backend = measurement_fit.run(
                    training_problem(data, task),
                    Policy.model_validate(plan["policy"]),
                    folder / "fit",
                    strategy=task["strategy"],
                )
            except (ValueError, RuntimeError, ArithmeticError) as error:
                backend = {
                    "status": "failed",
                    "identity": execution,
                    "parameters": None,
                    "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                    "cost": {
                        "wall_seconds": monotonic() - began,
                        "cpu_seconds": process_time() - cpu,
                        "accounting_complete": False,
                    },
                    "confidence_probability": None,
                }
            backend = backend | {"campaign_identity": identity}
            validate_backend(backend, identity, execution)
            seal(folder / "backend.json", backend)
        # No evaluator field reaches optimization, selection or verification.
        evaluation = evaluate_once(folder, data, task, backend, identity)
        result = task | {
            "identity": identity,
            "status": backend["status"],
            "backend": backend,
            "backend_sha256": public.content_sha256(backend),
            "evaluation": evaluation,
        }
        seal(folder / "result.json", result)
        return result


def report_row(root: Path, plan: dict, data: dict, task: dict) -> dict:
    """Keep known committed fit costs even when final evaluation was interrupted."""
    folder = root / "results" / task["task_id"]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    execution = fit_identity(plan, data, task)
    evaluator = data["evaluation"][task["common"]]
    if (folder / "result.json").exists():
        return completed(folder, identity, execution, evaluator)
    row = task | {"status": "missing"}
    backend = None
    if (folder / "backend.json").exists():
        backend = read_seal(folder / "backend.json")
    elif (folder / "fit/result.json").exists():
        backend = read_seal(folder / "fit/result.json") | {
            "campaign_identity": identity
        }
    if backend is not None:
        validate_backend(backend, identity, execution)
        row["backend"] = backend
        row["partial_backend_retained"] = True
        expected = evaluation_identity(identity, backend, evaluator)
        if (folder / "evaluation.json").exists():
            evaluation = read_seal(folder / "evaluation.json")
            validate_evaluation(evaluation, expected)
            if read_seal(folder / "evaluation-started.json") != expected:
                raise ValueError("measurement evaluation journal identity differs")
            row["evaluation"] = evaluation
        elif (folder / "evaluation-started.json").exists():
            if read_seal(folder / "evaluation-started.json") != expected:
                raise ValueError("measurement evaluation journal identity differs")
            row["evaluation"] = {
                "status": "unfinished",
                "wall_seconds": None,
                "cpu_seconds": None,
                "accounting_complete": False,
            }
    return row


def report(root: Path) -> dict:
    """Retain missing/failed arms in denominators and compare matched objectives."""
    plan, data = verify(root, runtime=False)
    rows = [report_row(root, plan, data, task) for task in plan["tasks"]]
    groups = []
    for case in measurement_fixtures.CASES:
        for strategy in STRATEGIES:
            selected = [
                r for r in rows if r["case"] == case and r["strategy"] == strategy
            ]
            groups.append(
                {
                    "case": case,
                    "strategy": strategy,
                    "expected": len(selected),
                    "recorded": sum(r["status"] != "missing" for r in selected),
                    "status_counts": dict(Counter(r["status"] for r in selected)),
                    "analytic_objective_passed": sum(
                        r.get("evaluation", {}).get("analytic_objective_passed") is True
                        for r in selected
                    ),
                    "solver_agreement_passed": sum(
                        r.get("backend", {}).get("verification", {}).get("passed")
                        is True
                        for r in selected
                    ),
                    "analytic_rollout_objective_agreement_passed": sum(
                        r.get("evaluation", {}).get(
                            "analytic_rollout_objective_agreement_passed"
                        )
                        is True
                        for r in selected
                    ),
                }
            )
    pairs = []
    for common in sorted({r["common"] for r in rows}):
        pair = [r for r in rows if r["common"] == common]
        losses = [r.get("backend", {}).get("objective_half_sum_squares") for r in pair]
        available = all(loss is not None for loss in losses)
        gap = abs(losses[0] - losses[1]) if available else None
        pairs.append(
            {
                "common": common,
                "available": available,
                "absolute_objective_difference": gap,
                "objective_agreement_passed": (
                    gap <= 1e-5 * (1 + max(abs(loss) for loss in losses))
                    if available
                    else None
                ),
                "maximum_parameter_difference": (
                    max(
                        abs(
                            pair[0]["backend"]["parameters"][name]
                            - pair[1]["backend"]["parameters"][name]
                        )
                        for name in pair[0]["backend"]["parameters"]
                    )
                    if pair[0]["case"] != "nonlinear_decay"
                    and all(r.get("backend", {}).get("parameters") for r in pair)
                    else None
                ),
            }
        )
    counts = dict(Counter(r["status"] for r in rows))
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if "missing" not in counts else "incomplete",
        "expected": len(rows),
        "recorded": sum(r["status"] != "missing" for r in rows),
        "status_counts": counts,
        "evaluation_status_counts": dict(
            Counter(r.get("evaluation", {}).get("status", "missing") for r in rows)
        ),
        "rows": rows,
        "groups": groups,
        "paired_objectives": pairs,
        "fit_wall_seconds_observed": sum(
            r.get("backend", {}).get("cost", {}).get("wall_seconds", 0) for r in rows
        ),
        "evaluation_wall_seconds_observed": sum(
            r.get("evaluation", {}).get("wall_seconds") or 0 for r in rows
        ),
        "evaluation_cpu_seconds_observed": sum(
            r.get("evaluation", {}).get("cpu_seconds") or 0 for r in rows
        ),
        "cost_complete": all(
            r["status"] != "missing"
            and r.get("backend", {}).get("cost", {}).get("accounting_complete", False)
            and r.get("evaluation", {}).get("accounting_complete", False)
            for r in rows
        ),
        "cost_scope": (
            "Known committed fit stages/setup and analytic evaluation; interrupted "
            "unknown usage is omitted, not zero. Preparation/reporting are excluded."
        ),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "confidence_probability": None,
        "limitation": (
            "Small known-equation fixed-sigma qualification, not a diagnosis study. "
            "No confidence calibration, specification/data-fault attribution or "
            "noiseless parameter-recovery gate. Analytic references and clean "
            "predictions are evaluator-only. Completion includes failed attempts."
        ),
    }
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
