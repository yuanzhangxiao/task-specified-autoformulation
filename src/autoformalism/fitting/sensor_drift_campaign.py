"""M36 shared sensor drift with immutable inputs and post-fit evaluation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import monotonic, process_time

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import (
    drift_information,
    measurement_fit,
    sensor_drift_fixtures,
)
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.robust_measurement import Loss
from autoformalism.fitting.robust_measurement_campaign import _cost_summary
from autoformalism.schemas.sensor_drift import DriftProblem

PROTOCOL = "phase-c-sensor-drift-fitting-1"
INPUT_PROTOCOL = "phase-c-sensor-drift-fitting-inputs-1"
STRATEGIES = ("direct", "profiled")
LOSSES = ("gaussian", "huber")
HYPOTHESES = ("no_drift", "shared_drift")
DIAGNOSTIC_SECONDS = 15.0
Policy = measurement_fit.Policy


def loss_definition(kind: str) -> Loss:
    """Use the prespecified threshold; neither truth nor residuals can tune it."""
    return Loss(kind=kind, delta=sensor_drift_fixtures.HUBER_DELTA)


def roster(data: dict) -> list[dict]:
    """Validate six cases, three seeds and eight matched observation/loss arms."""
    expected = {
        f"{name}_s{seed}"
        for name in sensor_drift_fixtures.CASES
        for seed in sensor_drift_fixtures.SEEDS
    }
    if (
        data["protocol"] != INPUT_PROTOCOL
        or data["test_data_opened"] is not False
        or set(data["problems"]) != expected
        or set(data["evaluation"]) != expected
    ):
        raise ValueError("sensor drift qualification roster differs")
    sensor_drift_fixtures.verify_raw_artifacts(data)
    return [
        {
            "task_id": f"{name}_s{seed}__{hypothesis}__{kind}__{strategy}",
            "common": f"{name}_s{seed}",
            "case": name,
            "seed": seed,
            "strategy": strategy,
            "hypothesis": hypothesis,
            "loss": loss_definition(kind).model_dump(mode="json"),
        }
        for name in sensor_drift_fixtures.CASES
        for seed in sensor_drift_fixtures.SEEDS
        for hypothesis in HYPOTHESES
        for kind in LOSSES
        for strategy in STRATEGIES
    ]


def training_problem(data: dict, task: dict):
    """Pass only the explicitly sealed sensor drift problem to the fitter."""
    problem = DriftProblem.model_validate(data["problems"][task["common"]])
    if task["hypothesis"] not in HYPOTHESES:
        raise ValueError("unknown sensor drift observation hypothesis")
    return problem if task["hypothesis"] == "shared_drift" else problem.base


def diagnostic_policy() -> dict:
    """Bind the separate post-fit diagnostic and its allowance to the plan."""
    return {
        "protocol": drift_information.PROTOCOL,
        "policy": drift_information.POLICY,
        "maximum_seconds": DIAGNOSTIC_SECONDS,
    }


def prepare(root: Path, inputs: Path | None, policy: Policy) -> dict:
    """Freeze generated controls once, or import their exact portable export."""
    if inputs is None and (root / "inputs.json").exists():
        inputs = root / "inputs.json"
    data = (
        read_seal(inputs) if inputs is not None else sensor_drift_fixtures.make_inputs()
    )
    tasks = roster(data)
    for task in tasks[::4]:
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
        "losses": [loss_definition(kind).model_dump(mode="json") for kind in LOSSES],
        "matched_policy_and_starts": True,
        "diagnostic": diagnostic_policy(),
        "observation_hypothesis_selection": False,
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
        or plan["losses"]
        != [loss_definition(kind).model_dump(mode="json") for kind in LOSSES]
        or plan["matched_policy_and_starts"] is not True
        or plan["diagnostic"] != diagnostic_policy()
        or plan["observation_hypothesis_selection"] is not False
    ):
        raise ValueError("sensor drift campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("sensor drift source/runtime differs")
    return plan, data


def fit_identity(plan: dict, data: dict, task: dict) -> dict:
    """Derive the fitter's exact identity without passing evaluator data to it."""
    problem = training_problem(data, task)
    return {
        "protocol": "sensor-drift-fitting-execution-1"
        if task["hypothesis"] == "shared_drift"
        else "robust-measurement-fitting-execution-1",
        "source_sha256": plan["source_sha256"],
        "runtime": plan["runtime"],
        "problem_sha256": public.content_sha256(problem.model_dump(mode="json")),
        "base_request_sha256": public.content_sha256(
            problem.request.model_dump(mode="json")
        ),
        "policy": plan["policy"],
        "strategy": task["strategy"],
        "objective_definition": task["loss"],
    }


def validate_backend(backend: dict, identity: dict, execution: dict) -> None:
    """Reject a copied/stale intermediate backend before scoring or terminal reuse."""
    if (
        backend.get("identity") != execution
        or backend.get("campaign_identity") != identity
    ):
        raise ValueError("sensor drift backend identity differs")


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
        raise ValueError("sensor drift evaluation identity differs")


def diagnostic_identity(identity: dict, backend: dict) -> dict:
    """Bind local sensitivity evidence to the frozen endpoint and public policy."""
    return {
        "task_identity": identity,
        "backend_sha256": public.content_sha256(backend),
        "diagnostic": diagnostic_policy(),
    }


def _diagnostic_progress(folder: Path, expected: dict) -> dict:
    """Retain known observed counters without treating progress as final usage."""
    path = folder / "diagnostic-progress.json"
    if not path.exists():
        return {"calls": None, "completed_calls": None}
    progress = public._read(path)
    if progress.get("identity") != expected:
        raise ValueError("sensor drift diagnostic progress identity differs")
    counts = {key: progress.get(key) for key in ("calls", "completed_calls")}
    if any(
        value is not None and (type(value) is not int or value < 0)
        for value in counts.values()
    ):
        raise ValueError("invalid sensor drift diagnostic progress counters")
    return counts


def read_diagnostic(
    folder: Path, identity: dict, backend: dict, *, terminal: bool = False
) -> dict | None:
    """Verify a completed diagnostic or show interrupted usage without rerunning."""
    expected = diagnostic_identity(identity, backend)
    path, started = folder / "diagnostic.json", folder / "diagnostic-started.json"
    if path.exists():
        result = read_seal(path)
        if result.get("identity") != expected or read_seal(started) != expected:
            raise ValueError("sensor drift diagnostic identity differs")
        return result
    if terminal:
        raise ValueError("terminal sensor drift diagnostic missing")
    if not started.exists():
        return None
    if read_seal(started) != expected:
        raise ValueError("sensor drift diagnostic journal identity differs")
    return {
        "status": "unfinished",
        "identity": expected,
        "information_status": "unavailable",
        "wall_seconds": None,
        "cpu_seconds": None,
        "accounting_complete": False,
        "budget_charge_seconds": DIAGNOSTIC_SECONDS,
        "observed_progress": _diagnostic_progress(folder, expected),
    }


def diagnose_once(
    folder: Path, data: dict, task: dict, backend: dict, identity: dict
) -> dict:
    """Spend one diagnostic allowance, preserving failed/interrupted outcomes."""
    previous = read_diagnostic(folder, identity, backend)
    if previous is not None:
        if previous["status"] != "unfinished":
            return previous
        result = previous | {
            "status": "interrupted",
            "reason": (
                "Started diagnostic has unknown usage; its allowance is consumed "
                "and it was not repeated."
            ),
        }
    else:
        expected = diagnostic_identity(identity, backend)
        seal(folder / "diagnostic-started.json", expected)
        began, cpu = monotonic(), process_time()

        def save(value):
            public._write(
                folder / "diagnostic-progress.json", value | {"identity": expected}
            )

        save({"calls": 0, "completed_calls": 0})
        try:
            if backend.get("parameters") is None:
                result = {
                    "status": "no_parameters",
                    "information_status": "unavailable",
                    "calls": 0,
                    "completed_calls": 0,
                }
            else:
                result = drift_information.evaluate(
                    training_problem(data, task),
                    backend["parameters"],
                    loss=Loss.model_validate(task["loss"]),
                    deadline=began + DIAGNOSTIC_SECONDS,
                    save=save,
                )
                result = {"status": "evaluated"} | result
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
            result = {
                "status": "failed",
                "information_status": "unavailable",
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
                "observed_progress": _diagnostic_progress(folder, expected),
            }
        wall = monotonic() - began
        result |= {
            "identity": expected,
            "wall_seconds": wall,
            "cpu_seconds": process_time() - cpu,
            "accounting_complete": True,
            "budget_charge_seconds": min(wall, DIAGNOSTIC_SECONDS),
        }
    seal(folder / "diagnostic.json", result)
    return result


def completed(folder: Path, identity: dict, execution: dict, evaluator: dict) -> dict:
    """Verify the final fit and evaluator seals before reusing a terminal result."""
    result = read_seal(folder / "result.json")
    if result["identity"] != identity:
        raise ValueError("sensor drift result identity differs")
    backend = read_seal(folder / "backend.json")
    validate_backend(backend, identity, execution)
    evaluation = read_seal(folder / "evaluation.json")
    expected_evaluation = evaluation_identity(identity, backend, evaluator)
    validate_evaluation(evaluation, expected_evaluation)
    if read_seal(folder / "evaluation-started.json") != expected_evaluation:
        raise ValueError("sensor drift evaluation journal identity differs")
    if (
        result["backend_sha256"] != public.content_sha256(backend)
        or result["backend"] != backend
        or result["evaluation"] != evaluation
    ):
        raise ValueError("sensor drift backend/evaluation differs")
    diagnostic = read_diagnostic(folder, identity, backend, terminal=True)
    if result["diagnostic"] != diagnostic:
        raise ValueError("sensor drift diagnostic differs")
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
            raise ValueError("sensor drift evaluation journal identity differs")
        return result
    if started.exists():
        if read_seal(started) != expected:
            raise ValueError("sensor drift evaluation journal identity differs")
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
    result = sensor_drift_fixtures.evaluate(
        data["evaluation"][task["common"]],
        backend.get("parameters"),
        task["hypothesis"],
        task["loss"]["kind"],
        task["loss"]["delta"],
    )
    reported = backend.get("objective_value")
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
        raise ValueError("unknown sensor drift task index")
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
                    loss=Loss.model_validate(task["loss"]),
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
        # The numerical diagnostic sees the chosen public hypothesis only.
        diagnostic = diagnose_once(folder, data, task, backend, identity)
        # No evaluator field reaches optimization, selection or diagnostics.
        evaluation = evaluate_once(folder, data, task, backend, identity)
        result = task | {
            "identity": identity,
            "status": backend["status"],
            "backend": backend,
            "backend_sha256": public.content_sha256(backend),
            "evaluation": evaluation,
            "diagnostic": diagnostic,
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
        diagnostic = read_diagnostic(folder, identity, backend)
        if diagnostic is not None:
            row["diagnostic"] = diagnostic
        expected = evaluation_identity(identity, backend, evaluator)
        if (folder / "evaluation.json").exists():
            evaluation = read_seal(folder / "evaluation.json")
            validate_evaluation(evaluation, expected)
            if read_seal(folder / "evaluation-started.json") != expected:
                raise ValueError("sensor drift evaluation journal identity differs")
            row["evaluation"] = evaluation
        elif (folder / "evaluation-started.json").exists():
            if read_seal(folder / "evaluation-started.json") != expected:
                raise ValueError("sensor drift evaluation journal identity differs")
            row["evaluation"] = {
                "status": "unfinished",
                "wall_seconds": None,
                "cpu_seconds": None,
                "accounting_complete": False,
            }
    return row


def _paired_objectives(rows: list[dict]) -> list[dict]:
    """Compare strategies within the same hypothesis and loss, retaining failures."""
    pairs = []
    for common in sorted({row["common"] for row in rows}):
        for hypothesis in HYPOTHESES:
            for kind in LOSSES:
                pair = [
                    row
                    for row in rows
                    if row["common"] == common
                    and row["hypothesis"] == hypothesis
                    and row["loss"]["kind"] == kind
                ]
                values = [row.get("backend", {}).get("objective_value") for row in pair]
                available = all(value is not None for value in values)
                gap = abs(values[0] - values[1]) if available else None
                pairs.append(
                    {
                        "common": common,
                        "hypothesis": hypothesis,
                        "loss": kind,
                        "available": available,
                        "absolute_objective_difference": gap,
                        "objective_agreement_passed": gap
                        <= 1e-5 * (1 + max(map(abs, values)))
                        if available
                        else None,
                        "parameter_agreement_required": False,
                        "interpretation": (
                            "Equal objective values do not establish equal or "
                            "identified physical parameters."
                        ),
                    }
                )
    return pairs


def _hypothesis_comparisons(rows: list[dict]) -> list[dict]:
    """Display each hypothesis without choosing one from training loss or truth."""
    comparisons = []
    for common in sorted({row["common"] for row in rows}):
        for kind in LOSSES:
            for strategy in STRATEGIES:
                selected = [
                    row
                    for row in rows
                    if row["common"] == common
                    and row["loss"]["kind"] == kind
                    and row["strategy"] == strategy
                ]
                comparisons.append(
                    {
                        "common": common,
                        "loss": kind,
                        "strategy": strategy,
                        "hypotheses": {
                            row["hypothesis"]: {
                                "status": row["status"],
                                "objective_value": row.get("backend", {}).get(
                                    "objective_value"
                                ),
                                "clean_physical_rmse_evaluator_only": row.get(
                                    "evaluation", {}
                                ).get("clean_physical_prediction_rmse_evaluator_only"),
                                "clean_measured_rmse_evaluator_only": row.get(
                                    "evaluation", {}
                                ).get("clean_measured_prediction_rmse_evaluator_only"),
                                "physical_errors_evaluator_only": row.get(
                                    "evaluation", {}
                                ).get("physical_parameter_truth_error_evaluator_only"),
                                "sensor_drift_truth_error_evaluator_only": row.get(
                                    "evaluation", {}
                                ).get("sensor_drift_truth_error_evaluator_only"),
                                "information_status": row.get("diagnostic", {}).get(
                                    "information_status", "missing"
                                ),
                            }
                            for row in selected
                        },
                        "selected_hypothesis": None,
                        "data_fault_attribution": None,
                    }
                )
    return comparisons


def _information_summary(rows: list[dict]) -> dict:
    """Keep public local-information findings separate from evaluator recovery."""
    diagnostics = [row.get("diagnostic", {}) for row in rows]
    counters = {}
    for name in ("calls", "completed_calls"):
        values = [value.get(name) for value in diagnostics]
        counters[name + "_observed"] = sum(
            value.get(name)
            if value.get(name) is not None
            else value.get("observed_progress", {}).get(name) or 0
            for value in diagnostics
        )
        counters[name + "_unknown_or_missing"] = sum(value is None for value in values)
        counters[name + "_total"] = (
            sum(values) if all(value is not None for value in values) else None
        )
    return {
        "expected": len(rows),
        "status_counts": dict(
            Counter(value.get("status", "missing") for value in diagnostics)
        ),
        "information_status_counts": dict(
            Counter(value.get("information_status", "missing") for value in diagnostics)
        ),
        "derivatives_passed": sum(
            value.get("derivatives_passed") is True for value in diagnostics
        ),
        "physical_conditional_rank_counts": dict(
            Counter(
                str(value["information"]["physical_after_nuisance"]["rank"])
                for value in diagnostics
                if isinstance(value.get("information"), dict)
            )
        ),
        "wall_seconds_observed": sum(
            value.get("wall_seconds") or 0 for value in diagnostics
        ),
        "cpu_seconds_observed": sum(
            value.get("cpu_seconds") or 0 for value in diagnostics
        ),
        "budget_charge_seconds": sum(
            value.get("budget_charge_seconds") or 0 for value in diagnostics
        ),
        "accounting_complete": all(
            value.get("accounting_complete", False)
            and value.get("calls") is not None
            and value.get("completed_calls") is not None
            for value in diagnostics
        ),
    } | counters


def report(root: Path) -> dict:
    """Preserve failures, ambiguity, distinct outputs and diagnostic costs."""
    plan, data = verify(root, runtime=False)
    rows = [report_row(root, plan, data, task) for task in plan["tasks"]]
    groups = []
    for case in sensor_drift_fixtures.CASES:
        for hypothesis in HYPOTHESES:
            for kind in LOSSES:
                for strategy in STRATEGIES:
                    selected = [
                        row
                        for row in rows
                        if row["case"] == case
                        and row["hypothesis"] == hypothesis
                        and row["loss"]["kind"] == kind
                        and row["strategy"] == strategy
                    ]
                    groups.append(
                        {
                            "case": case,
                            "hypothesis": hypothesis,
                            "loss": kind,
                            "strategy": strategy,
                            "expected": len(selected),
                            "recorded": sum(
                                row["status"] != "missing" for row in selected
                            ),
                            "status_counts": dict(
                                Counter(row["status"] for row in selected)
                            ),
                            "analytic_objective_applicable": 0
                            if case == "drift_nonlinear_decay"
                            else len(selected),
                            "analytic_objective_evaluated": sum(
                                row.get("evaluation", {}).get(
                                    "analytic_objective_passed"
                                )
                                is not None
                                for row in selected
                            ),
                            "analytic_objective_passed": sum(
                                row.get("evaluation", {}).get(
                                    "analytic_objective_passed"
                                )
                                is True
                                for row in selected
                            ),
                            "analytic_rollout_objective_agreement_passed": sum(
                                row.get("evaluation", {}).get(
                                    "analytic_rollout_objective_agreement_passed"
                                )
                                is True
                                for row in selected
                            ),
                            "solver_agreement_passed": sum(
                                row.get("backend", {})
                                .get("verification", {})
                                .get("passed")
                                is True
                                for row in selected
                            ),
                            "profile_initial_fallback_count": sum(
                                strategy == "profiled"
                                and row.get("backend", {})
                                .get("route", {})
                                .get("selected")
                                == "joint_rollout"
                                for row in selected
                            ),
                            "profile_runtime_fallback_count": sum(
                                any(
                                    attempt.get("profile_failures")
                                    for attempt in row.get("backend", {})
                                    .get("search", {})
                                    .get("attempts", [])
                                )
                                for row in selected
                            ),
                            "cost": _cost_summary(selected),
                            "diagnostic": _information_summary(selected),
                        }
                    )
    counts = dict(Counter(row["status"] for row in rows))
    diagnostic = _information_summary(rows)
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if "missing" not in counts else "incomplete",
        "expected": len(rows),
        "recorded": sum(row["status"] != "missing" for row in rows),
        "status_counts": counts,
        "evaluation_status_counts": dict(
            Counter(row.get("evaluation", {}).get("status", "missing") for row in rows)
        ),
        "diagnostic_status_counts": diagnostic["status_counts"],
        "rows": rows,
        "groups": groups,
        "paired_objectives": _paired_objectives(rows),
        "observation_hypothesis_comparisons": _hypothesis_comparisons(rows),
        "information_by_hypothesis": [
            {"hypothesis": hypothesis}
            | _information_summary(
                [row for row in rows if row["hypothesis"] == hypothesis]
            )
            for hypothesis in HYPOTHESES
        ],
        "cost_by_hypothesis_loss_and_strategy": [
            {"hypothesis": hypothesis, "loss": kind, "strategy": strategy}
            | _cost_summary(
                [
                    row
                    for row in rows
                    if row["hypothesis"] == hypothesis
                    and row["loss"]["kind"] == kind
                    and row["strategy"] == strategy
                ]
            )
            for hypothesis in HYPOTHESES
            for kind in LOSSES
            for strategy in STRATEGIES
        ],
        "fit_wall_seconds_observed": sum(
            row.get("backend", {}).get("cost", {}).get("wall_seconds", 0)
            for row in rows
        ),
        "evaluation_wall_seconds_observed": sum(
            row.get("evaluation", {}).get("wall_seconds") or 0 for row in rows
        ),
        "evaluation_cpu_seconds_observed": sum(
            row.get("evaluation", {}).get("cpu_seconds") or 0 for row in rows
        ),
        "diagnostic_wall_seconds_observed": diagnostic["wall_seconds_observed"],
        "diagnostic_cpu_seconds_observed": diagnostic["cpu_seconds_observed"],
        "diagnostic_budget_charge_seconds": diagnostic["budget_charge_seconds"],
        "diagnostic_calls_observed": diagnostic["calls_observed"],
        "diagnostic_completed_calls_observed": diagnostic["completed_calls_observed"],
        "diagnostic_calls_total": diagnostic["calls_total"],
        "diagnostic_completed_calls_total": diagnostic["completed_calls_total"],
        "diagnostic_calls_unknown_or_missing": diagnostic["calls_unknown_or_missing"],
        "cost_complete": _cost_summary(rows)["accounting_complete"]
        and diagnostic["accounting_complete"]
        and all(
            row.get("evaluation", {}).get("accounting_complete", False) for row in rows
        ),
        "cost_scope": (
            "Known committed fit/setup, independent evaluator and separate bounded "
            "information diagnostic. Interrupted unknown usage is omitted, not "
            "zero; diagnostic allowance remains charged. Preparation/reporting "
            "excluded."
        ),
        "observation_hypothesis_selected": None,
        "data_fault_attribution": None,
        "confidence_probability": None,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": (
            "Small shared linear observation-drift qualification with known sigma, "
            "not an automatic sensor-fault or model-selection procedure. "
            "Objectives are compared only across strategies with identical "
            "hypothesis and loss. Local information is conditional on the "
            "supplied model and coordinate/rank conventions, not confidence or "
            "global identification. Equal fit does not identify physical "
            "parameters in the confounded control. Affine reference optima do not "
            "assert unique parameters; nonlinear global optima are unassessed. "
            "All missing and failed attempts remain in denominators."
        ),
    }
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
