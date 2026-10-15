"""M39 matched Gaussian fits with separately journaled practical profile evidence."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from time import monotonic, process_time

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import measurement_fit, practical_profile
from autoformalism.fitting import practical_uncertainty_fixtures as fixtures
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.robust_measurement import Loss
from autoformalism.fitting.robust_measurement_campaign import _cost_summary
from autoformalism.fitting.sensor_drift_campaign import validate_backend
from autoformalism.schemas.sensor_drift import DriftProblem

PROTOCOL = "phase-c-practical-uncertainty-fitting-1"
INPUT_PROTOCOL = fixtures.INPUT_PROTOCOL
STRATEGIES = ("direct", "profiled")
PROFILE_SECONDS = 30.0
COUNTERS = (
    "calls",
    "completed_calls",
    "solve_calls",
    "rollout_calls",
    "inner_iterations",
    "nodes_completed",
    "failed_calls",
)
Policy = measurement_fit.Policy


def diagnostic_policy() -> dict:
    """Freeze the target, support convention, profile solver and separate budget."""
    return {
        "protocol": practical_profile.PROTOCOL,
        "policy": practical_profile.POLICY,
        "target": "g",
        "maximum_seconds": PROFILE_SECONDS,
    }


def roster(data: dict) -> list[dict]:
    """Validate the full 75-acquisition roster and paired strategy denominator."""
    expected = {
        fixtures.case_key(separation, sigma, seed)
        for separation in fixtures.SEPARATIONS
        for sigma in fixtures.SIGMAS
        for seed in fixtures.SEEDS
    }
    if (
        data["protocol"] != INPUT_PROTOCOL
        or data["test_data_opened"] is not False
        or set(data["problems"]) != expected
        or set(data["evaluation"]) != expected
    ):
        raise ValueError("practical uncertainty acquisition roster differs")
    fixtures.verify_raw_artifacts(data)
    return [
        {
            "task_id": fixtures.case_key(separation, sigma, seed) + "__" + strategy,
            "common": fixtures.case_key(separation, sigma, seed),
            "separation": separation,
            "sigma": sigma,
            "seed": seed,
            "strategy": strategy,
        }
        for separation in fixtures.SEPARATIONS
        for sigma in fixtures.SIGMAS
        for seed in fixtures.SEEDS
        for strategy in STRATEGIES
    ]


def training_problem(data: dict, task: dict) -> DriftProblem:
    """Read only the public frozen acquisition and bounded candidate hypothesis."""
    return DriftProblem.model_validate(data["problems"][task["common"]])


def prepare(root: Path, inputs: Path | None, policy: Policy) -> dict:
    """Generate once or import exact sealed inputs; later preparations never redraw."""
    if inputs is None and (root / "inputs.json").exists():
        inputs = root / "inputs.json"
    data = read_seal(inputs) if inputs is not None else fixtures.make_inputs()
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
        "objective_definition": Loss(kind="gaussian").model_dump(mode="json"),
        "diagnostic": diagnostic_policy(),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "qualification_only": True,
        "truth_used_for_selection": False,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(tasks)}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Reject changed inputs, protocol, objective, budgets or execution identity."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != roster(data)
        or plan["objective_definition"] != Loss(kind="gaussian").model_dump(mode="json")
        or plan["diagnostic"] != diagnostic_policy()
        or plan["test_data_opened"] is not False
        or plan["live_llm_calls"] != 0
        or plan["qualification_only"] is not True
        or plan["truth_used_for_selection"] is not False
    ):
        raise ValueError("practical uncertainty campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("practical uncertainty source/runtime differs")
    return plan, data


def fit_identity(plan: dict, data: dict, task: dict) -> dict:
    """Match M36 execution identity while keeping the evaluator outside the fitter."""
    problem = training_problem(data, task)
    return {
        "protocol": "sensor-drift-fitting-execution-1",
        "source_sha256": plan["source_sha256"],
        "runtime": plan["runtime"],
        "problem_sha256": public.content_sha256(problem.model_dump(mode="json")),
        "base_request_sha256": public.content_sha256(
            problem.request.model_dump(mode="json")
        ),
        "policy": plan["policy"],
        "strategy": task["strategy"],
        "objective_definition": plan["objective_definition"],
    }


def stage_identity(identity: dict, backend: dict, definition: dict) -> dict:
    """Bind each post-fit stage to the exact selected endpoint and its own protocol."""
    return {
        "task_identity": identity,
        "backend_sha256": public.content_sha256(backend),
        "definition": definition,
    }


def _progress(folder: Path, name: str, identity: dict) -> dict:
    """Keep only identity-checked observed usage and point events after interruption."""
    path = folder / f"{name}-progress.jsonl"
    value, events, truncated = {}, [], False
    if path.exists():
        lines = path.read_bytes().splitlines(keepends=True)
        identity_sha = public.content_sha256(identity)
        for sequence, line in enumerate(lines):
            if not line.endswith(b"\n"):
                # A torn final write contributes no invented work. Earlier
                # durable start/finish counters remain lower bounds only.
                truncated = True
                break
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or record.get("identity_sha256") != identity_sha
                or record.get("sequence") != sequence
                or not isinstance(record.get("value"), dict)
            ):
                raise ValueError("practical uncertainty progress identity differs")
            value = record["value"]
            if "event" in value:
                events.append(value)
    counters = {key: value.get(key) for key in COUNTERS}
    if any(
        item is not None and (type(item) is not int or item < 0)
        for item in counters.values()
    ):
        raise ValueError("invalid practical uncertainty progress counters")
    if not isinstance(events, list) or any(
        not isinstance(event, dict) for event in events
    ):
        raise ValueError("invalid practical uncertainty point event history")
    return counters | {
        "events": events,
        "inner_iterations_complete": value.get("inner_iterations_complete"),
        "truncated_final_record": truncated,
    }


def read_stage(
    folder: Path,
    name: str,
    identity: dict,
    *,
    allowance: float | None,
    terminal: bool = False,
) -> dict | None:
    """Check final evidence or retain interrupted stage allowance and partial points."""
    path, started = folder / f"{name}.json", folder / f"{name}-started.json"
    if path.exists():
        result = read_seal(path)
        if result.get("identity") != identity or read_seal(started) != identity:
            raise ValueError("practical uncertainty stage identity differs")
        return result
    if terminal:
        raise ValueError("terminal practical uncertainty stage missing")
    if not started.exists():
        return None
    if read_seal(started) != identity:
        raise ValueError("practical uncertainty stage journal identity differs")
    return {
        "status": "unfinished",
        "identity": identity,
        "wall_seconds": None,
        "cpu_seconds": None,
        "accounting_complete": False,
        "budget_charge_seconds": allowance,
        "observed_progress": _progress(folder, name, identity),
    }


def _stage_once(
    folder: Path,
    name: str,
    identity: dict,
    operation: Callable,
    allowance: float | None,
) -> dict:
    """Run a post-fit stage once; interrupted work never receives renewed allowance."""
    previous = read_stage(folder, name, identity, allowance=allowance)
    if previous is not None:
        if previous["status"] != "unfinished":
            return previous
        result = previous | {
            "status": "interrupted",
            "reason": "Started stage has unknown usage and is not repeated.",
        }
    else:
        seal(folder / f"{name}-started.json", identity)
        began, cpu = monotonic(), process_time()
        events = []
        sequence = 0
        journal = folder / f"{name}-progress.jsonl"
        identity_sha = public.content_sha256(identity)

        def save(value):
            nonlocal sequence
            record = {
                "identity_sha256": identity_sha,
                "sequence": sequence,
                "value": value,
            }
            # Append one compact durable record, rather than rewriting the
            # growing history at every node. This keeps files and I/O bounded.
            with journal.open("a") as stream:
                stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            sequence += 1
            if "event" in value:
                events.append(value)

        save({"calls": 0, "completed_calls": 0})
        try:
            result = operation(
                began + allowance if allowance is not None else float("inf"), save
            )
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
            result = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
                "observed_progress": _progress(folder, name, identity),
            }
        wall = monotonic() - began
        result |= {
            "identity": identity,
            "wall_seconds": wall,
            "cpu_seconds": process_time() - cpu,
            "accounting_complete": True,
            "event_history": events,
            "budget_charge_seconds": min(wall, allowance)
            if allowance is not None
            else None,
        }
    seal(folder / f"{name}.json", result)
    return result


def evaluate(data: dict, task: dict, backend: dict) -> dict:
    """Compare only a sealed fit with exact independent Gaussian references."""
    value = fixtures.evaluate(
        data["evaluation"][task["common"]], backend.get("parameters")
    )
    reported, independent = (
        backend.get("objective_value"),
        value.get("independent_analytic_endpoint_objective"),
    )
    difference = (
        abs(reported - independent)
        if reported is not None and independent is not None
        else None
    )
    return value | {
        "analytic_rollout_objective_difference": difference,
        "analytic_rollout_objective_agreement_passed": difference
        <= 1e-5 * (1 + abs(independent))
        if difference is not None
        else None,
    }


def _stage_definitions(plan: dict, data: dict, task: dict) -> dict:
    """Keep public profile policy separate from evaluator-only reference identity."""
    return {
        "profile": plan["diagnostic"],
        "evaluation": {
            "protocol": "practical-uncertainty-evaluation-1",
            "evaluator_sha256": public.content_sha256(
                data["evaluation"][task["common"]]
            ),
        },
    }


def completed(folder: Path, identity: dict, execution: dict, definitions: dict) -> dict:
    """Verify nested endpoint and stage seals before returning a terminal no-op."""
    result, backend = (
        read_seal(folder / "result.json"),
        read_seal(folder / "backend.json"),
    )
    validate_backend(backend, identity, execution)
    if (
        result.get("identity") != identity
        or result.get("backend") != backend
        or result.get("backend_sha256") != public.content_sha256(backend)
    ):
        raise ValueError("practical uncertainty terminal backend identity differs")
    for name, definition in definitions.items():
        value = read_stage(
            folder,
            name,
            stage_identity(identity, backend, definition),
            allowance=PROFILE_SECONDS if name == "profile" else None,
            terminal=True,
        )
        if result.get(name) != value:
            raise ValueError("practical uncertainty terminal stage differs")
    return result


def run_task(root: Path, index: int) -> dict:
    """Seal the selected endpoint, profile it using public data, then open truth."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown practical uncertainty task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    execution, definitions = (
        fit_identity(plan, data, task),
        _stage_definitions(plan, data, task),
    )
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return completed(folder, identity, execution, definitions)
        if (folder / "backend.json").exists():
            backend = read_seal(folder / "backend.json")
        else:
            began, cpu = monotonic(), process_time()
            try:
                backend = measurement_fit.run(
                    training_problem(data, task),
                    Policy.model_validate(plan["policy"]),
                    folder / "fit",
                    strategy=task["strategy"],
                    loss=Loss.model_validate(plan["objective_definition"]),
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
                }
            backend |= {"campaign_identity": identity}
            validate_backend(backend, identity, execution)
            seal(folder / "backend.json", backend)
        validate_backend(backend, identity, execution)

        def profile(deadline, save):
            if backend.get("parameters") is None:
                return {
                    "status": "no_parameters",
                    "inner_iterations_complete": True,
                } | dict.fromkeys(COUNTERS, 0)
            return practical_profile.evaluate(
                training_problem(data, task),
                backend,
                target=plan["diagnostic"]["target"],
                policy=practical_profile.Policy.model_validate(
                    plan["diagnostic"]["policy"]
                ),
                deadline=deadline,
                save=save,
            )

        def score(deadline, save):
            save({"calls": 1, "completed_calls": 0})
            value = evaluate(data, task, backend)
            save({"calls": 1, "completed_calls": 1})
            return value | {"calls": 1, "completed_calls": 1}

        profile_result = _stage_once(
            folder,
            "profile",
            stage_identity(identity, backend, definitions["profile"]),
            profile,
            PROFILE_SECONDS,
        )
        evaluation = _stage_once(
            folder,
            "evaluation",
            stage_identity(identity, backend, definitions["evaluation"]),
            score,
            None,
        )
        result = task | {
            "identity": identity,
            "status": backend["status"],
            "backend": backend,
            "backend_sha256": public.content_sha256(backend),
            "profile": profile_result,
            "evaluation": evaluation,
        }
        seal(folder / "result.json", result)
        return result


def report_row(root: Path, plan: dict, data: dict, task: dict) -> dict:
    """Retain partial committed endpoints and known stage usage in all reports."""
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    execution, definitions = (
        fit_identity(plan, data, task),
        _stage_definitions(plan, data, task),
    )
    folder = root / "results" / task["task_id"]
    if (folder / "result.json").exists():
        return completed(folder, identity, execution, definitions)
    row = task | {"status": "missing"}
    if (folder / "backend.json").exists():
        backend = read_seal(folder / "backend.json")
    elif (folder / "fit/result.json").exists():
        backend = read_seal(folder / "fit/result.json") | {
            "campaign_identity": identity
        }
    else:
        return row
    validate_backend(backend, identity, execution)
    row |= {"backend": backend, "partial_backend_retained": True}
    for name, definition in definitions.items():
        value = read_stage(
            folder,
            name,
            stage_identity(identity, backend, definition),
            allowance=PROFILE_SECONDS if name == "profile" else None,
        )
        if value is not None:
            row[name] = value
    return row


def _statistics(values: list) -> dict:
    """Report available medians and median absolute deviations without imputations."""
    available = np.array([value for value in values if value is not None], dtype=float)
    if not len(available):
        return {"available": 0, "missing": len(values), "median": None, "mad": None}
    median = float(np.median(available))
    return {
        "available": len(available),
        "missing": len(values) - len(available),
        "median": median,
        "mad": float(np.median(abs(available - median))),
    }


def _support_summary(rows: list[dict], threshold: float, name: str) -> dict:
    """Separate bounded or nuisance-assisted supports from interior data limits."""
    supports = []
    for row in rows:
        matches = [
            record
            for record in row.get("profile", {}).get("supports", [])
            if record["threshold"] == threshold
            and row.get("profile", {}).get("status") == "complete"
        ]
        supports.append(matches[0].get(name) if matches else None)

    def data_limited(value):
        return (
            name == "profiled"
            and value is not None
            and not value.get("domain_clipped", True)
            and all(
                not value.get(side, {}).get("nuisance_active_bounds")
                and value.get(side, {}).get("status") == "threshold_crossed"
                for side in ("left", "right")
            )
        )

    return {
        "threshold": threshold,
        "kind": name,
        "width": _statistics(
            [value.get("width") if value is not None else None for value in supports]
        ),
        "interior_data_limited_width": _statistics(
            [value.get("width") if data_limited(value) else None for value in supports]
        ),
        "domain_clipped": sum(
            bool(value and value.get("domain_clipped")) for value in supports
        ),
        "nuisance_domain_assisted": sum(
            bool(
                value
                and any(
                    value.get(side, {}).get("nuisance_active_bounds")
                    for side in ("left", "right")
                )
            )
            for value in supports
        ),
        "classification_counts": dict(
            Counter(
                value.get("classification", "unclassified")
                if value is not None
                else "unavailable"
                for value in supports
            )
        ),
        "interpretation": (
            "Frozen-nuisance slices are conditional contrasts; support widths "
            "are not calibrated confidence intervals."
        ),
    }


def _stage_cost(rows: list[dict], name: str) -> dict:
    """Count observed work separately from unavailable completed-stage totals."""
    values = [row.get(name, {}) for row in rows]
    result = {
        "wall_seconds_observed": sum(
            value.get("wall_seconds") or 0 for value in values
        ),
        "cpu_seconds_observed": sum(value.get("cpu_seconds") or 0 for value in values),
        "budget_charge_seconds": sum(
            value.get("budget_charge_seconds") or 0 for value in values
        ),
        "accounting_complete": all(
            value.get("accounting_complete", False) for value in values
        ),
    }
    for field in COUNTERS:
        counts = [value.get(field) for value in values]
        if field == "inner_iterations":
            counts = [
                count if value.get("inner_iterations_complete") is True else None
                for count, value in zip(counts, values, strict=True)
            ]
        result[field + "_observed"] = sum(
            value.get(field)
            if value.get(field) is not None
            else value.get("observed_progress", {}).get(field) or 0
            for value in values
        )
        result[field + "_unknown_or_missing"] = sum(value is None for value in counts)
        result[field + "_total"] = (
            sum(counts) if all(value is not None for value in counts) else None
        )
    result["accounting_complete"] &= all(
        value.get("calls") is not None and value.get("completed_calls") is not None
        for value in values
    )
    if name == "profile":
        result["accounting_complete"] &= all(
            value.get("inner_iterations_complete") is True for value in values
        )
    return result


def _group(rows: list[dict]) -> dict:
    """Aggregate all five fixed seeds, keeping error evidence evaluator-only."""
    evaluations = [row.get("evaluation", {}) for row in rows]
    curvatures = [row.get("profile", {}).get("curvature") or {} for row in rows]
    validation = [curvature_validation(row) for row in rows]
    return {
        "expected": len(rows),
        "recorded": sum(row["status"] != "missing" for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "profile_status_counts": dict(
            Counter(row.get("profile", {}).get("status", "missing") for row in rows)
        ),
        "analytic_objective_passed": sum(
            value.get("analytic_objective_passed") is True for value in evaluations
        ),
        "analytic_objective_evaluated": sum(
            value.get("analytic_objective_passed") is not None for value in evaluations
        ),
        "parameter_errors_evaluator_only": {
            parameter: _statistics(
                [
                    value.get("parameter_truth_error_evaluator_only", {}).get(parameter)
                    for value in evaluations
                ]
            )
            for parameter in fixtures.PARAMETERS
        },
        "absolute_parameter_errors_evaluator_only": {
            parameter: _statistics(
                [
                    value.get("absolute_parameter_truth_error_evaluator_only", {}).get(
                        parameter
                    )
                    for value in evaluations
                ]
            )
            for parameter in fixtures.PARAMETERS
        },
        "clean_physical_rmse_evaluator_only": _statistics(
            [
                value.get("clean_physical_prediction_rmse_evaluator_only")
                for value in evaluations
            ]
        ),
        "clean_measured_rmse_evaluator_only": _statistics(
            [
                value.get("clean_measured_prediction_rmse_evaluator_only")
                for value in evaluations
            ]
        ),
        "conditional_information": _statistics(
            [value.get("conditional_information") for value in curvatures]
        ),
        "conditional_information_analytic_passed": sum(
            value["conditional_passed"] is True for value in validation
        ),
        "frozen_information_analytic_passed": sum(
            value["frozen_passed"] is True for value in validation
        ),
        "curvature_analytic_evaluated": sum(
            value["conditional_passed"] is not None for value in validation
        ),
        "frozen_information": _statistics(
            [value.get("frozen_information") for value in curvatures]
        ),
        "joint_rank_counts": dict(
            Counter(
                str(value.get("joint_rank")) if value else "unavailable"
                for value in curvatures
            )
        ),
        "conditional_rank_counts": dict(
            Counter(
                str(value.get("conditional_rank")) if value else "unavailable"
                for value in curvatures
            )
        ),
        "supports": [
            _support_summary(rows, threshold, name)
            for threshold in practical_profile.POLICY["thresholds"]
            for name in ("profiled", "frozen")
        ],
        "cost": {
            "fit": _cost_summary(rows),
            "profile": _stage_cost(rows, "profile"),
            "evaluation": _stage_cost(rows, "evaluation"),
        },
    }


def curvature_validation(row: dict) -> dict:
    """Check exact public-design curvature on the frozen 17-point schedule."""
    separation, sigma = row["separation"], row["sigma"]
    expected = {
        "conditional": 187 * separation**2 / sigma**2,
        "frozen": 93.5 * (3 + 2 * separation**2) / sigma**2,
    }
    curvature = row.get("profile", {}).get("curvature") or {}
    result = {
        "expected_" + name + "_information": value for name, value in expected.items()
    }
    for name, value in expected.items():
        observed = curvature.get(name + "_information")
        result[name + "_passed"] = (
            abs(observed - value) <= 1e-7 * (1 + abs(value))
            if observed is not None
            else None
        )
    result["scope"] = (
        "Exact public acquisition design and known sigma; no generating coefficients."
    )
    return result


def _paired_objectives(rows: list[dict]) -> list[dict]:
    """Use the same Gaussian objective for numerical pairing, never generating truth."""
    result = []
    for common in sorted({row["common"] for row in rows}):
        pair = [row for row in rows if row["common"] == common]
        values = [row.get("backend", {}).get("objective_value") for row in pair]
        available = all(value is not None for value in values)
        gap = abs(values[0] - values[1]) if available else None
        result.append(
            {
                "common": common,
                "available": available,
                "absolute_objective_difference": gap,
                "objective_agreement_passed": gap <= 1e-5 * (1 + max(map(abs, values)))
                if available
                else None,
                "parameter_agreement_required": False,
            }
        )
    return result


def report(root: Path) -> dict:
    """Report every arm, support limitation and separate numerical overhead."""
    plan, data = verify(root, runtime=False)
    rows = [report_row(root, plan, data, task) for task in plan["tasks"]]
    rows = [row | {"curvature_validation": curvature_validation(row)} for row in rows]
    counts = dict(Counter(row["status"] for row in rows))
    cost = {
        "fit": _cost_summary(rows),
        "profile": _stage_cost(rows, "profile"),
        "evaluation": _stage_cost(rows, "evaluation"),
    }
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if "missing" not in counts else "incomplete",
        "expected": len(rows),
        "recorded": sum(row["status"] != "missing" for row in rows),
        "status_counts": counts,
        "profile_status_counts": dict(
            Counter(row.get("profile", {}).get("status", "missing") for row in rows)
        ),
        "evaluation_status_counts": dict(
            Counter(row.get("evaluation", {}).get("status", "missing") for row in rows)
        ),
        "rows": rows,
        "groups": [
            {"separation": separation, "sigma": sigma, "strategy": strategy}
            | _group(
                [
                    row
                    for row in rows
                    if row["separation"] == separation
                    and row["sigma"] == sigma
                    and row["strategy"] == strategy
                ]
            )
            for separation in fixtures.SEPARATIONS
            for sigma in fixtures.SIGMAS
            for strategy in STRATEGIES
        ],
        "paired_objectives": _paired_objectives(rows),
        "cost": cost,
        "cost_complete": all(value["accounting_complete"] for value in cost.values()),
        "confidence_probability": None,
        "data_fault_attribution": None,
        "truth_used_for_selection": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": (
            "Fixed known-sigma affine qualification. Profile objective rises are "
            "declared support conventions, not calibrated confidence probabilities. "
            "Finite supports can be limited by gain or nuisance domains; no "
            "parameter recovery is inferred from bounded ridge solutions. "
            "Frozen-nuisance slices do not account for confounding. Truth is "
            "evaluated only after fitting/profiling and cannot choose endpoints or "
            "thresholds. All failed/missing arms remain in denominators; unavailable "
            "costs are not imputed as zero totals."
        ),
    }
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
