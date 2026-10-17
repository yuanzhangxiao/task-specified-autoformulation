"""M41 paired fits with immutable endpoints and separately resumed diagnostics."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from time import monotonic, process_time

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import measurement_fit
from autoformalism.fitting import nonlinear_practical_fixtures as fixtures
from autoformalism.fitting import nonlinear_practical_profile as profiles
from autoformalism.fitting import practical_uncertainty_campaign as stages
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.robust_measurement import Loss
from autoformalism.fitting.robust_measurement_campaign import _cost_summary
from autoformalism.fitting.sensor_drift_campaign import validate_backend
from autoformalism.schemas.sensor_drift import DriftProblem

PROTOCOL = "phase-c-nonlinear-practical-uncertainty-1"
INPUT_PROTOCOL = fixtures.INPUT_PROTOCOL
STRATEGIES = ("direct", "profiled")
Policy = measurement_fit.Policy
fit_identity = stages.fit_identity
stage_identity = stages.stage_identity


def diagnostic_policy() -> dict:
    """Freeze a diagnostic policy independently of fit selection and truth."""
    return {
        "protocol": profiles.PROTOCOL,
        "policy": profiles.Policy().model_dump(mode="json"),
    }


def roster(data: dict) -> list[dict]:
    """Check every acquisition before fixing the 36 paired strategy tasks."""
    expected = {
        fixtures.case_key(delta, sigma, seed)
        for delta in fixtures.SEPARATIONS
        for sigma in fixtures.SIGMAS
        for seed in fixtures.SEEDS
    }
    if (
        data["protocol"] != INPUT_PROTOCOL
        or data["test_data_opened"] is not False
        or set(data["problems"]) != expected
        or set(data["evaluation"]) != expected
    ):
        raise ValueError("nonlinear practical acquisition roster differs")
    fixtures.verify_raw_artifacts(data)
    return [
        {
            "task_id": fixtures.case_key(delta, sigma, seed) + "__" + strategy,
            "common": fixtures.case_key(delta, sigma, seed),
            "separation": delta,
            "sigma": sigma,
            "seed": seed,
            "strategy": strategy,
        }
        for delta in fixtures.SEPARATIONS
        for sigma in fixtures.SIGMAS
        for seed in fixtures.SEEDS
        for strategy in STRATEGIES
    ]


def training_problem(data: dict, task: dict) -> DriftProblem:
    """Consume only the public candidate and observed training acquisition."""
    return DriftProblem.model_validate(data["problems"][task["common"]])


def prepare(
    root: Path, inputs: Path | None = None, policy: Policy | None = None
) -> dict:
    """Generate once or import sealed acquisitions; never redraw on resume."""
    policy = Policy() if policy is None else policy
    if inputs is None and (root / "inputs.json").exists():
        inputs = root / "inputs.json"
    data = read_seal(inputs) if inputs is not None else fixtures.make_inputs()
    tasks = roster(data)
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
    """Reject changed protocols, acquisitions, policies, source or runtime."""
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
        raise ValueError("nonlinear practical campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("nonlinear practical source/runtime differs")
    return plan, data


def _checkpoint(folder: Path, identity: dict) -> tuple[dict | None, list[dict], bool]:
    """Verify compact journal chaining and the latest atomically saved snapshot."""
    journal, path = (
        folder / "profile-progress.jsonl",
        folder / "profile-checkpoint.json",
    )
    records, truncated = [], False
    identity_sha = public.content_sha256(identity)
    for line in (
        journal.read_bytes().splitlines(keepends=True) if journal.exists() else []
    ):
        if not line.endswith(b"\n"):
            truncated = True
            break
        record = json.loads(line)
        if (
            record.get("identity_sha256") != identity_sha
            or record.get("sequence") != len(records)
            or record.get("event_sha256")
            != public.content_sha256(
                {key: value for key, value in record.items() if key != "event_sha256"}
            )
            or record.get("previous_event_sha256")
            != (records[-1]["event_sha256"] if records else None)
            or record.get("previous_checkpoint_sha256")
            != (records[-1]["checkpoint_sha256"] if records else None)
        ):
            raise ValueError("nonlinear practical profile journal differs")
        records.append(record)
    checkpoint = read_seal(path) if path.exists() else None
    if checkpoint is None:
        if records:
            raise ValueError("nonlinear practical profile checkpoint missing")
        return None, records, truncated
    if (
        checkpoint.get("identity") != identity
        or checkpoint["event"]["identity_sha256"] != identity_sha
        or checkpoint["event"].get("event_sha256")
        != public.content_sha256(
            {
                key: value
                for key, value in checkpoint["event"].items()
                if key != "event_sha256"
            }
        )
        or checkpoint["event"]["checkpoint_sha256"]
        != public.content_sha256(checkpoint["state"])
    ):
        raise ValueError("nonlinear practical profile checkpoint identity differs")
    sequence = checkpoint["event"]["sequence"]
    if sequence == len(records) - 1 and records[-1] == checkpoint["event"]:
        pass
    elif (
        sequence == len(records)
        and checkpoint["event"]["previous_checkpoint_sha256"]
        == (records[-1]["checkpoint_sha256"] if records else None)
        and checkpoint["event"]["previous_event_sha256"]
        == (records[-1]["event_sha256"] if records else None)
    ):
        # The durable snapshot can precede its compact journal append when killed.
        pass
    else:
        raise ValueError("nonlinear practical profile checkpoint sequence differs")
    return checkpoint, records, truncated


def _append_event(path: Path, event: dict) -> None:
    """Append and fsync one compact progress record."""
    with path.open("a") as stream:
        stream.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _profile_once(
    folder: Path,
    identity: dict,
    problem: DriftProblem,
    parameters: dict | None,
    policy: profiles.Policy,
) -> dict:
    """Resume only checkpointed diagnostics, preserving the diagnostic ledger."""
    final, started = folder / "profile.json", folder / "profile-started.json"
    if final.exists():
        value = read_seal(final)
        if value.get("identity") != identity or read_seal(started) != identity:
            raise ValueError("nonlinear practical profile identity differs")
        return value
    previous = started.exists()
    seal(started, identity)
    checkpoint, records, truncated = _checkpoint(folder, identity)
    if previous and checkpoint is None:
        value = {
            "status": "interrupted",
            "accounting_complete": False,
            "budget_charge_seconds": policy.wall_seconds,
            "reason": "Started diagnostic has no checkpoint; allowance consumed.",
        }
    elif parameters is None:
        value = {
            "status": "no_parameters",
            "calls": 0,
            "completed_calls": 0,
            "failed_calls": 0,
            "timing": {"total_wall_seconds": 0, "total_cpu_seconds": 0},
            "accounting_complete": True,
            "budget_charge_seconds": 0,
        }
    else:
        journal = folder / "profile-progress.jsonl"
        if truncated:
            with journal.open("rb+") as stream:
                content = stream.read()
                stream.truncate(content.rfind(b"\n") + 1)
        if checkpoint is not None and checkpoint["event"]["sequence"] == len(records):
            _append_event(journal, checkpoint["event"])
            records.append(checkpoint["event"])

        def save(state):
            event = {
                "identity_sha256": public.content_sha256(identity),
                "sequence": len(records),
                "previous_checkpoint_sha256": records[-1]["checkpoint_sha256"]
                if records
                else None,
                "previous_event_sha256": records[-1]["event_sha256"]
                if records
                else None,
                "checkpoint_sha256": public.content_sha256(state),
                "status": state.get("status"),
                "calls": state.get("calls"),
                "completed_calls": state.get("completed_calls"),
                "budget_charge_seconds": state.get("budget_charge_seconds"),
            }
            event["event_sha256"] = public.content_sha256(event)
            snapshot = {"identity": identity, "state": state, "event": event}
            public._write(
                folder / "profile-checkpoint.json",
                {"sha256": public.content_sha256(snapshot), "value": snapshot},
            )
            _append_event(journal, event)
            records.append(event)

        try:
            value = profiles.run(
                problem,
                parameters,
                policy=policy,
                state=checkpoint["state"] if checkpoint else None,
                save=save,
            )
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
            saved, _, _ = _checkpoint(folder, identity)
            value = {
                "status": "failed",
                "accounting_complete": False,
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
                "observed_progress": saved["state"] if saved else {},
                "budget_charge_seconds": policy.wall_seconds,
            }
    observed = value.get("observed_progress", value)
    result = {
        "identity": identity,
        "status": value["status"],
        "diagnostic": value,
        "wall_seconds": observed.get("timing", {}).get("total_wall_seconds"),
        "cpu_seconds": observed.get("timing", {}).get("total_cpu_seconds"),
        "accounting_complete": value.get("accounting_complete", False),
        "budget_charge_seconds": value.get("budget_charge_seconds"),
        **{
            name: observed.get(name)
            for name in ("calls", "completed_calls", "failed_calls")
        },
    }
    seal(final, result)
    return result


def evaluate(data: dict, task: dict, backend: dict) -> dict:
    """Compare sealed endpoints with exact responses, then evaluator-only truth."""
    value = fixtures.evaluate(
        data["evaluation"][task["common"]], backend.get("parameters")
    )
    exact, reported = (
        value.get("independent_analytic_endpoint_objective"),
        backend.get("objective_value"),
    )
    gap = abs(exact - reported) if exact is not None and reported is not None else None
    return value | {
        "analytic_rollout_objective_difference": gap,
        "analytic_rollout_objective_agreement_passed": gap <= 1e-5 * (1 + abs(exact))
        if gap is not None
        else None,
    }


def _definitions(plan: dict, data: dict, task: dict) -> dict:
    """Bind truth evaluation separately from the public diagnostic definition."""
    return {
        "profile": plan["diagnostic"],
        "evaluation": {
            "protocol": "nonlinear-practical-evaluation-1",
            "evaluator_sha256": public.content_sha256(
                data["evaluation"][task["common"]]
            ),
        },
    }


def _read_profile(folder: Path, identity: dict) -> dict | None:
    """Read verified terminal evidence or expose observed interrupted progress."""
    if not (folder / "profile-started.json").exists():
        return None
    if read_seal(folder / "profile-started.json") != identity:
        raise ValueError("nonlinear practical profile identity differs")
    checkpoint, _, _ = _checkpoint(folder, identity)
    if (folder / "profile.json").exists():
        value = read_seal(folder / "profile.json")
        if value.get("identity") != identity:
            raise ValueError("nonlinear practical profile identity differs")
        return value
    state = checkpoint["state"] if checkpoint else {}
    return {
        "status": "unfinished",
        "diagnostic": state,
        "accounting_complete": False,
        "wall_seconds": state.get("timing", {}).get("total_wall_seconds"),
        "cpu_seconds": state.get("timing", {}).get("total_cpu_seconds"),
        "budget_charge_seconds": state.get("budget_charge_seconds")
        if checkpoint
        else identity["definition"]["policy"]["wall_seconds"],
        **{
            name: state.get(name)
            for name in ("calls", "completed_calls", "failed_calls")
        },
    }


def report_row(root: Path, plan: dict, data: dict, task: dict) -> dict:
    """Verify all available stages, including endpoints from interrupted tasks."""
    folder = root / "results" / task["task_id"]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    row = task | {"status": "missing"}
    if (folder / "backend.json").exists():
        backend = read_seal(folder / "backend.json")
    elif (folder / "fit/result.json").exists():
        backend = read_seal(folder / "fit/result.json") | {
            "campaign_identity": identity
        }
    else:
        return row
    validate_backend(backend, identity, fit_identity(plan, data, task))
    row["backend"] = backend
    definitions = _definitions(plan, data, task)
    profile = _read_profile(
        folder, stage_identity(identity, backend, definitions["profile"])
    )
    evaluation = stages.read_stage(
        folder,
        "evaluation",
        stage_identity(identity, backend, definitions["evaluation"]),
        allowance=None,
    )
    if profile is not None:
        row["profile"] = profile
    if evaluation is not None:
        row["evaluation"] = evaluation
    if (folder / "result.json").exists():
        receipt = read_seal(folder / "result.json")
        expected = {
            "identity": identity,
            "status": backend["status"],
            "backend_sha256": public.content_sha256(backend),
            "profile_sha256": public.content_sha256(profile),
            "evaluation_sha256": public.content_sha256(evaluation),
        }
        if (
            receipt != expected
            or profile is None
            or evaluation is None
            or profile["status"] == "unfinished"
            or evaluation["status"] == "unfinished"
        ):
            raise ValueError("nonlinear practical terminal stage differs")
        row["status"] = backend["status"]
    return row


def run_task(root: Path, index: int) -> dict:
    """Seal fitted endpoints before public diagnostics, and seal those before truth."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown nonlinear practical task index")
    task = plan["tasks"][index]
    folder = root / "results" / task["task_id"]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    execution, definitions = (
        fit_identity(plan, data, task),
        _definitions(plan, data, task),
    )
    with public._lock(folder):
        if (folder / "result.json").exists():
            return report_row(root, plan, data, task)
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
        profile = _profile_once(
            folder,
            stage_identity(identity, backend, definitions["profile"]),
            training_problem(data, task),
            backend.get("parameters"),
            profiles.Policy.model_validate(plan["diagnostic"]["policy"]),
        )

        def score(deadline, save):
            save({"calls": 1, "completed_calls": 0})
            value = evaluate(data, task, backend)
            save({"calls": 1, "completed_calls": 1})
            return value | {"calls": 1, "completed_calls": 1}

        evaluation = stages._stage_once(
            folder,
            "evaluation",
            stage_identity(identity, backend, definitions["evaluation"]),
            score,
            None,
        )
        seal(
            folder / "result.json",
            {
                "identity": identity,
                "status": backend["status"],
                "backend_sha256": public.content_sha256(backend),
                "profile_sha256": public.content_sha256(profile),
                "evaluation_sha256": public.content_sha256(evaluation),
            },
        )
        return task | {
            "status": backend["status"],
            "backend": backend,
            "profile": profile,
            "evaluation": evaluation,
        }


def _compact(row: dict) -> dict:
    """Keep rows small; full node and fit evidence remains in stage files."""
    value = {
        key: row[key]
        for key in (
            "task_id",
            "common",
            "separation",
            "sigma",
            "seed",
            "strategy",
            "status",
        )
    }
    value["evidence_directory"] = "results/" + row["task_id"]
    if "backend" in row:
        backend = row["backend"]
        value["fit"] = {
            key: backend.get(key)
            for key in ("status", "parameters", "objective_value", "selected_origin")
        }
        value["fit"]["verification_passed"] = backend.get("verification", {}).get(
            "passed"
        )
    if "profile" in row:
        profile = row["profile"]
        diagnostic = profile.get("diagnostic", {})
        value["profile"] = {
            "status": profile["status"],
            **{
                key: diagnostic.get(key)
                for key in (
                    "endpoint_qualification",
                    "supports",
                    "curvature",
                    "independent_rollout",
                )
            },
        }
    if "evaluation" in row:
        value["evaluation"] = {
            key: item
            for key, item in row["evaluation"].items()
            if key not in {"identity", "event_history"}
        }
    return value


def _diagnostic_cost(rows: list[dict]) -> dict:
    """Distinguish measured diagnostic overhead from incomplete totals."""
    values = [row.get("profile", {}) for row in rows]
    result = {
        "call_count_scope": (
            "calls_observed is the recorded budget ledger, including reserved "
            "calls charged after interruption; it is not an actual evaluation "
            "count when accounting is incomplete."
        ),
        "interrupted_calls_charged_observed": sum(
            value.get("diagnostic", {})
            .get("observed_progress", value.get("diagnostic", {}))
            .get("interrupted_calls_charged", 0)
            for value in values
        ),
        "accounting_complete": all(
            value.get("accounting_complete", False)
            and value.get("wall_seconds") is not None
            and value.get("cpu_seconds") is not None
            for value in values
        )
    }
    for field in (
        "wall_seconds",
        "cpu_seconds",
        "calls",
        "completed_calls",
        "failed_calls",
        "budget_charge_seconds",
    ):
        counts = [value.get(field) for value in values]
        result[field + "_observed"] = sum(value or 0 for value in counts)
        result[field + "_total"] = (
            sum(counts)
            if all(value is not None for value in counts)
            and (result["accounting_complete"] or field == "budget_charge_seconds")
            else None
        )
        result[field + "_unknown_or_missing"] = sum(value is None for value in counts)
    for field in (
        "numerical_wall_seconds",
        "numerical_cpu_seconds",
        "save_callback_wall_seconds",
        "save_callback_cpu_seconds",
    ):
        diagnostics = [value.get("diagnostic", {}) for value in values]
        counts = [
            value.get("observed_progress", value).get("timing", {}).get(field)
            for value in diagnostics
        ]
        result[field + "_observed"] = sum(value or 0 for value in counts)
        result[field + "_total"] = (
            sum(counts)
            if all(value is not None for value in counts)
            and result["accounting_complete"]
            else None
        )
    return result


def _group(rows: list[dict]) -> dict:
    """Retain all three seed denominators and separate truth errors from diagnostics."""
    diagnostics = [row.get("profile", {}).get("diagnostic", {}) for row in rows]
    evaluations = [row.get("evaluation", {}) for row in rows]
    return {
        "expected": len(rows),
        "recorded": sum(row["status"] != "missing" for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "profile_status_counts": dict(
            Counter(row.get("profile", {}).get("status", "missing") for row in rows)
        ),
        "numerically_qualified": sum(
            (value.get("endpoint_qualification") or {}).get("numerically_qualified")
            is True
            for value in diagnostics
        ),
        "retained_fit_inadequate": sum(
            (value.get("endpoint_qualification") or {}).get("retained_fit_inadequate")
            is True
            for value in diagnostics
        ),
        "parameter_errors_evaluator_only": {
            name: stages._statistics(
                [
                    value.get("parameter_truth_error_evaluator_only", {}).get(name)
                    for value in evaluations
                ]
            )
            for name in fixtures.PARAMETERS
        },
        "clean_physical_rmse_evaluator_only": stages._statistics(
            [
                value.get("clean_physical_prediction_rmse_evaluator_only")
                for value in evaluations
            ]
        ),
        "clean_measured_rmse_evaluator_only": stages._statistics(
            [
                value.get("clean_measured_prediction_rmse_evaluator_only")
                for value in evaluations
            ]
        ),
    }


def report(root: Path) -> dict:
    """Publish a concise report with full stage evidence retained by task path."""
    plan, data = verify(root, runtime=False)
    rows = [report_row(root, plan, data, task) for task in plan["tasks"]]
    counts = dict(Counter(row["status"] for row in rows))
    cost = {
        "fit": _cost_summary(rows),
        "profile": _diagnostic_cost(rows),
        "evaluation": stages._stage_cost(rows, "evaluation"),
    }
    fit_cpu = cost["fit"].get("observed", {}).get("cpu_seconds", 0)
    cost["profile"]["cpu_over_fit_observed_ratio"] = (
        cost["profile"]["cpu_seconds_observed"] / fit_cpu if fit_cpu else None
    )
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
        "rows": [_compact(row) for row in rows],
        "groups": [
            {"separation": delta, "sigma": sigma, "strategy": strategy}
            | _group(
                [
                    row
                    for row in rows
                    if row["separation"] == delta
                    and row["sigma"] == sigma
                    and row["strategy"] == strategy
                ]
            )
            for delta in fixtures.SEPARATIONS
            for sigma in fixtures.SIGMAS
            for strategy in STRATEGIES
        ],
        "paired_objectives": stages._paired_objectives(rows),
        "cost": cost,
        "cost_complete": all(value["accounting_complete"] for value in cost.values()),
        "confidence_probability": None,
        "data_fault_attribution": None,
        "truth_used_for_selection": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitations": [
            "Synthetic known-sigma control; production fitting defaults unchanged.",
            "Finite multi-start profiles and references do not certify global optima.",
            "Sampled support and brackets are not calibrated confidence intervals.",
            "Zero input separation can retain exponential-curvature information.",
            "Parameter-domain limits and unresolved diagnostic work remain explicit.",
            "Truth is opened after fit and diagnostic seals; it selects neither.",
            "Missing/failed arms remain in denominators; unknown total costs are null.",
        ],
    }
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
