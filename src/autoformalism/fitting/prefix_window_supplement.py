"""M45: verify saved M44 endpoints with separately accounted persistence time.

No optimizer is called. Historical fits, successes, failures and costs are frozen
in a portable snapshot; only failed verification and its blocked forecasts run.
"""

from __future__ import annotations

import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import monotonic, process_time
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import practical_uncertainty_supplement as snapshot
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_verification as verifier
from autoformalism.fitting import prefix_window_selection as previous
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "phase-c-prefix-window-supplement-1"


class Policy(StrictSchema):
    """An additional verification allowance; scientific selection stays frozen."""

    protocol: Literal["phase-c-prefix-window-supplement-policy-1"] = (
        "phase-c-prefix-window-supplement-policy-1"
    )
    selection: Literal["failed_verification_only"] = "failed_verification_only"
    execution: verifier.ExecutionPolicy = Field(
        default_factory=verifier.ExecutionPolicy
    )


def _source_rows(source: Path) -> tuple[dict, dict, list[dict]]:
    """Validate complete historical receipts without writing to the source root."""
    plan, data = previous.verify(source, runtime=False)
    policy = previous.Policy.model_validate(plan["policy"])
    rows = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        row = previous._completed(
            source / "results" / task["task_id"],
            identity,
            data["records"][task["task_id"]],
            policy,
        )
        if not row["training_fit"] or not row["training_fit"].get("fit_success"):
            raise ValueError("supplement requires verified frozen training fits")
        rows.append(row)
    return plan, data, rows


def _gaps(row: dict) -> list[dict]:
    """Identify recoverable receipts by availability, never by fitted accuracy."""
    gaps = []
    for window in row["windows"]:
        for entry in window["trajectories"]:
            for attempt in entry["attempts"]:
                inference, forecast = attempt["inference"], attempt["forecast"]
                if (
                    inference
                    and inference.get("fit_success")
                    and forecast
                    and forecast.get("forecast_success")
                ):
                    continue
                if not inference or (
                    not inference.get("fit_success")
                    and (
                        not inference.get("verification_error")
                        or not (inference.get("search") or {}).get("best")
                    )
                ):
                    raise ValueError(
                        "missing saved inference candidate; no search retry authorized"
                    )
                gaps.append(
                    {
                        "label": attempt["label"],
                        "duration": window["duration"],
                        "trajectory_id": entry["trajectory_id"],
                        "verify_inference": not inference["fit_success"],
                        "inference_sha256": public.content_sha256(inference),
                        "forecast_sha256": public.content_sha256(forecast),
                    }
                )
    return gaps


def audit(source: Path) -> dict:
    """Bind each saved fit/result and identify only recoverable verification gaps."""
    plan, _, rows = _source_rows(source)
    records = [
        {
            "task_id": row["task_id"],
            "source_result_sha256": public.content_sha256(row),
            "training_fit_sha256": public.content_sha256(row["training_fit"]),
            "gaps": _gaps(row),
        }
        for row in rows
    ]
    return {
        "status": "passed",
        "source_plan_sha256": public.content_sha256(plan),
        "rows": records,
        "source_tasks": len(rows),
        "selected_task_ids": [r["task_id"] for r in records if r["gaps"]],
        "forecast_gaps": sum(len(r["gaps"]) for r in records),
        "inference_verifications": sum(
            g["verify_inference"] for r in records for g in r["gaps"]
        ),
        "truth_used_for_selection": False,
        "fitting_performed": False,
    }


def prepare(root: Path, source: Path | None, policy: Policy) -> dict:
    """Freeze a separate portable campaign; source files are never modified."""
    root = root.resolve()
    if source is None:
        if not (root / "plan.json").exists():
            raise ValueError("an existing M44 source root is required")
        plan, _ = verify(root)
        if plan["policy"] != policy.model_dump(mode="json"):
            raise ValueError("supplement policy differs")
        return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}
    source = source.resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("supplement and source require separate directories")
    print(
        f"M45: reading historical receipts from {source}", file=sys.stderr, flush=True
    )
    manifest = snapshot._files(source)
    with public._lock(root):
        if (root / "plan.json").exists():
            plan, frozen = verify(root)
            if manifest != frozen or plan["policy"] != policy.model_dump(mode="json"):
                raise ValueError("supplement source/policy differs")
        else:
            source_audit = audit(source)
            source_plan = read_seal(source / "plan.json")
            print(
                f"M45: preserving {len(manifest)} source files",
                file=sys.stderr,
                flush=True,
            )
            snapshot._copy_snapshot(source, root / "source_snapshot", manifest)
            # Validate the actual copied receipts as well as their individual bytes.
            if audit(root / "source_snapshot") != source_audit:
                raise ValueError("supplement source changed during audit")
            selected = set(source_audit["selected_task_ids"])
            plan = {
                "protocol": PROTOCOL,
                "policy": policy.model_dump(mode="json"),
                "tasks": [t for t in source_plan["tasks"] if t["task_id"] in selected],
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "inputs_sha256": public.content_sha256(manifest),
                "audit_sha256": public.content_sha256(source_audit),
                "source_plan_sha256": public.content_sha256(source_plan),
                "test_data_opened": False,
                "live_llm_calls": 0,
                "fitting_performed": False,
                "truth_used_for_selection": False,
            }
            seal(root / "inputs.json", manifest)
            seal(root / "portable_audit.json", source_audit)
            seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Validate the source snapshot and bind new work to its own code/runtime."""
    plan, manifest = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    frozen = read_seal(root / "portable_audit.json")
    source_plan = read_seal(root / "source_snapshot/plan.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(manifest)
        or snapshot._files(root / "source_snapshot") != manifest
        or plan["audit_sha256"] != public.content_sha256(frozen)
        or audit(root / "source_snapshot") != frozen
        or plan["source_plan_sha256"] != public.content_sha256(source_plan)
        or plan["tasks"]
        != [
            t
            for t in source_plan["tasks"]
            if t["task_id"] in frozen["selected_task_ids"]
        ]
        or plan["test_data_opened"] is not False
        or plan["live_llm_calls"] != 0
        or plan["fitting_performed"] is not False
        or plan["truth_used_for_selection"] is not False
    ):
        raise ValueError("supplement snapshot/identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("supplement source/runtime differs")
    return plan, manifest


def _operation(folder: Path, identity: dict, name: str, seconds: float, work) -> dict:
    """Persist attempts once; interrupted work is retained without automatic retry."""
    directory = folder / "operations" / name
    bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
    with public._lock(directory):
        start, result_path = directory / "started.json", directory / "result.json"
        if start.exists() and read_seal(start) != bound:
            raise ValueError("supplement operation identity differs")
        if result_path.exists():
            saved = read_seal(result_path)
            if saved["identity"] != bound:
                raise ValueError("supplement operation result differs")
            return saved
        if start.exists():
            result = {
                "status": "interrupted",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "calls": None,
                "accounting_complete": False,
                "budget_charge_seconds": seconds,
            }
        else:
            seal(start, bound)
            begun, cpu = monotonic(), process_time()
            progress = {}

            def save(value):
                progress.update(value)
                public._write(directory / "progress.json", progress)

            interrupt = None
            try:
                value = work(save)
                result = {"status": "complete", "value": value}
            except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
                result = {
                    "status": "unavailable",
                    "value": None,
                    "error": str(error)[-1000:],
                }
            except BaseException as error:
                interrupt = error
                receipt = getattr(error, "verification_receipt", None)
                if receipt is not None:
                    save(receipt)
                result = {"status": "interrupted", "value": receipt}
            elapsed = monotonic() - begun
            result.update(
                wall_seconds=elapsed,
                cpu_seconds=process_time() - cpu,
                calls=0,
                accounting_complete=True,
                budget_charge_seconds=elapsed,
            )
            if interrupt is not None:
                seal(
                    result_path, result | {"identity": bound, "budget_restarted": False}
                )
                raise interrupt
        saved = result | {"identity": bound, "budget_restarted": False}
        seal(result_path, saved)
        return saved


def _identity(plan: dict, task: dict, source: dict) -> dict:
    return {
        "plan_sha256": public.content_sha256(plan),
        "task": task,
        "source_result_sha256": public.content_sha256(source),
        "training_fit_sha256": public.content_sha256(source["training_fit"]),
    }


def _supplement_cost(operations: dict) -> dict:
    # Stage names deliberately exclude *_inference: retained search history is
    # provenance, not new search work. Actual optimization calls are exactly zero.
    result = previous._cost(operations)
    timings = [
        (op.get("value") or op.get("observed_progress") or {}).get("timing")
        for op in operations.values()
    ]
    complete = [
        op["status"] == "complete" and bool((op.get("value") or {}).get("timing"))
        for op in operations.values()
    ]
    for field in (
        "callback_wall_seconds",
        "callback_cpu_seconds",
        "numerical_active_wall_seconds",
        "numerical_wall_seconds",
        "numerical_cpu_seconds",
    ):
        values = [t.get(field) if t else None for t in timings]
        result[field + "_observed"] = sum(v for v in values if v is not None)
        result[field + "_total"] = (
            sum(values)
            if all(complete) and all(v is not None for v in values)
            else None
        )
    for field in ("verification_calls_attempted", "verification_calls_failed"):
        observed = [
            (op.get("value") or op.get("observed_progress") or {}).get(field, 0)
            for op in operations.values()
        ]
        result[field + "_observed"] = sum(observed)
        result[field + "_total"] = sum(observed) if all(complete) else None
    result["verification_start_semantics"] = (
        "attempted=checkpoint reservation; started=actual integrator invocation; "
        "completed=returned invocation, which may still fail qualification"
    )
    result["scope"] = (
        "Supplement operation wall/CPU includes setup and progress writes. Timing "
        "subcategories are nested, never additive to totals. Final receipt sealing "
        "and source audit/snapshot costs are excluded; old search is not repeated."
    )
    return result


def _compose(
    source: dict, record: dict, policy: previous.Policy, operations: dict
) -> dict:
    """Rebuild outcomes from saved endpoints/receipts, without numerical execution."""
    windows = deepcopy(source["windows"])
    gaps = {g["label"]: g for g in _gaps(source)}
    consumed, recovered = set(), 0
    rows = {
        r.trajectory_id: r
        for r in PublicSplit.model_validate(record["validation"]).rows
    }
    fit = source["training_fit"]
    for window in windows:
        for entry in window["trajectories"]:
            old_chosen = entry["chosen_start"]
            old_hash = (
                public.content_sha256(entry["attempts"][old_chosen]["inference"])
                if old_chosen is not None
                else None
            )
            for attempt in entry["attempts"]:
                label = attempt["label"]
                if label not in gaps:
                    continue
                gap = gaps[label]
                if gap["verify_inference"]:
                    op = operations.get(label + "_verify")
                    consumed.add(label + "_verify")
                    attempt["inference"] = (
                        op.get("value") if op and op["status"] == "complete" else None
                    )
                inference = attempt["inference"]
                forecast_op = operations.get(label + "_forecast")
                if forecast_op:
                    if not inference or not inference.get("fit_success"):
                        raise ValueError(
                            "forecast operation without verified inference"
                        )
                    consumed.add(label + "_forecast")
                forecast = (
                    forecast_op.get("value")
                    if forecast_op and forecast_op["status"] == "complete"
                    else None
                )
                attempt["forecast"] = forecast
                row = rows[entry["trajectory_id"]]
                prefix_policy = kernel.PrefixPolicy(duration=window["duration"])
                prefix = kernel.make_prefix(row, prefix_policy)
                schedule = kernel.make_forecast_schedule(row, prefix_policy)
                if inference and gap["verify_inference"]:
                    if (
                        inference.get("original_inference_sha256")
                        != gap["inference_sha256"]
                        or inference.get("fit_sha256") != public.content_sha256(fit)
                        or inference.get("prefix_sha256")
                        != public.content_sha256(prefix)
                        or inference.get("shared_parameters")
                        != fit["shared_parameters"]
                        or inference.get("training_scales") != fit["training_scales"]
                        or inference.get("no_search_performed") is not True
                    ):
                        raise ValueError("supplement inference identity differs")
                    old = next(
                        a["inference"]
                        for w in source["windows"]
                        for e in w["trajectories"]
                        for a in e["attempts"]
                        if a["label"] == label
                    )
                    if (
                        inference.get("fit_success")
                        and inference["roots"] != old["search"]["best"]["parameters"]
                    ):
                        raise ValueError("supplement changed saved root candidate")
                if forecast and (
                    forecast["inference_sha256"] != public.content_sha256(inference)
                    or forecast["schedule_sha256"] != public.content_sha256(schedule)
                ):
                    raise ValueError("supplement forecast identity differs")
                attempt["common_score"] = (
                    previous.diagnostic.common_horizon_score(
                        forecast, row, policy.durations[-1], fit["training_scales"]
                    )
                    if forecast
                    else None
                )
                attempt["own_suffix_score_descriptive_only"] = (
                    kernel.score_forecast(
                        forecast, row, prefix.cutoff, fit["training_scales"]
                    )
                    if forecast
                    else None
                )
                recovered += bool(forecast and forecast.get("forecast_success"))
            entry["chosen_start"] = previous._choose_prefix_start(entry["attempts"])
            chosen = entry["chosen_start"]
            new_hash = (
                public.content_sha256(entry["attempts"][chosen]["inference"])
                if chosen is not None
                else None
            )
            if old_hash != new_hash:
                entry["historical_information"] = entry["information"]
                entry["information"] = {
                    "available": False,
                    "status": "selected_receipt_changed",
                    "reason": (
                        "Historical information retained separately; "
                        "not rebound to a new endpoint."
                    ),
                }
        window["metrics"] = previous.window_metrics(
            window["duration"], window["trajectories"], policy
        )
    if set(operations) - consumed:
        raise ValueError("unexpected supplemental operation")
    selection = previous.select_duration([w["metrics"] for w in windows], policy)
    return {
        "windows": windows,
        "selection": selection,
        "status": "complete" if selection["status"] != "unavailable" else "incomplete",
        "recovered_forecast_gaps": recovered,
        "training_fit_sha256": public.content_sha256(fit),
        "supplemental_cost": _supplement_cost(operations),
    }


def _completed(folder, identity, source, record, policy):
    operations = previous._read_operations(folder, identity)
    expected = {"identity": identity, **_compose(source, record, policy, operations)}
    result = read_seal(folder / "result.json")
    if result != expected or result["selection"] != read_seal(
        folder / "selection.json"
    ):
        raise ValueError("supplement terminal artifact differs")
    return result


def run_task(root: Path, index: int) -> dict:
    """Reverify only existing candidates and produce their missing forecasts."""
    plan, _ = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown supplement task index")
    source_plan, data, source_rows = _source_rows(root / "source_snapshot")
    policy = previous.Policy.model_validate(source_plan["policy"])
    execution = Policy.model_validate(plan["policy"]).execution
    task = plan["tasks"][index]
    source = next(r for r in source_rows if r["task_id"] == task["task_id"])
    record = data["records"][task["task_id"]]
    identity = _identity(plan, task, source)
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return _completed(folder, identity, source, record, policy)
        problem = kernel.PrefixFitProblem.model_validate(record["problem"])
        fit = source["training_fit"]
        rows = {
            r.trajectory_id: r
            for r in PublicSplit.model_validate(record["validation"]).rows
        }
        for gap in _gaps(source):
            label = gap["label"]
            print(f"M45: {task['task_id']} {label}", file=sys.stderr, flush=True)
            old = next(
                a
                for w in source["windows"]
                for e in w["trajectories"]
                for a in e["attempts"]
                if a["label"] == label
            )
            row = rows[gap["trajectory_id"]]
            prefix_policy = kernel.PrefixPolicy(duration=gap["duration"])
            prefix = kernel.make_prefix(row, prefix_policy)
            schedule = kernel.make_forecast_schedule(row, prefix_policy)
            inference = old["inference"]
            if gap["verify_inference"]:
                op = _operation(
                    folder,
                    identity,
                    label + "_verify",
                    execution.overall_seconds,
                    lambda save,
                    inference=inference,
                    prefix=prefix: verifier.verify_inference(
                        problem,
                        fit,
                        prefix,
                        inference,
                        policy.prefix_numerical,
                        execution,
                        checkpoint=save,
                    ),
                )
                inference = op["value"] if op["status"] == "complete" else None
            if inference and inference.get("fit_success"):
                _operation(
                    folder,
                    identity,
                    label + "_forecast",
                    execution.overall_seconds,
                    lambda save,
                    inference=inference,
                    schedule=schedule: verifier.forecast(
                        problem,
                        fit,
                        inference,
                        schedule,
                        policy.prefix_numerical,
                        execution,
                        checkpoint=save,
                    ),
                )
        operations = previous._read_operations(folder, identity)
        result = {"identity": identity, **_compose(source, record, policy, operations)}
        seal(folder / "selection.json", result["selection"])
        seal(folder / "result.json", result)
        return result


def report(root: Path) -> dict:
    """Combine recovered outcomes while retaining historical failures and charges."""
    plan, _ = verify(root, runtime=False)
    source_plan, data, source_rows = _source_rows(root / "source_snapshot")
    policy = previous.Policy.model_validate(source_plan["policy"])
    source_ops, new_ops, rows, effective = {}, {}, [], []
    selected = {t["task_id"] for t in plan["tasks"]}
    for source in source_rows:
        task = next(
            t for t in source_plan["tasks"] if t["task_id"] == source["task_id"]
        )
        name = task["task_id"]
        source_ops.update({name + "/" + k: v for k, v in source["operations"].items()})
        if name not in selected:
            effective.append(source)
            continue
        identity = _identity(plan, task, source)
        folder = root / "results" / name
        ops = previous._read_operations(folder, identity)
        new_ops.update({name + "/" + k: v for k, v in ops.items()})
        row = (
            _completed(folder, identity, source, data["records"][name], policy)
            if (folder / "result.json").exists()
            else {"status": "missing", "windows": [], "selection": None}
        )
        rows.append(task | row)
        effective.append(task | row)
    selections = []
    for case in ("clean", "noisy"):
        members = [r for r in effective if r["case"] == case]
        metrics = []
        for index, duration in enumerate(policy.durations):
            found = [
                r["windows"][index]["metrics"]
                for r in members
                if len(r["windows"]) == len(policy.durations)
            ]
            complete = len(found) == len(members) and all(m["complete"] for m in found)
            metrics.append(
                {
                    "duration": duration,
                    "complete": complete,
                    "adequate": complete and all(m["adequate"] for m in found),
                }
            )
        selections.append({"case": case, **previous.select_duration(metrics, policy)})
    recorded = sum(r["status"] != "missing" for r in rows)
    new_cost = _supplement_cost(new_ops)
    combined = previous._cost(
        {
            **{"source/" + k: v for k, v in source_ops.items()},
            **{"supplement/" + k: v for k, v in new_ops.items()},
        }
    )
    for key in list(combined):
        if key.startswith("verification_calls_started_"):
            combined[
                key.replace("verification_calls_started_", "mixed_start_counter_")
            ] = combined.pop(key)
    combined["verification_start_semantics"] = (
        "Mixed historical reservations and new executed calls. mixed_start_counter "
        "is a bookkeeping sum, not a physical-integration count. See source_cost "
        "and supplemental_cost separately."
    )
    if recorded != len(rows):
        for cost in (new_cost, combined):
            cost["accounting_complete"] = False
            for key in list(cost):
                if key.endswith("_total"):
                    cost[key] = None
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if recorded == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": recorded,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "source_status_counts": dict(Counter(r["status"] for r in source_rows)),
        "source_cost": previous._cost(source_ops),
        "supplemental_cost": new_cost,
        "combined_cost": combined,
        "rows": rows,
        "selections": selections,
        "recovered_forecast_gaps": sum(
            r.get("recovered_forecast_gaps", 0) for r in rows
        ),
        "source_tasks_retained": len(source_rows),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "fitting_performed": False,
        "limitation": (
            "Verification-only supplement at saved endpoints. Original failed ledger "
            "remains unchanged. Validation-tuned common late horizon, not independent "
            "test or immediate forecast qualification; no new search, confidence "
            "probabilities or production-default change."
        ),
    }
    public._write(root / "summary.json", result)
    return result
