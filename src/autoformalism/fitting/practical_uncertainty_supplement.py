"""M40: separately budgeted profiles of immutable, verified M39 endpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from time import monotonic, process_time
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import practical_profile
from autoformalism.fitting import practical_uncertainty_campaign as previous
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.base import FiniteFloat, StrictSchema

PROTOCOL = "phase-c-practical-uncertainty-supplement-1"
COUNTERS = (*previous.COUNTERS, "frozen_calls", "frozen_nodes_completed")
JOURNAL_TIMES = (
    "serialization_wall_seconds",
    "serialization_cpu_seconds",
    "write_fsync_wall_seconds",
    "write_fsync_cpu_seconds",
)


class Policy(StrictSchema):
    """Freeze the incremental allowance and execution implementation, not science."""

    protocol: Literal["phase-c-practical-uncertainty-supplement-policy-1"] = (
        "phase-c-practical-uncertainty-supplement-policy-1"
    )
    maximum_seconds: FiniteFloat = Field(default=90.0, gt=0, le=3600)
    selection: Literal["incomplete_profiles"] = "incomplete_profiles"
    execution: practical_profile.ExecutionPolicy = Field(
        default_factory=practical_profile.ExecutionPolicy
    )


def _files(source: Path) -> dict[str, str]:
    """Bind retained experiment artifacts, including partial diagnostic journals."""
    paths = [source / "plan.json", source / "inputs.json"]
    paths += sorted(
        path
        for path in (source / "results").rglob("*")
        if path.suffix in {".json", ".jsonl"}
    )
    result = {}
    for path in paths:
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(source.resolve())
        ):
            raise ValueError("source artifacts must be regular, contained files")
        result[str(path.relative_to(source))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


def _source_rows(source: Path) -> tuple[dict, dict, list[dict]]:
    """Read historical seals without fitting, profiling, scoring or source writes."""
    plan, data = previous.verify(source, runtime=False)
    rows = [previous.report_row(source, plan, data, task) for task in plan["tasks"]]
    for row in rows:
        backend = row.get("backend", {})
        parameters = backend.get("parameters")
        if (
            backend.get("status") != "complete"
            or backend.get("verification", {}).get("passed") is not True
            or not isinstance(parameters, dict)
            or set(parameters) != set(previous.fixtures.PARAMETERS)
            or not all(math.isfinite(value) for value in parameters.values())
            or not isinstance(backend.get("objective_value"), (float, int))
            or not math.isfinite(backend["objective_value"])
        ):
            raise ValueError("supplement requires every source fit to be verified")
    return plan, data, rows


def audit(source: Path) -> dict:
    """Select solely by diagnostic availability after all retained fits qualify."""
    plan, _, rows = _source_rows(source)
    records = [
        {
            "task_id": row["task_id"],
            "row_sha256": public.content_sha256(row),
            "backend_sha256": public.content_sha256(row["backend"]),
            "source_profile_status": row.get("profile", {}).get("status", "missing"),
        }
        for row in rows
    ]
    return {
        "status": "passed",
        "source_plan_sha256": public.content_sha256(plan),
        "expected": len(rows),
        "verified_fits": len(rows),
        "profile_status_counts": dict(
            Counter(row["source_profile_status"] for row in records)
        ),
        "selected_task_ids": [
            row["task_id"]
            for row in records
            if row["source_profile_status"] != "complete"
        ],
        "rows": records,
        "selection_uses_truth": False,
        "fitting_performed": False,
        "numerical_calls": 0,
    }


def _copy_snapshot(source: Path, destination: Path, manifest: dict) -> None:
    """Retain exact bytes once and reject concurrent or partial-copy changes."""
    for name, digest in manifest.items():
        payload = (source / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("source changed during supplement snapshot")
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError("partial supplement snapshot differs")
        else:
            temporary = path.with_suffix(path.suffix + ".copying")
            with temporary.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
    if _files(source) != manifest or _files(destination) != manifest:
        raise ValueError("source/snapshot changed during supplement preparation")


def prepare(root: Path, source: Path, policy: Policy) -> dict:
    """Freeze a portable historical snapshot and a distinct diagnostic allowance."""
    root, source = root.resolve(), source.resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("supplement and source must use separate directories")
    manifest = _files(source)
    with public._lock(root):
        if (root / "plan.json").exists():
            plan, frozen = verify(root)
            if frozen != manifest or plan["policy"] != policy.model_dump(mode="json"):
                raise ValueError("supplement source/policy differs")
        else:
            source_audit = audit(source)
            source_plan = read_seal(source / "plan.json")
            _copy_snapshot(source, root / "source_snapshot", manifest)
            selected = set(source_audit["selected_task_ids"])
            plan = {
                "protocol": PROTOCOL,
                "policy": policy.model_dump(mode="json"),
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "inputs_sha256": public.content_sha256(manifest),
                "audit_sha256": public.content_sha256(source_audit),
                "source_plan_sha256": public.content_sha256(source_plan),
                "diagnostic": source_plan["diagnostic"],
                "tasks": [
                    task for task in source_plan["tasks"] if task["task_id"] in selected
                ],
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
    """Verify every frozen source byte; execution additionally binds new runtime."""
    plan, manifest = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    frozen = read_seal(root / "portable_audit.json")
    source_plan = read_seal(root / "source_snapshot/plan.json")
    selected = set(frozen["selected_task_ids"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(manifest)
        or _files(root / "source_snapshot") != manifest
        or plan["audit_sha256"] != public.content_sha256(frozen)
        or plan["source_plan_sha256"] != public.content_sha256(source_plan)
        or plan["source_plan_sha256"] != frozen["source_plan_sha256"]
        or plan["diagnostic"] != source_plan["diagnostic"]
        or plan["tasks"]
        != [task for task in source_plan["tasks"] if task["task_id"] in selected]
        or frozen["status"] != "passed"
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


def _identity(plan: dict, task: dict, backend: dict) -> dict:
    """Bind the new allowance to exactly one original selected endpoint."""
    return {
        "plan_sha256": public.content_sha256(plan),
        "task": task,
        "backend_sha256": public.content_sha256(backend),
    }


def _progress(folder: Path, identity: dict) -> dict:
    """Recover complete durable records; a torn final record never invents cost."""
    path = folder / "profile-progress.jsonl"
    value, events, truncated, records = {}, [], False, 0
    if path.exists():
        for sequence, line in enumerate(path.read_bytes().splitlines(keepends=True)):
            if not line.endswith(b"\n"):
                truncated = True
                break
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or record.get("identity_sha256") != public.content_sha256(identity)
                or record.get("sequence") != sequence
                or not isinstance(record.get("value"), dict)
            ):
                raise ValueError("supplement progress identity differs")
            candidate = record["value"]
            for name in COUNTERS:
                count = candidate.get(name)
                if count is not None and (
                    type(count) is not int or count < value.get(name, 0)
                ):
                    raise ValueError("supplement progress counter differs")
            value = candidate
            records += 1
            if "event" in value:
                events.append(value)
    return value | {
        "events": events,
        "journal_records": records,
        "truncated_final_record": truncated,
    }


def _read_profile(folder: Path, identity: dict, allowance: float) -> dict | None:
    """Return immutable terminal evidence or a conservative interrupted view."""
    started, terminal = folder / "profile-started.json", folder / "profile.json"
    if terminal.exists():
        value = read_seal(terminal)
        if read_seal(started) != identity or value.get("identity") != identity:
            raise ValueError("supplement profile identity differs")
        return value
    if not started.exists():
        if (folder / "profile-progress.jsonl").exists():
            raise ValueError("supplement progress lacks a durable start")
        return None
    if read_seal(started) != identity:
        raise ValueError("supplement started identity differs")
    return {
        "status": "unfinished",
        "identity": identity,
        "wall_seconds": None,
        "cpu_seconds": None,
        "budget_charge_seconds": allowance,
        "accounting_complete": False,
        "observed_progress": _progress(folder, identity),
        "timing_scope": "Terminal artifact serialization/write is excluded.",
    }


def _profile_once(folder: Path, identity: dict, allowance: float, operation) -> dict:
    """Charge one durable attempt, retaining interrupted work without retrying it."""
    previous_result = _read_profile(folder, identity, allowance)
    if previous_result is not None:
        if previous_result["status"] != "unfinished":
            return previous_result
        result = previous_result | {
            "status": "interrupted",
            "reason": "Started extension has unknown usage and is not repeated.",
        }
        seal(folder / "profile.json", result)
        return result
    began, cpu = monotonic(), process_time()
    seal(folder / "profile-started.json", identity)
    started_write_wall, started_write_cpu = monotonic() - began, process_time() - cpu
    journal = dict.fromkeys(JOURNAL_TIMES, 0.0) | {
        "records": 0,
        "bytes": 0,
        "persistence_complete": True,
    }
    digest = public.content_sha256(identity)
    latest = {}

    def save(value):
        nonlocal latest
        before, before_cpu = monotonic(), process_time()
        try:
            encoded = (
                json.dumps(
                    {
                        "identity_sha256": digest,
                        "sequence": journal["records"],
                        "value": value,
                    },
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode()
        except Exception:
            journal["persistence_complete"] = False
            raise
        finally:
            journal["serialization_wall_seconds"] += monotonic() - before
            journal["serialization_cpu_seconds"] += process_time() - before_cpu
        before, before_cpu = monotonic(), process_time()
        try:
            with (folder / "profile-progress.jsonl").open("ab") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            journal["persistence_complete"] = False
            raise
        finally:
            journal["write_fsync_wall_seconds"] += monotonic() - before
            journal["write_fsync_cpu_seconds"] += process_time() - before_cpu
        journal["records"] += 1
        journal["bytes"] += len(encoded)
        latest = value

    save(dict.fromkeys(COUNTERS, 0))
    failed = False
    try:
        result = operation(began + allowance, save)
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
        failed = True
        result = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error)[:1000],
            "observed_progress": _progress(folder, identity),
        }
    consistent = all(
        type(result.get(name)) is int and result[name] >= latest.get(name, 0)
        for name in COUNTERS
    )
    wall, elapsed_cpu = monotonic() - began, process_time() - cpu
    result |= {
        "identity": identity,
        "wall_seconds": wall,
        "cpu_seconds": elapsed_cpu,
        "budget_charge_seconds": min(wall, allowance),
        "accounting_complete": not failed
        and result.get("accounting_complete", True)
        and result.get("journal_persistence_complete", True)
        and journal["persistence_complete"]
        and consistent
        and result.get("calls") == result.get("completed_calls")
        and result.get("calls")
        == sum(
            result.get(name, 0)
            for name in ("solve_calls", "rollout_calls", "frozen_calls")
        )
        and result.get("inner_iterations_complete") is True,
        "journal_timing": journal,
        "journal_persistence_complete": result.get("journal_persistence_complete", True)
        and journal["persistence_complete"],
        "started_write_wall_seconds": started_write_wall,
        "started_write_cpu_seconds": started_write_cpu,
        "timing_scope": (
            "Stage includes durable start, numerical execution and progress writes. "
            "Terminal artifact serialization/write is excluded; profile timing is "
            "nested within stage timing, not additive to it."
        ),
    }
    seal(folder / "profile.json", result)
    return result


def _completed(folder: Path, identity: dict, allowance: float) -> dict:
    """Reject copied result bodies and verify the separately sealed diagnostic."""
    result = read_seal(folder / "result.json")
    profile = _read_profile(folder, identity, allowance)
    if (
        result.get("identity") != identity
        or profile is None
        or result.get("profile") != profile
        or result.get("status") != profile.get("status")
        or any(result.get(key) != value for key, value in identity["task"].items())
    ):
        raise ValueError("supplement terminal identity differs")
    return result


def _group(rows: list[dict]) -> dict:
    """Summarize effective evidence without presenting replacement costs as totals."""
    summary = previous._group(rows)
    summary.pop("cost")
    summary["cost_scope"] = "Costs are retained separately in campaign cost summaries."
    return summary


def run_task(root: Path, index: int) -> dict:
    """Profile one retained optimum; no fitting or evaluator entry point is called."""
    plan, _ = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown practical uncertainty supplement task index")
    policy = Policy.model_validate(plan["policy"])
    source = root / "source_snapshot"
    source_plan, data = previous.verify(source, runtime=False)
    task = plan["tasks"][index]
    row = previous.report_row(source, source_plan, data, task)
    backend = row["backend"]
    identity = _identity(plan, task, backend)
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return _completed(folder, identity, policy.maximum_seconds)

        def operation(deadline, save):
            return practical_profile.evaluate(
                previous.training_problem(data, task),
                backend,
                target=plan["diagnostic"]["target"],
                policy=practical_profile.Policy.model_validate(
                    plan["diagnostic"]["policy"]
                ),
                execution=policy.execution,
                deadline=deadline,
                save=save,
            )

        profile = _profile_once(folder, identity, policy.maximum_seconds, operation)
        result = task | {
            "identity": identity,
            "status": profile["status"],
            "profile": profile,
        }
        seal(folder / "result.json", result)
        return result


def _cost(profiles: list[dict]) -> dict:
    """Keep known observations, unknown totals and nested timing separately."""
    values = [{"profile": value} for value in profiles]
    result = previous._stage_cost(values, "profile")
    for field in ("wall_seconds", "cpu_seconds", *COUNTERS):
        counts = [value.get(field) for value in profiles]
        if field == "inner_iterations":
            counts = [
                count if value.get("inner_iterations_complete") is True else None
                for value, count in zip(profiles, counts, strict=True)
            ]
        result[field + "_observed"] = sum(
            count
            if count is not None
            else value.get("observed_progress", {}).get(field) or 0
            for value, count in zip(profiles, counts, strict=True)
        )
        result[field + "_unknown_or_missing"] = sum(value is None for value in counts)
        result[field + "_total"] = (
            sum(counts) if all(value is not None for value in counts) else None
        )
    result["profiles"] = len(profiles)
    result["scope"] = (
        "Observed values are lower bounds where totals are unknown. M39 did not "
        "record frozen-only operation counters or the new timing breakdown; their "
        "absence is unavailable historical instrumentation, not zero work."
    )
    for category in ("timing", "journal_timing"):
        names = sorted(
            {
                key
                for value in profiles
                for key in value.get(category, {})
                if key.endswith("seconds")
            }
        )
        result[category] = {}
        for name in names:
            counts = [value.get(category, {}).get(name) for value in profiles]
            result[category][name] = {
                "observed": sum(value for value in counts if value is not None),
                "total": sum(counts)
                if all(value is not None for value in counts)
                else None,
                "unknown_or_missing": sum(value is None for value in counts),
            }
    return result


def report(root: Path) -> dict:
    """Preserve all source outcomes and charge old and new diagnostics additively."""
    plan, _ = verify(root, runtime=False)
    source = root / "source_snapshot"
    if audit(source) != read_seal(root / "portable_audit.json"):
        raise ValueError("supplement portable source audit differs")
    _, _, source_rows = _source_rows(source)
    policy = Policy.model_validate(plan["policy"])
    selected = {task["task_id"] for task in plan["tasks"]}
    rows, extension_profiles = [], []
    for row in source_rows:
        extension = None
        if row["task_id"] in selected:
            task = next(
                task for task in plan["tasks"] if task["task_id"] == row["task_id"]
            )
            identity = _identity(plan, task, row["backend"])
            folder = root / "results" / row["task_id"]
            extension = (
                _completed(folder, identity, policy.maximum_seconds)["profile"]
                if (folder / "result.json").exists()
                else _read_profile(folder, identity, policy.maximum_seconds)
            )
            extension_profiles.append(extension or {"status": "missing"})
        effective = (
            extension
            if extension and extension["status"] == "complete"
            else row.get("profile", {})
        )
        rows.append(
            row
            | {
                "source_profile": row.get("profile"),
                "extension_profile": extension,
                "profile": effective,
            }
        )
    source_profiles = [row.get("profile", {}) for row in source_rows]
    counts = dict(Counter(value["status"] for value in extension_profiles))
    recorded = sum(
        value["status"] not in {"missing", "unfinished"} for value in extension_profiles
    )
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if recorded == len(selected) else "incomplete",
        "expected": len(selected),
        "recorded": recorded,
        "status_counts": counts,
        "source_expected": len(source_rows),
        "source_verified_fits": len(source_rows),
        "source_profile_status_counts": dict(
            Counter(value.get("status", "missing") for value in source_profiles)
        ),
        "combined_profile_status_counts": dict(
            Counter(row["profile"].get("status", "missing") for row in rows)
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
            for separation in previous.fixtures.SEPARATIONS
            for sigma in previous.fixtures.SIGMAS
            for strategy in previous.STRATEGIES
        ],
        "paired_objectives": previous._paired_objectives(rows),
        "cost": {
            "source_fit_unchanged": previous._cost_summary(source_rows),
            "source_diagnostic": _cost(source_profiles),
            "extension_diagnostic": _cost(extension_profiles),
            "combined_diagnostic": _cost(source_profiles + extension_profiles),
            "source_evaluation_unchanged": previous._stage_cost(
                source_rows, "evaluation"
            ),
        },
        "fitting_performed": False,
        "test_data_opened": False,
        "truth_used_for_selection": False,
        "live_llm_calls": 0,
        "confidence_probability": None,
        "limitation": (
            "Only incomplete source diagnostics receive a distinct frozen allowance. "
            "All original fits, profiles and costs remain retained. Completed "
            "extensions supply missing support evidence; domain-assisted ranges "
            "are not confidence intervals or proof of identification. Timing is not "
            "additive; terminal artifact writes are excluded. Cross-run wall times "
            "do not establish an execution speedup. Unknown work remains unknown."
        ),
    }
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", summary)
    return summary
