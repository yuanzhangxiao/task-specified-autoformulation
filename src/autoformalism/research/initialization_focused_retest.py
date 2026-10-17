"""Freeze selected saved initializer failures for a separate bounded ACES retest."""

from __future__ import annotations

from pathlib import Path

from autoformalism.llm.construction import MEASURED_RESERVATION
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import initialization_handoff_audit as audit
from autoformalism.research import initialized_topology_repair as live
from autoformalism.research import saved_topology_repair as saved
from autoformalism.research.actionable_repair_status import ALL_SEEN_WITNESSES
from autoformalism.search import construction_initialized_repair as api
from autoformalism.staged_topology import content_hash


def _last_feedback(directory: Path, parent: dict, task: dict, result: dict):
    """Bind the final rejected reply to its actual event and retained predecessor."""
    count = result["event_count"]
    if count < 1:
        raise ValueError("focused retest requires a saved reply")
    identity = saved.namespace(parent, task)
    event = sealed_read(directory / "events" / f"{count - 1:03d}.json")
    records = _cache_records(directory / "calls", identity)
    record = next(
        (r for r in records if r["request_hash"] == event["request_hash"]), None
    )
    if (
        record is None
        or content_hash(record) != event["record_sha256"]
        or event["identity"] != identity
        or event["index"] != count - 1
        or event["after"] != result["draft"]
        or event["initialization_after"] != result["initializer_handoff"]
        or event["scientific_state"] != result["scientific_state"]
    ):
        raise ValueError("saved final event or response differs")
    if event["accepted"]:
        return None, None, event["artifact_sha256"]
    if event["before"] != event["after"] or (
        event["initialization_before"] != event["initialization_after"]
    ):
        raise ValueError("rejected historical transaction changed the draft")
    raw = visible_response(record)
    last = {
        "status": "rejected",
        "transaction_applied": False,
        "error": event["error"],
        "stage_complete": False,
        "compatibility_decisions": [],
        "required_predicates_lost": [],
        "edit_effects": None,
        "rejected_reply": raw,
        "normalizations": event["normalizations"],
        "initialization_before": event["initialization_before"],
        "initialization_after": event["initialization_after"],
    }
    repetition = {
        "consecutive_unchanged": event["consecutive_unchanged"],
        "previous_reply": raw,
        "previous_error": event["error"],
        "remaining_error_codes": sorted(
            {e["code"] for e in event["structural_assessment"]["errors"]}
        ),
        "instruction": "This saved reply was rejected without changing the model. "
        "Correct its listed record errors; do not repeat the rejected declarations.",
        "historical_receipt": True,
    }
    return last, repetition, event["artifact_sha256"]


def freeze(source: Path, destination: Path, task_ids: list[str]) -> dict:
    """Preserve final drafts, initializers and witnesses; select by named defect only.

    This is a new separately budgeted experiment, not continuation with a hidden
    budget reset. Historical calls and scores never enter model selection.
    """
    from autoformalism.search.topology_initialization import INITIAL_TIME_POLICY

    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("write the focused plan outside its historical source")
    parent = sealed_read(source / "plan.json")
    if (
        parent.get("protocol") != live.PROTOCOL_V2
        or parent.get("actionable_policy") != api.POLICY_V2
        or parent.get("test_data_opened") is not False
        or parent["settings"]["maximum_requests"] != saved.MAX_CALLS
        or parent["settings"]["maximum_total_tokens"] != saved.MAX_TOKENS
    ):
        raise ValueError("requires a sealed public-only initializer v2 campaign")
    if not 1 <= len(task_ids) <= 3 or len(set(task_ids)) != len(task_ids):
        raise ValueError("select one to three distinct saved task IDs")
    roster = {audit._name(t["task_id"]): t for t in parent["tasks"]}
    if len(roster) != len(parent["tasks"]) or set(task_ids) - set(roster):
        raise ValueError("unknown or duplicate historical task")
    starts, tasks = {}, []
    for name in task_ids:
        task = roster[audit._name(name)]
        original = parent["starts"][audit._name(task["source_task"])]
        directory = source / "results" / name
        result = sealed_read(directory / "result.json")
        if (
            result["identity"] != saved.namespace(parent, task)
            or result["task"] != task
            or result["source_draft_sha256"] != content_hash(original["draft"])
            or result.get("test_data_opened") is not False
        ):
            raise ValueError("saved final result identity differs")
        problem = saved.Problem(parent["cells"][task["benchmark_id"]], original["task"])
        draft = api.Draft.model_validate(result["draft"])
        last, repetition, event_hash = _last_feedback(directory, parent, task, result)
        starts[name] = {
            "task": original["task"],
            "draft": result["draft"],
            "assessment": problem.assess(draft),
            "diagnostics": problem.diagnose(draft),
            "previous_conflicts": audit._saved_conflicts(original, result),
            "initializer_handoff": result["initializer_handoff"],
            "initial_information_contract": result["initial_information_contract"],
            "last_edit": last,
            "repetition_feedback": repetition,
            "source_result_sha256": result["artifact_sha256"],
            "source_final_event_sha256": event_hash,
            "historical_cost": result["cost"],
        }
        live.Session(problem, starts[name], policy=api.POLICY_V3)
        tasks.append({**task, "task_id": name + "_focused", "source_task": name})
    return sealed_write(
        destination / "plan.json",
        {
            "protocol": live.PROTOCOL_V3,
            "actionable_policy": api.POLICY_V3,
            "source_identity": saved.source_identity(),
            "source_plan_sha256": parent["artifact_sha256"],
            "execution_platform": "aces-h100x1",
            "initial_time_policy": INITIAL_TIME_POLICY,
            "witness_policy": ALL_SEEN_WITNESSES,
            "budget_reservation_policy": MEASURED_RESERVATION,
            "settings": parent["settings"],
            "cells": {
                k: v
                for k, v in parent["cells"].items()
                if k in {t["benchmark_id"] for t in tasks}
            },
            "starts": starts,
            "tasks": tasks,
            "selection": "Explicit saved cases affected by feedback expansion or "
            "initial-time conditioning. Final drafts, initializers and witnesses "
            "retained. Separately budgeted ACES diagnostic retest; no score-based "
            "selection or model promotion.",
            "maximum_calls_per_arm": saved.MAX_CALLS,
            "maximum_tokens_per_arm": saved.MAX_TOKENS,
            "initialization_calls_within_total_budget": 3,
            "historical_cost_scope": "Retained per start; excluded from new-call "
            "accounting.",
            "automatic_followup": False,
            "test_data_opened": False,
            "optimizer_calls": 0,
        },
    )
