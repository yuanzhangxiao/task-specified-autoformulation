"""Freeze explicit recovered V3 finals for a separately budgeted equality retest."""

from __future__ import annotations

from pathlib import Path

from autoformalism.llm.construction import MEASURED_RESERVATION
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_training as training
from autoformalism.research import initialization_handoff_audit as audit
from autoformalism.research import initialization_provider_recovery as recovery
from autoformalism.research import initialized_topology_repair as live
from autoformalism.research import saved_topology_repair as saved
from autoformalism.research.actionable_repair_status import ALL_SEEN_WITNESSES
from autoformalism.search import construction_initialized_repair as api
from autoformalism.search.topology_initialization import INITIAL_TIME_POLICY
from autoformalism.staged_topology import content_hash

BUDGET_POLICY = "separate-new-retest-budget-1"


def _episode(
    directory: Path, parent: dict, task: dict, start: dict, result: dict
) -> tuple[list[dict], dict]:
    """Bind every saved response and offset-indexed transaction to its predecessor."""
    identity = saved.namespace(parent, task)
    proof = parent["inherited_provider_outcomes"][task["task_id"]]
    offset = proof["physical_request_offset"]
    count = result["event_count"]
    if (
        type(count) is not int
        or not 1 <= count <= saved.MAX_CALLS - offset
        or result.get("physical_request_offset") != offset
        or result.get("provider_recovery_policy") != recovery.POLICY
    ):
        raise ValueError("saved recovery event count or offset differs")
    paths = sorted((directory / "events").glob("*.json"))
    if [path.name for path in paths] != [
        f"{index:03d}.json" for index in range(offset, offset + count)
    ]:
        raise ValueError("saved recovery event coverage differs")
    events = [sealed_read(path) for path in paths]
    records = _cache_records(directory / "calls", identity)
    by_hash = {record["request_hash"]: record for record in records}
    if (
        len(records) != count
        or len({event["request_hash"] for event in events}) != count
        or {event["request_hash"] for event in events} != set(by_hash)
    ):
        raise ValueError("saved recovery call coverage differs")
    settings = {
        **parent["settings"],
        "maximum_requests": saved.MAX_CALLS - offset,
        "maximum_total_tokens": saved.MAX_TOKENS - proof["cost"]["budget_charge"],
    }
    before, initial = start["draft"], start["initializer_handoff"]
    for index, event in enumerate(events, offset):
        record = by_hash[event["request_hash"]]
        if (
            event["identity"] != identity
            or event["index"] != index
            or event["record_sha256"] != content_hash(record)
            or record["attempt"] != index
            or record["step"] != f"{event['stage']}_{index:03d}"
            or record["request"]["settings"] != settings
            or record["request"].get("budget_reservation_policy")
            != MEASURED_RESERVATION
            or event["before"] != before
            or event["initialization_before"] != initial
            or event.get(
                "initial_information_contract", result["initial_information_contract"]
            )
            != result["initial_information_contract"]
            or (
                event["accepted"] is not True
                and (
                    event["after"] != before
                    or event["initialization_after"] != initial
                )
            )
        ):
            raise ValueError("saved recovery event or response predecessor differs")
        before, initial = event["after"], event["initialization_after"]
    final = events[-1]
    if any(
        final[event_key] != result[result_key]
        for event_key, result_key in (
            ("after", "draft"),
            ("initialization_after", "initializer_handoff"),
            ("structural_assessment", "assessment"),
            ("scientific_state", "scientific_state"),
        )
    ):
        raise ValueError("saved final event differs from result")
    cost = _cost(records)
    cumulative = recovery.cumulative_cost(proof["cost"], cost)
    if (
        result.get("new_cost") != cost
        or result.get("inherited_cost") != proof["cost"]
        or result["cost"] != cumulative
        or result.get("cumulative_cost") != cumulative
    ):
        raise ValueError("saved recovery historical accounting differs")
    return events, by_hash[final["request_hash"]]


def _last_feedback(event: dict, record: dict) -> tuple[dict | None, dict | None]:
    """Retain the final rejected scientific reply, excluding transport failures."""
    if event.get("scientific_reply_received") is False:
        return (
            event.get("retained_last_edit"),
            event.get("retained_repetition_feedback"),
        )
    if event["accepted"]:
        return None, None
    try:
        raw = visible_response(record)
    except ValueError:
        # A truncated or invalid JSON response has an error receipt but no edit.
        raw = None
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
            {error["code"] for error in event["structural_assessment"]["errors"]}
        ),
        "instruction": "This saved reply was rejected without changing the model. "
        "Correct its listed record errors; do not repeat the rejected declarations.",
        "historical_receipt": True,
    }
    return last, repetition


def freeze(source: Path, destination: Path, task_ids: list[str]) -> dict:
    """Preserve one or two named recovered finals under an explicit new allowance.

    Historical seals and response chains are checked without requiring their
    archived source identity to equal today's code. No source artifact is changed,
    and historical charges are not passed to the new episode's budget counter.
    """
    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("write the equality retest outside its historical source")
    if not 1 <= len(task_ids) <= 2 or len(set(task_ids)) != len(task_ids):
        raise ValueError("select one or two distinct saved task IDs")
    names = [audit._name(name) for name in task_ids]
    parent = sealed_read(source / "plan.json")
    if (
        parent.get("protocol") != live.PROTOCOL_V3
        or parent.get("actionable_policy") != api.POLICY_V3
        or parent.get("provider_recovery_policy") != recovery.POLICY
        or parent.get("test_data_opened") is not False
        or parent.get("optimizer_calls") != 0
        or parent.get("witness_policy") != ALL_SEEN_WITNESSES
        or parent.get("initial_time_policy") != INITIAL_TIME_POLICY
    ):
        raise ValueError("requires a sealed public-only recovered initializer v3 plan")
    recovery.validate_plan(parent)
    roster = {audit._name(task["task_id"]): task for task in parent["tasks"]}
    if len(roster) != len(parent["tasks"]) or set(names) - set(roster):
        raise ValueError("unknown or duplicate historical task")
    starts, tasks = {}, []
    for name in names:
        task = roster[name]
        original = parent["starts"][audit._name(task["source_task"])]
        directory = source / "results" / name
        result = sealed_read(directory / "result.json")
        if (
            result["identity"] != saved.namespace(parent, task)
            or result["task"] != task
            or result["source_draft_sha256"] != content_hash(original["draft"])
            or result.get("test_data_opened") is not False
            or result.get("optimizer_calls") != 0
            or result["initial_information_contract"]
            != original["initial_information_contract"]
        ):
            raise ValueError("saved final result identity or public-only scope differs")
        events, record = _episode(directory, parent, task, original, result)
        last, repetition = _last_feedback(events[-1], record)
        problem = saved.Problem(parent["cells"][task["benchmark_id"]], original["task"])
        draft = api.Draft.model_validate(result["draft"])
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
            "source_final_event_sha256": events[-1]["artifact_sha256"],
            "source_event_sha256": [event["artifact_sha256"] for event in events],
            "source_physical_request_offset": result["physical_request_offset"],
            "historical_cost": result["cost"],
            "predecessor_historical_cost": original.get("historical_cost"),
        }
        live.Session(problem, starts[name], policy=api.POLICY_V4)
        tasks.append({**task, "task_id": name + "_equality", "source_task": name})
    return sealed_write(
        destination / "plan.json",
        {
            "protocol": live.PROTOCOL_V4,
            "actionable_policy": api.POLICY_V4,
            "question_identity_policy": training.MATHEMATICAL_QUESTION_IDENTITY,
            "source_identity": saved.source_identity(),
            "source_plan_sha256": parent["artifact_sha256"],
            "execution_platform": "aces-h100x1",
            "initial_time_policy": INITIAL_TIME_POLICY,
            "witness_policy": ALL_SEEN_WITNESSES,
            "budget_reservation_policy": MEASURED_RESERVATION,
            "budget_policy": BUDGET_POLICY,
            "settings": parent["settings"],
            "cells": {
                key: value
                for key, value in parent["cells"].items()
                if key in {task["benchmark_id"] for task in tasks}
            },
            "starts": starts,
            "tasks": tasks,
            "selection": "One or two explicitly named recovered V3 final drafts; "
            "initializers and all qualified witnesses retained. Separately budgeted "
            "ACES equality diagnostic retest; no score-based selection or promotion.",
            "maximum_calls_per_arm": saved.MAX_CALLS,
            "maximum_tokens_per_arm": saved.MAX_TOKENS,
            "initialization_calls_within_total_budget": 3,
            "historical_cost_scope": "Per-start historical_cost is the recovered V3 "
            "cumulative cost, including its provider-only predecessor. Older costs "
            "remain in predecessor_historical_cost. All are excluded from the new "
            "ten-call, 262144-token allowance; this is not a continuation or refund.",
            "automatic_followup": False,
            "test_data_opened": False,
            "optimizer_calls": 0,
        },
    )
