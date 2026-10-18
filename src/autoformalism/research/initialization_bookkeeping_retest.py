"""Freeze V5 finals and complete historical charges for a bounded V6 retest."""

from __future__ import annotations

from pathlib import Path

from autoformalism.llm.construction import MEASURED_RESERVATION
from autoformalism.llm.staged_topology import StagedModelSettings, visible_response
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_training as training
from autoformalism.research import initialization_equality_retest as equality
from autoformalism.research import initialization_evidence_retest as evidence
from autoformalism.research import initialization_handoff_audit as audit
from autoformalism.research import initialized_topology_repair as live
from autoformalism.research import saved_topology_repair as saved
from autoformalism.research.actionable_repair_status import ALL_SEEN_WITNESSES
from autoformalism.search import construction_initialized_repair as api
from autoformalism.search.topology_initialization import INITIAL_TIME_POLICY
from autoformalism.staged_topology import content_hash


def _source(root: Path) -> dict:
    """Validate the historical seal and scientific boundaries, not today's source."""
    parent = sealed_read(root / "plan.json")
    required = {
        "protocol": live.PROTOCOL_V5,
        "actionable_policy": api.POLICY_V5,
        "question_identity_policy": training.MATHEMATICAL_QUESTION_IDENTITY,
        "initial_time_policy": INITIAL_TIME_POLICY,
        "witness_policy": ALL_SEEN_WITNESSES,
        "budget_reservation_policy": MEASURED_RESERVATION,
        "budget_policy": equality.BUDGET_POLICY,
        "execution_platform": "aces-h100x1",
        "maximum_calls_per_arm": saved.MAX_CALLS,
        "maximum_tokens_per_arm": saved.MAX_TOKENS,
        "initialization_calls_within_total_budget": 3,
    }
    if (
        any(parent.get(key) != value for key, value in required.items())
        or parent.get("test_data_opened") is not False
        or parent.get("automatic_followup") is not False
        or type(parent.get("optimizer_calls")) is not int
        or parent["optimizer_calls"] != 0
        or "provider_recovery_policy" in parent
        or "inherited_provider_outcomes" in parent
        or not 1 <= len(parent["tasks"]) <= 2
    ):
        raise ValueError("requires a sealed public-only separately budgeted V5 plan")
    settings = StagedModelSettings.model_validate(parent["settings"])
    if (
        settings.maximum_requests != saved.MAX_CALLS
        or settings.maximum_total_tokens != saved.MAX_TOKENS
    ):
        raise ValueError("historical V5 request or token allowance differs")
    return parent


def _call(record: dict, settings: dict) -> dict | None:
    """Check original charges; a length-limited response supplies no scientific edit.

    This freezer admits recorded complete JSON objects and output-length failures.
    Other provider failures or malformed complete responses require their own
    provenance review. In particular, partial text is never repaired or parsed.
    """
    usage = record.get("raw_response", {}).get("usage", {}).get("total_tokens")
    if (
        record.get("status") != "responded"
        or record["request"]["settings"] != settings
        or record["request"].get("budget_reservation_policy") != MEASURED_RESERVATION
        or record.get("budget_charge_basis") != "provider_total_tokens"
        or type(record.get("observed_total_tokens")) is not int
        or record["observed_total_tokens"] < 0
        or type(record.get("budget_charge")) is not int
        or record.get("budget_charge") != record["observed_total_tokens"]
        or type(usage) is not int
        or usage != record["observed_total_tokens"]
    ):
        raise ValueError("historical provider settings, response or charge differs")
    choices = record["raw_response"].get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("historical provider must return exactly one choice")
    if choices[0].get("finish_reason") == "length":
        return None
    reply = visible_response(record)
    if not isinstance(reply, dict):
        raise ValueError("historical provider reply is not a complete JSON object")
    return reply


def _episode(
    directory: Path, parent: dict, task: dict, original: dict, result: dict
) -> tuple[list[dict], dict[str, dict], list[dict]]:
    """Verify all zero-based V5 transactions, cached calls and final saved fields."""
    count = result["event_count"]
    if (
        type(count) is not int
        or not 1 <= count <= saved.MAX_CALLS
        or any(
            key in result
            for key in (
                "physical_request_offset",
                "provider_recovery_policy",
                "inherited_cost",
            )
        )
    ):
        raise ValueError("historical V5 event count or accounting scope differs")
    paths = sorted((directory / "events").glob("*.json"))
    if [path.name for path in paths] != [f"{index:03d}.json" for index in range(count)]:
        raise ValueError("historical V5 event coverage differs")
    events = [sealed_read(path) for path in paths]
    identity = saved.namespace(parent, task)
    records = _cache_records(directory / "calls", identity)
    by_hash = {record["request_hash"]: record for record in records}
    if (
        len(records) != count
        or len({event["request_hash"] for event in events}) != count
        or {event["request_hash"] for event in events} != set(by_hash)
    ):
        raise ValueError("historical V5 call coverage differs")
    before, initial = original["draft"], original["initializer_handoff"]
    availability = []
    for index, event in enumerate(events):
        record = by_hash[event["request_hash"]]
        reply = _call(record, parent["settings"])
        if (
            event["identity"] != identity
            or event["index"] != index
            or event["record_sha256"] != content_hash(record)
            or record["attempt"] != index
            or event["stage"] not in {"initialization", "repair"}
            or record["step"] != f"{event['stage']}_{index:03d}"
            or type(event["accepted"]) is not bool
            or event["before"] != before
            or event["initialization_before"] != initial
            or event["initial_information_contract"]
            != original["initial_information_contract"]
            or (
                not event["accepted"]
                and (
                    event["after"] != before or event["initialization_after"] != initial
                )
            )
        ):
            raise ValueError("historical event or response predecessor differs")
        if reply is None and (
            event["accepted"]
            or event["stage_complete"] is not False
            or event["error"] != "incomplete provider response: length"
            or event["normalizations"] != []
            or event["decisions"] != []
            or event["required_predicates_lost"] != []
            or event["semantic_model_unchanged"] is not True
            or (
                index > 0
                and any(
                    event[key] != events[index - 1][key]
                    for key in ("structural_assessment", "scientific_state")
                )
            )
        ):
            raise ValueError(
                "truncated historical response cannot supply a scientific edit"
            )
        availability.append(
            {
                "event_index": index,
                "event_sha256": event["artifact_sha256"],
                "request_hash": record["request_hash"],
                "record_sha256": event["record_sha256"],
                "status": record["status"],
                "finish_reason": record["raw_response"]["choices"][0]["finish_reason"],
                "scientific_reply_available": reply is not None,
                "budget_charge": record["budget_charge"],
                "recorded_event_error": event["error"],
                "recorded_consecutive_unchanged": event["consecutive_unchanged"],
            }
        )
        before, initial = event["after"], event["initialization_after"]
    final = events[-1]
    if any(
        final[event_key] != result[result_key]
        for event_key, result_key in (
            ("after", "draft"),
            ("initialization_after", "initializer_handoff"),
            ("initial_information_contract", "initial_information_contract"),
            ("structural_assessment", "assessment"),
            ("scientific_state", "scientific_state"),
        )
    ):
        raise ValueError("historical final event differs from result")
    if result["cost"] != _cost(records):
        raise ValueError("historical result cost differs from recorded charges")
    return events, by_hash, availability


def _feedback(
    events: list[dict], records: dict[str, dict], availability: list[dict]
) -> tuple[dict | None, dict | None, dict]:
    """Use the latest complete reply, never the historical truncation-as-edit receipt.

    An accepted latest reply needs no rejected-edit reminder. A source containing
    only unavailable replies carries neutral feedback with explicit provenance.
    Historical counters and stop labels remain archived, not reinterpreted as
    evidence of three scientific repetitions under V6.
    """
    usable = [row for row in availability if row["scientific_reply_available"]]
    selected = usable[-1] if usable else None
    last, repetition = None, None
    if selected is not None:
        event = events[selected["event_index"]]
        last, repetition = equality._last_feedback(
            event, records[event["request_hash"]]
        )
    return (
        last,
        repetition,
        {
            "policy": "latest-complete-visible-scientific-reply-1",
            "selected_event_index": selected["event_index"] if selected else None,
            "selected_event_sha256": selected["event_sha256"] if selected else None,
            "selected_request_hash": selected["request_hash"] if selected else None,
            "unavailable_event_indices": [
                row["event_index"]
                for row in availability
                if not row["scientific_reply_available"]
            ],
            "last_edit_carried": last is not None,
            "interpretation": "The latest complete scientific reply supplies feedback; "
            "an accepted reply or no available reply supplies neutral edit feedback. "
            "Length-limited text supplies no edit. Historical V5 truncation errors, "
            "repetition counts and stop labels are retained as recorded historical "
            "behavior, including its availability-classification defect. All calls "
            "remain charged; the new episode has a separate bounded allowance.",
        },
    )


def _freeze(source: Path, destination: Path, names: list[str]) -> dict:
    """Assemble a checked plan; the public wrapper normalizes malformed artifacts."""
    parent = _source(source)
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
            or result.get("status") != "assessed"
            or result.get("test_data_opened") is not False
            or type(result.get("optimizer_calls")) is not int
            or result["optimizer_calls"] != 0
            or result.get("scientific_accuracy_established") is not False
            or result["initial_information_contract"]
            != original["initial_information_contract"]
        ):
            raise ValueError("historical result identity or scientific scope differs")
        events, records, availability = _episode(
            directory, parent, task, original, result
        )
        evidence._terminal(result, events[-1])
        problem = saved.Problem(parent["cells"][task["benchmark_id"]], original["task"])
        draft = api.Draft.model_validate(result["draft"])
        if problem.assess(draft, qualified_memory=True) != result["assessment"]:
            raise ValueError("historical final structural assessment differs")
        last, repetition, feedback_source = _feedback(events, records, availability)
        start = {
            "task": original["task"],
            "draft": result["draft"],
            "assessment": problem.assess(draft),
            "diagnostics": problem.diagnose(draft),
            "previous_conflicts": evidence._conflicts(original, result),
            "initializer_handoff": result["initializer_handoff"],
            "initial_information_contract": result["initial_information_contract"],
            "last_edit": last,
            "repetition_feedback": repetition,
            "source_result_sha256": result["artifact_sha256"],
            "source_final_event_sha256": events[-1]["artifact_sha256"],
            "source_event_sha256": [event["artifact_sha256"] for event in events],
            "source_response_availability": availability,
            "source_feedback": feedback_source,
            "source_recorded_stop_reason": result["stop_reason"],
            "predecessor_provenance": {
                key: value
                for key, value in original.items()
                if key.startswith("source_")
            },
            "historical_cost": result["cost"],
            "predecessor_historical_cost": {
                "historical_cost": original["historical_cost"],
                "predecessor_historical_cost": original["predecessor_historical_cost"],
            },
        }
        historical = live.Session(problem, start, policy=api.POLICY_V5)
        if (
            any(
                result[key] != value
                for key, value in historical.result(draft, result["assessment"]).items()
            )
            or not result["initialization_status"]["complete"]
        ):
            raise ValueError("historical initializer or interaction handoff differs")
        live.Session(problem, start, policy=api.POLICY_V6)
        starts[name] = start
        tasks.append({**task, "task_id": name + "_bookkeeping", "source_task": name})
    return sealed_write(
        destination / "plan.json",
        {
            "protocol": live.PROTOCOL_V6,
            "actionable_policy": api.POLICY_V6,
            "question_identity_policy": training.MATHEMATICAL_QUESTION_IDENTITY,
            "source_identity": saved.source_identity(),
            "source_plan_sha256": parent["artifact_sha256"],
            "execution_platform": "aces-h100x1",
            "initial_time_policy": INITIAL_TIME_POLICY,
            "witness_policy": ALL_SEEN_WITNESSES,
            "budget_reservation_policy": MEASURED_RESERVATION,
            "budget_policy": equality.BUDGET_POLICY,
            "settings": parent["settings"],
            "cells": {
                key: value
                for key, value in parent["cells"].items()
                if key in {task["benchmark_id"] for task in tasks}
            },
            "starts": starts,
            "tasks": tasks,
            "selection": "One or two explicitly selected latest V5 final models; "
            "initializers and all qualified witnesses retained. Separately budgeted "
            "ACES bookkeeping diagnostic retest, without score-based selection, "
            "model promotion or a controlled causal comparison claim.",
            "maximum_calls_per_arm": saved.MAX_CALLS,
            "maximum_tokens_per_arm": saved.MAX_TOKENS,
            "initialization_calls_within_total_budget": 3,
            "historical_cost_scope": "Per-start historical_cost contains all V5 "
            "episode charges, including truncated responses. Nested "
            "predecessor_historical_cost retains each earlier episode's charges "
            "and recovered provider failures in its original scope. These scopes "
            "are not merged or refunded; all are excluded from the new ten-call, "
            "262144-token allowance.",
            "automatic_followup": False,
            "test_data_opened": False,
            "optimizer_calls": 0,
        },
    )


def freeze(source: Path, destination: Path, task_ids: list[str]) -> dict:
    """Freeze one or two latest V5 finals without modifying historical artifacts.

    Known-usage complete replies and length-limited responses retain their full
    charges. Truncated text supplies no scientific edit or carried feedback.
    Source seals, recorded settings and event chains establish historical
    provenance; archived code need not match the current V6 implementation.
    """
    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("write the bookkeeping retest outside its historical source")
    if not 1 <= len(task_ids) <= 2 or len(set(task_ids)) != len(task_ids):
        raise ValueError("select one or two distinct saved task IDs")
    try:
        return _freeze(source, destination, [audit._name(name) for name in task_ids])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("malformed historical V5 bookkeeping retest source") from exc
