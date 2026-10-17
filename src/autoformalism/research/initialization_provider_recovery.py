"""Preserve failed ACES attempts while spending only their original remainder."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from autoformalism.llm.construction import MEASURED_RESERVATION
from autoformalism.llm.response_revision import SAFETY_TOKENS
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash

POLICY = "aces-provider-recovery-1"
_ADDED_FIELDS = {
    "artifact_sha256",
    "source_identity",
    "provider_recovery_policy",
    "recovery_source",
    "inherited_provider_outcomes",
}


def _sealed(value: dict) -> None:
    """Verify an embedded immutable artifact without rewriting its original seal."""
    if value.get("artifact_sha256") != content_hash(
        {key: item for key, item in value.items() if key != "artifact_sha256"}
    ):
        raise ValueError("recovery predecessor artifact digest differs")


def _name(value: str) -> str:
    """Keep imported task identifiers within the result directory."""
    if not isinstance(value, str) or Path(value).name != value or value in {".", ".."}:
        raise ValueError("unsafe recovery task identifier")
    return value


def _validate_outcome(parent: dict, task: dict, proof: dict) -> None:
    """Admit only three known provider failures with unchanged scientific state."""
    from autoformalism.research import saved_topology_repair as saved

    start = parent["starts"][_name(task["source_task"])]
    identity = saved.namespace(parent, task)
    result, events, calls = proof["result"], proof["events"], proof["calls"]
    receipts = proof["budget_preflight"]
    _sealed(result)
    if (
        len(events) != 3
        or len(calls) != 3
        or len(receipts) != 3
        or proof["physical_request_offset"] != 3
        or result["event_count"] != 3
        or result["identity"] != identity
        or result["task"] != task
        or result["source_draft_sha256"] != content_hash(start["draft"])
        or result["draft"] != start["draft"]
        or result["initializer_handoff"] != start["initializer_handoff"]
        or result["initial_information_contract"]
        != start["initial_information_contract"]
        or result["stop_reason"] != "three_repeated_repairs"
        or result["status"] != "assessed"
        or result.get("test_data_opened") is not False
        or result.get("optimizer_calls") != 0
    ):
        raise ValueError("recovery requires an unchanged provider-only final result")
    charged = 0
    seen = set()
    for index, (event, call, receipt) in enumerate(
        zip(events, calls, receipts, strict=True)
    ):
        _sealed(event)
        key = call["request_hash"]
        request = call["request"]
        reservation = call["budget_charge"]
        preflight = receipt["context_preflight"]
        if (
            key in seen
            or key != content_hash(request)
            or request["namespace"] != identity
            or request["settings"] != parent["settings"]
            or request.get("budget_reservation_policy") != MEASURED_RESERVATION
            or call["status"] != "provider_failure"
            or "raw_response" in call
            or call.get("observed_total_tokens") is not None
            or not call.get("error")
            or call["step"] != f"repair_{index:03d}"
            or call["attempt"] != index
            or call["budget_charge_basis"] != MEASURED_RESERVATION
            or type(reservation) is not int
            or reservation <= 0
            or event["identity"] != identity
            or event["index"] != index
            or event["request_hash"] != key
            or event["record_sha256"] != content_hash(call)
            or event["before"] != start["draft"]
            or event["after"] != start["draft"]
            or event["initialization_before"] != start["initializer_handoff"]
            or event["initialization_after"] != start["initializer_handoff"]
            or event["scientific_state"] != result["scientific_state"]
            or event["structural_assessment"] != result["assessment"]
            or event["stage"] != "repair"
            or event["accepted"] is not False
            or event["stage_complete"] is not False
            or event["semantic_model_unchanged"] is not True
            or event["consecutive_unchanged"] != index + 1
            or event["error"] != call["error"]
            or event["decisions"]
            or event["normalizations"]
            or event["required_predicates_lost"]
            or receipt["request_hash"] != key
            or receipt["status"] != "ready"
            or receipt["blocking_reasons"]
            or receipt["charged_before_call"] != charged
            or receipt["requests_before_call"] != index
            or receipt["active_request_ceiling"] != saved.MAX_CALLS
            or receipt["active_token_ceiling"] != saved.MAX_TOKENS
            or receipt["reservation"] != reservation
            or receipt["reservation_basis"] != MEASURED_RESERVATION
            or preflight["status"] != "ready"
            or preflight["scientific_context_truncated"] is not False
            or preflight["context_tokens"] != 32768
            or preflight["input_limit"] != 24000
            or not 0 <= preflight["input_tokens"] <= 24000
            or preflight["output_allowance"] != parent["settings"]["max_output_tokens"]
            or reservation
            != preflight["input_tokens"] + preflight["output_allowance"] + SAFETY_TOKENS
        ):
            raise ValueError("recovery provider-only call or event provenance differs")
        seen.add(key)
        charged += reservation
    cost = _cost(calls)
    if (
        result["cost"] != cost
        or proof["cost"] != cost
        or cost["budget_charge"] >= saved.MAX_TOKENS
        or cost["observed_tokens"] != 0
        or cost["requests_with_unknown_usage"] != 3
    ):
        raise ValueError("recovery inherited accounting differs")


def validate_plan(plan: dict) -> None:
    """Recheck self-contained source proofs and reject changed science or caps."""
    from autoformalism.research import saved_topology_repair as saved

    try:
        source = plan["recovery_source"]
        parent, execution = source["plan"], source["execution"]
        _sealed(parent)
        _sealed(execution)
        if (
            plan["provider_recovery_policy"] != POLICY
            or parent.get("provider_recovery_policy") is not None
            or parent["protocol"] != saved.PROTOCOL_INITIALIZATION_V3
            or parent.get("actionable_policy") != saved.initialized.POLICY_V3
            or parent.get("budget_reservation_policy") != MEASURED_RESERVATION
            or parent.get("execution_platform") != "aces-h100x1"
            or parent.get("test_data_opened") is not False
            or parent.get("automatic_followup") is not False
            or parent["settings"]["maximum_requests"] != saved.MAX_CALLS
            or parent["settings"]["maximum_total_tokens"] != saved.MAX_TOKENS
            or not 1 <= len(parent["tasks"]) <= 3
            or source["plan_sha256"] != parent["artifact_sha256"]
            or source["source_identity"] != parent["source_identity"]
            or source["execution_sha256"] != execution["artifact_sha256"]
            or execution["plan_sha256"] != parent["artifact_sha256"]
            or execution["protocol"] != "aces-saved-initialization-execution-1"
            or execution["platform"] != "aces-h100x1"
            or execution["model"] != parent["settings"]["model"]
            or execution["model_revision"] != parent["settings"]["model_revision"]
            or execution["served_context_tokens"] != 32768
            or execution.get("test_data_opened") is not False
            or execution.get("automatic_followup") is not False
            or {k: v for k, v in plan.items() if k not in _ADDED_FIELDS}
            != {k: v for k, v in parent.items() if k not in _ADDED_FIELDS}
        ):
            raise ValueError("recovery source policy, science or original caps differ")
        names = [_name(task["task_id"]) for task in parent["tasks"]]
        if len(names) != len(set(names)) or set(names) != set(
            plan["inherited_provider_outcomes"]
        ):
            raise ValueError("recovery roster differs")
        for task in parent["tasks"]:
            _validate_outcome(
                parent, task, plan["inherited_provider_outcomes"][task["task_id"]]
            )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("malformed provider recovery proof") from exc


def freeze(source: Path, destination: Path) -> dict:
    """Freeze one explicit continuation without altering or refunding its source."""
    from autoformalism.research import saved_topology_repair as saved

    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("write provider recovery outside its historical source")
    parent = sealed_read(source / "plan.json")
    execution = sealed_read(source / "execution.json")
    outcomes = {}
    for task in parent["tasks"]:
        folder = source / "results" / _name(task["task_id"])
        records = _cache_records(folder / "calls", saved.namespace(parent, task))
        events = [
            sealed_read(path) for path in sorted((folder / "events").glob("*.json"))
        ]
        by_hash = {record["request_hash"]: record for record in records}
        if {event["request_hash"] for event in events} != set(by_hash):
            raise ValueError("recovery source has unaccounted provider calls")
        calls = [by_hash[event["request_hash"]] for event in events]
        receipts = [
            json.loads(
                (
                    folder
                    / "calls"
                    / "budget_preflight"
                    / f"{call['request_hash']}.json"
                ).read_text()
            )
            for call in calls
        ]
        outcomes[task["task_id"]] = {
            "result": sealed_read(folder / "result.json"),
            "events": events,
            "calls": calls,
            "budget_preflight": receipts,
            "cost": _cost(calls),
            "physical_request_offset": len(calls),
        }
    plan = {
        **{k: deepcopy(v) for k, v in parent.items() if k != "artifact_sha256"},
        "source_identity": saved.source_identity(),
        "provider_recovery_policy": POLICY,
        "recovery_source": {
            "plan_sha256": parent["artifact_sha256"],
            "source_identity": parent["source_identity"],
            "execution_sha256": execution["artifact_sha256"],
            "plan": parent,
            "execution": execution,
        },
        "inherited_provider_outcomes": outcomes,
    }
    validate_plan(plan)
    return sealed_write(destination / "plan.json", plan)


def cumulative_cost(inherited: dict, current: dict) -> dict:
    """Add distinct inherited and new physical outcomes without refunding usage."""
    from collections import Counter

    return {
        **{
            key: inherited[key] + current[key] for key in inherited if key != "statuses"
        },
        "statuses": dict(Counter(inherited["statuses"]) + Counter(current["statuses"])),
    }


def cumulative_costs(costs: list[dict]) -> dict:
    """Aggregate explicit cost summaries without collapsing physical identities."""
    total = _cost([])
    for cost in costs:
        total = cumulative_cost(total, cost)
    return total
