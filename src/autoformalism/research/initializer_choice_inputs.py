"""Freeze replay-verified V6 finals for a separate initializer-choice diagnostic."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from autoformalism.llm.construction import MEASURED_RESERVATION
from autoformalism.llm.staged_topology import StagedModelSettings, atomic_json
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_training as training
from autoformalism.research import initialization_bookkeeping_retest as bookkeeping
from autoformalism.research import initialization_equality_retest as equality
from autoformalism.research import initialization_evidence_retest as evidence
from autoformalism.research import initialization_handoff_audit as audit
from autoformalism.research import initialized_topology_repair as live
from autoformalism.research import saved_topology_repair as saved
from autoformalism.research.actionable_repair_status import ALL_SEEN_WITNESSES
from autoformalism.search import construction_initialized_repair as api
from autoformalism.search.topology_initialization import INITIAL_TIME_POLICY
from autoformalism.staged_topology import content_hash

PROTOCOL = "saved-initializer-choice-diagnostic-1"
BUDGETS = {
    "maximum_choice_calls_per_case": 2,
    "maximum_choice_tokens_per_case": 65536,
    "completion_probe_calls": 2,
    "completion_probe_tokens_per_call": 32768,
    "maximum_calls": 6,
    "maximum_tokens": 196608,
}
ENTRYPOINTS = (
    "scripts/initializer_choice_aces.py",
    "scripts/submit_initializer_choice_aces.py",
    "scripts/submit_review_continuation.py",
    "scripts/hpc/run_initializer_choice_aces.sh",
    "scripts/hpc/start_initializer_choice_aces.sh",
    "scripts/harmony_vocab.py",
    "scripts/saved_initialization_aces.py",
)
SCOPE = {
    "execution_platform": "aces-h100x1",
    "automatic_followup": False,
    "test_data_opened": False,
    "optimizer_calls": 0,
    "model_promotion": False,
    "topology_edits_allowed": False,
    "historical_budget_refunded": False,
}


def source_identity() -> str:
    """Bind package modules and all diagnostic entrypoints without runner imports."""
    try:
        entrypoints = {
            name: hashlib.sha256((saved.REPO / name).read_bytes()).hexdigest()
            for name in ENTRYPOINTS
        }
    except FileNotFoundError as exc:
        raise ValueError("initializer-choice entrypoint is unavailable") from exc
    return content_hash(
        {"saved_repair": saved.source_identity(), "entrypoints": entrypoints}
    )


def _sealed(value: dict) -> dict:
    """Check an embedded historical seal using the original canonical JSON rule."""
    if value.get("artifact_sha256") != content_hash(
        {key: item for key, item in value.items() if key != "artifact_sha256"}
    ):
        raise ValueError("embedded historical artifact digest differs")
    return value


def _parent(value: dict) -> dict:
    """Require the bounded, public-only V6 source; allow its archived code identity."""
    parent = _sealed(value)
    required = {
        "protocol": live.PROTOCOL_V6,
        "actionable_policy": api.POLICY_V6,
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
        any(
            type(parent.get(key)) is not type(item) or parent[key] != item
            for key, item in required.items()
        )
        or parent.get("test_data_opened") is not False
        or parent.get("automatic_followup") is not False
        or type(parent.get("optimizer_calls")) is not int
        or parent["optimizer_calls"] != 0
        or "provider_recovery_policy" in parent
        or "inherited_provider_outcomes" in parent
        or not 1 <= len(parent["tasks"]) <= 2
    ):
        raise ValueError("requires a sealed public-only bounded V6 source")
    settings = StagedModelSettings.model_validate(parent["settings"])
    if (
        settings.maximum_requests != saved.MAX_CALLS
        or settings.maximum_total_tokens != saved.MAX_TOKENS
        or settings.max_output_tokens != 8192
    ):
        raise ValueError("historical V6 serving settings or allowances differ")
    names = [audit._name(task["task_id"]) for task in parent["tasks"]]
    if len(names) != len(set(names)) or any(
        task["arm"] != "actionable" for task in parent["tasks"]
    ):
        raise ValueError("historical task roster is not distinct and actionable")
    return parent


def _blocked(*args, **kwargs):
    """A freezer or verifier may never issue a provider or tokenizer request."""
    raise ValueError("historical replay attempted uncached provider/tokenizer access")


def _replay(parent: dict, task: dict, result: dict, events: list, calls: list) -> None:
    """Replay original V6 inputs in a temporary cache with both transports blocked."""
    _sealed(result)
    count = result["event_count"]
    if (
        type(count) is not int
        or not 1 <= count <= saved.MAX_CALLS
        or len(events) != count
        or len(calls) != count
        or [event["index"] for event in events] != list(range(count))
        or len({record["request_hash"] for record in calls}) != count
    ):
        raise ValueError("historical event/call coverage differs")
    for event in events:
        _sealed(event)
    for record in calls:
        bookkeeping._call(record, parent["settings"])
        if record["request_hash"] != content_hash(record["request"]):
            raise ValueError("historical request hash differs")
    if result["cost"] != _cost(calls):
        raise ValueError("historical charges differ from all recorded requests")
    with TemporaryDirectory(prefix="initializer-choice-replay-") as temporary:
        root = Path(temporary)
        atomic_json(root / "plan.json", parent)
        directory = root / "results" / audit._name(task["task_id"])
        atomic_json(directory / "result.json", result)
        for event in events:
            atomic_json(directory / "events" / f"{event['index']:03d}.json", event)
        for record in calls:
            atomic_json(directory / "calls" / f"{record['request_hash']}.json", record)
        replayed = saved.run_episode(
            root,
            deepcopy(parent),
            task,
            base_url="http://blocked.invalid",
            transport=_blocked,
            token_transport=_blocked,
        )
        if replayed != result:
            raise ValueError("historical terminal result differs on exact V6 replay")


def _case(parent: dict, task: dict, result: dict, events: list, calls: list) -> dict:
    """Preserve an unchanged final model and the exact request that saw that model."""
    _replay(parent, task, result, events, calls)
    original = parent["starts"][audit._name(task["source_task"])]
    final = events[-1]
    if (
        result["task"] != task
        or result.get("test_data_opened") is not False
        or result.get("optimizer_calls") != 0
        or result.get("scientific_accuracy_established") is not False
        or result["draft"] != original["draft"]
        or result["initializer_handoff"] != original["initializer_handoff"]
        or final["before"] != result["draft"]
        or final["initialization_before"] != result["initializer_handoff"]
        or not result["initialization_status"]["complete"]
    ):
        raise ValueError("diagnostic requires unchanged latest initialized V6 finals")
    by_hash = {record["request_hash"]: record for record in calls}
    record = by_hash[final["request_hash"]]
    messages = record["request"]["body"]["messages"]
    if len(messages) != 2 or messages[1]["role"] != "user":
        raise ValueError("historical request has no unique exact user packet")
    packet = json.loads(messages[1]["content"])
    editable = packet["initialization"]["editable_states"]
    if len(editable) != 1 or packet["prompt_policy"] != api.POLICY_V6:
        raise ValueError("diagnostic requires exactly one editable initializer state")
    last, repetition = equality._last_feedback(final, record)
    start = {
        "task": original["task"],
        "draft": result["draft"],
        "assessment": original["assessment"],
        "diagnostics": original["diagnostics"],
        "previous_conflicts": evidence._conflicts(original, result),
        "initializer_handoff": result["initializer_handoff"],
        "initial_information_contract": result["initial_information_contract"],
        "scientific_state": result["scientific_state"],
        "last_edit": last,
        "repetition_feedback": repetition,
        "historical_cost": result["cost"],
        "predecessor_historical_cost": {
            "historical_cost": original["historical_cost"],
            "predecessor_historical_cost": original["predecessor_historical_cost"],
        },
    }
    return {
        "task": task,
        "start": start,
        "cell": parent["cells"][task["benchmark_id"]],
        "source_result": result,
        "source_events": events,
        "source_calls": calls,
        "latest_request_record": record,
        "packet": packet,
        "editable_state": editable[0],
        "source_input_hashes": {
            "result": result["artifact_sha256"],
            "events": [event["artifact_sha256"] for event in events],
            "records": [content_hash(item) for item in calls],
        },
    }


def _probe(cases: dict) -> dict:
    """Select the sole recorded truncated terminal request for two exact repeats."""
    failed = [
        (name, case)
        for name, case in cases.items()
        if case["source_events"][-1].get("scientific_reply_received") is False
    ]
    if len(failed) != 1:
        raise ValueError("requires exactly one saved unavailable completion request")
    name, case = failed[0]
    record = case["latest_request_record"]
    if record["raw_response"]["choices"][0]["finish_reason"] != "length":
        raise ValueError("completion probe requires a recorded output-length failure")
    return {
        "case_id": name,
        "event_index": case["source_events"][-1]["index"],
        "request_hash": record["request_hash"],
        "record_sha256": content_hash(record),
        "request": record["request"],
        "step": record["step"],
        "attempt": record["attempt"],
        "repeats": 2,
        "interpretation": "Two independently charged, preplanned repeats of one "
        "exact original body, schema, seed and settings. No scientific edit is "
        "applied. Outcomes cannot isolate model behavior from decoder behavior.",
    }


def _assemble(parent: dict, cases: dict) -> dict:
    """Declare independent budgets without refunding any historical work."""
    return {
        "protocol": PROTOCOL,
        "source_identity": source_identity(),
        "source_plan_sha256": parent["artifact_sha256"],
        "parent_plan": parent,
        "cases": cases,
        "completion_probe": _probe(cases),
        "budgets": dict(BUDGETS),
        **SCOPE,
        "selection": "At most two latest unchanged V6 finals with one editable "
        "initializer each. Independent diagnostic calls; no controlled causal "
        "comparison or scientific model promotion is claimed.",
        "historical_cost_scope": "Each case start retains all V6 charges, including "
        "unavailable truncated responses. Nested predecessor scopes and complete "
        "historical source artifacts remain separate. None counts against or is "
        "refunded from this explicitly new six-call, 196608-token maximum.",
    }


def freeze(source: Path, destination: Path) -> dict:
    """Freeze all one or two V6 cases, leaving original caches and plans untouched."""
    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("write diagnostic inputs outside the historical source")
    try:
        parent = _parent(sealed_read(source / "plan.json"))
        cases = {}
        for task in parent["tasks"]:
            name = audit._name(task["task_id"])
            directory = source / "results" / name
            paths = sorted((directory / "events").glob("*.json"))
            events = [sealed_read(path) for path in paths]
            if [path.name for path in paths] != [
                f"{i:03d}.json" for i in range(len(paths))
            ]:
                raise ValueError("historical event filenames differ")
            calls = _cache_records(directory / "calls", saved.namespace(parent, task))
            cases[name] = _case(
                parent, task, sealed_read(directory / "result.json"), events, calls
            )
        return sealed_write(destination / "plan.json", _assemble(parent, cases))
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("malformed initializer-choice source") from exc


def verify(root: Path) -> dict:
    """Recheck policy, embedded history and offline replay before any new call."""
    try:
        plan = sealed_read(root / "plan.json")
        if (
            plan.get("protocol") != PROTOCOL
            or plan.get("source_identity") != source_identity()
            or plan.get("budgets") != BUDGETS
            or any(type(value) is not int for value in plan["budgets"].values())
            or any(
                type(plan.get(key)) is not type(value) or plan[key] != value
                for key, value in SCOPE.items()
            )
        ):
            raise ValueError(
                "initializer-choice source identity, policy or budget differs"
            )
        parent = _parent(plan["parent_plan"])
        if set(plan["cases"]) != {task["task_id"] for task in parent["tasks"]}:
            raise ValueError("initializer-choice case coverage differs")
        cases = {}
        for task in parent["tasks"]:
            case = plan["cases"][task["task_id"]]
            cases[task["task_id"]] = _case(
                parent,
                task,
                case["source_result"],
                case["source_events"],
                case["source_calls"],
            )
        expected = _assemble(parent, cases)
        if {
            key: value for key, value in plan.items() if key != "artifact_sha256"
        } != expected:
            raise ValueError(
                "initializer-choice frozen case or request binding differs"
            )
        return plan
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("malformed initializer-choice plan") from exc
