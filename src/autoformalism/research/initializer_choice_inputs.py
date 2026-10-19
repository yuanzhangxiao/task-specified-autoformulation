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
PROTOCOL_V2 = "saved-initializer-choice-diagnostic-2"
BUDGETS_V2 = {
    **BUDGETS,
    "completion_probe_calls": 0,
    "maximum_calls": 4,
    "maximum_tokens": 131072,
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


def _verify_original(plan: dict, *, archived: bool = False) -> dict:
    """Verify V1 content; archived plans retain their old source identity."""
    try:
        _sealed(plan)
        if (
            plan.get("protocol") != PROTOCOL
            or (not archived and plan.get("source_identity") != source_identity())
            or not isinstance(plan.get("source_identity"), str)
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
        if archived:
            expected["source_identity"] = plan["source_identity"]
        if {
            key: value for key, value in plan.items() if key != "artifact_sha256"
        } != expected:
            raise ValueError(
                "initializer-choice frozen case or request binding differs"
            )
        return plan
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("malformed initializer-choice plan") from exc


def _completed(snapshot: dict) -> dict:
    """Replay complete V1 history without refunding costs or using a provider."""
    from autoformalism.research import initializer_choice_diagnostic as diagnostic

    plan = _verify_original(snapshot["plan"], archived=True)
    expected_units = {unit for unit, _ in diagnostic.units(plan)}
    if set(snapshot["units"]) != expected_units:
        raise ValueError("completed diagnostic unit coverage differs")
    with TemporaryDirectory(prefix="initializer-readable-replay-") as temporary:
        root = Path(temporary)
        for unit, name in diagnostic.units(plan):
            saved_unit = snapshot["units"][unit]
            directory = root / "results" / unit
            if [event["index"] for event in saved_unit["events"]] != list(
                range(len(saved_unit["events"]))
            ) or any(type(event["index"]) is not int for event in saved_unit["events"]):
                raise ValueError("completed diagnostic event indices differ")
            atomic_json(directory / "result.json", _sealed(saved_unit["result"]))
            for event in saved_unit["events"]:
                atomic_json(
                    directory / "events" / f"{event['index']:03d}.json", _sealed(event)
                )
            for record in saved_unit["calls"]:
                if record["request_hash"] != content_hash(record["request"]):
                    raise ValueError("completed diagnostic request hash differs")
                atomic_json(
                    directory / "calls" / f"{record['request_hash']}.json", record
                )
            for record in saved_unit["tokenization"]:
                if (
                    type(record.get("count")) is not int
                    or record["count"] <= 0
                    or type(record.get("max_model_len")) is not int
                    or record["max_model_len"] != 32768
                    or "error" in record
                ):
                    raise ValueError("completed diagnostic tokenizer metadata differs")
                token_key = content_hash(record["request"])
                atomic_json(
                    directory / "calls" / "tokenization" / f"{token_key}.json", record
                )
            kwargs = {
                "base_url": "http://blocked.invalid",
                "transport": _blocked,
                "token_transport": _blocked,
            }
            if name is None:
                replayed = diagnostic.run_probe(root, plan, unit, **kwargs)
                if not replayed.get("schema_passed"):
                    raise ValueError(
                        "completed diagnostic requires successful availability probes"
                    )
            else:
                replayed = diagnostic.run_choice(root, plan, unit, name, **kwargs)
                if (
                    replayed["candidate_initializer_handoff"]
                    != plan["cases"][name]["start"]["initializer_handoff"]
                ):
                    raise ValueError(
                        "readable retest requires unchanged initializer choices"
                    )
            if replayed != saved_unit["result"] or replayed["status"] != "assessed":
                raise ValueError("completed diagnostic differs on exact cache replay")
            actual_events = [
                sealed_read(p) for p in sorted((directory / "events").glob("*.json"))
            ]
            if actual_events != saved_unit["events"]:
                raise ValueError("completed diagnostic events differ on replay")
        summary = diagnostic.report(root, plan)
        if summary != snapshot["summary"]:
            raise ValueError("completed diagnostic summary differs on replay")
    return snapshot


def _assemble_readable(snapshot: dict) -> dict:
    """Keep all previous work separate from this new presentation-only allowance."""
    original = snapshot["plan"]
    plan = _assemble(original["parent_plan"], original["cases"])
    plan.update(
        protocol=PROTOCOL_V2,
        budgets=dict(BUDGETS_V2),
        completion_probe=None,
        prior_diagnostic=snapshot,
        prompt_policy="readable-initializer-choice-1",
        selection="The same unchanged saved models and seeds from the complete V1 "
        "diagnostic. Only request presentation changes; no controlled causal "
        "comparison is claimed.",
        historical_cost_scope="All V6/predecessor costs remain in cases. The complete "
        "V1 diagnostic and its costs are retained in prior_diagnostic. No work is "
        "refunded. This new allowance is at most four calls and 131072 charged tokens; "
        "successful completion probes are not repeated.",
    )
    return plan


def refine_completed(source: Path, destination: Path) -> dict:
    """Freeze the two unchanged choices after verifying the completed V1 diagnostic."""
    from autoformalism.research import initializer_choice_diagnostic as diagnostic

    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("write readable inputs outside the historical source")
    try:
        plan = sealed_read(source / "plan.json")
        snapshot = {
            "plan": plan,
            "summary": json.loads((source / "summary.json").read_text()),
            "units": {},
        }
        for unit, _ in diagnostic.units(plan):
            directory = source / "results" / unit
            calls, events, result = diagnostic._checked_unit(source, plan, unit)
            if result is None:
                raise ValueError("source diagnostic is incomplete")
            snapshot["units"][unit] = {
                "calls": calls,
                "events": events,
                "result": result,
                "tokenization": [
                    json.loads(path.read_text())
                    for path in sorted(
                        (directory / "calls" / "tokenization").glob("*.json")
                    )
                ],
            }
        return sealed_write(
            destination / "plan.json", _assemble_readable(_completed(snapshot))
        )
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("malformed completed diagnostic") from exc


def verify(root: Path) -> dict:
    """Bind the executing source and replay historical records before new calls."""
    plan = sealed_read(root / "plan.json")
    if plan.get("protocol") == PROTOCOL:
        return _verify_original(plan)
    try:
        if (
            plan.get("protocol") != PROTOCOL_V2
            or plan.get("source_identity") != source_identity()
        ):
            raise ValueError("initializer-choice source identity or protocol differs")
        expected = _assemble_readable(_completed(plan["prior_diagnostic"]))
        if (
            any(type(value) is not int for value in plan["budgets"].values())
            or {key: value for key, value in plan.items() if key != "artifact_sha256"}
            != expected
        ):
            raise ValueError(
                "readable initializer-choice policy, history or budget differs"
            )
        return plan
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("malformed readable initializer-choice plan") from exc
