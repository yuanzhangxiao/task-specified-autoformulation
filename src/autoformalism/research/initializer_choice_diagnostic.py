"""Bounded initializer decisions and independent saved-request completion probes.

These diagnostic candidates never overwrite the source models or launch fitting.
Every physical outcome is cached, charged, and replayed against its predecessor.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from autoformalism.llm.construction import MEASURED_RESERVATION, ConstructionClient
from autoformalism.llm.response_revision import PromptPreflightError
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    atomic_json,
    strict_provider_schema,
    visible_response,
)
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.research import initialized_topology_repair as initialized
from autoformalism.research import initializer_choice_inputs as inputs
from autoformalism.research import initializer_choice_prompt as prompt
from autoformalism.research import initializer_choice_readable as readable
from autoformalism.research import saved_topology_repair as saved
from autoformalism.schemas.base import StrictSchema
from autoformalism.search import construction_initialized_repair as api
from autoformalism.search.topology_initialization import InitializerDependencyHandoff
from autoformalism.staged_topology import content_hash


def units(plan: dict) -> list[tuple[str, str | None]]:
    """Return the frozen order; probe repeats are independent planned units."""
    return [
        (f"completion-{i}", None)
        for i in range(plan["budgets"]["completion_probe_calls"])
    ] + [(f"choice-{name}", name) for name in plan["cases"]]


def namespace(plan: dict, unit: str) -> str:
    """Separate diagnostic calls from their historical sources and other units."""
    return content_hash([plan["artifact_sha256"], unit])


def request_body(settings, seed, step, attempt, system, user, schema):
    """Match the existing client before generation, including its exact seed."""
    return {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "reasoning_effort": settings.reasoning_effort,
        "temperature": settings.temperature,
        "max_tokens": settings.max_output_tokens,
        "seed": int(content_hash([seed, step, attempt])[:8], 16) % (2**31),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": strict_provider_schema(schema.model_json_schema()),
            },
        },
    }


def choice_settings(plan: dict) -> StagedModelSettings:
    """Keep historical serving settings under the separate diagnostic call cap."""
    return StagedModelSettings.model_validate(
        {
            **plan["parent_plan"]["settings"],
            "maximum_requests": plan["budgets"]["maximum_choice_calls_per_case"],
            "maximum_total_tokens": plan["budgets"]["maximum_choice_tokens_per_case"],
        }
    )


def _client(root, plan, unit, settings, seed, base_url, **kwargs):
    directory = root / "results" / unit / "calls"
    identity = namespace(plan, unit)
    _cache_records(directory, identity)
    client = ConstructionClient(
        settings=settings,
        directory=directory,
        namespace=identity,
        seed=seed,
        base_url=base_url,
        reservation_policy=MEASURED_RESERVATION,
        **kwargs,
    )
    client.budget_diagnostics_enabled = True
    return client


def _request(client, system, packet, schema, step, index, *, exact_body=None):
    user = packet if isinstance(packet, str) else json.dumps(packet, ensure_ascii=False)
    expected = request_body(
        client.settings, client.seed, step, index, system, user, schema
    )
    if exact_body is not None and expected != exact_body:
        raise ValueError("completion probe differs from the original HTTP body")
    expected_identity = {
        "protocol": "scientific-staged-topology-1",
        "namespace": client.namespace,
        "settings": client.settings.model_dump(mode="json"),
        "body": expected,
        **client.request_policy(),
    }
    cached = _cache_records(client.directory, client.namespace)
    linked = {
        sealed_read(path)["request_hash"]
        for path in (client.directory.parent / "events").glob("*.json")
    }
    cached_hashes = {record["request_hash"] for record in cached}
    if content_hash(expected_identity) not in cached_hashes and cached_hashes - linked:
        raise ValueError("orphan cached request differs from reconstructed next call")
    record = client.call(
        system=system, user=user, response_model=schema, step=step, attempt=index
    )
    if record["request"] != expected_identity:
        raise ValueError("cached diagnostic request differs from its predecessor")
    return record


def _available(record):
    try:
        return visible_response(record), None
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return None, str(exc)[:2000]


def _finish(root, plan, unit, events, result):
    directory = root / "results" / unit
    records = _cache_records(directory / "calls", namespace(plan, unit))
    if {r["request_hash"] for r in records} != {
        event["request_hash"] for event in events
    }:
        raise ValueError("unaccounted diagnostic call; inspect before continuing")
    if sorted(p.name for p in (directory / "events").glob("*.json")) != [
        f"{i:03d}.json" for i in range(len(events))
    ]:
        raise ValueError("diagnostic event coverage differs")
    return sealed_write(
        directory / "result.json",
        {
            "identity": namespace(plan, unit),
            "unit": unit,
            "event_count": len(events),
            "cost": _cost(records),
            "source_model_changes": 0,
            "optimizer_calls": 0,
            "test_data_opened": False,
            "automatic_followup": False,
            **result,
        },
    )


def _frozen_schema(body):
    """Supply a frozen data schema to the client; never execute response text."""
    contract = body["response_format"]["json_schema"]

    def ordered(value):
        # JSON artifacts sort object keys, but the provider's required arrays
        # retain their original order. Restore that order before the existing
        # strict-schema converter regenerates those arrays from properties.
        if isinstance(value, list):
            return [ordered(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: ordered(item) for key, item in value.items()}
        if "properties" in result:
            required = result.get("required", [])
            if len(required) != len(result["properties"]) or set(required) != set(
                result["properties"]
            ):
                raise ValueError("saved completion schema is not fully strict")
            result["properties"] = {key: result["properties"][key] for key in required}
        return result

    schema = ordered(contract["schema"])

    class FrozenReply(StrictSchema):
        @classmethod
        def model_json_schema(cls, *args, **kwargs):
            return deepcopy(schema)

    FrozenReply.__name__ = contract["name"]
    return FrozenReply


def run_probe(root, plan, unit, *, base_url, **kwargs):
    """Repeat one original request for output availability only, without edits."""
    _, _, terminal = _checked_unit(root, plan, unit)
    if terminal and terminal["status"] == "preflight_blocked":
        return terminal
    probe = plan["completion_probe"]
    original = probe["request"]
    body = original["body"]
    settings = StagedModelSettings.model_validate(
        {**original["settings"], "maximum_requests": 1, "maximum_total_tokens": 32768}
    )
    case = plan["cases"][probe["case_id"]]
    source_call = case["latest_request_record"]
    client = _client(
        root, plan, unit, settings, case["task"]["seed"], base_url, **kwargs
    )
    events = []
    try:
        record = _request(
            client,
            body["messages"][0]["content"],
            body["messages"][1]["content"],
            _frozen_schema(body),
            source_call["step"],
            source_call["attempt"],
            exact_body=body,
        )
    except (RepairBudgetExceeded, PromptPreflightError) as exc:
        return _finish(
            root,
            plan,
            unit,
            events,
            {
                "kind": "completion_probe",
                "status": "preflight_blocked",
                "error": str(exc),
            },
        )
    raw, error = _available(record)
    schema_error = None
    if error is None:
        try:
            api.response_model("repair").model_validate(raw)
        except ValueError as exc:
            schema_error = str(exc)[:2000]
    status = "provider_unavailable" if error else "assessed"
    event = sealed_write(
        root / "results" / unit / "events/000.json",
        {
            "identity": client.namespace,
            "index": 0,
            "request_hash": record["request_hash"],
            "record_sha256": content_hash(record),
            "original_http_body_equal": True,
            "status": status,
            "response_error": error,
            "schema_error": schema_error,
            "scientific_assessment": "not_performed",
            "source_model_changes": 0,
        },
    )
    return _finish(
        root,
        plan,
        unit,
        [event],
        {
            "kind": "completion_probe",
            "status": status,
            "complete_json": error is None,
            "schema_passed": error is None and schema_error is None,
            "scientific_status": "not_assessed",
            "original_request_hash": probe["request_hash"],
        },
    )


def _session(case):
    start = case["start"]
    problem = saved.Problem(case["cell"], start["task"])
    draft = api.Draft.model_validate(start["draft"])
    session = initialized.Session(problem, start, policy=api.POLICY_V6)
    check = problem.assess(draft, qualified_memory=True)
    tracked = case["source_result"]["scientific_state"]
    report = session.diagnose(draft, tracked)
    state = session.state(draft, report, tracked, check["eligible"], {})
    return problem, draft, session, check, state


def choice_packet(case, problem, draft, session, state, previous, index, charged):
    full = saved.make_packet(
        api,
        problem,
        draft,
        None,
        state,
        None,
        index,
        charged,
        stage="repair",
        policy=api.POLICY_V6,
        initialization=session,
    )
    return prompt.build_packet(full, case["editable_state"], previous=previous)


def choice_message(plan, packet, evidence):
    """Keep V1 requests identical and render V2 evidence as readable text."""
    if plan["protocol"] == inputs.PROTOCOL_V2:
        return readable.render(packet, evidence)
    return json.dumps(packet, ensure_ascii=False)


def assess_choice(case, problem, draft, session, check, state, raw):
    """Validate one editable initializer and recheck every retained witness."""
    reply = prompt.ChoiceReply.model_validate(raw)
    if reply.decision != "propose_initializer":
        return reply, state, False
    records = session.handoff.model_dump(mode="json")["records"]
    records = [r for r in records if r["state"] != case["editable_state"]]
    records.append(
        {"state": case["editable_state"], "rule": reply.rule.model_dump(mode="json")}
    )
    candidate = InitializerDependencyHandoff.model_validate({"records": records})
    api.handoff_status(draft, problem.evidence, session.contract, candidate)
    before = session.handoff
    try:
        session.handoff = candidate
        report = session.diagnose(draft, state)
        next_state = session.state(draft, report, state, check["eligible"], {})
    except Exception:
        session.handoff = before
        raise
    return reply, next_state, True


def run_choice(root, plan, unit, name, *, base_url, **kwargs):
    """Make at most two initializer-only attempts; retain topology unchanged."""
    _, _, terminal = _checked_unit(root, plan, unit)
    if terminal and terminal["status"] == "preflight_blocked":
        return terminal
    case = plan["cases"][name]
    problem, draft, session, check, state = _session(case)
    settings = choice_settings(plan)
    client = _client(
        root, plan, unit, settings, case["task"]["seed"], base_url, **kwargs
    )
    previous, events = None, []
    stop, status, decision = "choice_budget_exhausted", "assessed", None
    for index in range(plan["budgets"]["maximum_choice_calls_per_case"]):
        charged = sum(event["budget_charge"] for event in events)
        packet = choice_packet(
            case, problem, draft, session, state, previous, index, charged
        )
        try:
            record = _request(
                client,
                prompt.SYSTEM,
                choice_message(plan, packet, problem.evidence),
                prompt.ChoiceReply,
                f"initializer_choice_{index:03d}",
                index,
            )
        except (RepairBudgetExceeded, PromptPreflightError) as exc:
            status, stop = "preflight_blocked", str(exc)
            break
        raw, availability_error = _available(record)
        before = session.handoff.model_dump(mode="json")
        accepted, error, decision = False, None, None
        if availability_error:
            status, stop = "provider_unavailable", "provider_unavailable"
        else:
            try:
                reply, state, accepted = assess_choice(
                    case, problem, draft, session, check, state, raw
                )
                decision = reply.decision
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)[:2000]
            previous = {
                "reply": raw,
                "error": error,
                "status": "invalid_choice" if error else state["repair_status"],
            }
        event = sealed_write(
            root / "results" / unit / "events" / f"{index:03d}.json",
            {
                "identity": client.namespace,
                "index": index,
                "request_hash": record["request_hash"],
                "record_sha256": content_hash(record),
                "budget_charge": record["budget_charge"],
                "accepted_initializer": accepted,
                "decision": decision,
                "error": error,
                "response_error": availability_error,
                "initializer_before": before,
                "initializer_after": session.handoff.model_dump(mode="json"),
                "scientific_state": state,
                "frozen_draft_sha256": content_hash(case["start"]["draft"]),
                "source_model_changes": 0,
            },
        )
        events.append(event)
        if availability_error:
            break
        if decision in {"topology_change_needed", "insufficient_information"}:
            stop = decision
            break
        if accepted and state["repair_status"] != "repair_required":
            stop = "initializer_assessed"
            break
    return _finish(
        root,
        plan,
        unit,
        events,
        {
            "kind": "initializer_choice",
            "case_id": name,
            "status": status,
            "stop_reason": stop,
            "scientific_status": state["repair_status"],
            "last_decision": decision,
            "candidate_initializer_handoff": session.handoff.model_dump(mode="json"),
            "candidate_scientific_state": state,
            "frozen_draft_sha256": content_hash(case["start"]["draft"]),
            "historical_cost": case["source_result"]["cost"],
            "scientific_accuracy_established": False,
            "interpretation": "A validated diagnostic initializer candidate is not "
            "a promoted model. Deferred evidence is not a pass; no fitting occurred.",
        },
    )


def run(root: Path, plan: dict, *, base_url: str, can_start=lambda: True, **kwargs):
    """Run independent planned units; availability failures never trigger retries."""
    try:
        for unit, name in units(plan):
            try:
                if name is None:
                    run_probe(
                        root,
                        plan,
                        unit,
                        base_url=base_url,
                        can_start=can_start,
                        **kwargs,
                    )
                else:
                    run_choice(
                        root,
                        plan,
                        unit,
                        name,
                        base_url=base_url,
                        can_start=can_start,
                        **kwargs,
                    )
            except DeferredCall:
                break
            report(root, plan)
    finally:
        summary = report(root, plan)
    return summary


def _checked_unit(root: Path, plan: dict, unit: str):
    """Fail before generation if saved events or results have lost their calls."""
    directory = root / "results" / unit
    identity = namespace(plan, unit)
    calls = _cache_records(directory / "calls", identity)
    by_hash = {call["request_hash"]: call for call in calls}
    event_paths = sorted((directory / "events").glob("*.json"))
    if [path.name for path in event_paths] != [
        f"{index:03d}.json" for index in range(len(event_paths))
    ]:
        raise ValueError("diagnostic event filenames differ")
    events = [sealed_read(path) for path in event_paths]
    for index, event in enumerate(events):
        if (
            event["identity"] != identity
            or event["index"] != index
            or event["request_hash"] not in by_hash
            or event["record_sha256"] != content_hash(by_hash[event["request_hash"]])
        ):
            raise ValueError("diagnostic event/call provenance differs")
    path = directory / "result.json"
    result = sealed_read(path) if path.exists() else None
    if result and (
        result["identity"] != identity
        or result["unit"] != unit
        or result["cost"] != _cost(calls)
        or result["event_count"] != len(events)
        or {event["request_hash"] for event in events} != set(by_hash)
    ):
        raise ValueError("diagnostic result provenance differs")
    return calls, events, result


def report(root: Path, plan: dict) -> dict:
    """Read saved outcomes and partial costs without making any provider call."""
    rows, records = [], []
    for unit, name in units(plan):
        calls, _, result = _checked_unit(root, plan, unit)
        records.extend(calls)
        rows.append(
            {
                "unit": unit,
                "kind": "completion_probe" if name is None else "initializer_choice",
                "status": result["status"] if result else "pending",
                "scientific_status": result.get("scientific_status")
                if result
                else None,
                "stop_reason": result.get("stop_reason") if result else None,
                "schema_passed": result.get("schema_passed") if result else None,
                "cost": _cost(calls),
            }
        )
    counts = dict(Counter(row["status"] for row in rows))
    available = "available"
    if counts.get("provider_unavailable") or counts.get("preflight_blocked"):
        available = "unavailable"
    elif counts.get("pending"):
        available = "pending"
    cost = _cost(records)
    if cost["physical_requests"] > plan["budgets"]["maximum_calls"]:
        raise ValueError("diagnostic request cap exceeded")
    summary = {
        "protocol": plan["protocol"],
        "plan_sha256": plan["artifact_sha256"],
        "status": "pending" if counts.get("pending") else "complete",
        "availability_status": available,
        "status_counts": counts,
        "planned_units": len(rows),
        "rows": rows,
        "cost": cost,
        "physical_requests": cost["physical_requests"],
        "observed_tokens": cost["observed_tokens"],
        "unmeasured_requests": cost["requests_with_unknown_usage"],
        "maximum_calls": plan["budgets"]["maximum_calls"],
        "maximum_charged_tokens": plan["budgets"]["maximum_tokens"],
        "cost_scope": "New diagnostic units only; source costs remain "
        "separately frozen in plan.cases.",
        "source_model_changes": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "interpretation": "Output completion and conditional scientific assessment "
        "are separate. These diagnostic candidates are not promoted models.",
    }
    atomic_json(root / "summary.json", summary)
    lines = [
        "# Initializer choice and output completion diagnostics",
        "",
        summary["interpretation"],
        "",
        "| Unit | Status | Scientific status | Calls | Tokens |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['unit']} | {row['status']} | {row['scientific_status']} | "
            f"{row['cost']['physical_requests']} | {row['cost']['observed_tokens']} |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (root / "REPORT.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Initializer diagnostic</title>'
        "<style>body{font:16px/1.5 system-ui;margin:40px}"
        "pre{white-space:pre-wrap}</style>"
        "<h1>Initializer choice diagnostic</h1>"
        f"<pre>{html.escape(json.dumps(summary, indent=2))}</pre>"
    )
    return summary


def preview(plan: dict, output: Path) -> dict:
    """Render exact initial choice requests and the unchanged completion probe."""
    output.mkdir(parents=True, exist_ok=True)
    links, sizes = [], []
    for name, case in plan["cases"].items():
        problem, draft, session, _, state = _session(case)
        packet = choice_packet(case, problem, draft, session, state, None, 0, 0)
        message = choice_message(plan, packet, problem.evidence)
        body = request_body(
            choice_settings(plan),
            case["task"]["seed"],
            "initializer_choice_000",
            0,
            prompt.SYSTEM,
            message,
            prompt.ChoiceReply,
        )
        label = f"choice-{len(links)}"
        (output / f"{label}.json").write_text(
            json.dumps(body, indent=2, ensure_ascii=False) + "\n"
        )
        (output / f"{label}.html").write_text(
            '<!doctype html><meta charset="utf-8"><style>'
            "body{font:16px/1.5 system-ui;max-width:1100px;margin:40px auto}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
            f"<h1>{html.escape(name)}</h1><h2>System</h2>"
            f"<pre>{html.escape(prompt.SYSTEM)}</pre>"
            "<h2>Focused request</h2>"
            "<pre>"
            + html.escape(
                message
                if plan["protocol"] == inputs.PROTOCOL_V2
                else json.dumps(packet, indent=2, ensure_ascii=False)
            )
            + "</pre>"
            "<details><summary>Output schema</summary>"
            f"<pre>{html.escape(json.dumps(body['response_format'], indent=2))}</pre>"
            "</details>"
        )
        sizes.append(
            {
                "case": name,
                "focused_user_bytes": len(body["messages"][1]["content"].encode()),
                "source_user_bytes": len(
                    case["latest_request_record"]["request"]["body"]["messages"][1][
                        "content"
                    ].encode()
                ),
            }
        )
        links.append(
            f'<li><a href="{label}.html">{html.escape(name)}</a> · '
            f'<a href="{label}.json">Exact request body</a></li>'
        )
    if plan["completion_probe"]:
        (output / "completion-probe.json").write_text(
            json.dumps(plan["completion_probe"]["request"]["body"], indent=2) + "\n"
        )
    availability = (
        "Separate availability check: two planned repeats of the exact previously "
        "incomplete request. These are not repairs and do not change models."
        if plan["completion_probe"]
        else "Changed: readable equations and paired initial readings; fixed runtime "
        "rules are distinguished from the editable latent initializer. The complete "
        "V1 diagnostic is retained with its costs. Its successful completion probes "
        "are not repeated. This retest allows at most four new calls."
    )
    (output / "PROMPTS.html").write_text(
        '<!doctype html><meta charset="utf-8"><style>'
        "body{font:17px/1.6 system-ui;max-width:1000px;margin:45px auto}"
        "pre{white-space:pre-wrap}</style>"
        "<h1>Focused initializer-choice diagnostic</h1>"
        "<p>Changed: initializer decisions only; equations are frozen. "
        "Only shared_parameter or common_map can be proposed. Topology change or "
        "insufficient information can be reported, but cannot mark the conflict "
        "resolved. Each case allows two attempts; prior evidence and costs "
        "remain unchanged.</p><p>"
        + availability
        + "</p><ul>"
        + "".join(links)
        + "</ul>"
        + (
            '<p><a href="completion-probe.json">Unchanged completion request '
            "(both repeats)</a></p>"
            if plan["completion_probe"]
            else ""
        )
        + f"<pre>{html.escape(json.dumps(sizes, indent=2))}</pre>"
        "<p>Byte sizes are not token counts. Serving-token preflight remains "
        "mandatory.</p>"
    )
    return {
        "cases": len(sizes),
        "sizes": sizes,
        "llm_calls": 0,
        "test_data_opened": False,
    }
