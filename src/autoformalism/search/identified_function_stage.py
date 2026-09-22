"""Opt-in slot delivery, retaining the existing function/initialization pipeline."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from autoformalism.expressions import ModelValidationError
from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.search import function_delivery as delivery
from autoformalism.search import process_assembly_revision as revision
from autoformalism.search import process_revision_runner as repair
from autoformalism.search import signed_processes as signed

ERRORS = (ValueError, TypeError, KeyError, ModelValidationError)


def slots(source):
    return [
        f"term_{i}_{j}"
        for i, e in enumerate(source["equations"])
        for j in range(len(e["terms"]))
    ]


def batch(brief, context, source, known, requested, raw):
    """Independently interpret a keyed batch; preserve every admissible named slot."""
    mapped, errors = delivery.unpack(raw, requested)
    outcomes = []
    for name in requested:
        if name not in mapped:
            continue
        value = {"interaction_id": name, "reply": mapped[name]}
        try:
            tx, labels = delivery.interpret(
                brief, context, source, known, name, mapped[name]
            )
            value.update(transaction=tx, delivery_normalizations=labels, error=None)
            source, known = tx["effective_source"], tx["functions"]
        except ERRORS as exc:
            value.update(
                transaction=None, delivery_normalizations=[], error=str(exc)[:6000]
            )
        outcomes.append(value)
    return (
        source,
        known,
        {
            "requested": requested,
            "raw": raw,
            "delivery_errors": errors,
            "outcomes": outcomes,
        },
    )


def run(brief, context, source, client, output: Path, *, public_brief: dict):
    """Reuse batch calls and per-slot repair caps; initialize/finalize in the caller."""
    current, known, ledger, events = deepcopy(source), {}, [], []
    failures, audits = {}, []

    def save():
        atomic_json(
            output / "progress.json",
            {
                "policy": delivery.POLICY,
                "effective_source": current,
                "functions": known,
                "ledger": ledger,
                "events": events,
            },
        )

    error = None
    try:
        for name in slots(source):
            selected = revision.selected_slot(current, name)
            automatic = signed.automatic_function(selected)
            if automatic is not None:
                tx, _ = delivery.interpret(
                    brief,
                    context,
                    current,
                    known,
                    name,
                    automatic.model_dump(mode="json"),
                )
                ledger.append(
                    {"kind": "automatic", "interaction_id": name, "transaction": tx}
                )
                current, known = tx["effective_source"], tx["functions"]
        save()
        for i in range(len(source["equations"])):
            equation_slots = [
                n for n in slots(current) if revision.dep.slot(current, n)[0] == i
            ]
            pending = [n for n in equation_slots if n not in known]
            delivered = set()
            for attempt in range(client.settings.attempts_per_step):
                requested = [
                    n for n in pending if n not in delivered and n not in known
                ]
                if not requested:
                    break
                payload = {
                    "policy": delivery.POLICY,
                    "public_brief": public_brief,
                    "frozen_inventory": current["inventory"],
                    "equation_context": current["equations"][i],
                    "requested_slots": {
                        n: revision.selected_slot(current, n) for n in requested
                    },
                    "runtime_supplied_slots": {
                        n: known[n]
                        for n in equation_slots
                        if signed.automatic_function(revision.selected_slot(current, n))
                        is not None
                    },
                    "retained_functions": known,
                    "diagnostics": {
                        "errors": ledger[-1]["result"]["delivery_errors"],
                        "instruction": "Return only remaining requested slots. "
                        "Previously delivered slots are retained.",
                    }
                    if attempt and ledger
                    else None,
                }
                record = client.call(
                    system=delivery.system_prompt(),
                    user=json.dumps(payload, sort_keys=True),
                    response_model=delivery.FunctionBatch,
                    step=f"identified_functions_{i}",
                    attempt=attempt,
                )
                try:
                    raw = visible_response(record)
                except ValueError as exc:
                    raw = {"delivery_error": str(exc)[:2000]}
                current, known, result = batch(
                    brief, context, current, known, requested, raw
                )
                ledger.append(
                    {
                        "kind": "batch",
                        "result": result,
                        "request_hash": record["request_hash"],
                    }
                )
                for value in result["outcomes"]:
                    name = value["interaction_id"]
                    delivered.add(name)
                    if value["error"]:
                        failures[name] = value["error"]
                events.append(
                    {
                        "step": f"identified_functions_{i}",
                        "attempt": attempt,
                        "request_hash": record["request_hash"],
                        "accepted": not result["delivery_errors"]
                        and len(result["outcomes"]) == len(requested)
                        and all(o["error"] is None for o in result["outcomes"]),
                        "error": "; ".join(result["delivery_errors"]) or None,
                    }
                )
                save()
            for name in pending:
                if name in known:
                    continue
                initial_error = failures.get(
                    name,
                    "MISSING_FUNCTION_DELIVERY: return this scalar function only",
                )
                result = repair.run(
                    brief,
                    context,
                    current,
                    name,
                    known,
                    client,
                    output / "repairs" / name,
                    initial_error=initial_error,
                    normalize_delivery=True,
                    public_brief=public_brief,
                )
                ledger.append(
                    {
                        "kind": "repair",
                        "interaction_id": name,
                        "initial_error": initial_error,
                        "result": result,
                    }
                )
                events.extend(
                    {"step": f"assembly_revision_{name}_{e['scope']}", **e}
                    for e in result["events"]
                )
                if result["transaction"] is None:
                    save()
                    raise ValueError(f"bounded function repair exhausted for {name}")
                tx = result["transaction"]
                current, known = tx["effective_source"], tx["functions"]
                save()
    except ERRORS as exc:
        error = str(exc)[:6000]
    topology, draft = revision.bind_known(brief, context, current, known)
    result = {
        "policy": delivery.POLICY,
        "ledger": ledger,
        "events": events,
        "effective_source": current,
        "functions": known,
        "topology": topology.model_dump(mode="json"),
        "draft": draft.model_dump(mode="json"),
        "error": error,
        "complete": error is None and set(known) == set(slots(current)),
        "batch_term_audits": audits,
    }
    atomic_json(output / "result.json", result)
    return result


def replay(brief, context, original, result):
    """Verify every deterministic commit, including changed retained companions."""
    if result["policy"] != delivery.POLICY:
        raise ValueError("unknown identified-function policy")
    source, known = deepcopy(original), {}
    for event in result["ledger"]:
        if event["kind"] == "batch":
            saved = event["result"]
            if any(n in known or n not in slots(source) for n in saved["requested"]):
                raise ValueError("batch requests already filled or unknown slots")
            source, known, expected = batch(
                brief, context, source, known, saved["requested"], saved["raw"]
            )
            if expected != saved:
                raise ValueError("identified batch interpretation differs")
        elif event["kind"] == "automatic":
            name = event["interaction_id"]
            automatic = signed.automatic_function(revision.selected_slot(source, name))
            if automatic is None or name in known:
                raise ValueError("invalid automatic function event")
            tx, _ = delivery.interpret(
                brief, context, source, known, name, automatic.model_dump(mode="json")
            )
            if tx != event["transaction"]:
                raise ValueError("runtime supplied function differs")
            source, known = tx["effective_source"], tx["functions"]
        elif event["kind"] == "repair":
            name, repaired = event["interaction_id"], event["result"]
            if name in known or name not in slots(source):
                raise ValueError("repair focus is not an unfilled slot")
            if repaired["transaction"] is not None:
                repair._interpret_delivery(
                    brief, context, source, name, known, repaired
                )
                tx = revision.replay(
                    brief, context, source, known, repaired["transaction"]
                )
                source, known = tx["effective_source"], tx["functions"]
        else:
            raise ValueError("unknown function delivery event")
    topology, draft = revision.bind_known(brief, context, source, known)
    if (
        source != result["effective_source"]
        or known != result["functions"]
        or topology.model_dump(mode="json") != result["topology"]
        or draft.model_dump(mode="json") != result["draft"]
    ):
        raise ValueError("identified function replay differs")
    return source, known


def records(brief, context, source, functions):
    """Use the existing canonical record and outer-gain interpretation for reporting."""
    from autoformalism.search.staged_function_runner import _accepted_function_record
    from autoformalism.staged_topology import lower_topology

    topology, draft = revision.bind_known(brief, context, source, functions)
    _, aliases = lower_topology(
        brief,
        tuple(
            revision.ScientificVariable.model_validate(v) for v in source["inventory"]
        ),
        tuple(
            revision.EquationDefinition.model_validate(e) for e in source["equations"]
        ),
        context,
    )
    accepted, local = [], []
    for name in slots(source):
        if name not in functions:
            continue
        selected = revision.selected_slot(source, name)
        law, _ = revision.repair_certified_outer_gain_role(
            revision.InteractionFunctionReply.model_validate(functions[name]),
            set(selected["sources"]),
            outer_weight_sign=selected["outer_weight_sign"],
        )
        local.append({"selected_term": selected, **law.model_dump(mode="json")})
        accepted.append(_accepted_function_record(draft, name, selected, aliases))
    return topology, aliases, draft, accepted, local
