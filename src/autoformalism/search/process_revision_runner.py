"""Cached repair of a construction slot; routing spends the same retry budget."""

from __future__ import annotations

import json
from pathlib import Path

from autoformalism.expressions import ModelValidationError
from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.search import process_assembly_revision as revision

SYSTEM = """Repair the displayed construction slot using its public scientific brief.
Return only the requested JSON. Define one intrinsic law, before its consumer
signs and conversions. You may omit a supplied fixed covariate belonging only to
the consumer conversion; do not insert dummy factors just to mention it.
For other dependency changes set revise_dependencies=true. Runtime derives names
from your expression. Signs, shared identities and consumer targets stay fixed.
Use consumer_conversions to revise a conversion together with this process law;
each edit names an existing target and a positive fixed-factor expression, or null
for unresolved. No fitted parameter or state belongs in a conversion.
Inspect assembled contributions, not the law in isolation. If an overlapping
factor is genuinely intrinsic, retain_intrinsic_for must list the actual affected
targets. Such a decision remains scientifically unverified; it is not a repair
certificate. Empty retention lists are normal. Explanations cannot override rules.
In local scope leave companion_laws empty. In topology scope you may explicitly
revise the displayed ordinary companion slots and their actual dependencies to
restore required pathways. Each companion supplies its full function/parameters.
Do not invent a dummy path. No new variables, sign changes, process consumer
changes or initial-condition changes are allowed in this bounded repair.
Use only the restricted expression grammar and displayed variables. Numeric
parameter values are assigned by the fitter, not by this repair.
"""


def error_category(error: str) -> str:
    """Separate public hard predicates, process guards, conversion and delivery."""
    if "PROCESS_TARGET_PATH_LOST" in error:
        return "process_target_path"
    if "PUBLIC_PATHWAY_REPAIR_REQUIRED" in error or "public pathways" in error:
        return "public_pathway"
    if "CONVERSION_REPAIR_REQUIRED" in error or "CONSUMER_CONVERSION_OVERLAP" in error:
        return "conversion_overlap"
    if (
        "DEPENDENCY_DECISION_REQUIRED" in error
        or "dependency revision required" in error
    ):
        return "dependency_decision"
    if (
        "equation function count mismatch" in error
        or "validation error" in error.lower()
    ):
        return "response_schema"
    return "other"


def request_context(brief, context, source, identifier, known, diagnostic, scope):
    """Expose only existing slots; return their exact assembled-use contracts."""
    slots = {}
    for i, equation in enumerate(source["equations"]):
        for j in range(len(equation["terms"])):
            name = f"term_{i}_{j}"
            selected = revision.selected_slot(source, name)
            if (
                name != identifier
                and not selected.get("shared_process_use")
                and not selected["assembly_contract"]["conversions"]
            ):
                slots[name] = selected
    return {
        "policy": revision.POLICY,
        "scope": scope,
        "public_brief": brief.model_dump(mode="json"),
        "selected_slot": revision.selected_slot(source, identifier),
        "conversion_owned_fixed_covariates": sorted(
            revision.conversion_sources(revision.selected_slot(source, identifier))
            & revision.dep.fixed_sources(brief, context, source)
        ),
        "available_dependencies": [
            v["name"] for v in source["inventory"] if v["definition"] != "unused"
        ],
        "equations": source["equations"],
        "retained_functions": known,
        "ordinary_companion_slots": slots if scope == "topology" else {},
        "diagnostic": diagnostic,
    }


def run(
    brief,
    context,
    source: dict,
    identifier: str,
    known: dict,
    client,
    output: Path,
    *,
    initial_error: str,
    attempt_offset: int = 0,
) -> dict:
    """Use only remaining attempts; cached/uncertain outcomes still consume a slot.

    The existing client enforces total construction token/request/wall-clock gates.
    This API does not reset them, launch a model search, or mutate the parent.
    """
    maximum = client.settings.attempts_per_step
    if type(attempt_offset) is not int or not 0 <= attempt_offset <= maximum:
        raise ValueError("invalid consumed repair attempt count")
    freeze = {
        "policy": revision.POLICY,
        "runtime_sha256": runtime_source_hash(),
        "brief": brief.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "source": source,
        "known_functions": known,
        "interaction_id": identifier,
        "settings": client.settings.model_dump(mode="json"),
        "namespace": client.namespace,
        "seed": client.seed,
        "initial_error": initial_error,
        "attempt_offset": attempt_offset,
    }
    with _lock(output):
        sealed_write(output / "freeze.json", freeze)
        if (output / "result.json").exists():
            result = sealed_read(output / "result.json")
            if result["transaction"] is not None:
                revision.replay(brief, context, source, known, result["transaction"])
            return result
        diagnostic, scope, events, transaction = initial_error, "local", [], None
        for attempt in range(attempt_offset, maximum):
            if error_category(diagnostic) in {"public_pathway", "process_target_path"}:
                scope = "topology"
            record = client.call(
                system=SYSTEM,
                user=json.dumps(
                    request_context(
                        brief, context, source, identifier, known, diagnostic, scope
                    ),
                    sort_keys=True,
                ),
                response_model=revision.RevisionReply,
                step=f"assembly_revision_{identifier}_{scope}",
                attempt=attempt,
            )
            error = None
            try:
                reply = revision.RevisionReply.model_validate(visible_response(record))
                transaction = revision.prepare(
                    brief,
                    context,
                    source,
                    identifier,
                    reply,
                    known_functions=known,
                    allow_topology_repair=scope == "topology",
                )
            except (ValueError, TypeError, KeyError, ModelValidationError) as exc:
                error = str(exc)[:6000]
                diagnostic = error
            events.append(
                {
                    "attempt": attempt,
                    "scope": scope,
                    "request_hash": record["request_hash"],
                    "accepted": transaction is not None,
                    "error": error,
                    "error_category": error_category(error) if error else None,
                    "observed_total_tokens": record.get("observed_total_tokens"),
                }
            )
            atomic_json(output / "progress.json", {"events": events})
            if transaction is not None:
                break
        return sealed_write(
            output / "result.json",
            {
                "policy": revision.POLICY,
                "status": "repaired" if transaction else "repair_exhausted",
                "transaction": transaction,
                "events": events,
                "remaining_attempts_at_entry": maximum - attempt_offset,
                "attempts_consumed": len(events),
                "optimizer_calls": 0,
                "automatic_followup": False,
                "scientific_validity_certified": False,
            },
        )
