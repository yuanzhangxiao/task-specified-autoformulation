"""Saved-v6 interface audit: no LLM calls, fitting, trajectories or model promotion."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal import process_assembly_audit as previous
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.process_handoff_audit import _read, request_payload
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import function_dependencies as dep
from autoformalism.search import process_assembly_contract as assembly
from autoformalism.search import process_assembly_revision as revision
from autoformalism.search.process_revision_runner import error_category
from autoformalism.staged_topology import content_hash

PROTOCOL = "process-revision-audit-1"


def source_files(source: Path, plan: dict) -> list[Path]:
    """Include all function-stage events, including failed equation-batch delivery."""
    files = set(previous.source_files(source, plan))
    for path in list(files):
        if path.name != "function_stage.json":
            continue
        functions = sealed_read(path)["result"]
        for event in functions["events"]:
            key = event.get("request_hash", "")
            if not re.fullmatch(r"[0-9a-f]{64}", key):
                raise ValueError("invalid function event request hash")
            call = path.parent.parent / "calls" / f"{key}.json"
            if call.exists():
                files.add(call)
    return sorted(files)


def assess(brief, context, source, identifier, raw, accepted) -> dict:
    """Replay expressions exactly; never invent conversion or topology decisions."""
    row = {
        "interaction_id": identifier,
        "historically_accepted": accepted,
        "whole_model_recovered": False,
        "reply": raw,
        "newly_mechanically_valid": False,
    }
    try:
        # The legacy flag is recorded but cannot imply a new targeted decision.
        legacy = assembly.AssemblyFunctionReply.model_validate(raw)
        current = revision.RevisionReply(
            expression=legacy.expression,
            parameters=legacy.parameters,
            revise_dependencies=legacy.revise_dependencies,
        )
        row["historical_intrinsic_flag"] = legacy.conversion_factor_is_intrinsic
        transaction = revision.prepare(brief, context, source, identifier, current)
        row.update(
            classification="conversion_dependencies_separated"
            if any(
                c["conversion_only_omissions"]
                for c in transaction["dependency_changes"]
            )
            else "mechanically_valid",
            dependency_changes=transaction["dependency_changes"],
            sign_normalizations=transaction["sign_normalizations"],
            newly_mechanically_valid=not accepted,
        )
    except (ValueError, TypeError, KeyError, ModelValidationError) as exc:
        error = str(exc)[:6000]
        category = error_category(error)
        row.update(
            error=error,
            error_category=category,
            classification={
                "conversion_overlap": "conversion_decision_required",
                "dependency_decision": "dependency_decision_required",
                "public_pathway": "topology_repair_required",
                "process_target_path": "topology_repair_required",
            }.get(category, "still_blocked"),
        )
    return row


def historical_counters(functions: dict) -> dict:
    """Use separate denominators for committed decisions and rejected attempts."""
    decisions = functions.get("assembly_decisions", [])
    errors = [
        b["batch_error"] for b in functions["batch_term_audits"] if b.get("batch_error")
    ]
    errors += [e.get("error", "") for e in functions["events"] if not e["accepted"]]
    confirmed = [d for d in decisions if d["intrinsic_factor_confirmed"]]
    return {
        "committed_slot_decisions": len(decisions),
        "confirmed_actual_overlaps": sum(
            bool(d["consumer_conversion_overlap"]) for d in confirmed
        ),
        "intrinsic_flags_without_overlap": sum(
            not d["consumer_conversion_overlap"] for d in confirmed
        ),
        "conversion_rejected_attempts": sum(
            error_category(e) == "conversion_overlap" for e in errors
        ),
        "public_pathway_rejected_attempts": sum(
            error_category(e) == "public_pathway" for e in errors
        ),
        "process_target_path_rejected_attempts": sum(
            error_category(e) == "process_target_path" for e in errors
        ),
        "batch_delivery_rejected_attempts": sum(
            not e["accepted"] and e["step"].startswith("equation_functions_")
            for e in functions["events"]
        ),
    }


def _route(source, base, route, brief, context) -> dict:
    """Verify original request identities and contexts before testing new semantics."""
    functions = sealed_read(base / route / "function_stage.json")["result"]
    original = sealed_read(base / route / "topology_stage.json")["result"]
    diagnostics = {}
    contexts = previous._contexts(
        brief, context, original, functions, diagnostics=diagnostics
    )
    attempts = [
        {
            "stage": "batch",
            **assess(
                brief,
                context,
                contexts[b["interaction_id"]],
                b["interaction_id"],
                b["batch_function"],
                b["batch_accepted"],
            ),
        }
        for b in functions["batch_term_audits"]
    ]
    event_rows, unavailable = [], 0
    for event in functions["events"]:
        path = base / "calls" / (event["request_hash"] + ".json")
        if not path.exists():
            unavailable += 1
            event_rows.append({**event, "saved_response": "unavailable"})
            continue
        record = _read(path, source)
        if record.get("request_hash") != event["request_hash"]:
            raise ValueError("saved event/request identity differs")
        payload = request_payload(record)
        try:
            raw = visible_response(record)
        except (ValueError, KeyError, TypeError):
            raw = None
        if not event["accepted"]:
            event_rows.append(
                {
                    **event,
                    "category": error_category(event.get("error", "")),
                    "saved_response": raw,
                }
            )
        if not event["step"].startswith("atomic_repair_"):
            continue
        identifier = event["step"].removeprefix("atomic_repair_")
        prior = contexts[identifier]
        i, j = dep.slot(prior, identifier)
        if (
            payload["frozen_equation_sketch"] != prior["equations"]
            or payload["frozen_inventory"] != prior["inventory"]
            or set(payload["selected_term"]["sources"])
            != set(prior["equations"][i]["terms"][j]["sources"])
        ):
            raise ValueError("saved repair context differs from verified history")
        attempts.append(
            {
                "stage": "repair",
                "attempt": event["attempt"],
                "request_hash": event["request_hash"],
                **assess(brief, context, prior, identifier, raw, event["accepted"]),
            }
        )
    return {
        "route": route,
        "historical_status": functions["status"],
        "historical_error": functions.get("error"),
        "dependency_replay": diagnostics,
        "attempts": attempts,
        "rejected_events": event_rows,
        "historical_counters": historical_counters(functions),
        "unavailable_saved_attempts": unavailable,
    }


def audit(source: Path, output: Path) -> dict:
    """Checkpoint metadata-only analysis, rejecting source drift on resume."""
    source, output = source.resolve(), output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("audit output must be separate from historical source")
    _read(source / "plan.json", source)
    plan = sealed_read(source / "plan.json")
    if (
        plan["protocol"] != "detention-process-pilot-3"
        or plan.get("process_assembly_policy") != assembly.POLICY
        or plan.get("function_dependency_policy") != dep.POLICY
        or plan.get("test_data_opened")
        or plan.get("private_reference_opened")
    ):
        raise ValueError("requires the public v6 assembly-policy pilot")
    paths = source_files(source, plan)
    hashes = {}
    for path in paths:
        _read(path, source)
        hashes[str(path.relative_to(source))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    freeze = {
        "protocol": PROTOCOL,
        "policy": revision.POLICY,
        "source": str(source),
        "source_plan_sha256": plan["artifact_sha256"],
        "files": hashes,
        "runtime_sha256": runtime_source_hash(),
        "llm_calls": 0,
        "optimizer_calls": 0,
        "trajectory_values_used": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    with public._lock(output):
        sealed_write(output / "freeze.json", freeze)
        rows = []
        tasks = {t["construction_task"]["task_id"]: t for t in plan["tasks"]}
        for name, task in sorted(tasks.items()):
            checkpoint = output / "checkpoints" / f"{name}.json"
            if checkpoint.exists():
                rows.append(sealed_read(checkpoint))
                continue
            cell = plan["cells"][task["case"]]
            brief = PublicScientificBrief.model_validate(cell["brief"])
            context = ValidationContext.model_validate(cell["context"])
            base = source / "construction/results" / name
            routes = [
                _route(source, base, route, brief, context)
                for route in ("review", "fallback")
                if (base / route / "function_stage.json").exists()
            ]
            rows.append(sealed_write(checkpoint, {"task": name, "routes": routes}))
        if source_files(source, plan) != paths or any(
            hashlib.sha256(p.read_bytes()).hexdigest()
            != hashes[str(p.relative_to(source))]
            for p in paths
        ):
            raise ValueError("historical files changed during audit")
        attempts = [
            {"task": r["task"], "route": s["route"], **a}
            for r in rows
            for s in r["routes"]
            for a in s["attempts"]
        ]
        counters = Counter()
        for row in rows:
            for stage in row["routes"]:
                counters.update(stage["historical_counters"])
        summary = {
            "protocol": PROTOCOL,
            "identity": content_hash(freeze),
            "constructions": rows,
            "classification_counts": dict(
                Counter(a["classification"] for a in attempts)
            ),
            "historical_counters": dict(counters),
            "newly_mechanically_valid_attempts": sum(
                a["newly_mechanically_valid"] for a in attempts
            ),
            "accepted_requiring_new_conversion_decision": [
                a
                for a in attempts
                if a["historically_accepted"]
                and a["classification"] == "conversion_decision_required"
            ],
            "unexpected_acceptance_regressions": [
                a
                for a in attempts
                if a["historically_accepted"]
                and a["classification"]
                in {
                    "still_blocked",
                    "topology_repair_required",
                    "dependency_decision_required",
                }
            ],
            "unavailable_saved_attempts": sum(
                s["unavailable_saved_attempts"] for r in rows for s in r["routes"]
            ),
            "missing_constructions": [r["task"] for r in rows if not r["routes"]],
            "whole_models_recovered": 0,
            "llm_calls": 0,
            "optimizer_calls": 0,
            "test_data_opened": False,
            "automatic_followup": False,
        }
        saved = sealed_write(output / "summary.json", summary)
        lines = [
            "# Saved process revision audit",
            "",
            "Same saved replies in their original contexts. No fitting, LLM calls, "
            "trajectory reads or model promotion. "
            "A reply becoming valid is not a recovered model.",
            "",
            f"Classifications: {summary['classification_counts']}",
            f"Historical counters: {dict(counters)}",
            f"Newly valid replies: {summary['newly_mechanically_valid_attempts']}",
            "Unexpected acceptance regressions: "
            f"{len(summary['unexpected_acceptance_regressions'])}",
            f"Unavailable saved attempts: {summary['unavailable_saved_attempts']}",
            f"Missing constructions: {summary['missing_constructions']}",
            "",
            "Actual retained overlaps remain scientifically unverified. Legacy Boolean "
            "flags are never converted into new target-specific decisions. "
            "Topology repair and conversion edits require a fresh proposer decision; "
            "this audit invents neither.",
        ]
        (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
        return saved
