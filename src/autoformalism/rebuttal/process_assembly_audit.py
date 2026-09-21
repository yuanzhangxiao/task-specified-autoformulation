"""Counterfactual v5 assembly checks; never refit, promote or rewrite a saved model."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.function_dependency_audit import source_files
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.process_handoff_audit import _read, request_payload
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas.construction import FunctionalDraft
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search import function_dependencies as dep
from autoformalism.search import process_assembly_contract as assembly
from autoformalism.search import shared_process_contract as shared
from autoformalism.search.staged_function_runner import _obligation, _selected_term
from autoformalism.staged_functions import (
    bind_function_reply,
    repair_certified_outer_gain_role,
)
from autoformalism.staged_topology import content_hash, lower_topology
from autoformalism.staging import topology_commitment_sha256

PROTOCOL = "process-assembly-audit-1"


def assess(brief, context, source, identifier, raw, accepted):
    """Evaluate a recorded decision only; never invent intrinsic-factor approval."""
    row = {
        "interaction_id": identifier,
        "reply": raw,
        "historically_accepted": accepted,
        "whole_model_recovered": False,
        "counterfactual_only": True,
    }
    try:
        i, j = dep.slot(source, identifier)
        equations = tuple(
            EquationDefinition.model_validate(e) for e in source["equations"]
        )
        inventory = tuple(
            ScientificVariable.model_validate(v) for v in source["inventory"]
        )
        bindings = shared.validate_contract(
            source.get("shared_process_contract"), equations, inventory
        )
        selected = assembly.decorate(
            shared.decorate_function_term(
                _selected_term(
                    equations[i],
                    equations[i].terms[j],
                    parameter_identity_policy="interaction_local",
                ),
                bindings,
            ),
            bindings,
        )
        reply = assembly.AssemblyFunctionReply.model_validate(raw)
        normalized, decision = assembly.prepare(reply, selected)
        row["interpretation"] = decision
        revised, event = dep.prepare(
            brief,
            context,
            source,
            identifier,
            dep.DependencyFunctionReply(
                **normalized.model_dump(mode="json"),
                revise_dependencies=reply.revise_dependencies,
            ),
            preserve_process_paths=True,
        )
        row["dependency_change"] = event
        topology, aliases = lower_topology(
            brief,
            inventory,
            tuple(EquationDefinition.model_validate(e) for e in revised["equations"]),
            context,
        )
        selected["sources"] = revised["equations"][i]["terms"][j]["sources"]
        shared.validate_function(
            normalized.expression, selected.get("shared_process_use")
        )
        prepared, _ = repair_certified_outer_gain_role(
            normalized,
            set(selected["sources"]),
            outer_weight_sign=selected["outer_weight_sign"],
        )
        bind_function_reply(
            topology,
            FunctionalDraft(
                topology_commitment_sha256=topology_commitment_sha256(topology)
            ),
            identifier,
            prepared,
            context,
            aliases,
            _obligation(selected),
        )
        row["classification"] = (
            "outer_sign_normalized"
            if decision["sign_normalizations"]
            else "unchanged_valid"
        )
    except (ValueError, KeyError, TypeError, ModelValidationError) as exc:
        error = str(exc)
        row["error"] = error[:6000]
        row["classification"] = (
            "conversion_review_required"
            if error.startswith("CONSUMER_CONVERSION_OVERLAP")
            else "target_path_repair_required"
            if error.startswith("PROCESS_TARGET_PATH_LOST")
            else "still_blocked"
        )
    return row


def _contexts(brief, context, source, functions):
    """Recover the exact source state preceding each saved batch/atomic attempt."""
    dep.replay(brief, context, source, functions)
    changes = {
        e["interaction_id"]: e for e in functions.get("dependency_revisions", [])
    }
    current, contexts = source, {}
    for batch in functions["batch_term_audits"]:
        identifier = batch["interaction_id"]
        contexts[identifier] = current
        if identifier in changes:
            e = changes[identifier]
            current, actual = dep.prepare(
                brief,
                context,
                current,
                identifier,
                dep.DependencyFunctionReply(
                    **e["reply"], revise_dependencies=e["revise_dependencies"]
                ),
            )
            if e != actual:
                raise ValueError("historical dependency event differs")
    if current != functions.get("effective_source", source):
        raise ValueError("historical attempt coverage differs")
    return contexts


def audit(source: Path, output: Path) -> dict:
    """Seal input hashes and checkpoint the audit without accessing trajectory files."""
    source, output = source.resolve(), output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("audit output must be separate from historical source")
    _read(source / "plan.json", source)
    plan = sealed_read(source / "plan.json")
    if (
        plan["protocol"] != "detention-process-pilot-3"
        or plan.get("function_dependency_policy") != dep.POLICY
        or plan.get("test_data_opened")
        or plan.get("private_reference_opened")
    ):
        raise ValueError("requires the public v5 dependency-policy pilot")
    paths = source_files(source, plan)
    hashes = {}
    for p in paths:
        _read(p, source)
        hashes[str(p.relative_to(source))] = hashlib.sha256(p.read_bytes()).hexdigest()
    identity = {
        "protocol": PROTOCOL,
        "assembly_policy": assembly.POLICY,
        "source": str(source),
        "source_plan_sha256": plan["artifact_sha256"],
        "runtime_sha256": runtime_source_hash(),
        "files": hashes,
        "llm_calls": 0,
        "optimizer_calls": 0,
        "trajectory_values_used": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    with public._lock(output):
        sealed_write(output / "freeze.json", identity)
        tasks = {t["construction_task"]["task_id"]: t for t in plan["tasks"]}
        rows = []
        for name, task in sorted(tasks.items()):
            checkpoint = output / "checkpoints" / f"{name}.json"
            if checkpoint.exists():
                rows.append(sealed_read(checkpoint))
                continue
            cell = plan["cells"][task["case"]]
            brief = PublicScientificBrief.model_validate(cell["brief"])
            context = ValidationContext.model_validate(cell["context"])
            base = source / "construction/results" / name
            routes = []
            for route in ("review", "fallback"):
                path = base / route / "function_stage.json"
                if not path.exists():
                    continue
                functions = sealed_read(path)["result"]
                source_path = base / route / "topology_stage.json"
                if not source_path.exists():
                    raise ValueError("function stage has no saved topology")
                original = sealed_read(source_path)["result"]
                contexts = _contexts(brief, context, original, functions)
                attempts = []
                for b in functions["batch_term_audits"]:
                    identifier = b["interaction_id"]
                    attempts.append(
                        {
                            "stage": "batch",
                            **assess(
                                brief,
                                context,
                                contexts[identifier],
                                identifier,
                                b["batch_function"],
                                b["batch_accepted"],
                            ),
                        }
                    )
                for event in functions["events"]:
                    if not event["step"].startswith("atomic_repair_"):
                        continue
                    path = base / "calls" / (event["request_hash"] + ".json")
                    if not path.exists():
                        attempts.append(
                            {
                                "stage": "repair",
                                "classification": "unavailable",
                                **event,
                            }
                        )
                        continue
                    record = _read(path, source)
                    if record.get("request_hash") != event["request_hash"]:
                        raise ValueError("saved request hash differs")
                    payload = request_payload(record)
                    identifier = event["step"].removeprefix("atomic_repair_")
                    prior = contexts[identifier]
                    i, j = dep.slot(prior, identifier)
                    if (
                        payload["frozen_equation_sketch"] != prior["equations"]
                        or set(payload["selected_term"]["sources"])
                        != set(prior["equations"][i]["terms"][j]["sources"])
                        or payload["frozen_inventory"] != prior["inventory"]
                    ):
                        raise ValueError("saved atomic request context differs")
                    try:
                        raw = visible_response(record)
                    except (ValueError, TypeError, KeyError):
                        raw = None
                    attempts.append(
                        {
                            "stage": "repair",
                            "attempt": event["attempt"],
                            "request_hash": event["request_hash"],
                            **assess(
                                brief,
                                context,
                                prior,
                                identifier,
                                raw,
                                event["accepted"],
                            ),
                        }
                    )
                routes.append(
                    {
                        "route": route,
                        "historical_status": functions["status"],
                        "attempts": attempts,
                        "historical_connectivity": dep.connectivity(
                            functions.get("candidate")
                        ),
                    }
                )
            rows.append(sealed_write(checkpoint, {"task": name, "routes": routes}))
        if source_files(source, plan) != paths or any(
            hashlib.sha256(p.read_bytes()).hexdigest()
            != hashes[str(p.relative_to(source))]
            for p in paths
        ):
            raise ValueError("historical files changed during audit")
        attempts = [
            {"task": r["task"], "route": rt["route"], **a}
            for r in rows
            for rt in r["routes"]
            for a in rt["attempts"]
        ]
        counts = dict(Counter(a["classification"] for a in attempts))
        summary = {
            "protocol": PROTOCOL,
            "identity": content_hash(identity),
            "constructions": rows,
            "classification_counts": counts,
            "historically_accepted_requiring_repair": [
                a
                for a in attempts
                if a.get("historically_accepted")
                and a["classification"]
                in ("conversion_review_required", "target_path_repair_required")
            ],
            "unexpected_acceptance_regressions": [
                a
                for a in attempts
                if a.get("historically_accepted")
                and a["classification"] == "still_blocked"
            ],
            "unavailable_saved_attempts": counts.get("unavailable", 0),
            "missing_constructions": [r["task"] for r in rows if not r["routes"]],
            "whole_models_recovered": 0,
            "llm_calls": 0,
            "optimizer_calls": 0,
            "test_data_opened": False,
            "automatic_followup": False,
        }
        sealed_write(output / "summary.json", summary)
        lines = [
            "# Saved process assembly audit",
            "",
            "Historical responses under a prospective convention; no model promotion, "
            "LLM calls, fitting or trajectory reads. Sign normalization is a declared "
            "interpretation change, not algebraic equivalence. Conversion questions "
            "require fresh proposer decisions. No intrinsic confirmation is invented.",
            "",
            f"Classification counts: {counts}",
            "Accepted replies requiring repair: "
            f"{len(summary['historically_accepted_requiring_repair'])}",
            "Unexpected acceptance regressions: "
            f"{len(summary['unexpected_acceptance_regressions'])}",
            f"Unavailable replies: {summary['unavailable_saved_attempts']}",
            f"Missing constructions: {summary['missing_constructions']}",
            "",
            "| Construction | Route | Historical | Reply classifications |",
            "| --- | --- | --- | --- |",
        ]
        for row in rows:
            for rt in row["routes"]:
                counts = dict(Counter(a["classification"] for a in rt["attempts"]))
                lines.append(
                    f"| {row['task']} | {rt['route']} | "
                    f"{rt['historical_status']} | {counts} |"
                )
        (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
        return summary
