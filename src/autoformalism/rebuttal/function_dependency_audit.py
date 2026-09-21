"""Replay saved function replies without guessing a proposer's dependency decision."""

from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting import public_fitting as public
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.process_handoff_audit import _read, request_payload
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas.construction import FunctionalDraft
from autoformalism.schemas.staged_functions import InteractionFunctionReply
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search import function_dependencies as dep
from autoformalism.search import shared_process_contract as shared
from autoformalism.search.staged_function_runner import _obligation, _selected_term
from autoformalism.staged_functions import (
    bind_function_reply,
    repair_certified_outer_gain_role,
)
from autoformalism.staged_topology import content_hash, lower_topology
from autoformalism.staging import topology_commitment_sha256

PROTOCOL = "function-dependency-audit-1"


def check(brief, context, source, identifier, raw, *, mode):
    """Check one reply in its original topology, never promote a complete model."""
    reply = InteractionFunctionReply.model_validate(raw)
    if mode == "explicit":
        reply = dep.DependencyFunctionReply(**raw, revise_dependencies=True)
    effective, change = (
        (source, None)
        if mode == "strict"
        else dep.prepare(brief, context, source, identifier, reply)
    )
    inventory = tuple(
        ScientificVariable.model_validate(v) for v in effective["inventory"]
    )
    equations = tuple(
        EquationDefinition.model_validate(e) for e in effective["equations"]
    )
    if mode != "strict" and any(
        not c["passed"] for c in dep.required_checks(brief, equations)
    ):
        raise ValueError("effective dependency source fails public pathways")
    bindings = shared.validate_contract(
        effective.get("shared_process_contract"), equations, inventory
    )
    topology, aliases = lower_topology(brief, inventory, equations, context)
    i, j = dep.slot(effective, identifier)
    selected = shared.decorate_function_term(
        _selected_term(
            equations[i],
            equations[i].terms[j],
            parameter_identity_policy="interaction_local",
        ),
        bindings,
    )
    shared.validate_function(reply.expression, selected.get("shared_process_use"))
    prepared, _ = repair_certified_outer_gain_role(
        dep.plain(reply),
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
    return change


def assess(brief, context, source, identifier, raw, historically_accepted):
    """Separate mechanically valid replies from unmade scientific decisions."""
    row = {
        "interaction_id": identifier,
        "reply": raw,
        "historically_accepted": historically_accepted,
        "whole_model_recovered": False,
    }
    for mode in ("strict", "automatic", "explicit"):
        try:
            row[mode + "_change"] = check(
                brief, context, source, identifier, raw, mode=mode
            )
            row[mode + "_valid"] = True
        except (ValueError, KeyError, TypeError, ModelValidationError) as exc:
            row[mode + "_valid"] = False
            row[mode + "_error"] = str(exc)[:6000]
    row["classification"] = (
        "already_valid"
        if row["strict_valid"]
        else "public_covariate_addition"
        if row["automatic_valid"]
        else "requires_proposer_revision"
        if row["explicit_valid"]
        else "blocked"
    )
    row["newly_mechanically_valid"] = (
        not historically_accepted and row["automatic_valid"]
    )
    row["counterfactual_only"] = row["classification"] == "requires_proposer_revision"
    try:
        tree = RestrictedParser().parse(raw["expression"], location="audit").tree
        row["self_division_factors"] = [
            ast.unparse(n)
            for n in ast.walk(tree)
            if isinstance(n, ast.BinOp)
            and isinstance(n.op, ast.Div)
            and ast.dump(n.left) == ast.dump(n.right)
        ]
        i, _ = dep.slot(source, identifier)
        lhs = source["equations"][i]["name"]
        symbols = RestrictedParser().parse(raw["expression"], location="audit").symbols
        row["consumer_conversion_overlap"] = [
            {
                "target": u["target"],
                "conversion": u["conversion"],
                "symbols": sorted(
                    set(symbols)
                    & set(
                        RestrictedParser()
                        .parse(u["conversion"], location="conversion")
                        .symbols
                    )
                ),
            }
            for b in (source.get("shared_process_contract") or {}).get("bindings", [])
            if b["proposal"]["name"] == lhs
            for u in b.get("signed_declaration", {}).get("uses", [])
            if u["conversion"] is not None
            and set(symbols)
            & set(
                RestrictedParser().parse(u["conversion"], location="conversion").symbols
            )
        ]
    except (ValueError, KeyError, TypeError, ModelValidationError):
        row["syntax_diagnostics_unavailable"] = True
    return row


def source_files(source, plan):
    """Only metadata, saved stages and referenced function requests are read."""
    files = {source / "plan.json"}
    names = sorted({t["construction_task"]["task_id"] for t in plan["tasks"]})
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError("invalid construction name")
        base = source / "construction/results" / name
        for route in ("review", "fallback"):
            for kind in ("topology", "function"):
                path = base / route / f"{kind}_stage.json"
                if path.exists():
                    files.add(path)
            path = base / route / "function_stage.json"
            if not path.exists():
                continue
            for event in _read(path, source)["result"]["events"]:
                if not event["step"].startswith("atomic_repair_"):
                    continue
                key = event.get("request_hash", "")
                if not re.fullmatch(r"[0-9a-f]{64}", key):
                    raise ValueError("invalid request hash")
                path = base / "calls" / f"{key}.json"
                if path.exists():
                    files.add(path)
    return sorted(files)


def audit(source: Path, output: Path):
    """Seal a CPU-only diagnostic with deterministic resume and no fresh requests."""
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("audit output must be separate from source")
    _read(source / "plan.json", source)
    plan = sealed_read(source / "plan.json")
    if (
        plan["protocol"] != "detention-process-pilot-3"
        or plan.get("test_data_opened")
        or plan.get("private_reference_opened")
    ):
        raise ValueError("requires a public gain-policy pilot")
    paths = source_files(source, plan)
    hashes = {}
    for path in paths:
        _read(path, source)
        hashes[str(path.relative_to(source))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    identity = {
        "protocol": PROTOCOL,
        "dependency_policy": dep.POLICY,
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
                if functions.get("dependency_policy", "strict") != "strict":
                    raise ValueError("audit expects historical strict-source responses")
                original = sealed_read(base / route / "topology_stage.json")["result"]
                attempts = []
                for batch in functions["batch_term_audits"]:
                    attempts.append(
                        {
                            "stage": "batch",
                            **assess(
                                brief,
                                context,
                                original,
                                batch["interaction_id"],
                                batch["batch_function"],
                                batch["batch_accepted"],
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
                        raise ValueError("saved function event request hash differs")
                    payload = request_payload(record)
                    identifier = event["step"].removeprefix("atomic_repair_")
                    i, j = dep.slot(original, identifier)
                    if (
                        set(payload["selected_term"]["sources"])
                        != set(original["equations"][i]["terms"][j]["sources"])
                        or payload["frozen_equation_sketch"] != original["equations"]
                        or payload["frozen_inventory"] != original["inventory"]
                    ):
                        raise ValueError("saved function request sources differ")
                    try:
                        raw = visible_response(record)
                    except (ValueError, TypeError, KeyError):
                        raw = None  # Retain malformed delivery as a blocked attempt.
                    attempts.append(
                        {
                            "stage": "repair",
                            "attempt": event["attempt"],
                            "request_hash": event["request_hash"],
                            **assess(
                                brief,
                                context,
                                original,
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
                        "connectivity": dep.connectivity(functions.get("candidate")),
                    }
                )
            rows.append(sealed_write(checkpoint, {"task": name, "routes": routes}))
        if source_files(source, plan) != paths or any(
            hashlib.sha256(p.read_bytes()).hexdigest()
            != hashes[str(p.relative_to(source))]
            for p in paths
        ):
            raise ValueError("source changed during audit")
        attempts = [
            {"task": r["task"], "route": rt["route"], **a}
            for r in rows
            for rt in r["routes"]
            for a in rt["attempts"]
        ]
        summary = {
            "protocol": PROTOCOL,
            "identity": content_hash(identity),
            "constructions": rows,
            "classification_counts": dict(
                Counter(a["classification"] for a in attempts)
            ),
            "newly_mechanically_valid": sum(
                a.get("newly_mechanically_valid", False) for a in attempts
            ),
            "previously_accepted_now_blocked": [
                a
                for a in attempts
                if a.get("historically_accepted") and not a["automatic_valid"]
            ],
            "unavailable_saved_attempts": sum(
                a["classification"] == "unavailable" for a in attempts
            ),
            "whole_models_recovered": 0,
            "llm_calls": 0,
            "optimizer_calls": 0,
            "test_data_opened": False,
            "automatic_followup": False,
        }
        sealed_write(output / "summary.json", summary)
        lines = [
            "# Saved function dependency audit",
            "",
            "No LLM calls, fitting or model promotion. Explicit dependency revisions "
            "are counterfactuals requiring a fresh proposer decision.",
            "",
            f"Classification counts: {summary['classification_counts']}",
            f"Newly mechanically valid attempts: {summary['newly_mechanically_valid']}",
            "Previously accepted now blocked: "
            f"{len(summary['previously_accepted_now_blocked'])}",
            f"Unavailable attempts: {summary['unavailable_saved_attempts']}",
            "",
            "| Construction | Route | Historical | Classification counts |",
            "| --- | --- | --- | --- |",
        ]
        for row in rows:
            for route in row["routes"]:
                counts = dict(Counter(a["classification"] for a in route["attempts"]))
                lines.append(
                    f"| {row['task']} | {route['route']} | "
                    f"{route['historical_status']} | {counts} |"
                )
        lines += [
            "",
            "Counts are replies/slots, not recovered models. Conversion overlap, "
            "self-division and disconnected processes are advisory syntax findings, "
            "not scientific verdicts.",
        ]
        (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
        return summary
