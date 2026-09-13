"""Frozen-context admissibility replay of repair-comparison replies, not search."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ModelValidationError
from autoformalism.fitting.initialization import LatentInitializationPlan
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.repair_drafts import (
    KINDS,
    RepairActionV2,
    advance,
    diagnostic,
    empty_draft,
)
from autoformalism.rebuttal.repair_transactions import default_initialization
from autoformalism.rebuttal.revision_decision import (
    expressions,
    parameter_aliases,
    translate_names,
)
from autoformalism.rebuttal.staged_prefit_fitting_campaign import load_public_data
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash


def adapt_legacy(
    raw: dict, parent, context, *, inventory_assumption: bool
) -> tuple[dict, dict]:
    """Never guess away a conflicting keep without reporting the explicit assumption."""
    reply = copy.deepcopy(raw)
    keep = reply.pop("keep", [])
    if not isinstance(keep, list) or any(not isinstance(n, str) for n in keep):
        raise ValueError("legacy keep is not a list of identifiers")
    changed = {e["component"] for e in reply.get("equations", [])}
    removed = set(reply.get("remove", []))
    if removed & set(keep):
        raise ValueError("legacy keep/removal conflict is not normalized")
    known = set(expressions(parent)) | set(context.forcing_channels) | changed
    if set(keep) - known:
        raise ValueError("legacy keep contains unknown symbols")
    overlap = sorted(changed & set(keep))
    if overlap and not inventory_assumption:
        raise ValueError(
            "AMBIGUOUS_LEGACY_KEEP: retained variable versus unchanged equation"
        )
    return reply, {
        "removed_keep_entries": keep,
        "overlap": overlap,
        "legacy_keep_as_inventory_assumption": bool(overlap),
    }


def captured_draft(request: dict) -> dict:
    """Preserve historical provisional actions, including invalid related edits."""
    draft = empty_draft()
    draft["edits"]["equation"] = copy.deepcopy(request.get("provisional_edits", {}))
    for kind, (field, key) in KINDS.items():
        if kind != "equation":
            for entry in request.get("provisional_related_edits", {}).get(field, []):
                draft["edits"][kind][entry[key] if key else entry] = entry
    # Old requests did not persist per-equation scope. Require unchanged signed
    # interactions for old provisional equations unless model scope was recorded.
    previous = request.get("retry_findings", [])
    previous_reply = (previous[-1].get("response") or {}) if previous else {}
    scope = request.get("draft_scope", previous_reply.get("scope", "function"))
    if scope == "no_change":
        scope = "function"
    draft["scope"] = scope
    draft["equation_scopes"] = dict.fromkeys(draft["edits"]["equation"], scope)
    return draft


def replay(source: Path, output: Path, arm="redesigned_runtime") -> dict:
    """Check both conservative and explicitly assumed adapters on every stored reply."""
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source):
        raise ValueError("replay output must be outside the frozen source")
    hashes = {}

    def source_bytes(relative):
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError("source path escapes frozen root")
        data = path.read_bytes()
        hashes[str(path.relative_to(source))] = hashlib.sha256(data).hexdigest()
        return data

    def read(relative):
        return json.loads(source_bytes(relative))

    def verified(relative):
        plan = read(relative)
        if (
            content_hash({k: v for k, v in plan.items() if k != "plan_sha256"})
            != plan["plan_sha256"]
        ):
            raise ValueError("source plan digest differs")
        if (
            plan.get("test_data_opened") is not False
            or plan.get("private_reference_opened") is not False
        ):
            raise ValueError("source must be public-only")
        return plan

    plan = verified("plan.json")
    if plan["schema_version"] != "repair-feedback-comparison-1":
        raise ValueError("replay requires the historical comparison-1 source")
    inputs = verified("inputs/plan.json")
    if inputs["plan_sha256"] != plan["input_plan_sha256"]:
        raise ValueError("input plan identity differs")
    public_root = source / "inputs/frozen/public"
    ledger = inputs["public_asset_ledger"]
    if content_hash(ledger) != inputs["public_asset_ledger_sha256"]:
        raise ValueError("public asset ledger digest differs")
    for relative, expected in ledger.items():
        path = (source / "inputs" / relative).resolve()
        if not path.is_relative_to(public_root) or any(
            re.search(r"(^|[_\-.])(test|private|hidden)([_\-.]|$)", part.lower())
            for part in Path(relative).parts
        ):
            raise ValueError("ledger contains a non-development public asset")
        key = str(path.relative_to(source))
        source_bytes(key)
        if hashes[key] != expected:
            raise ValueError("public asset digest differs")
    if any(
        str(path.relative_to(source / "inputs")) not in ledger
        for path in public_root.rglob("*")
        if path.is_file()
    ):
        raise ValueError("untracked public asset in frozen source")
    contexts = {}
    rows = []
    selected = [t for t in plan["tasks"] if t["arm"] == arm]
    if not selected:
        raise ValueError("selected arm has no tasks")
    for task in selected:
        candidate_path = Path("inputs") / task["candidate_path"]
        if (
            not (source / candidate_path)
            .resolve()
            .is_relative_to(source / "inputs/frozen/candidates")
        ):
            raise ValueError("candidate path outside frozen public candidates")
        parent = CandidateModel.model_validate(read(candidate_path))
        if hashes[str(candidate_path)] != task["candidate_file_sha256"]:
            raise ValueError("frozen candidate digest differs")
        context_key = (task["benchmark_id"], task["tier"])
        if context_key not in contexts:
            _, contexts[context_key] = load_public_data(public_root, *context_key)
        context = contexts[context_key]
        initial = default_initialization(parent, context)
        directory = source / "results" / task["task_id"]
        if not directory.resolve().is_relative_to(source / "results"):
            raise ValueError("invalid task path")
        for transaction_path in sorted(directory.glob("round_*/transaction.json")):
            transaction = read(transaction_path.relative_to(source))
            for attempt in transaction["attempts"]:
                call = read(
                    (
                        directory / "calls" / (attempt["request_hash"] + ".json")
                    ).relative_to(source)
                )
                if content_hash(call["request"]) != attempt["request_hash"]:
                    raise ValueError("cached request digest differs")
                request = json.loads(call["request"]["body"]["messages"][1]["content"])
                request = translate_names(
                    request, {v: k for k, v in parameter_aliases(parent).items()}
                )
                if request["equations"] != expressions(parent):
                    raise ValueError("historical parent equations differ from request")
                initial = LatentInitializationPlan.model_validate(
                    request["initialization_plan"]
                )
                row = {
                    "task_id": task["task_id"],
                    "round": transaction_path.parent.name,
                    "attempt": attempt["attempt"],
                    "request_hash": attempt["request_hash"],
                    "historical_diagnostics": attempt["diagnostics"],
                    "views": {},
                }
                for assume in (False, True):
                    mode = "keep_inventory_assumption" if assume else "conservative"
                    try:
                        if not isinstance(attempt["response"], dict):
                            raise ValueError(
                                "historical response has no visible JSON object"
                            )
                        adapted, audit = adapt_legacy(
                            translate_names(
                                attempt["response"],
                                {v: k for k, v in parameter_aliases(parent).items()},
                            ),
                            parent,
                            context,
                            inventory_assumption=assume,
                        )
                        _, _, draft, findings = advance(
                            parent,
                            initial,
                            context,
                            captured_draft(request),
                            RepairActionV2.model_validate(adapted),
                            tuple(plan["config"]["nonlinear_targets"]),
                            tuple(plan["config"]["memory_targets"]),
                        )
                        row["views"][mode] = {
                            "status": draft["status"],
                            "diagnostics": findings,
                            "normalization": audit,
                            "accepted_equations": sorted(draft["accepted"]),
                        }
                    except (
                        ValueError,
                        ModelValidationError,
                        TypeError,
                        KeyError,
                    ) as exc:
                        row["views"][mode] = {
                            "status": "rejected",
                            "diagnostics": [diagnostic(exc)],
                        }
                rows.append(row)
            # Only the actual historical commit advances the context; never an
            # admissible replay alternative. No counterfactual follow-up exists.
            if transaction["status"] == "committed":
                parent = CandidateModel.model_validate(transaction["candidate"])
                initial = LatentInitializationPlan.model_validate(
                    transaction["initialization"]
                )
    result = {
        "schema_version": "repair-action-replay-2",
        "status": "complete" if rows else "empty",
        "source_plan_sha256": plan["plan_sha256"],
        "arm": arm,
        "stored_reply_count": len(rows),
        "source_file_sha256": hashes,
        "source_bundle_sha256": content_hash(hashes),
        "views": {
            mode: dict(Counter(r["views"][mode]["status"] for r in rows))
            for mode in ("conservative", "keep_inventory_assumption")
        },
        "historical_diagnostic_counts": dict(
            Counter(d["code"] for r in rows for d in r["historical_diagnostics"])
        ),
        "rows": rows,
        "new_llm_calls": 0,
        "parameter_fitting_performed": False,
        "public_development_context_reconstructed": True,
        "test_data_opened": False,
        "private_reference_opened": False,
        "counterfactual_search_claimed": False,
    }
    if output.exists() and json.loads(output.read_text()) != result:
        raise ValueError("existing replay differs; use a new output file")
    atomic_json(output, result)
    return result
