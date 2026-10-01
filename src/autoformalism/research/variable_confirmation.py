"""Fresh variable-only calls on the saved public roster; no fitting or selection."""

from __future__ import annotations

import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import StagedModelSettings, atomic_json
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.research import construction_baseline as baseline
from autoformalism.research import construction_contract as contract
from autoformalism.research import construction_trace
from autoformalism.search.staged_topology_runner import run_staged_topology
from autoformalism.search.training_evidence import TrainingEvidence
from autoformalism.search.variable_bindings import POLICY as BINDING_POLICY
from autoformalism.search.variable_inventory_review import POLICY as REVIEW_POLICY

PROTOCOL = "phase-c-variable-confirmation-1"
REPO = baseline.REPO


class Config(baseline.Config):
    """Reuse frozen proposer settings; inherited fit settings are never executed."""

    protocol: Literal["phase-c-variable-confirmation-1"] = PROTOCOL
    served_context_tokens: Literal[32768] = 32768
    review_inventory: bool = False


def source_identity() -> dict:
    value = baseline.source_identity()
    value["variable_launchers"] = {
        name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        for name in (
            "scripts/phase_c_variables.py",
            "scripts/hpc/start_phase_c_variables.sh",
        )
    }
    return value


def freeze(source_plan: Path, root: Path, *, review_inventory: bool = False) -> dict:
    """Project only public construction context from the original sealed plan."""
    source = sealed_read(source_plan)
    if source["protocol"] != baseline.PROTOCOL or source.get("test_data_opened"):
        raise ValueError("requires a development-only Phase C construction plan")
    config = Config.model_validate(
        {**source["config"], "protocol": PROTOCOL, "review_inventory": review_inventory}
    )
    cells = {
        name: contract.correct_roles(
            {k: cell[k] for k in ("brief", "context", "target_contract", "evidence")}
        )
        for name, cell in source["cells"].items()
    }
    with public._lock(root):
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "config": config.model_dump(mode="json"),
                "tasks": source["tasks"],
                "cells": cells,
                "source_identity": source_identity(),
                "source_plan_sha256": source["artifact_sha256"],
                "binding_policy": BINDING_POLICY,
                "inventory_review_policy": REVIEW_POLICY if review_inventory else None,
                "test_data_opened": False,
                "automatic_followup": False,
                "scope": (
                    "variables_only; no topology, functions, fitting or model selection"
                ),
            },
        )


def verify(root: Path) -> dict:
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != PROTOCOL or plan["source_identity"] != source_identity():
        raise ValueError("variable confirmation source/protocol differs")
    config = Config.model_validate(plan["config"])
    if plan["binding_policy"] != BINDING_POLICY:
        raise ValueError("binding policy differs")
    if plan.get("inventory_review_policy") != (
        REVIEW_POLICY if config.review_inventory else None
    ):
        raise ValueError("inventory review policy differs")
    return plan


def propose(root, plan, task, base_url, **kwargs):
    """Reuse inventory construction, call caching, and exact prompt/response traces."""
    directory = baseline.location(root, task)
    identity = baseline.namespace(plan, task)
    with public._lock(directory / "construction-lock"):
        _cache_records(directory / "calls", identity)
        previous = baseline.read_outcome(directory / "proposal.json", plan, task)
        if previous:
            construction_trace.render(directory, identity)
            return previous
        client = ConstructionClient(
            settings=StagedModelSettings.model_validate(
                plan["config"]["model_settings"]
            ),
            directory=directory / "calls",
            namespace=identity,
            seed=task["seed"],
            base_url=base_url,
            **kwargs,
        )
        cell = plan["cells"][task["benchmark_id"]]
        try:
            result = run_staged_topology(
                contract.visible_brief(cell, task),
                ValidationContext.model_validate(cell["context"]),
                client,
                directory / "construction" / "variables",
                hybrid_variable_construction=True,
                stop_after_inventory=True,
                explicit_mechanism_bindings=True,
                review_inventory=plan["config"].get("review_inventory", False),
                target_definitions=contract.target_definitions(cell),
                training_evidence=TrainingEvidence.model_validate(cell["evidence"])
                if task["arm"] == "full"
                else None,
                shared_process_guidance=task.get("shared_processes", True),
            )
            return sealed_write(
                directory / "proposal.json",
                {
                    "identity": identity,
                    **result,
                    "cost": _cost(client.records),
                },
            )
        except RepairBudgetExceeded as exc:
            return sealed_write(
                directory / "proposal.json",
                {
                    "identity": identity,
                    "status": "provider_budget_exhausted",
                    "error": str(exc),
                    "cost": _cost(client.records),
                },
            )
        finally:
            construction_trace.render(directory, identity)


def report(root: Path, plan: dict) -> dict:
    """Count mechanical outcomes separately from scientific adequacy and skips."""
    rows = []
    for task in plan["tasks"]:
        directory = baseline.location(root, task)
        outcome = baseline.read_outcome(directory / "proposal.json", plan, task) or {}
        progress = directory / "construction/variables/progress.json"
        saved = json.loads(progress.read_text()) if progress.exists() else {}
        if directory.exists():
            construction_trace.render(directory, baseline.namespace(plan, task))
        events = [e for e in saved.get("events", []) if e.get("request_hash")]
        agenda_events = [e for e in events if e["step"].startswith("variables_")]
        review_events = [e for e in events if e["step"] == "inventory_review"]
        first = [e for e in agenda_events if e["attempt"] == 0]
        rows.append(
            {
                "task": task["task_id"],
                "benchmark_id": task["benchmark_id"],
                "status": outcome.get("status", "pending"),
                "first_reply_accepted": sum(e["accepted"] for e in first),
                "first_reply_total": len(first),
                "repair_calls": sum(e["attempt"] > 0 for e in agenda_events),
                "inventory_review_calls": len(review_events),
                "inventory_review_repairs": sum(
                    e["attempt"] > 0 for e in review_events
                ),
                "inventory_review": saved.get("inventory_review"),
                "inventory": saved.get("inventory", []),
                "memory_bindings": saved.get("memory_candidates", {}),
                "events": events,
                "cost": _cost(
                    _cache_records(directory / "calls", baseline.namespace(plan, task))
                ),
                "scientific_adequacy": "not_assessed",
            }
        )
    result = {
        "protocol": PROTOCOL,
        "identity": plan["artifact_sha256"],
        "rows": rows,
        "planned_constructions": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "first_reply_accepted": sum(r["first_reply_accepted"] for r in rows),
        "first_reply_total": sum(r["first_reply_total"] for r in rows),
        "repair_calls": sum(r["repair_calls"] for r in rows),
        "inventory_review_calls": sum(r["inventory_review_calls"] for r in rows),
        "inventory_review_repairs": sum(r["inventory_review_repairs"] for r in rows),
        "inventory_review_status_counts": dict(
            Counter(
                (r["inventory_review"] or {}).get("status", "not_recorded")
                for r in rows
            )
        ),
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "scientific_adequacy": (
            "requires inventory and binding review; "
            "mechanical success is not compliance"
        ),
    }
    atomic_json(root / "summary.json", result)
    text = [
        "# Variable-only confirmation",
        "",
        "Mechanical acceptance is not scientific compliance. No equations or fits.",
        "",
        f"Status counts: {result['status_counts']}",
        f"Final inventory reviews: {result['inventory_review_status_counts']}",
        f"Review calls / repairs: {result['inventory_review_calls']} / "
        f"{result['inventory_review_repairs']}",
        "",
        "| Task | Status | First accepted / called | Repairs | Memory bindings |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    panels = [
        "<!doctype html><meta charset='utf-8'><title>Variable inspection</title>",
        "<style>body{max-width:1100px;margin:2em auto;font-family:system-ui}"
        "pre{white-space:pre-wrap}</style>",
        "<h1>Variable-only confirmation</h1><p>Inspect scientific adequacy "
        "separately from runtime acceptance.</p>",
    ]
    for r in rows:
        text.append(
            f"| {r['task']} | {r['status']} | {r['first_reply_accepted']} / "
            f"{r['first_reply_total']} | {r['repair_calls']} | {r['memory_bindings']} |"
        )
        panels.append(
            f"<h2>{html.escape(r['task'])}</h2>"
            f"<a href='results/{html.escape(r['task'], quote=True)}/TRACE.html'>"
            "Exact prompts and replies</a>"
            f"<pre>{html.escape(json.dumps(r, indent=2))}</pre>"
        )
    (root / "SUMMARY.md").write_text("\n".join(text) + "\n")
    (root / "VARIABLES.html").write_text("\n".join(panels))
    return result
