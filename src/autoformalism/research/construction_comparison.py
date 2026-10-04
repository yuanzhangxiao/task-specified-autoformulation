"""Matched public-only comparison of separate, joint and adaptive construction."""

from __future__ import annotations

import errno
import hashlib
import html
import json
import statistics
import warnings
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_baseline as baseline
from autoformalism.research import construction_contract as contract
from autoformalism.research import construction_trace, topology_confirmation
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_schedules as schedules
from autoformalism.search.training_evidence import TrainingEvidence, evidence_brief
from autoformalism.staged_topology import content_hash

PROTOCOL = "phase-c-construction-comparison-2"
REPO = baseline.REPO


class Config(baseline.Config):
    """Identical total budgets, with a reserved bounded repair allowance per arm."""

    protocol: Literal["phase-c-construction-comparison-2"] = PROTOCOL
    repair_requests: int = Field(default=3, ge=1, le=5)
    repair_tokens: int = Field(default=131072, ge=256)

    @model_validator(mode="after")
    def positive_initial_budget(self):
        if (
            self.model_settings.maximum_requests <= self.repair_requests
            or self.model_settings.maximum_total_tokens <= self.repair_tokens + 256
        ):
            raise ValueError("reserve positive construction and repair budgets")
        return self


def source_identity() -> dict:
    """Bind source, new CLI and shared scheduler dispatch to the frozen plan."""
    return {
        **baseline.source_identity(),
        "comparison_launchers": {
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
            for name in (
                "scripts/phase_c_construction_comparison.py",
                "scripts/hpc/start_phase_c_construction_comparison.sh",
            )
        },
    }


def freeze(source: Path, root: Path) -> dict:
    """Reuse public contexts/settings only; all three arms start from empty drafts."""
    old = sealed_read(source / "plan.json")
    if (
        old["protocol"] != topology_confirmation.PROTOCOL
        or old.get("test_data_opened") is not False
    ):
        raise ValueError("requires the sealed development topology plan")
    topology_confirmation._roster(
        old["tasks"], topology_confirmation.phase_c_inputs.ROSTER
    )
    config = Config.model_validate(
        {
            **{
                k: old["config"][k]
                for k in baseline.Config.model_fields
                if k != "protocol"
            },
            "protocol": PROTOCOL,
        }
    )
    cells = {}
    for name in topology_confirmation.phase_c_inputs.ROSTER:
        cell = contract.correct_roles(
            {
                k: old["cells"][name][k]
                for k in ("brief", "context", "target_contract", "evidence")
            }
        )
        if cell["brief"]["limits"] != config.limits.model_dump(mode="json"):
            raise ValueError("public construction limits differ from frozen settings")
        # Validate the exact public context/packet without opening trajectory files.
        brief = contract.visible_brief(cell, {"arm": "full"})
        evidence_brief(
            brief.model_dump(mode="json"),
            ValidationContext.model_validate(cell["context"]),
            TrainingEvidence.model_validate(cell["evidence"]),
        )
        cells[name] = cell
    tasks = []
    for i, task in enumerate(old["tasks"]):
        # Rotate policies within matched blocks so draining does not always omit
        # the same arm. This is scheduling, never score-based task selection.
        policies = schedules.POLICIES[i % 3 :] + schedules.POLICIES[: i % 3]
        tasks.extend(
            {
                **task,
                "task_id": f"{task['task_id']}_{policy}",
                "policy": policy,
                "matched_task_id": task["task_id"],
            }
            for policy in policies
        )
    with public._lock(root):
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "config": config.model_dump(mode="json"),
                "cells": cells,
                "tasks": tasks,
                "source_identity": source_identity(),
                "source_plan_sha256": old["artifact_sha256"],
                "scope": (
                    "Fresh variables and topology only; no historical inventories, "
                    "equations, fitting or scientific critic."
                ),
                "automatic_followup": False,
                "test_data_opened": False,
            },
        )


def verify(root: Path) -> dict:
    """Resume only the same code and complete matched 96-task matrix."""
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != PROTOCOL or plan["source_identity"] != source_identity():
        raise ValueError("construction comparison source/protocol differs")
    Config.model_validate(plan["config"])
    if len(plan["tasks"]) != 96 or {t["policy"] for t in plan["tasks"]} != set(
        schedules.POLICIES
    ):
        raise ValueError("requires three policies and 96 tasks")
    for policy in schedules.POLICIES:
        topology_confirmation._roster(
            [t for t in plan["tasks"] if t["policy"] == policy],
            topology_confirmation.phase_c_inputs.ROSTER,
        )
    if len({t["task_id"] for t in plan["tasks"]}) != 96:
        raise ValueError("duplicate task IDs")
    return plan


def checked_records(directory: Path, identity: str) -> list[dict]:
    """Do not accept a resumed result without its exact cached call evidence."""
    records = _cache_records(directory / "calls", identity)
    by_hash = {r["request_hash"]: r for r in records}
    drafts, used = [ledger.Draft().model_dump(mode="json")], []
    for index, path in enumerate(
        sorted((directory / "construction" / "events").glob("*.json"))
    ):
        event = sealed_read(path)
        record = by_hash.get(event["request_hash"])
        if record is None or content_hash(record) != event["record_sha256"]:
            raise ValueError("missing or changed construction call records")
        displayed = json.loads(record["request"]["body"]["messages"][1]["content"])
        if (
            path.name != f"{index:03d}.json"
            or event["index"] != index
            or event["before"] != drafts[-1]
            or displayed["current_draft"]["declarations"] != event["before"]
            or event["request_hash"] in used
        ):
            raise ValueError("construction transaction chain differs")
        drafts.append(event["after"])
        used.append(event["request_hash"])
    before_path = directory / "construction/before_repair.json"
    if before_path.exists():
        before = sealed_read(before_path)
        index = before["event_count"]
        if (
            index >= len(drafts)
            or before["draft"] != drafts[index]
            or before["cost"] != _cost([by_hash[h] for h in sorted(used[:index])])
        ):
            raise ValueError("initial draft differs from its transaction prefix")
    outcome_path = directory / "proposal.json"
    if outcome_path.exists():
        result = sealed_read(outcome_path)
        if (
            result["identity"] != identity
            or result["event_count"] != len(used)
            or len(used) != len(records)
            or result["draft"] != drafts[-1]
        ):
            raise ValueError("terminal result lacks its complete transaction history")
    return records


def propose(root: Path, plan: dict, task: dict, base_url: str, **kwargs) -> dict:
    """Run the selected schedule through the same cache, ledger and public checks."""
    directory, identity = baseline.location(root, task), baseline.namespace(plan, task)
    with public._lock(directory / "construction-lock"):
        records = checked_records(directory, identity)
        previous = baseline.read_outcome(directory / "proposal.json", plan, task)
        if previous:
            if previous["cost"] != _cost(records):
                raise ValueError("result cost differs from cached call records")
            render_trace(directory, identity)
            return previous
        config = Config.model_validate(plan["config"])
        client = ConstructionClient(
            settings=config.model_settings,
            directory=directory / "calls",
            namespace=identity,
            seed=task["seed"],
            base_url=base_url,
            **kwargs,
        )
        cell = plan["cells"][task["benchmark_id"]]
        brief = contract.visible_brief(cell, task)
        context = ValidationContext.model_validate(cell["context"])
        evidence = (
            TrainingEvidence.model_validate(cell["evidence"])
            if task["arm"] == "full"
            else None
        )
        try:
            result = schedules.run(
                brief,
                context,
                contract.target_definitions(cell),
                evidence_brief(brief.model_dump(mode="json"), context, evidence),
                client,
                directory / "construction",
                task["policy"],
                repair_requests=config.repair_requests,
                repair_tokens=config.repair_tokens,
            )
            return sealed_write(
                directory / "proposal.json",
                {
                    **result,
                    "identity": identity,
                    "function_generation_performed": False,
                    "parameter_fitting_performed": False,
                },
            )
        finally:
            render_trace(directory, identity)


def render_trace(directory: Path, identity: str) -> dict:
    """A derived view cannot mask a saved result or a primary checkpoint error."""
    try:
        construction_trace.render(directory, identity, compact=True)
    except OSError as exc:
        if exc.errno not in {errno.EDQUOT, errno.ENOSPC}:
            raise
        warnings.warn(f"Trace unavailable due to storage limit: {exc}", stacklevel=2)
        return {"status": "unavailable", "errno": exc.errno}
    return {"status": "available", "presentation": "linked-original-records-1"}


def spread(values: list[int]) -> dict:
    """Unscaled median absolute deviation; missing work is never a zero cost."""
    median = statistics.median(values) if values else None
    return {
        "median": median,
        "MAD": statistics.median(abs(v - median) for v in values) if values else None,
    }


def _skeleton(assessment: dict) -> str:
    lines = []
    bindings = {b["proposal"]["name"]: b for b in assessment["shared_process_bindings"]}
    for i, e in enumerate(assessment["equations"]):
        terms = []
        for j, term in enumerate(e["terms"]):
            sources = term["sources"]
            p = bindings.get(sources[0]) if len(sources) == 1 else None
            uses = (
                [u for u in p["signed_declaration"]["uses"] if u["target"] == e["name"]]
                if p
                else []
            )
            if uses:
                law = (
                    f"({uses[0]['conversion'] or 'unknown_conversion'}) * {sources[0]}"
                )
            else:
                law = f"phi_{i}_{j}({', '.join(sources)})"
            sign = {
                "positive": "+ magnitude",
                "negative": "- magnitude",
                "unrestricted": "+ signed_weight",
            }[term["outer_weight_sign"]]
            terms.append(f"{sign} * {law}")
        lhs = f"d({e['name']})/dt" if e["definition"] == "differential" else e["name"]
        lines.append(f"{lhs} = {' '.join(terms)}")
    return (
        "\n".join(lines) or "No assembled skeleton available; inspect raw declarations."
    )


def report(root: Path, plan: dict) -> dict:
    """Show before/after skeletons and every trace; never label syntax as science."""
    rows, sections = [], []

    def block(value):
        return (
            "<pre>"
            + html.escape(json.dumps(value, indent=2, ensure_ascii=False))
            + "</pre>"
        )

    for task in plan["tasks"]:
        directory, identity = (
            baseline.location(root, task),
            baseline.namespace(plan, task),
        )
        records = checked_records(directory, identity)
        value = baseline.read_outcome(directory / "proposal.json", plan, task)
        if value and value["cost"] != _cost(records):
            raise ValueError("result accounting differs from retained calls")
        initial = (value or {}).get("before_repair")
        if initial is None and (directory / "construction/before_repair.json").exists():
            initial = sealed_read(directory / "construction/before_repair.json")
        events = [
            sealed_read(p)
            for p in sorted((directory / "construction/events").glob("*.json"))
        ]
        batch_sizes = [
            {
                "stage": e["stage"],
                "accepted": e["accepted"],
                "selected_lhs": e["selected_lhs"],
                "changed_variables": sum(
                    v not in e["before"]["variables"] for v in e["after"]["variables"]
                ),
                "changed_equations": sum(
                    v not in e["before"]["equations"] for v in e["after"]["equations"]
                ),
                "pending": e["pending_after"],
            }
            for e in events
        ]
        row = {
            **task,
            "status": value["status"] if value else "pending",
            "cost": _cost(records),
            "batch_history": batch_sizes,
            "stage_outcomes": initial.get("stage_outcomes") if initial else None,
            "equation_stage_reached": any(e["stage"] == "equations" for e in events),
            "initial_complete": bool(
                initial["assessment"]["eligible"] and initial["ready_requested"]
            )
            if initial
            else None,
            "initial_errors": initial["assessment"]["errors"] if initial else None,
            "final_errors": value["assessment"]["errors"] if value else None,
            "repair_calls": sum(e["stage"] == "repair" for e in events),
            "scientific_adequacy": "requires independent equation inspection",
        }
        rows.append(row)
        if records:
            row["trace"] = render_trace(directory, identity)
        title = html.escape(task["task_id"])
        sections.append(
            f'<h2>{title}</h2><a href="results/{title}/TRACE.html">'
            "Exact requests, responses and decisions</a>" + block(row)
        )
        for label, saved in (
            ("Before overall repair", initial),
            ("After overall repair", value),
        ):
            if saved:
                sections.append(
                    f"<h3>{label}</h3><pre>{html.escape(_skeleton(saved['assessment']))}</pre>"
                    + block(saved["draft"])
                    + block(saved["assessment"]["errors"])
                )
    groups = {}
    for policy in schedules.POLICIES:
        for arm in ("full", "brief_only"):
            selected = [r for r in rows if r["policy"] == policy and r["arm"] == arm]
            done = [r for r in selected if r["status"] != "pending"]
            groups[f"{policy}:{arm}"] = {
                "planned": len(selected),
                "finished": len(done),
                "equation_stage_reached": sum(
                    r["equation_stage_reached"] for r in done
                ),
                "initial_complete": sum(
                    r["initial_complete"] is True for r in selected
                ),
                "final_complete": sum(
                    r["status"] == "topology_complete" for r in selected
                ),
                "tokens": spread([r["cost"]["observed_tokens"] for r in done]),
                "calls": spread([r["cost"]["physical_requests"] for r in done]),
            }
    result = {
        "protocol": PROTOCOL,
        "identity": plan["artifact_sha256"],
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "groups": groups,
        "observed_total_tokens": sum(r["cost"]["observed_tokens"] for r in rows),
        "physical_requests": sum(r["cost"]["physical_requests"] for r in rows),
        "unmeasured_requests": sum(
            r["cost"]["requests_with_unknown_usage"] for r in rows
        ),
        "rows": rows,
        "test_data_opened": False,
        "optimizer_calls": 0,
        "scientific_adequacy": "not_assessed",
        "automatic_followup": False,
    }
    atomic_json(root / "summary.json", result)
    lines = [
        "# Phase C construction schedules",
        "",
        "Fresh variables and topology; no functions or fitting.",
        "Completion means declared structural checks passed, not scientific adequacy.",
        "Costs include variable construction and repairs. Brackets show unscaled "
        "MAD over finished tasks; pending tasks are not zeros.",
        "",
        "| Policy / prompt | Finished / planned | Initial complete | Final complete "
        "| Tokens median [MAD] | Calls median [MAD] |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, g in groups.items():
        lines.append(
            f"| {name} | {g['finished']}/{g['planned']} | {g['initial_complete']} "
            f"| {g['final_complete']} | {g['tokens']['median']} [{g['tokens']['MAD']}] "
            f"| {g['calls']['median']} [{g['calls']['MAD']}] |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (root / "TOPOLOGY.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Construction comparison</title>"
        "<style>body{max-width:1200px;margin:2em auto;font-family:system-ui}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;"
        "padding:1em}</style><h1>Construction comparison</h1>"
        "<p>Structural eligibility is not scientific correctness. Functions, "
        "initialization, NMSE and fitted mechanism compliance remain unassessed.</p>"
        + "\n".join(sections)
    )
    return result
