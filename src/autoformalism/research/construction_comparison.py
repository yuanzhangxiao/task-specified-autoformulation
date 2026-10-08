"""Matched public-only comparison of separate, joint and adaptive construction."""

from __future__ import annotations

import errno
import hashlib
import html
import json
import os
import statistics
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_baseline as baseline
from autoformalism.research import construction_contract as contract
from autoformalism.research import construction_trace, topology_confirmation
from autoformalism.research.construction_obligations import reviewed_contract
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_bookkeeping as bookkeeping
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_prompts as prompts
from autoformalism.search import construction_schedules as schedules
from autoformalism.search.public_graph_obligations import PublicGraphContract
from autoformalism.search.training_evidence import TrainingEvidence, evidence_brief
from autoformalism.staged_topology import content_hash

PROTOCOL = "phase-c-construction-comparison-10"
REPO = baseline.REPO
Study = Literal[
    "comparison",
    "live_confirmation",
    "basin_confirmation",
    "shared_law_comparison",
    "prompt_comparison",
    "refinement_confirmation",
    "variable_checklist_confirmation",
]
STUDIES = (
    "comparison",
    "live_confirmation",
    "basin_confirmation",
    "shared_law_comparison",
    "prompt_comparison",
    "refinement_confirmation",
    "variable_checklist_confirmation",
)
BASIN_STUDIES = ("basin_confirmation", "shared_law_comparison")
BASIN_CASES = tuple(topology_confirmation.phase_c_inputs.basin.BASINS)


class Config(baseline.Config):
    """Identical total budgets, with a reserved bounded repair allowance per arm."""

    protocol: Literal["phase-c-construction-comparison-10"] = PROTOCOL
    repair_requests: int = Field(default=3, ge=1, le=5)
    repair_tokens: int = Field(default=131072, ge=256)
    bookkeeping_policy: bookkeeping.Policy = "legacy"

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
                "scripts/hpc/start_phase_c_construction_live.sh",
                "scripts/hpc/start_phase_c_construction_basin.sh",
                "scripts/hpc/start_phase_c_shared_laws.sh",
                "scripts/hpc/start_phase_c_minimal_prompts.sh",
                "scripts/hpc/start_phase_c_prompt_refinement.sh",
                "scripts/hpc/start_phase_c_variable_checklist.sh",
            )
        },
    }


def freeze(
    source: Path,
    root: Path,
    *,
    study: Study = "comparison",
    bookkeeping_policy: bookkeeping.Policy = "legacy",
    repair_source: Path | None = None,
) -> dict:
    """Freeze public construction tasks and any explicitly selected saved repairs."""
    if study not in STUDIES:
        raise ValueError("unknown construction study")
    if (repair_source is not None) != (study == "refinement_confirmation"):
        raise ValueError(
            "refinement confirmation requires its saved repair source only"
        )
    if study == "refinement_confirmation" and bookkeeping_policy != "legacy":
        raise ValueError("refinement policies are frozen separately by task")
    if study == "variable_checklist_confirmation":
        if bookkeeping_policy not in {"legacy", bookkeeping.CHECKLIST_POLICY}:
            raise ValueError("variable checklist study fixes its minimal policy")
        bookkeeping_policy = bookkeeping.CHECKLIST_POLICY
    bookkeeping.validate_policy(
        bookkeeping_policy,
        "minimal"
        if study == "variable_checklist_confirmation"
        else "current"
        if study == "prompt_comparison"
        else None,
    )
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
            "bookkeeping_policy": bookkeeping_policy,
            **({"wall_seconds": 10800} if study != "comparison" else {}),
        }
    )
    roster = (
        BASIN_CASES
        if study in BASIN_STUDIES
        else topology_confirmation.phase_c_inputs.ROSTER
    )
    cells = {}
    for name in roster:
        cell = contract.correct_roles(
            {
                k: old["cells"][name][k]
                for k in ("brief", "context", "target_contract", "evidence")
            }
        )
        if cell["brief"]["limits"] != config.limits.model_dump(mode="json"):
            raise ValueError("public construction limits differ from frozen settings")
        cell["public_graph_contract"] = reviewed_contract(
            name, PublicScientificBrief.model_validate(cell["brief"])
        ).model_dump(mode="json")
        # Validate the exact public context/packet without opening trajectory files.
        brief = contract.visible_brief(cell, {"arm": "full"})
        evidence_brief(
            brief.model_dump(mode="json"),
            ValidationContext.model_validate(cell["context"]),
            TrainingEvidence.model_validate(cell["evidence"]),
        )
        cells[name] = cell
    source_tasks = old["tasks"]
    if study != "comparison":
        by_case = {
            t["benchmark_id"]: t
            for t in source_tasks
            if t["seed"] == 0 and t["arm"] == "full"
        }
        source_tasks = [by_case[name] for name in roster]
    blocks = []
    for i, task in enumerate(source_tasks):
        if study in {"refinement_confirmation", "variable_checklist_confirmation"}:
            blocks.append(
                [
                    {
                        **task,
                        "task_id": f"{task['task_id']}_minimal_"
                        + (
                            "checklist"
                            if study == "variable_checklist_confirmation"
                            else "clarity"
                        ),
                        "policy": "joint_adaptive",
                        "process_question": "dedicated",
                        "prompt_family": "minimal",
                        "stage_schedule": prompts.SCHEDULE,
                        "matched_task_id": task["task_id"],
                        **(
                            {"bookkeeping_policy": bookkeeping.MINIMAL_POLICY}
                            if study == "refinement_confirmation"
                            else {}
                        ),
                    }
                ]
            )
            continue
        if study == "prompt_comparison":
            families = prompts.FAMILIES if i % 2 == 0 else prompts.FAMILIES[::-1]
            blocks.append(
                [
                    {
                        **task,
                        "task_id": f"{task['task_id']}_{family}",
                        "policy": "joint_adaptive",
                        "process_question": "dedicated",
                        "prompt_family": family,
                        "stage_schedule": prompts.SCHEDULE,
                        "matched_task_id": task["task_id"],
                    }
                    for family in families
                ]
            )
            continue
        # Rotate policies within matched blocks so draining does not always omit
        # the same arm. This is scheduling, never score-based task selection.
        policies = schedules.POLICIES[i % 3 :] + schedules.POLICIES[: i % 3]
        block = []
        for j, policy in enumerate(policies):
            placements = (
                schedules.PROCESS_QUESTIONS
                if study == "shared_law_comparison"
                else ("integrated",)
            )
            if (i + j) % 2:
                placements = placements[::-1]
            block.extend(
                [
                    {
                        **task,
                        "task_id": f"{task['task_id']}_{policy}"
                        + (f"_{placement}" if study == "shared_law_comparison" else ""),
                        "policy": policy,
                        "process_question": placement,
                        "matched_task_id": task["task_id"],
                    }
                    for placement in placements
                ]
            )
        blocks.append(block)
    # Expose every selected case before starting its second policy.
    # Every case receives every policy; rotation is independent of results.
    tasks = (
        [block[j] for j in range(len(blocks[0])) for block in blocks]
        if study != "comparison"
        else [task for block in blocks for task in block]
    )
    if study == "refinement_confirmation":
        from autoformalism.research.construction_refinement import repair_tasks

        tasks.extend(repair_tasks(repair_source, cells))
    with public._lock(root):
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "study": study,
                "selection_rule": "eight-fresh-minimal-and-three-saved-repairs-1"
                if study == "refinement_confirmation"
                else "all-eight-cases-seed0-full-variable-checklist-1"
                if study == "variable_checklist_confirmation"
                else ("all-eight-cases-seed0-full-two-prompt-families-1")
                if study == "prompt_comparison"
                else ("both-basins-seed0-full-three-policies-two-placements-1")
                if study == "shared_law_comparison"
                else "both-basins-seed0-full-three-policies-1"
                if study == "basin_confirmation"
                else "all-eight-cases-seed0-full-three-policies-1"
                if study != "comparison"
                else "full-case-seed-prompt-roster-1",
                "config": config.model_dump(mode="json"),
                "cells": cells,
                "tasks": tasks,
                "source_identity": source_identity(),
                "source_plan_sha256": old["artifact_sha256"],
                "scope": (
                    "Eight fresh minimal constructions and three saved repairs; "
                    "separate diagnostics, not a ranking. No functions or fitting."
                    if study == "refinement_confirmation"
                    else "Fresh variables and topology only; no old inventories, "
                    "equations, fitting or scientific critic."
                ),
                "automatic_followup": False,
                "test_data_opened": False,
            },
        )


def verify(root: Path) -> dict:
    """Resume only the frozen source and exact declared matched roster."""
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != PROTOCOL or plan["source_identity"] != source_identity():
        raise ValueError("construction comparison source/protocol differs")
    config = Config.model_validate(plan["config"])
    bookkeeping.validate_policy(
        config.bookkeeping_policy,
        "minimal"
        if plan.get("study") == "variable_checklist_confirmation"
        else "current"
        if plan.get("study") == "prompt_comparison"
        else None,
    )
    for name, cell in plan["cells"].items():
        expected = reviewed_contract(
            name, PublicScientificBrief.model_validate(cell["brief"])
        ).model_dump(mode="json")
        if cell.get("public_graph_contract") != expected:
            raise ValueError("reviewed public graph contract differs")
    study = plan.get("study", "comparison")
    if study not in STUDIES:
        raise ValueError("unknown construction study")
    if study != "comparison" and plan["config"]["wall_seconds"] != 10800:
        raise ValueError("live confirmation uses a three-hour worker window")
    roster = (
        BASIN_CASES
        if study in BASIN_STUDIES
        else topology_confirmation.phase_c_inputs.ROSTER
    )
    if set(plan["cells"]) != set(roster):
        raise ValueError("public cells differ from study roster")
    expected = {
        (case, seed, arm, policy, placement)
        for case in roster
        for seed in ((0,) if study != "comparison" else (0, 1))
        for arm in (("full",) if study != "comparison" else ("full", "brief_only"))
        for policy in schedules.POLICIES
        for placement in (
            schedules.PROCESS_QUESTIONS
            if study == "shared_law_comparison"
            else ("integrated",)
        )
    }
    actual = [
        (t["benchmark_id"], t["seed"], t["arm"], t["policy"], t.get("process_question"))
        for t in plan["tasks"]
    ]
    if study == "refinement_confirmation":
        from autoformalism.research.construction_refinement import validate_tasks

        validate_tasks(plan)
        return plan
    if any(
        "bookkeeping_policy" in t or "starting_checkpoint" in t for t in plan["tasks"]
    ):
        raise ValueError("per-task refinement requires its dedicated study")
    if study in {"prompt_comparison", "variable_checklist_confirmation"}:
        if study == "variable_checklist_confirmation" and (
            config.bookkeeping_policy != bookkeeping.CHECKLIST_POLICY
        ):
            raise ValueError("variable checklist policy differs")
        expected = {
            (case, 0, "full", "joint_adaptive", "dedicated", family, prompts.SCHEDULE)
            for case in roster
            for family in (
                ("minimal",)
                if study == "variable_checklist_confirmation"
                else prompts.FAMILIES
            )
        }
        actual = [
            (*row, task.get("prompt_family"), task.get("stage_schedule"))
            for row, task in zip(actual, plan["tasks"], strict=True)
        ]
    elif any(
        "prompt_family" in task or "stage_schedule" in task for task in plan["tasks"]
    ):
        raise ValueError("prompt families require the dedicated matched study")
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f"requires the exact {len(expected)}-task {study} roster")
    ids = [t["task_id"] for t in plan["tasks"]]
    if len(set(ids)) != len(ids) or any(
        Path(n).name != n or n in {".", ".."} for n in ids
    ):
        raise ValueError("task IDs must be unique simple names")
    if any(
        not t.get("shared_processes") or not t.get("scientific_verifier")
        for t in plan["tasks"]
    ):
        raise ValueError("construction study retains shared processes and verifier")
    return plan


def check_storage(root: Path, *, probe_bytes: int = 128 * 1024 * 1024) -> dict:
    """Test actual quota-limited writes, without reserving space or deleting records."""
    if probe_bytes <= 0:
        raise ValueError("probe_bytes must be positive")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=root, prefix=".storage-probe-") as stream:
        chunk = bytes(min(probe_bytes, 1024 * 1024))
        remaining = probe_bytes
        while remaining > 0:
            remaining -= stream.write(chunk[:remaining])
        stream.flush()
        os.fsync(stream.fileno())
    return {"write_probe_bytes": probe_bytes, "space_reserved": False}


def delivery_counts(records: list[dict]) -> dict:
    """Count observed delivery outcomes, without a scientific interpretation."""
    finish = Counter()
    whitespace = 0
    for record in records:
        raw = record.get("raw_response")
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
        ):
            finish["unavailable"] += 1
            continue
        reason = choices[0].get("finish_reason")
        finish[reason if isinstance(reason, str) else "unavailable"] += 1
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if reason == "length" and isinstance(content, str) and content:
            whitespace += (len(content) - len(content.rstrip())) / len(content) >= 0.97
    return {
        "finish_reason_counts": dict(finish),
        "length_limited_responses": finish["length"],
        "whitespace_heavy_length_responses": whitespace,
    }


def checked_records(
    directory: Path, identity: str, starting_draft: dict | None = None
) -> list[dict]:
    """Do not accept a resumed result without its exact cached call evidence."""
    records = _cache_records(directory / "calls", identity)
    by_hash = {r["request_hash"]: r for r in records}
    drafts, used = (
        [
            starting_draft
            if starting_draft is not None
            else ledger.Draft().model_dump(mode="json")
        ],
        [],
    )
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
            or ledger.Draft.model_validate(event["before"])
            != ledger.Draft.model_validate(drafts[-1])
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
        starting = task.get("starting_checkpoint", {}).get("draft")
        records = checked_records(directory, identity, starting)
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
                process_question=task["process_question"],
                prompt_family=task.get("prompt_family"),
                bookkeeping_policy=task.get(
                    "bookkeeping_policy", config.bookkeeping_policy
                ),
                starting_draft=ledger.Draft.model_validate(starting)
                if starting is not None
                else None,
                repair_requests=config.repair_requests,
                repair_tokens=config.repair_tokens,
                graph_contract=PublicGraphContract.model_validate(
                    cell["public_graph_contract"]
                ),
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


def process_history(records: list[dict], events: list[dict]) -> list[dict]:
    """Distinguish absent, rejected and committed law declarations from raw replies.

    Consumer counts below describe proposals, not verified assembly or science.
    Unreadable delivery stays unknown rather than becoming an empty decision.
    """
    by_hash = {r["request_hash"]: r for r in records}
    history = []
    for event in events:
        record = by_hash[event["request_hash"]]
        payload = json.loads(record["request"]["body"]["messages"][1]["content"])
        declarations = None
        removals = None
        try:
            raw = visible_response(record)
            if isinstance(raw, dict):
                items = raw.get("processes")
                if isinstance(items, list):
                    declarations = []
                    for p in items:
                        uses = p.get("uses") if isinstance(p, dict) else None
                        targets = (
                            sorted(
                                {
                                    u["target"]
                                    for u in uses
                                    if isinstance(u, dict)
                                    and isinstance(u.get("target"), str)
                                }
                            )
                            if isinstance(uses, list)
                            else None
                        )
                        declarations.append(
                            {
                                "name": p.get("name") if isinstance(p, dict) else None,
                                "kind": p.get("kind") if isinstance(p, dict) else None,
                                "declared_consumers": targets,
                                "declared_shared": len(targets) >= 2
                                if targets is not None
                                else None,
                            }
                        )
                removals = raw.get("remove_processes")
        except (ValueError, TypeError, KeyError):
            pass
        history.append(
            {
                "event": event["index"],
                "stage": event["stage"],
                "shared_law_question_displayed": bool(
                    payload.get("shared_law_question")
                    or (
                        payload.get("prompt_family") == "minimal"
                        and payload["stage"] == "shared_laws"
                    )
                ),
                "request_hash": event["request_hash"],
                "accepted": event["accepted"],
                "error": event["error"],
                "proposed_processes": declarations,
                "requested_removals": removals,
                "retained_names": [p["name"] for p in event["after"]["processes"]],
            }
        )
    return history


def report(root: Path, plan: dict) -> dict:
    """Show before/after skeletons and every trace; never label syntax as science."""
    rows, sections = [], []
    deliveries = []

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
        records = checked_records(
            directory, identity, task.get("starting_checkpoint", {}).get("draft")
        )
        delivery = delivery_counts(records)
        deliveries.append(delivery)
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
            "delivery": delivery,
            "rejections_by_stage": dict(
                Counter(e["stage"] for e in events if not e["accepted"])
            ),
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
            "initial_clarifications": initial["assessment"].get(
                "clarification_requests"
            )
            if initial
            else None,
            "final_clarifications": value["assessment"].get("clarification_requests")
            if value
            else None,
            "contribution_overlaps": value["assessment"].get("contribution_overlaps")
            if value
            else None,
            "graph_check_status": value["assessment"]["graph_check_status"]
            if value
            else None,
            "unresolved_public_predicates": value["assessment"][
                "unresolved_public_predicates"
            ]
            if value
            else None,
            "process_usage": value["assessment"]["process_usage"] if value else None,
            "initial_process_usage": initial["assessment"]["process_usage"]
            if initial
            else None,
            "process_history": process_history(records, events),
            "reviewed_public_graph_checks": value["assessment"][
                "reviewed_public_graph_checks"
            ]
            if value
            else None,
            "deferred_scientific_checks": plan["cells"][task["benchmark_id"]][
                "public_graph_contract"
            ]["deferred_scientific_checks"],
            "accepted_reply_normalizations": sum(
                len(e["normalizations"]) for e in events if e["accepted"]
            ),
            "normalizations_by_code": dict(
                Counter(
                    n["code"]
                    for e in events
                    if e["accepted"]
                    for n in e["normalizations"]
                )
            ),
            "repair_calls": sum(e["stage"] == "repair" for e in events),
            "scientific_adequacy": "requires independent equation inspection",
        }
        row["process_counts"] = {
            "replies_with_declarations": sum(
                bool(h["proposed_processes"]) for h in row["process_history"]
            ),
            "rejected_replies_with_declarations": sum(
                bool(h["proposed_processes"]) and not h["accepted"]
                for h in row["process_history"]
            ),
            "unreadable_process_lists": sum(
                h["proposed_processes"] is None for h in row["process_history"]
            ),
            "retained_shared_laws": sum(
                p["scope"] == "shared" for p in row["process_usage"]
            )
            if value
            else None,
            "scope": (
                "Replies may repeat declarations; retained counts verify assembly only."
            ),
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
                    + block(saved["assessment"].get("clarification_requests", []))
                )
    groups = {}
    group_labels = sorted(
        {
            (r.get("prompt_family", r["policy"]), r["arm"], r["process_question"])
            for r in rows
        }
    )
    for label, arm, placement in group_labels:
        selected = [
            r
            for r in rows
            if (r.get("prompt_family", r["policy"]), r["arm"], r["process_question"])
            == (label, arm, placement)
        ]
        done = [r for r in selected if r["status"] != "pending"]
        key = f"{label}:{arm}"
        if plan.get("study") == "shared_law_comparison":
            key += f":{placement}"
        groups[key] = {
            "planned": len(selected),
            "finished": len(done),
            "equation_stage_reached": sum(r["equation_stage_reached"] for r in done),
            "initial_complete": sum(r["initial_complete"] is True for r in selected),
            "final_complete": sum(r["status"] == "topology_complete" for r in selected),
            "tokens": spread([r["cost"]["observed_tokens"] for r in done]),
            "calls": spread([r["cost"]["physical_requests"] for r in done]),
            "models_retaining_shared_laws": sum(
                r["process_counts"]["retained_shared_laws"] > 0 for r in done
            ),
            "replies_with_process_declarations": sum(
                r["process_counts"]["replies_with_declarations"] for r in done
            ),
            "rejected_replies_with_process_declarations": sum(
                r["process_counts"]["rejected_replies_with_declarations"] for r in done
            ),
        }
    result = {
        "protocol": PROTOCOL,
        "bookkeeping_policy": "per-task (minimal-clarity-1 / current-repair-fidelity-1)"
        if plan.get("study") == "refinement_confirmation"
        else plan["config"].get("bookkeeping_policy", "legacy"),
        "study": plan.get("study", "comparison"),
        "identity": plan["artifact_sha256"],
        "planned_tasks": len(rows),
        "all_tasks_terminal": all(r["status"] != "pending" for r in rows),
        "delivery": {
            "length_limited_responses": sum(
                d["length_limited_responses"] for d in deliveries
            ),
            "whitespace_heavy_length_responses": sum(
                d["whitespace_heavy_length_responses"] for d in deliveries
            ),
        },
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "groups": groups,
        "observed_total_tokens": sum(r["cost"]["observed_tokens"] for r in rows),
        "physical_requests": sum(r["cost"]["physical_requests"] for r in rows),
        "unmeasured_requests": sum(
            r["cost"]["requests_with_unknown_usage"] for r in rows
        ),
        "normalizations_by_code": dict(
            sum((Counter(r["normalizations_by_code"]) for r in rows), Counter())
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
        "Eight fresh minimal constructions and three saved repairs; not a ranking."
        if plan.get("study") == "refinement_confirmation"
        else "Fresh variables and topology; no functions or fitting.",
        f"Bookkeeping policy: {result['bookkeeping_policy']}.",
        "Eight fresh minimal cases plus three diagnosed saved repair episodes. "
        "Saved repairs are not fresh-current controls or a matched comparison."
        if plan.get("study") == "refinement_confirmation"
        else "Eight fresh minimal cases, Full/seed 0, with a per-turn variable "
        "checklist and bounded completion repair. Assignments do not certify "
        "mechanisms. Compare stage evidence with earlier runs; not a matched ranking."
        if plan.get("study") == "variable_checklist_confirmation"
        else "Matched current/minimal wording: all eight cases, Full/seed 0. "
        "Both use variables -> shared processes -> remaining topology -> repair, "
        "adaptive batches, the same response schemas, checks and budgets. "
        "The control is current wording under this common schedule, not a replay "
        "of the historical schedule."
        if plan.get("study") == "prompt_comparison"
        else "Shared-law question: integrated versus dedicated; both basins, "
        "three schedules, "
        "Full/seed 0. Same total budgets; the dedicated call is charged, not free. "
        "A small paired pilot, not a strategy ranking."
        if plan.get("study") == "shared_law_comparison"
        else (
            "Basin confirmation: both cases, seed 0, Full only; not a strategy ranking."
        )
        if plan.get("study") == "basin_confirmation"
        else (
            "Live confirmation: all eight cases, seed 0, Full only; "
            "not a strategy ranking."
        )
        if plan.get("study") == "live_confirmation"
        else "Full matched construction comparison.",
        "Completion means declared structural checks passed, not scientific adequacy.",
        f"All tasks terminal: {result['all_tasks_terminal']}. "
        f"Length-limited responses: {result['delivery']['length_limited_responses']}; "
        "with at least 97% trailing whitespace: "
        f"{result['delivery']['whitespace_heavy_length_responses']}.",
        (
            "Costs include new calls only; saved repair episodes exclude their "
            "historical construction cost. "
            if plan.get("study") == "refinement_confirmation"
            else "Costs include variable construction and repairs. "
        )
        + "Brackets show unscaled "
        "MAD over finished tasks; pending tasks are not zeros.",
        "Process counts distinguish raw proposals, rejected edits "
        "and verified assembly. "
        "More shared laws is not inherently better; inspect the independent control. "
        "No conservation or function-equivalence claim follows from topology.",
        "",
        "| Policy / prompt | Finished / planned | Initial complete | Final complete "
        "| Tokens median [MAD] | Calls median [MAD] | Models retaining shared laws |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, g in groups.items():
        lines.append(
            f"| {name} | {g['finished']}/{g['planned']} | {g['initial_complete']} "
            f"| {g['final_complete']} | {g['tokens']['median']} [{g['tokens']['MAD']}] "
            f"| {g['calls']['median']} [{g['calls']['MAD']}] "
            f"| {g['models_retaining_shared_laws']} |"
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
    if plan.get("study") == "variable_checklist_confirmation":
        from autoformalism.research import construction_variable_report

        construction_variable_report.report(root, plan)
    return result
