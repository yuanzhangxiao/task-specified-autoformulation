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
from autoformalism.research import (
    construction_trace,
    construction_training,
    topology_confirmation,
)
from autoformalism.research.construction_boundary_contracts import (
    reviewed_contract as reviewed_boundary_contract,
)
from autoformalism.research.construction_obligations import reviewed_contract
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_bookkeeping as bookkeeping
from autoformalism.search import construction_completion as completion_feedback
from autoformalism.search import construction_coverage as coverage
from autoformalism.search import construction_decisions as decisions
from autoformalism.search import construction_edit_feedback as edit_feedback
from autoformalism.search import construction_evidence as evidence_feedback
from autoformalism.search import construction_focus as focused
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_ownership_feedback as ownership_feedback
from autoformalism.search import construction_prompts as prompts
from autoformalism.search import construction_protocol as protocol_feedback
from autoformalism.search import construction_readable as readable
from autoformalism.search import construction_realizations as realization_prompts
from autoformalism.search import construction_schedules as schedules
from autoformalism.search import mechanism_realizations as realizations
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
    "stage_check_confirmation",
    "deferred_interaction_confirmation",
    "focused_prompt_confirmation",
    "readable_prompt_confirmation",
    "completion_feedback_confirmation",
    "public_coverage_confirmation",
    "coverage_decision_confirmation",
    "edit_feedback_confirmation",
    "ownership_feedback_confirmation",
    "protocol_clarity_confirmation",
    "mechanism_realization_confirmation",
    "evidence_feedback_confirmation",
]
STUDIES = (
    "comparison",
    "live_confirmation",
    "basin_confirmation",
    "shared_law_comparison",
    "prompt_comparison",
    "refinement_confirmation",
    "variable_checklist_confirmation",
    "stage_check_confirmation",
    "deferred_interaction_confirmation",
    "focused_prompt_confirmation",
    "readable_prompt_confirmation",
    "completion_feedback_confirmation",
    "public_coverage_confirmation",
    "coverage_decision_confirmation",
    "edit_feedback_confirmation",
    "ownership_feedback_confirmation",
    "protocol_clarity_confirmation",
    "mechanism_realization_confirmation",
    "evidence_feedback_confirmation",
)
CHECKLIST_STUDIES = {
    "variable_checklist_confirmation": bookkeeping.CHECKLIST_POLICY,
    "stage_check_confirmation": bookkeeping.STAGE_POLICY,
    "deferred_interaction_confirmation": bookkeeping.COMPACT_POLICY,
    "focused_prompt_confirmation": focused.POLICY,
    "readable_prompt_confirmation": readable.POLICY,
    "completion_feedback_confirmation": completion_feedback.POLICY,
    "public_coverage_confirmation": coverage.POLICY,
    "coverage_decision_confirmation": decisions.POLICY,
    "edit_feedback_confirmation": edit_feedback.POLICY,
    "ownership_feedback_confirmation": ownership_feedback.POLICY,
    "protocol_clarity_confirmation": protocol_feedback.POLICY,
    "mechanism_realization_confirmation": realization_prompts.POLICY,
    "evidence_feedback_confirmation": evidence_feedback.POLICY,
}
SHARED_STAGE_STUDIES = (
    "stage_check_confirmation",
    "deferred_interaction_confirmation",
    "focused_prompt_confirmation",
    "readable_prompt_confirmation",
    "completion_feedback_confirmation",
    "public_coverage_confirmation",
    "coverage_decision_confirmation",
    "edit_feedback_confirmation",
    "ownership_feedback_confirmation",
    "protocol_clarity_confirmation",
    "mechanism_realization_confirmation",
    "evidence_feedback_confirmation",
)
BASIN_STUDIES = ("basin_confirmation", "shared_law_comparison")
BASIN_CASES = tuple(topology_confirmation.phase_c_inputs.basin.BASINS)


def sampling_seeds(study: str, seeds: list[int] | None) -> tuple[int, ...]:
    """Validate a frozen seed roster; historical studies keep their old rosters."""
    if seeds is None:
        if study in {
            "coverage_decision_confirmation",
            "edit_feedback_confirmation",
            "ownership_feedback_confirmation",
            "protocol_clarity_confirmation",
            "mechanism_realization_confirmation",
            "evidence_feedback_confirmation",
        }:
            return (0, 1, 2, 3, 4)
        return (0, 1) if study == "comparison" else (0,)
    if study not in {
        "focused_prompt_confirmation",
        "readable_prompt_confirmation",
        "completion_feedback_confirmation",
        "public_coverage_confirmation",
        "coverage_decision_confirmation",
        "edit_feedback_confirmation",
        "ownership_feedback_confirmation",
        "protocol_clarity_confirmation",
        "mechanism_realization_confirmation",
        "evidence_feedback_confirmation",
    }:
        raise ValueError("explicit seeds require focused_prompt_confirmation")
    if (
        not isinstance(seeds, list)
        or not 1 <= len(seeds) <= 10
        or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be 1-10 distinct integers in [0, 2**31)")
    return tuple(seeds)


class ConstructionModelSettings(baseline.StagedModelSettings):
    """Permit the explicit extra repair calls without changing transport defaults."""

    maximum_requests: int = Field(default=64, ge=1, le=133)


class Config(baseline.Config):
    """Versioned budgets with an explicit repair reserve for each study."""

    model_settings: ConstructionModelSettings
    protocol: Literal["phase-c-construction-comparison-10"] = PROTOCOL
    platform: Literal["aces-h100x1", "jetstream2-h100x1-docker"] = "aces-h100x1"
    repair_requests: int = Field(default=3, ge=1, le=10)
    repair_tokens: int = Field(default=131072, ge=256)
    bookkeeping_policy: bookkeeping.Policy = "legacy"

    @model_validator(mode="after")
    def positive_initial_budget(self):
        if (
            self.repair_requests > 5 or self.model_settings.maximum_requests > 128
        ) and self.bookkeeping_policy not in {
            protocol_feedback.POLICY,
            realization_prompts.POLICY,
            evidence_feedback.POLICY,
        }:
            raise ValueError("extended repair requires the protocol clarity policy")
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
                "scripts/hpc/start_phase_c_stage_checks.sh",
                "scripts/hpc/start_phase_c_deferred_interaction.sh",
            )
        },
    }


def load_training_bundle(
    path: Path,
    cells: dict,
    roster: tuple[str, ...] | None = None,
    *,
    source_plan: dict | None = None,
) -> dict:
    """Validate a sealed complete training-only bundle before freezing its cells."""
    bundle = construction_training.load_bundle(path, source_plan)
    names = tuple(cells) if roster is None else roster
    if (
        bundle.get("protocol") != "construction-training-evidence-bundle-1"
        or bundle.get("test_data_opened") is not False
        or set(bundle.get("cells", {})) != set(names)
    ):
        raise ValueError("requires a complete training-only evidence bundle")
    for name in names:
        evidence = construction_training.load_evidence(bundle["cells"][name])
        if evidence.context != ValidationContext.model_validate(cells[name]["context"]):
            raise ValueError("complete training evidence context differs")
        source_digest = (
            bundle.get("source_files", {}).get(name, {}).get("historical_source_sha256")
        )
        if source_digest != cells[name]["evidence"]["source_sha256"]:
            raise ValueError("complete training evidence historical source differs")
    return bundle


def freeze(
    source: Path,
    root: Path,
    *,
    study: Study = "comparison",
    bookkeeping_policy: bookkeeping.Policy = "legacy",
    repair_source: Path | None = None,
    seeds: list[int] | None = None,
    serving_environment: dict | None = None,
    training_evidence: Path | None = None,
) -> dict:
    """Freeze public construction tasks and any explicitly selected saved repairs."""
    if study not in STUDIES:
        raise ValueError("unknown construction study")
    if (training_evidence is not None) != (study == "evidence_feedback_confirmation"):
        raise ValueError("evidence feedback requires explicit training evidence only")
    if (
        study
        in {
            "coverage_decision_confirmation",
            "edit_feedback_confirmation",
            "ownership_feedback_confirmation",
            "protocol_clarity_confirmation",
            "mechanism_realization_confirmation",
            "evidence_feedback_confirmation",
        }
        and seeds is None
    ):
        seeds = list(sampling_seeds(study, None))
    selected_seeds = sampling_seeds(study, seeds)
    if serving_environment is not None and (
        study
        not in {
            "focused_prompt_confirmation",
            "readable_prompt_confirmation",
            "completion_feedback_confirmation",
            "public_coverage_confirmation",
            "coverage_decision_confirmation",
            "edit_feedback_confirmation",
            "ownership_feedback_confirmation",
            "protocol_clarity_confirmation",
            "mechanism_realization_confirmation",
            "evidence_feedback_confirmation",
        }
        or set(serving_environment) != {"platform", "serving_image_sha256"}
        or serving_environment["platform"] != "jetstream2-h100x1-docker"
    ):
        raise ValueError("serving override requires the focused Jetstream study")
    if (repair_source is not None) != (study == "refinement_confirmation"):
        raise ValueError(
            "refinement confirmation requires its saved repair source only"
        )
    if study == "refinement_confirmation" and bookkeeping_policy != "legacy":
        raise ValueError("refinement policies are frozen separately by task")
    if study in CHECKLIST_STUDIES:
        if bookkeeping_policy not in {"legacy", CHECKLIST_STUDIES[study]}:
            raise ValueError("checklist study fixes its minimal policy")
        bookkeeping_policy = CHECKLIST_STUDIES[study]
    bookkeeping.validate_policy(
        bookkeeping_policy,
        "minimal"
        if study in CHECKLIST_STUDIES
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
            **(
                {"repair_requests": focused.REPAIR_REQUESTS}
                if study
                in {
                    "focused_prompt_confirmation",
                    "readable_prompt_confirmation",
                    "completion_feedback_confirmation",
                    "public_coverage_confirmation",
                    "coverage_decision_confirmation",
                    "edit_feedback_confirmation",
                    "ownership_feedback_confirmation",
                    "protocol_clarity_confirmation",
                    "mechanism_realization_confirmation",
                    "evidence_feedback_confirmation",
                }
                else {}
            ),
            **(
                {
                    "repair_requests": protocol_feedback.REPAIR_REQUESTS,
                    "repair_tokens": protocol_feedback.REPAIR_TOKENS,
                    "model_settings": {
                        **old["config"]["model_settings"],
                        "maximum_requests": old["config"]["model_settings"][
                            "maximum_requests"
                        ]
                        + 5,
                        "maximum_total_tokens": old["config"]["model_settings"][
                            "maximum_total_tokens"
                        ]
                        + 131072,
                    },
                }
                if study
                in {
                    "protocol_clarity_confirmation",
                    "mechanism_realization_confirmation",
                    "evidence_feedback_confirmation",
                }
                else {}
            ),
            **({"wall_seconds": 10800} if study != "comparison" else {}),
            **(serving_environment or {}),
        }
    )
    roster = (
        BASIN_CASES
        if study in BASIN_STUDIES
        else topology_confirmation.phase_c_inputs.ROSTER
    )
    training_bundle = (
        load_training_bundle(training_evidence, old["cells"], roster, source_plan=old)
        if training_evidence is not None
        else None
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
        if training_bundle is not None:
            cell["training_evidence"] = training_bundle["cells"][name]
            cell["boundary_contract"] = reviewed_boundary_contract(
                name, brief
            ).model_dump(mode="json")
        cells[name] = cell
    source_tasks = old["tasks"]
    if study != "comparison":
        by_case = {
            t["benchmark_id"]: t
            for t in source_tasks
            if t["seed"] == 0 and t["arm"] == "full"
        }
        source_tasks = [by_case[name] for name in roster]
        if seeds is not None:
            case_names = tuple(roster)
            # Fresh construction, not reuse of any old seed's proposed inventory.
            # Interleave cases across seeds, rotating their order without scores.
            source_tasks = [
                {
                    **by_case[name],
                    "task_id": f"cell{case_names.index(name):02d}_seed{seed}_full",
                    "seed": seed,
                }
                for index, seed in enumerate(selected_seeds)
                for name in (
                    case_names[index % len(case_names) :]
                    + case_names[: index % len(case_names)]
                )
            ]
    blocks = []
    for i, task in enumerate(source_tasks):
        if study == "refinement_confirmation" or study in CHECKLIST_STUDIES:
            blocks.append(
                [
                    {
                        **task,
                        "task_id": f"{task['task_id']}_minimal_"
                        + (
                            "checklist"
                            if study == "variable_checklist_confirmation"
                            else "deferred_interaction"
                            if study == "deferred_interaction_confirmation"
                            else "stage_checks"
                            if study == "stage_check_confirmation"
                            else "evidence_feedback"
                            if study == "evidence_feedback_confirmation"
                            else "mechanism_realizations"
                            if study == "mechanism_realization_confirmation"
                            else "protocol_clarity"
                            if study == "protocol_clarity_confirmation"
                            else "ownership_feedback"
                            if study == "ownership_feedback_confirmation"
                            else "edit_feedback"
                            if study == "edit_feedback_confirmation"
                            else "decisions"
                            if study == "coverage_decision_confirmation"
                            else "coverage"
                            if study == "public_coverage_confirmation"
                            else "completion"
                            if study == "completion_feedback_confirmation"
                            else "readable"
                            if study == "readable_prompt_confirmation"
                            else "focused"
                            if study
                            in {
                                "focused_prompt_confirmation",
                                "readable_prompt_confirmation",
                                "completion_feedback_confirmation",
                                "public_coverage_confirmation",
                                "coverage_decision_confirmation",
                                "edit_feedback_confirmation",
                                "ownership_feedback_confirmation",
                                "protocol_clarity_confirmation",
                                "mechanism_realization_confirmation",
                                "evidence_feedback_confirmation",
                            }
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
                **(
                    {"sampling_seeds": list(selected_seeds)}
                    if seeds is not None
                    else {}
                ),
                "selection_rule": (
                    "all-eight-cases-declared-seeds-full-focused-prompts-1"
                )
                if seeds is not None
                else "eight-fresh-minimal-and-three-saved-repairs-1"
                if study == "refinement_confirmation"
                else "all-eight-cases-seed0-full-variable-checklist-1"
                if study == "variable_checklist_confirmation"
                else "all-eight-cases-seed0-full-stage-checks-1"
                if study == "stage_check_confirmation"
                else "all-eight-cases-seed0-full-deferred-interaction-1"
                if study == "deferred_interaction_confirmation"
                else "all-eight-cases-seed0-full-focused-prompts-1"
                if study
                in {
                    "focused_prompt_confirmation",
                    "readable_prompt_confirmation",
                    "completion_feedback_confirmation",
                    "public_coverage_confirmation",
                    "coverage_decision_confirmation",
                    "edit_feedback_confirmation",
                    "ownership_feedback_confirmation",
                    "protocol_clarity_confirmation",
                    "mechanism_realization_confirmation",
                    "evidence_feedback_confirmation",
                }
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
                **(
                    {
                        "training_evidence_bundle_sha256": training_bundle[
                            "artifact_sha256"
                        ]
                    }
                    if training_bundle is not None
                    else {}
                ),
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
        if plan.get("study") in CHECKLIST_STUDIES
        else "current"
        if plan.get("study") == "prompt_comparison"
        else None,
    )
    for name, cell in plan["cells"].items():
        if config.bookkeeping_policy == evidence_feedback.POLICY:
            complete = construction_training.load_evidence(cell["training_evidence"])
            if complete.context != ValidationContext.model_validate(cell["context"]):
                raise ValueError("complete training evidence context differs")
            boundary = reviewed_boundary_contract(
                name, contract.visible_brief(cell, {"arm": "full"})
            ).model_dump(mode="json")
            if cell.get("boundary_contract") != boundary:
                raise ValueError("reviewed public boundary contract differs")
        elif "training_evidence" in cell or "boundary_contract" in cell:
            raise ValueError("complete evidence is isolated to the evidence policy")
        expected = reviewed_contract(
            name, PublicScientificBrief.model_validate(cell["brief"])
        ).model_dump(mode="json")
        if cell.get("public_graph_contract") != expected:
            raise ValueError("reviewed public graph contract differs")
    study = plan.get("study", "comparison")
    if study not in STUDIES:
        raise ValueError("unknown construction study")
    selected_seeds = sampling_seeds(study, plan.get("sampling_seeds"))
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
        for seed in selected_seeds
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
    if study == "prompt_comparison" or study in CHECKLIST_STUDIES:
        if study in CHECKLIST_STUDIES and (
            config.bookkeeping_policy != CHECKLIST_STUDIES[study]
        ):
            raise ValueError("checklist policy differs")
        expected = {
            (
                case,
                seed,
                "full",
                "joint_adaptive",
                "dedicated",
                family,
                prompts.SCHEDULE,
            )
            for case in roster
            for seed in selected_seeds
            for family in (
                ("minimal",) if study in CHECKLIST_STUDIES else prompts.FAMILIES
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
    before_path = directory / "construction/before_repair.json"
    before = sealed_read(before_path) if before_path.exists() else None
    empty = ledger.Draft().model_dump(mode="json")
    if (
        starting_draft is None
        and before is not None
        and before["event_count"] == 0
        and "mechanism_realizations" in before["draft"]
    ):
        # A first-call preflight stop has no event carrying the policy. Read only
        # the explicit schema, then still require its exact empty transaction prefix.
        realizations.load_draft(before["draft"])
        empty = (
            evidence_feedback.Draft()
            if "external_exchanges" in before["draft"]
            else realizations.Draft()
        ).model_dump(mode="json")
    drafts, used = (
        [starting_draft if starting_draft is not None else empty],
        [],
    )
    for index, path in enumerate(
        sorted((directory / "construction" / "events").glob("*.json"))
    ):
        event = sealed_read(path)
        record = by_hash.get(event["request_hash"])
        if record is None or content_hash(record) != event["record_sha256"]:
            raise ValueError("missing or changed construction call records")
        if (
            index == 0
            and starting_draft is None
            and event.get("bookkeeping_policy")
            in {realization_prompts.POLICY, evidence_feedback.POLICY}
        ):
            drafts[0] = (
                evidence_feedback.Draft()
                if event.get("bookkeeping_policy") == evidence_feedback.POLICY
                else realizations.Draft()
            ).model_dump(mode="json")
        displayed = json.loads(record["request"]["body"]["messages"][1]["content"])
        displayed_matches = (
            displayed.get("model_view")
            == (
                evidence_feedback
                if event.get("bookkeeping_policy") == evidence_feedback.POLICY
                else realization_prompts
                if event.get("bookkeeping_policy") == realization_prompts.POLICY
                else protocol_feedback
                if event.get("bookkeeping_policy") == protocol_feedback.POLICY
                else ownership_feedback
                if event.get("bookkeeping_policy") == ownership_feedback.POLICY
                else edit_feedback
                if event.get("bookkeeping_policy") == edit_feedback.POLICY
                else decisions
                if event.get("bookkeeping_policy") == decisions.POLICY
                else readable
                if event.get("bookkeeping_policy")
                in {readable.POLICY, completion_feedback.POLICY, coverage.POLICY}
                else focused
            ).model_view(realizations.load_draft(event["before"]))
            if event.get("bookkeeping_policy")
            in {
                focused.POLICY,
                readable.POLICY,
                completion_feedback.POLICY,
                coverage.POLICY,
                decisions.POLICY,
                edit_feedback.POLICY,
                ownership_feedback.POLICY,
                protocol_feedback.POLICY,
                realization_prompts.POLICY,
                evidence_feedback.POLICY,
            }
            else displayed.get("current_draft", {}).get("declarations")
            == event["before"]
        )
        if (
            path.name != f"{index:03d}.json"
            or event["index"] != index
            or realizations.load_draft(event["before"])
            != realizations.load_draft(drafts[-1])
            or not displayed_matches
            or event["request_hash"] in used
        ):
            raise ValueError("construction transaction chain differs")
        if event.get("bookkeeping_policy") == evidence_feedback.POLICY:
            questions = displayed.get("compatibility_questions", [])
            if event.get("compatibility_questions") != questions:
                raise ValueError(
                    "compatibility question ledger differs from displayed evidence"
                )
            reply = visible_response(record) if event["accepted"] else {}
            decisions_ = reply.get("compatibility_decisions", [])
            if event.get("compatibility_decisions") != decisions_:
                raise ValueError(
                    "compatibility decision ledger differs from accepted reply"
                )
            schedules.validate_advisory_decisions(
                [
                    evidence_feedback.CompatibilityDecision.model_validate(d)
                    for d in decisions_
                ],
                questions,
            )
        drafts.append(event["after"])
        used.append(event["request_hash"])
    if before is not None:
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
                **(
                    {
                        "training_evidence": cell["training_evidence"],
                        "boundary_contract": cell["boundary_contract"],
                    }
                    if config.bookkeeping_policy == evidence_feedback.POLICY
                    else {}
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


def seed_results(rows: list[dict]) -> list[dict]:
    """Report every planned repetition without treating missing work as failure."""
    labels = sorted(
        {
            (
                r["benchmark_id"],
                r.get("prompt_family", r["policy"]),
                r["arm"],
                r["process_question"],
            )
            for r in rows
        }
    )
    result = []
    for case, policy, arm, placement in labels:
        selected = [
            r
            for r in rows
            if (
                r["benchmark_id"],
                r.get("prompt_family", r["policy"]),
                r["arm"],
                r["process_question"],
            )
            == (case, policy, arm, placement)
        ]
        done = [r for r in selected if r["status"] != "pending"]
        result.append(
            {
                "benchmark_id": case,
                "policy": policy,
                "arm": arm,
                "process_question": placement,
                "planned": len(selected),
                "finished": len(done),
                "status_counts": dict(Counter(r["status"] for r in selected)),
                "initial_complete": sum(
                    r["initial_complete"] is True for r in selected
                ),
                "final_complete": sum(
                    r["status"] == "topology_complete" for r in selected
                ),
                "tokens": spread([r["cost"]["observed_tokens"] for r in done]),
                "calls": spread([r["cost"]["physical_requests"] for r in done]),
                "seeds": [
                    {k: r[k] for k in ("task_id", "seed", "status", "initial_complete")}
                    for r in sorted(selected, key=lambda r: r["seed"])
                ],
            }
        )
    return result


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
            **(
                {"budget_stop": (value or {}).get("budget_stop")}
                if plan["config"].get("bookkeeping_policy")
                in {
                    completion_feedback.POLICY,
                    coverage.POLICY,
                    decisions.POLICY,
                    edit_feedback.POLICY,
                    ownership_feedback.POLICY,
                    protocol_feedback.POLICY,
                    realization_prompts.POLICY,
                    evidence_feedback.POLICY,
                }
                else {}
            ),
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
        if plan["config"].get("bookkeeping_policy") == evidence_feedback.POLICY:
            advice = (value or {}).get("assessment", {}).get("training_compatibility")
            row["training_compatibility"] = (
                None
                if advice is None
                else {
                    "review_status": advice["review_status"],
                    "conditional_conflicts": len(advice["advisory_questions"]),
                    "unanswered_questions": len(advice["pending_questions"]),
                    "decisions": advice["decisions"],
                    "evidence_sha256": advice["evidence_sha256"],
                    "scientific_correctness": "unverified",
                }
            )
            row["boundary_checks"] = (
                (value or {}).get("assessment", {}).get("boundary_checks")
            )
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
        "per_case_seed_results": seed_results(rows),
        "groups": groups,
        "observed_total_tokens": sum(r["cost"]["observed_tokens"] for r in rows),
        "physical_requests": sum(r["cost"]["physical_requests"] for r in rows),
        **(
            {
                "budget_stops": [
                    {"task": r["task_id"], **r["budget_stop"]}
                    for r in rows
                    if r.get("budget_stop")
                ]
            }
            if plan["config"].get("bookkeeping_policy")
            in {
                completion_feedback.POLICY,
                coverage.POLICY,
                decisions.POLICY,
                edit_feedback.POLICY,
                ownership_feedback.POLICY,
                protocol_feedback.POLICY,
                realization_prompts.POLICY,
                evidence_feedback.POLICY,
            }
            else {}
        ),
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
    if plan["config"].get("bookkeeping_policy") == evidence_feedback.POLICY:
        advice = [
            r["training_compatibility"] for r in rows if r.get("training_compatibility")
        ]
        result["training_compatibility"] = {
            "assessed_models": len(advice),
            "review_status_counts": dict(Counter(r["review_status"] for r in advice)),
            "models_with_conditional_conflicts": sum(
                r["conditional_conflicts"] > 0 for r in advice
            ),
            "decision_counts": dict(
                Counter(d["disposition"] for r in advice for d in r["decisions"])
            ),
            "unanswered_questions": sum(r["unanswered_questions"] for r in advice),
            "interpretation": "Structural completion and advisory disposition "
            "are separate. A retained approximation or disputed assumption "
            "is not verified scientific compatibility.",
        }
    atomic_json(root / "summary.json", result)
    seed_roster = list(sampling_seeds(plan.get("study"), plan.get("sampling_seeds")))
    lines = [
        "# Phase C construction schedules",
        *(
            [
                "Training conflicts are conditional advice, not fitting results. "
                "Structural completion does not imply all advice was answered; "
                "see training_compatibility in summary.json."
            ]
            if plan["config"].get("bookkeeping_policy") == evidence_feedback.POLICY
            else []
        ),
        "",
        "Eight fresh minimal constructions and three saved repairs; not a ranking."
        if plan.get("study") == "refinement_confirmation"
        else "Fresh variables and topology; no functions or fitting.",
        f"Bookkeeping policy: {result['bookkeeping_policy']}.",
        "Eight fresh minimal cases plus three diagnosed saved repair episodes. "
        "Saved repairs are not fresh-current controls or a matched comparison."
        if plan.get("study") == "refinement_confirmation"
        else f"Eight fresh minimal cases, Full/seeds {seed_roster}: "
        "one model view, specific "
        "feedback, bounded local recovery and up to "
        f"{plan['config']['repair_requests']} global repairs. "
        "Existing scientific acceptance checks remain. Provisional paths are "
        "advisory. Inspect VARIABLES.md and SHARED_PROCESSES.md. "
        "Not a matched-budget ranking against earlier policies."
        if plan.get("study")
        in {
            "focused_prompt_confirmation",
            "readable_prompt_confirmation",
            "completion_feedback_confirmation",
            "public_coverage_confirmation",
            "coverage_decision_confirmation",
            "edit_feedback_confirmation",
            "ownership_feedback_confirmation",
            "protocol_clarity_confirmation",
            "mechanism_realization_confirmation",
            "evidence_feedback_confirmation",
        }
        else "Eight fresh minimal cases, Full/seed 0, with matching-driver questions "
        "deferred to interaction and visible requirement status. Existing structural "
        "checks remain; no new stage wording or destructive-repair guard. Inspect "
        "VARIABLES.md and SHARED_PROCESSES.md. Not a matched ranking."
        if plan.get("study") == "deferred_interaction_confirmation"
        else "Eight fresh minimal cases, Full/seed 0, with variable and local "
        "shared-process completion checks and specific topology editing examples. "
        "Inspect SHARED_PROCESSES.md at each checkpoint; local checks do not "
        "certify scientific correctness or assembly. Not a matched ranking."
        if plan.get("study") == "stage_check_confirmation"
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
    if plan.get("sampling_seeds") is not None:
        lines.extend(
            [
                "",
                "## Results across seeds",
                "",
                "Each cell uses the frozen seed roster. Pending runs are separate; "
                "completion is structural, not scientific correctness. Costs use "
                "terminal runs only. Resolution in one seed is not a general fix.",
                "",
                "| Case | Finished / planned | Before global repair | Final complete "
                "| Tokens median [MAD] |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for group in result["per_case_seed_results"]:
            lines.append(
                f"| {group['benchmark_id']} | {group['finished']}/{group['planned']} "
                f"| {group['initial_complete']} | {group['final_complete']} "
                f"| {group['tokens']['median']} [{group['tokens']['MAD']}] |"
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
    if plan.get("study") in CHECKLIST_STUDIES:
        from autoformalism.research import construction_variable_report

        construction_variable_report.report(root, plan)
    if plan.get("study") in SHARED_STAGE_STUDIES:
        from autoformalism.research import construction_shared_report

        construction_shared_report.report(root, plan)
    return result
