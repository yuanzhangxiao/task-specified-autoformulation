"""Bounded, paired repair of frozen public topology drafts; no fitting or selection."""

from __future__ import annotations

import hashlib
import html
import json
import statistics
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.llm.construction import (
    MEASURED_RESERVATION,
    ConstructionClient,
    PromptPreflightError,
)
from autoformalism.llm.staged_topology import (
    StagedModelSettings,
    atomic_json,
    strict_provider_schema,
    visible_response,
)
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research import construction_comparison as campaign
from autoformalism.research import construction_training as training
from autoformalism.research import repair_stagnation
from autoformalism.research.actionable_repair_status import build_state
from autoformalism.search import construction_actionable_repair as actionable
from autoformalism.search import construction_evidence_repair as control
from autoformalism.search import construction_initialized_repair as initialized
from autoformalism.search import construction_protocol as protocol
from autoformalism.search import construction_stage_checks as checks
from autoformalism.search import mechanism_realizations as realizations
from autoformalism.search.construction_schedules import validate_advisory_decisions
from autoformalism.search.public_graph_obligations import PublicGraphContract
from autoformalism.search.training_evidence import TrainingEvidence, evidence_brief
from autoformalism.staged_topology import content_hash

PROTOCOL = "saved-topology-repair-comparison-1"
PROTOCOL_V2 = "saved-topology-repair-feedback-2"
PROTOCOL_INITIALIZATION = "saved-topology-initialization-1"
PROTOCOL_INITIALIZATION_V2 = "saved-topology-initialization-2"
PROTOCOL_INITIALIZATION_V3 = "saved-topology-initialization-3"
PROTOCOL_INITIALIZATION_V4 = "saved-topology-initialization-4"
PROTOCOL_INITIALIZATION_V5 = "saved-topology-initialization-5"
PROTOCOL_INITIALIZATION_V6 = "saved-topology-initialization-6"
INITIALIZATION_PROTOCOLS = {
    PROTOCOL_INITIALIZATION,
    PROTOCOL_INITIALIZATION_V2,
    PROTOCOL_INITIALIZATION_V3,
    PROTOCOL_INITIALIZATION_V4,
    PROTOCOL_INITIALIZATION_V5,
    PROTOCOL_INITIALIZATION_V6,
}
ARMS = ("control", "actionable")
REPO = Path(__file__).resolve().parents[3]
MAX_CALLS = 10
MAX_TOKENS = 262144


def source_identity() -> str:
    """Bind both package code and the dedicated entry points on every resume."""
    return content_hash(
        {
            "package": runtime_source_hash(),
            "entry_points": {
                name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                for name in (
                    "scripts/saved_topology_repair.py",
                    "scripts/jetstream/run_saved_topology_repair.sh",
                    "scripts/jetstream_construction_campaign.py",
                    "scripts/jetstream_construction_probe.py",
                    "scripts/jetstream_server_record.py",
                )
            },
        }
    )


class Problem:
    """Typed public context shared by both arms of one frozen starting draft."""

    def __init__(self, cell: dict, task: dict):
        self.brief = campaign.contract.visible_brief(cell, task)
        self.context = ValidationContext.model_validate(cell["context"])
        self.targets = campaign.contract.target_definitions(cell)
        self.graph = PublicGraphContract.model_validate(cell["public_graph_contract"])
        self.boundary = control.BoundaryContract.model_validate(
            cell["boundary_contract"]
        )
        self.evidence = training.load_evidence(cell["training_evidence"])
        if self.evidence.context != self.context:
            raise ValueError("training evidence context differs")
        self.graph.validate_public(self.brief)
        self.boundary.validate_public(self.brief)
        self.enriched = evidence_brief(
            self.brief.model_dump(mode="json"),
            self.context,
            TrainingEvidence.model_validate(cell["evidence"]),
        )
        self.highlights = training.initial_highlights(self.evidence)

    def assess(self, draft: control.Draft, *, qualified_memory: bool = False) -> dict:
        """Reproduce the full scientific-structure checks used by construction."""
        result = control.assess(
            self.brief,
            self.context,
            self.targets,
            draft,
            graph_contract=self.graph,
            boundary_contract=self.boundary,
            clarify_overlaps=False,
        )
        result["interaction_review_handoff"] = control.interaction_handoff(
            self.brief, draft
        )
        result["mechanism_realization_handoff"] = control.final_handoff(draft, result)
        variables = realizations.variable_checklist(
            self.brief, draft, self.targets, self.graph
        )
        result["variable_checklist"] = variables
        result["errors"].extend(
            {"code": "variable_declaration", "item": item}
            for item in variables["items"]
            if item["status"] in {"missing", "inconsistent"}
        )
        result["eligible"] &= variables["status"] == "ready"
        result["shared_process_checklist"] = checks.shared_checklist(self.brief, draft)
        result["eligible"] &= result["shared_process_checklist"]["status"] == "ready"
        if qualified_memory:
            result = realizations.qualify_required_memory_paths(
                self.brief, draft, result
            )
            result["mechanism_realization_handoff"] = control.final_handoff(
                draft, result
            )
            result["mechanism_realization_handoff"][
                "required_memory_path_qualification"
            ] = result["required_memory_path_qualification"]
        return result

    def diagnose(self, draft: control.Draft) -> dict:
        """Keep diagnostic assumptions and witnesses unchanged across arms."""
        return training.compatibility_feedback(
            control.project(draft),
            self.evidence,
            identity_policy=training.SEMANTIC_QUESTION_IDENTITY,
        )


def freeze(source: Path, destination: Path) -> dict:
    """Select all structural failures or conditional conflicts without ranking."""
    parent = sealed_read(source / "plan.json")
    if (
        parent.get("test_data_opened") is not False
        or parent["config"]["bookkeeping_policy"] != control.POLICY
    ):
        raise ValueError("requires the public-only evidence-repair-v2 source")
    starts, source_rows, cells, tasks = {}, [], {}, []
    for task in parent["tasks"]:
        name = task["task_id"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("unsafe source task")
        folder = source / "results" / name
        saved = sealed_read(folder / "proposal.json")
        campaign.checked_records(folder, campaign.baseline.namespace(parent, task))
        problem = Problem(parent["cells"][task["benchmark_id"]], task)
        draft = control.Draft.model_validate(saved["draft"])
        check, report = problem.assess(draft), problem.diagnose(draft)
        if check != {
            k: v
            for k, v in saved["assessment"].items()
            if k != "training_compatibility"
        }:
            raise ValueError(f"saved structural assessment differs: {name}")
        if (
            report["advisory_questions"]
            != saved["assessment"]["training_compatibility"]["advisory_questions"]
        ):
            raise ValueError(f"saved conditional evidence differs: {name}")
        selected = not check["eligible"] or bool(report["advisory_questions"])
        source_rows.append(
            {
                "task": name,
                "selected": selected,
                "structural_eligible": check["eligible"],
                "conditional_questions": len(report["advisory_questions"]),
                "source_proposal_sha256": saved["artifact_sha256"],
            }
        )
        if not selected:
            continue
        starts[name] = {
            "task": task,
            "draft": saved["draft"],
            "assessment": check,
            "diagnostics": report,
            "source_proposal_sha256": saved["artifact_sha256"],
        }
        cells[task["benchmark_id"]] = parent["cells"][task["benchmark_id"]]
        # Counterbalance arm order, but use the same sampling seed at each call index.
        for arm in ARMS if task["seed"] % 2 == 0 else reversed(ARMS):
            tasks.append(
                {
                    "task_id": f"{task['matched_task_id']}_{arm}",
                    "source_task": name,
                    "arm": arm,
                    "seed": task["seed"],
                    "benchmark_id": task["benchmark_id"],
                }
            )
    settings = {
        **parent["config"]["model_settings"],
        "maximum_requests": MAX_CALLS,
        "maximum_total_tokens": MAX_TOKENS,
    }
    StagedModelSettings.model_validate(settings)
    return sealed_write(
        destination / "plan.json",
        {
            "protocol": PROTOCOL,
            "source_identity": source_identity(),
            "source_plan_sha256": parent["artifact_sha256"],
            "settings": settings,
            "cells": cells,
            "starts": starts,
            "tasks": tasks,
            "source_rows": source_rows,
            "selection": "All saved final drafts with structural failures or "
            "conditional training conflicts; no fit/score selection.",
            "maximum_calls_per_arm": MAX_CALLS,
            "maximum_tokens_per_arm": MAX_TOKENS,
            "historical_answers_reused": False,
            "automatic_followup": False,
            "test_data_opened": False,
            "optimizer_calls": 0,
        },
    )


def refreeze(source: Path, destination: Path) -> dict:
    """Start a separate v2 repair on the same frozen starts, without historical calls.

    This is a fresh bounded qualification, not a budget reset or continuation of
    the original campaign. Original assessments and data remain in the plan.
    """
    parent = sealed_read(source / "plan.json")
    if parent["protocol"] != PROTOCOL or parent.get("test_data_opened") is not False:
        raise ValueError("requires an original public-only saved repair plan")
    tasks = [task for task in parent["tasks"] if task["arm"] == "actionable"]
    if len(tasks) != len(parent["starts"]) or {t["source_task"] for t in tasks} != set(
        parent["starts"]
    ):
        raise ValueError("original starts must have exactly one actionable task each")
    settings = StagedModelSettings.model_validate(parent["settings"])
    if (
        settings.maximum_requests != MAX_CALLS
        or settings.maximum_total_tokens != MAX_TOKENS
    ):
        raise ValueError("original repair allowances differ")
    for task in tasks:
        if Path(task["task_id"]).name != task["task_id"] or task["task_id"] in {
            ".",
            "..",
        }:
            raise ValueError("unsafe source task")
        start = parent["starts"][task["source_task"]]
        problem = Problem(parent["cells"][task["benchmark_id"]], start["task"])
        draft = control.Draft.model_validate(start["draft"])
        if (
            problem.assess(draft) != start["assessment"]
            or problem.diagnose(draft) != start["diagnostics"]
        ):
            raise ValueError("historical start assessment differs")
    return sealed_write(
        destination / "plan.json",
        {
            **{k: v for k, v in parent.items() if k != "artifact_sha256"},
            "protocol": PROTOCOL_V2,
            "actionable_policy": actionable.POLICY_V2,
            "source_identity": source_identity(),
            "historical_repair_plan_sha256": parent["artifact_sha256"],
            "tasks": tasks,
            "selection": "Same saved starts as the original paired repair; fresh v2 "
            "actionable episodes only. Historical arms are not rerun "
            "or equal-cost controls.",
        },
    )


def verify(root: Path) -> dict:
    """Require unchanged source and sealed public inputs before any call."""
    plan = sealed_read(root / "plan.json")
    if (
        plan["protocol"] not in {PROTOCOL, PROTOCOL_V2, *INITIALIZATION_PROTOCOLS}
        or plan["source_identity"] != source_identity()
        or plan["test_data_opened"] is not False
    ):
        raise ValueError("saved repair execution identity differs")
    if (
        plan["protocol"] == PROTOCOL_V2
        and plan.get("actionable_policy") != actionable.POLICY_V2
    ):
        raise ValueError("saved repair policy differs")
    if plan["protocol"] in INITIALIZATION_PROTOCOLS and (
        plan.get("actionable_policy")
        != (
            initialized.POLICY_V6
            if plan["protocol"] == PROTOCOL_INITIALIZATION_V6
            else initialized.POLICY_V5
            if plan["protocol"] == PROTOCOL_INITIALIZATION_V5
            else initialized.POLICY_V4
            if plan["protocol"] == PROTOCOL_INITIALIZATION_V4
            else initialized.POLICY_V3
            if plan["protocol"] == PROTOCOL_INITIALIZATION_V3
            else initialized.POLICY_V2
            if plan["protocol"] == PROTOCOL_INITIALIZATION_V2
            else initialized.POLICY
        )
        or plan.get("initialization_calls_within_total_budget") != 3
        or plan["settings"]["maximum_requests"] != MAX_CALLS
        or plan["settings"]["maximum_total_tokens"] != MAX_TOKENS
    ):
        raise ValueError("initializer repair policy or allowance differs")
    if plan["protocol"] in {
        PROTOCOL_INITIALIZATION_V2,
        PROTOCOL_INITIALIZATION_V3,
        PROTOCOL_INITIALIZATION_V4,
        PROTOCOL_INITIALIZATION_V5,
        PROTOCOL_INITIALIZATION_V6,
    } and (
        plan.get("budget_reservation_policy") != MEASURED_RESERVATION
        or plan.get("witness_policy") != "all-seen-witnesses-1"
    ):
        raise ValueError("initializer qualification policy differs")
    if plan["protocol"] in {
        PROTOCOL_INITIALIZATION_V3,
        PROTOCOL_INITIALIZATION_V4,
        PROTOCOL_INITIALIZATION_V5,
        PROTOCOL_INITIALIZATION_V6,
    }:
        from autoformalism.search.topology_initialization import INITIAL_TIME_POLICY

        if (
            plan.get("initial_time_policy") != INITIAL_TIME_POLICY
            or plan.get("execution_platform") != "aces-h100x1"
            or not 1 <= len(plan["tasks"]) <= 3
        ):
            raise ValueError("focused initializer retest policy differs")
    if plan["protocol"] in {
        PROTOCOL_INITIALIZATION_V4,
        PROTOCOL_INITIALIZATION_V5,
        PROTOCOL_INITIALIZATION_V6,
    }:
        from autoformalism.research import construction_training as training

        if (
            plan.get("question_identity_policy")
            != training.MATHEMATICAL_QUESTION_IDENTITY
            or len(plan["tasks"]) > 2
        ):
            raise ValueError("equality retest identity policy or roster differs")
    if plan["protocol"] in {
        PROTOCOL_INITIALIZATION_V5,
        PROTOCOL_INITIALIZATION_V6,
    } and (
        plan.get("budget_policy") != "separate-new-retest-budget-1"
        or plan.get("maximum_calls_per_arm") != MAX_CALLS
        or plan.get("maximum_tokens_per_arm") != MAX_TOKENS
        or plan.get("automatic_followup") is not False
        or plan.get("optimizer_calls") != 0
        or "provider_recovery_policy" in plan
    ):
        raise ValueError("evidence retest must declare a separate bounded allowance")
    if "provider_recovery_policy" in plan:
        from autoformalism.research.initialization_provider_recovery import (
            validate_plan,
        )

        validate_plan(plan)
    return plan


def namespace(plan: dict, task: dict) -> str:
    """Separate caches without changing paired sampling seeds."""
    return content_hash([plan["artifact_sha256"], task])


def control_state(report: dict, answers: dict) -> dict:
    """Reproduce the existing control's completion rule, without erasing conflicts."""
    questions = report["advisory_questions"]
    pending = [
        q
        for q in questions
        if answers.get(q["question_id"], {}).get("disposition")
        not in {"retain_approximation", "dispute_assumption"}
    ]
    return {
        **report,
        "pending_questions": pending,
        "decisions": [
            answers[q["question_id"]] for q in questions if q["question_id"] in answers
        ],
        "review_status": "unanswered"
        if pending
        else "addressed"
        if questions
        else "no_conflict_identified",
        "scientific_correctness": "unverified",
    }


def semantic_signature(draft: control.Draft, check: dict, report: dict) -> str:
    """Ignore prose-only changes while retaining topology, roles and reference edits."""

    def remove_prose(value):
        if isinstance(value, dict):
            return {
                k: remove_prose(v)
                for k, v in value.items()
                if k
                not in {
                    "scientific_role",
                    "scientific_meaning",
                    "explanation",
                    "reference_sha256",
                }
            }
        if isinstance(value, list):
            return [remove_prose(v) for v in value]
        return value

    return content_hash(
        [
            remove_prose(draft.model_dump(mode="json")),
            check["eligible"],
            sorted(q["question_id"] for q in report["advisory_questions"]),
        ]
    )


def make_packet(
    api,
    problem: Problem,
    draft,
    last,
    state,
    repetition,
    count: int,
    charged: int,
    *,
    policy: str | None = None,
    stage: str = "repair",
    initialization=None,
) -> dict:
    """Use the same complete public context; never truncate scientific evidence."""
    packet = api.payload(
        stage,
        problem.brief,
        problem.context,
        problem.targets,
        problem.enriched,
        draft,
        problem.graph,
        last,
        repetition.get("continuation") if repetition else None,
        boundary_contract=problem.boundary,
        training_highlights=problem.highlights,
        compatibility_questions=state["pending_questions"],
        compatibility_state=state,
        **(
            {"repetition_feedback": repetition}
            if api in {actionable, initialized}
            else {}
        ),
        **({"policy": policy} if api is actionable and policy is not None else {}),
        **(
            {"qualified_assessment": problem.assess(draft, qualified_memory=True)}
            if api is initialized
            or (api is actionable and policy == actionable.POLICY_V2)
            else {}
        ),
        **(initialization.packet_arguments() if initialization is not None else {}),
    )
    control.repair_budget(packet, MAX_CALLS, count, MAX_TOKENS, charged, MAX_TOKENS)
    if initialization is not None and initialization.policy in {
        initialized.POLICY_V3,
        initialized.POLICY_V4,
        initialized.POLICY_V5,
        initialized.POLICY_V6,
    }:
        from autoformalism.search.construction_initialization_focus import project

        packet = project(packet, draft)
        if initialization.policy == initialized.POLICY_V6:
            packet = initialized.bookkeeping_packet(packet)
        if initialization.policy in {
            initialized.POLICY_V4,
            initialized.POLICY_V5,
            initialized.POLICY_V6,
        }:
            from autoformalism.research.initialization_equality_feedback import (
                project as equality_project,
            )

            packet = equality_project(
                packet,
                draft,
                **(
                    {"consistent_runtime_rules": True}
                    if initialization.policy == initialized.POLICY_V6
                    else {}
                ),
            )
        if initialization.policy in {initialized.POLICY_V5, initialized.POLICY_V6}:
            from autoformalism.research.initialization_evidence_feedback import (
                project as evidence_project,
            )

            packet = evidence_project(
                packet,
                draft,
                problem.evidence,
                **(
                    {"consistent_runtime_rules": True}
                    if initialization.policy == initialized.POLICY_V6
                    else {}
                ),
            )
        packet["prompt_policy"] = initialization.policy
    return packet


def request_system(api, packet: dict) -> str:
    """Select versioned instructions without changing older cached requests."""
    return api.system_for(packet) if api is initialized else api.SYSTEM


def request_user(packet: dict) -> str:
    """Preserve the focused policy's feedback-first ordering in actual requests."""
    return json.dumps(
        packet,
        sort_keys=packet.get("prompt_policy")
        not in {
            initialized.POLICY_V3,
            initialized.POLICY_V4,
            initialized.POLICY_V5,
            initialized.POLICY_V6,
        },
    )


def expected_request(client, api, packet: dict, index: int, stage="repair") -> dict:
    """Verify resumed requests against exact settings, schema, seed and predecessor."""
    schema = api.response_model(stage)
    step = f"{stage}_{index:03d}"
    return {
        "protocol": "scientific-staged-topology-1",
        "namespace": client.namespace,
        "settings": client.settings.model_dump(mode="json"),
        **client.request_policy(),
        "body": {
            "model": client.settings.model,
            "messages": [
                {"role": "system", "content": request_system(api, packet)},
                {"role": "user", "content": request_user(packet)},
            ],
            "stream": False,
            "reasoning_effort": client.settings.reasoning_effort,
            "temperature": client.settings.temperature,
            "max_tokens": client.settings.max_output_tokens,
            "seed": int(content_hash([client.seed, step, index])[:8], 16) % (2**31),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": strict_provider_schema(schema.model_json_schema()),
                },
            },
        },
    }


def run_episode(
    root: Path, plan: dict, task: dict, *, base_url: str, **client_kwargs
) -> dict:
    """Replay cached transactions exactly, then spend only the remaining allowance."""
    directory = root / "results" / task["task_id"]
    identity = namespace(plan, task)
    cached = _cache_records(directory / "calls", identity)
    recovery = plan.get("provider_recovery_policy") is not None
    inherited_cost, offset = _cost([]), 0
    settings = StagedModelSettings.model_validate(plan["settings"])
    if recovery:
        from autoformalism.research.initialization_provider_recovery import (
            cumulative_cost,
            validate_plan,
        )

        validate_plan(plan)
        predecessor = plan["inherited_provider_outcomes"][task["task_id"]]
        inherited_cost = predecessor["cost"]
        offset = predecessor["physical_request_offset"]
        settings = StagedModelSettings.model_validate(
            {
                **plan["settings"],
                "maximum_requests": MAX_CALLS - offset,
                "maximum_total_tokens": MAX_TOKENS - inherited_cost["budget_charge"],
            }
        )
    client = ConstructionClient(
        settings=settings,
        directory=directory / "calls",
        namespace=identity,
        seed=task["seed"],
        base_url=base_url,
        **(
            {"reservation_policy": MEASURED_RESERVATION}
            if plan.get("protocol")
            in {
                PROTOCOL_INITIALIZATION_V2,
                PROTOCOL_INITIALIZATION_V3,
                PROTOCOL_INITIALIZATION_V4,
                PROTOCOL_INITIALIZATION_V5,
                PROTOCOL_INITIALIZATION_V6,
            }
            else {}
        ),
        **client_kwargs,
    )
    client.budget_diagnostics_enabled = True
    start = plan["starts"][task["source_task"]]
    problem = Problem(plan["cells"][task["benchmark_id"]], start["task"])
    draft = control.Draft.model_validate(start["draft"])
    api = actionable if task["arm"] == "actionable" else control
    initialization = None
    if plan.get("protocol") in INITIALIZATION_PROTOCOLS:
        from autoformalism.research.initialized_topology_repair import Session

        api = initialized
        initialization = Session(problem, start, policy=plan["actionable_policy"])
    v2 = initialization is not None or (
        plan.get("actionable_policy") == actionable.POLICY_V2 and api is actionable
    )
    answers, records, events = {}, [], []
    last, repetition = None, None
    if plan.get("protocol") in {
        PROTOCOL_INITIALIZATION_V3,
        PROTOCOL_INITIALIZATION_V4,
        PROTOCOL_INITIALIZATION_V5,
        PROTOCOL_INITIALIZATION_V6,
    }:
        last = start.get("last_edit")
        repetition = start.get("repetition_feedback")
    unchanged = 0
    previous_signature = None
    stop = "repair_budget_exhausted"
    check, report = problem.assess(draft), problem.diagnose(draft)
    if check != start["assessment"] or report != start["diagnostics"]:
        raise ValueError("frozen starting assessment differs")
    if v2:
        check = problem.assess(draft, qualified_memory=True)
    baseline = (
        start["diagnostics"] if initialization is None else initialization.baseline
    )
    if initialization is not None:
        report = initialization.diagnose(draft, baseline)
    tracked = build_state(report, baseline, check["eligible"], answers)
    if initialization is not None:
        tracked = initialization.state(
            draft, report, baseline, check["eligible"], answers
        )
    for index in range(offset, MAX_CALLS):
        state = (
            build_state(report, tracked, check["eligible"], answers)
            if api is not control
            else control_state(report, answers)
        )
        stage = "repair"
        if initialization is not None:
            state = initialization.state(
                draft, report, tracked, check["eligible"], answers
            )
            stage = initialization.stage(draft)
        packet = make_packet(
            api,
            problem,
            draft,
            last,
            state,
            repetition,
            index,
            inherited_cost["budget_charge"] + _cost(records)["budget_charge"],
            policy=plan.get("actionable_policy") if v2 else None,
            stage=stage,
            initialization=initialization,
        )
        event_path = directory / "events" / f"{index:03d}.json"
        prior = sealed_read(event_path) if event_path.exists() else None
        request = expected_request(client, api, packet, index, stage)
        if prior is not None:
            record = next(
                (r for r in cached if r["request_hash"] == prior["request_hash"]), None
            )
            if record is None or content_hash(record) != prior["record_sha256"]:
                raise ValueError("saved repair call missing or changed")
        else:
            try:
                record = client.call(
                    system=request_system(api, packet),
                    user=request_user(packet),
                    response_model=api.response_model(stage),
                    step=f"{stage}_{index:03d}",
                    attempt=index,
                )
            except (RepairBudgetExceeded, PromptPreflightError) as exc:
                stop = f"preflight_stop: {exc}"
                break
        if record["request"] != request:
            raise ValueError("saved request differs from exact repair predecessor")
        records.append(record)
        before, before_check, before_report = draft, check, report
        before_initialization = (
            initialization.handoff if initialization is not None else None
        )
        # Transport success is not a complete scientific reply. In V6 a truncated
        # or absent/unparseable final JSON stops the episode without consuming a
        # scientific-stagnation step. The measured call remains in the ledger.
        response_error = None
        available_raw = None
        if plan.get("protocol") == PROTOCOL_INITIALIZATION_V6:
            try:
                available_raw = visible_response(record)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                response_error = str(exc)[:6000]
        unavailable = (
            recovery
            or plan.get("protocol")
            in {
                PROTOCOL_INITIALIZATION_V4,
                PROTOCOL_INITIALIZATION_V5,
                PROTOCOL_INITIALIZATION_V6,
            }
        ) and record["status"] != "responded"
        if unavailable or response_error is not None:
            event = sealed_write(
                event_path,
                {
                    "identity": identity,
                    "index": index,
                    "request_hash": record["request_hash"],
                    "record_sha256": content_hash(record),
                    "before": draft.model_dump(mode="json"),
                    "after": draft.model_dump(mode="json"),
                    "accepted": False,
                    "stage_complete": False,
                    "error": None,
                    "transport_status": record["status"],
                    "transport_error": record.get("error"),
                    **(
                        {"response_error": response_error}
                        if plan.get("protocol") == PROTOCOL_INITIALIZATION_V6
                        else {}
                    ),
                    "scientific_reply_received": False,
                    "structural_assessment": check,
                    "scientific_state": state,
                    "semantic_model_unchanged": True,
                    "consecutive_unchanged": unchanged,
                    "retained_last_edit": last,
                    "retained_repetition_feedback": repetition,
                    "stage": stage,
                    "initialization_before": initialization.handoff.model_dump(
                        mode="json"
                    ),
                    "initialization_after": initialization.handoff.model_dump(
                        mode="json"
                    ),
                },
            )
            events.append(event)
            stop = "provider_unavailable"
            break
        if initialization is not None and stage == "initialization":
            initialization.initialization_calls += 1
        raw, error, accepted, complete = None, None, False, False
        normalized = None
        attempted_candidate = None
        normalizations, submitted = [], []
        try:
            raw = (
                available_raw
                if plan.get("protocol") == PROTOCOL_INITIALIZATION_V6
                else visible_response(record)
            )
            normalized, normalizations = api.normalize_reply(
                problem.brief,
                raw,
                graph_contract=problem.graph,
                draft=draft,
                ignore_definition_description=True,
                **(
                    {"boundary_contract": problem.boundary}
                    if api is not control
                    else {}
                ),
                **({"stage": stage} if initialization is not None else {}),
            )
            patch = api.DraftPatch.model_validate(normalized)
            submitted = validate_advisory_decisions(
                patch.compatibility_decisions, state["pending_questions"]
            )
            protocol.controls.validate_patch(patch, brief=problem.brief)
            candidate = api.apply_patch(problem.brief, draft, patch)
            candidate, consumer_log = api.normalize_consumers(candidate, patch, draft)
            normalizations.extend(consumer_log)
            if initialization is not None:
                initialized.initial.validate_public_bindings(
                    candidate, problem.brief.scientific_context, initialization.contract
                )
                if initialization.policy == initialized.POLICY_V6:
                    from autoformalism.research.initialization_bookkeeping import (
                        cleanup_inherited_handoff,
                    )

                    # Capture only a fully assembled topology alternative. If its
                    # initializer is invalid, this is still not a committed draft.
                    attempted_candidate = candidate
                    next_handoff = cleanup_inherited_handoff(
                        draft, candidate, initialization.handoff, patch
                    )
                    initialized.handoff_status(
                        candidate,
                        problem.evidence,
                        initialization.contract,
                        next_handoff,
                    )
                    if patch.initialization is None:
                        removed = sorted(
                            {r.state for r in initialization.handoff.records}
                            - {r.state for r in next_handoff.records}
                        )
                        if removed:
                            normalizations.append(
                                {
                                    "code": "inherited_initializers_removed",
                                    "states": removed,
                                "reason": "explicit variable removal or retyping "
                                "to algebraic",
                                }
                            )
                else:
                    next_handoff = initialized.changed_handoff(
                        candidate,
                        problem.evidence,
                        initialization.contract,
                        initialization.handoff,
                        patch,
                    )
                initialization.handoff = next_handoff
            draft = candidate
            answers.update({d["question_id"]: d for d in submitted})
            accepted, complete = True, patch.stage_complete
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)[:6000]
        check = (
            problem.assess(draft, qualified_memory=True)
            if v2
            else problem.assess(draft)
        )
        report = (
            problem.diagnose(draft)
            if initialization is None
            else initialization.diagnose(draft, tracked)
        )
        after_state = build_state(report, tracked, check["eligible"], answers)
        if initialization is not None:
            after_state = initialization.state(
                draft, report, tracked, check["eligible"], answers
            )
        tracked = after_state
        scientific_signature = (
            repair_stagnation.model_signature if v2 else semantic_signature
        )
        same = scientific_signature(
            before, before_check, before_report
        ) == scientific_signature(draft, check, report)
        if initialization is not None:
            if initialization.policy in {initialized.POLICY_V5, initialized.POLICY_V6}:
                same &= repair_stagnation.initializer_semantics(
                    before_initialization.model_dump(mode="json")
                ) == repair_stagnation.initializer_semantics(
                    initialization.handoff.model_dump(mode="json")
                )
            else:
                same &= before_initialization == initialization.handoff
        lost = control.lost_required_predicates(before_check, check) if accepted else []
        last = {
            "status": "accepted" if accepted else "rejected",
            "transaction_applied": accepted,
            "error": error,
            "stage_complete": complete,
            "compatibility_decisions": submitted if accepted else [],
            "required_predicates_lost": lost,
            "edit_effects": control.edit_effects(before, draft) if accepted else None,
            "rejected_reply": raw if not accepted else None,
            "normalizations": normalizations,
            **(
                {
                    "initialization_before": before_initialization.model_dump(
                        mode="json"
                    ),
                    "initialization_after": initialization.handoff.model_dump(
                        mode="json"
                    ),
                }
                if initialization is not None
                else {}
            ),
        }
        progress = not same
        if api is control:
            progress = protocol.substantive(before.model_dump(mode="json")) != (
                protocol.substantive(draft.model_dump(mode="json"))
            )
        if api is control and accepted:
            resolved = {
                decision["question_id"]
                for decision in submitted
                if decision["disposition"]
                in {"retain_approximation", "dispute_assumption"}
            }
            progress |= bool(
                resolved & {q["question_id"] for q in state["pending_questions"]}
            )
        signature = (
            repair_stagnation.attempt_signature(
                draft, check, report, error=error, normalized=normalized, raw=raw
            )
            if v2
            else content_hash(control.repair_signature(draft, last, check["errors"]))
        )
        if initialization is not None:
            proposed_initialization = (
                (normalized or raw or {}).get("initialization")
                if isinstance(normalized or raw, dict)
                else None
            )
            retained_initialization = initialization.handoff.model_dump(mode="json")
            if initialization.policy in {initialized.POLICY_V5, initialized.POLICY_V6}:
                if (
                    initialization.policy == initialized.POLICY_V6
                    and error is not None
                    and attempted_candidate is not None
                ):
                    from autoformalism.research.initialization_bookkeeping import (
                        rejected_initializer_attempt_signature,
                    )

                    signature = rejected_initializer_attempt_signature(
                        draft,
                        check,
                        report,
                        attempted_candidate,
                        retained=retained_initialization,
                        proposed=proposed_initialization,
                        error=error,
                    )
                else:
                    signature = repair_stagnation.initializer_attempt_signature(
                        signature,
                        retained_initialization,
                        proposed_initialization,
                        error=error,
                    )
            else:
                signature = content_hash(
                    [signature, retained_initialization, proposed_initialization]
                )
        unchanged = (
            (unchanged + 1 if signature == previous_signature else 1)
            if not progress
            else 0
        )
        previous_signature = signature
        event = sealed_write(
            event_path,
            {
                "identity": identity,
                "index": index,
                "request_hash": record["request_hash"],
                "record_sha256": content_hash(record),
                "before": before.model_dump(mode="json"),
                "after": draft.model_dump(mode="json"),
                "accepted": accepted,
                "stage_complete": complete,
                "error": error,
                "normalizations": normalizations,
                "decisions": submitted if accepted else [],
                "structural_assessment": check,
                "scientific_state": after_state,
                "semantic_model_unchanged": same,
                "consecutive_unchanged": unchanged,
                "required_predicates_lost": lost,
                **(
                    {
                        "stage": stage,
                        "initial_information_contract": (
                            initialization.contract.model_dump(mode="json")
                        ),
                        "initialization_before": before_initialization.model_dump(
                            mode="json"
                        ),
                        "initialization_after": initialization.handoff.model_dump(
                            mode="json"
                        ),
                    }
                    if initialization is not None
                    else {}
                ),
            },
        )
        events.append(event)
        repetition = (
            {
                "consecutive_unchanged": unchanged,
                "continuation": {
                    "no_progress": True,
                    "repeated_semantic_attempts": unchanged,
                    "instruction": "The last reply made no substantive "
                    "declaration or advisory "
                    "progress. Follow the repair_work status: record corrections "
                    "do not require equation changes. Answer pending scientific "
                    "questions or explicitly finish when ready; use topology edits "
                    "when scientifically justified. Rewording descriptions does "
                    "not reset the repetition counter.",
                },
                "previous_reply": raw,
                "previous_error": error,
                "remaining_error_codes": sorted({e["code"] for e in check["errors"]}),
                "instruction": "The preceding reply did not change the model or "
                "resolve the displayed checks. Do not repeat it; use a different "
                "concrete edit addressing the remaining error.",
            }
            if unchanged
            else None
        )
        if v2 and repetition:
            repetition["continuation"]["instruction"] = (
                "The retained model and unresolved obligations have not changed. "
                "Completion toggles, descriptions, and redundant restatements do not "
                "reset this counter. Correct the displayed record or propose a "
                "different scientific edit; finish only when repair_work allows it."
            )
        if api is control:
            done = (
                check["eligible"]
                and complete
                and not control_state(report, answers)["pending_questions"]
            )
        else:
            done = complete and after_state["completion_allowed"]
        if done:
            stop = "completed_requested_work"
            break
        if (
            api is not control
            and check["eligible"]
            and not after_state["actionable_questions"]
            and after_state["repair_status"] == "review_deferred"
            and complete
            and (initialization is None or initialization.status(draft)["complete"])
        ):
            stop = "scientific_review_deferred"
            break
        if unchanged >= 3 and stage != "initialization":
            stop = "three_repeated_repairs"
            break
    all_records = _cache_records(directory / "calls", identity)
    if {r["request_hash"] for r in all_records} != {r["request_hash"] for r in records}:
        raise ValueError("unaccounted cached repair request")
    if len(list((directory / "events").glob("*.json"))) != len(events):
        raise ValueError("saved event tail differs")
    final_state = build_state(report, tracked, check["eligible"], answers)
    if initialization is not None:
        final_state = initialization.state(
            draft, report, tracked, check["eligible"], answers
        )
    return sealed_write(
        directory / "result.json",
        {
            "identity": identity,
            "task": task,
            "source_draft_sha256": content_hash(start["draft"]),
            "draft": draft.model_dump(mode="json"),
            "assessment": check,
            "scientific_state": final_state,
            "stop_reason": stop,
            "event_count": len(events),
            "cost": cumulative_cost(inherited_cost, _cost(all_records))
            if recovery
            else _cost(all_records),
            "status": "provider_unavailable"
            if stop == "provider_unavailable"
            else "assessed",
            "optimizer_calls": 0,
            "test_data_opened": False,
            "scientific_accuracy_established": False,
            **(
                {
                    "provider_recovery_policy": plan["provider_recovery_policy"],
                    "physical_request_offset": offset,
                    "inherited_cost": inherited_cost,
                    "new_cost": _cost(all_records),
                    "cumulative_cost": cumulative_cost(
                        inherited_cost, _cost(all_records)
                    ),
                    "provider_availability": "unavailable"
                    if stop == "provider_unavailable"
                    else "responded"
                    if any(r["status"] == "responded" for r in records)
                    else "not_attempted",
                }
                if recovery
                else {}
            ),
            **(
                initialization.result(draft, check)
                if initialization is not None
                else {}
            ),
        },
    )


def report(root: Path, plan: dict) -> dict:
    """Report structural and scientific outcomes separately, including partial cost."""
    recovery = plan.get("provider_recovery_policy") is not None
    if recovery:
        from autoformalism.research.initialization_provider_recovery import (
            cumulative_cost,
            cumulative_costs,
            validate_plan,
        )

        validate_plan(plan)
    rows, parts = (
        [],
        [
            "# Saved topology repair comparison",
            "",
            "Paired frozen drafts; equal maximum budgets. No fitting or test access.",
            "A removed diagnostic is not fitted accuracy. "
            "Deferred evidence is not a pass.",
            "",
            "| Task | Arm | Structure | Scientific checks | Calls | Tokens |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ],
    )
    cards = []
    if plan.get("protocol") == PROTOCOL_V2:
        parts[2] = (
            "Fresh v2 repairs of the same frozen starts; historical controls "
            "are not rerun. No fitting or test access."
        )
    elif plan.get("protocol") in INITIALIZATION_PROTOCOLS:
        parts[2] = (
            "Initialization dependencies and joint repairs of saved v2 final drafts; "
            "historical parents have different budgets. No fitting or test access."
        )
    if plan.get("protocol") in {
        PROTOCOL_INITIALIZATION_V4,
        PROTOCOL_INITIALIZATION_V5,
        PROTOCOL_INITIALIZATION_V6,
    }:
        parts[2] = (
            "Equality feedback on saved final drafts; historical costs are retained "
            "separately. New cap: ten calls / 262,144 tokens per case. "
            "No fitting or test access."
        )
    if recovery:
        parts[2] = (
            "Provider recovery retains every failed attempt and its reservation "
            "within the original 10-call / 262,144-token allowance."
        )
        parts[5] = (
            "| Task | Arm | Structure | Scientific checks | Availability | "
            "Inherited calls | New calls | Total calls | Charged tokens |"
        )
        parts[6] = "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"
    for task in plan["tasks"]:
        folder = root / "results" / task["task_id"]
        records = _cache_records(folder / "calls", namespace(plan, task))
        new_cost = _cost(records)
        inherited_cost = (
            plan["inherited_provider_outcomes"][task["task_id"]]["cost"]
            if recovery
            else _cost([])
        )
        total_cost = cumulative_cost(inherited_cost, new_cost) if recovery else new_cost
        value = (
            sealed_read(folder / "result.json")
            if (folder / "result.json").exists()
            else None
        )
        if value:
            expected_identity = namespace(plan, task)
            expected_draft = content_hash(plan["starts"][task["source_task"]]["draft"])
            if (
                value["identity"] != expected_identity
                or value["task"] != task
                or value["source_draft_sha256"] != expected_draft
            ):
                raise ValueError("result identity differs")
            if value["cost"] != total_cost or (
                recovery
                and (
                    value["inherited_cost"] != inherited_cost
                    or value["new_cost"] != new_cost
                    or value["cumulative_cost"] != total_cost
                )
            ):
                raise ValueError("result accounting differs")
        row = {
            "task": task["task_id"],
            "arm": task["arm"],
            "source_task": task["source_task"],
            "status": value["status"]
            if value
            and (
                recovery
                or plan.get("protocol")
                in {
                    PROTOCOL_INITIALIZATION_V4,
                    PROTOCOL_INITIALIZATION_V5,
                    PROTOCOL_INITIALIZATION_V6,
                }
            )
            else "assessed"
            if value
            else "partial"
            if records
            else "pending",
            "structural_eligible": value["assessment"]["eligible"] if value else None,
            "scientific_status": value["scientific_state"]["repair_status"]
            if value
            else None,
            "stop_reason": value["stop_reason"] if value else None,
            "cost": total_cost,
        }
        if plan.get("protocol") in {
            PROTOCOL_INITIALIZATION_V4,
            PROTOCOL_INITIALIZATION_V5,
            PROTOCOL_INITIALIZATION_V6,
        }:
            row["historical_cost"] = plan["starts"][task["source_task"]].get(
                "historical_cost"
            )
            row["cost_scope"] = (
                "New retest attempts only; historical cost kept separately."
            )
        if recovery:
            row.update(
                inherited_cost=inherited_cost,
                new_cost=new_cost,
                cumulative_cost=total_cost,
                provider_availability=value["provider_availability"]
                if value
                else "pending",
            )
        if plan.get("protocol") in INITIALIZATION_PROTOCOLS:
            row["initialization_complete"] = (
                value["initialization_status"]["complete"] if value else None
            )
            row["initialization_unresolved_states"] = (
                value["initialization_status"]["unresolved_states"] if value else None
            )
            row["interaction_source_available"] = (
                value["interaction_source"] is not None if value else None
            )
        if plan.get("protocol") in {PROTOCOL_V2, *INITIALIZATION_PROTOCOLS}:
            row["diagnostic_status_counts"] = (
                dict(
                    Counter(
                        d["status"]
                        for d in value["scientific_state"]["full_report"]["diagnostics"]
                    )
                )
                if value
                else None
            )
            row["checks_clear_scope"] = (
                "A checks_clear status means no tracked repair obligation remains; "
                "deferred or inapplicable diagnostics are not scientific passes."
            )
        rows.append(row)
        parts.append(
            f"| {row['task']} | {row['arm']} | {row['structural_eligible']} "
            f"| {row['scientific_status']} | "
            + (
                f"{row['provider_availability']} | "
                f"{inherited_cost['physical_requests']} | "
                f"{new_cost['physical_requests']} | "
                f"{total_cost['physical_requests']} | "
                f"{total_cost['budget_charge']} |"
                if recovery
                else f"{row['cost']['physical_requests']} | "
                f"{row['cost']['observed_tokens']} |"
            )
        )
        if value:
            model = control.model_view(control.Draft.model_validate(value["draft"]))
            details = {
                "model": model,
                "scientific_state": value["scientific_state"],
                "structural_errors": value["assessment"]["errors"],
                **(
                    {
                        "provider_availability": row["provider_availability"],
                        "inherited_cost": inherited_cost,
                        "new_cost": new_cost,
                        "cumulative_cost": total_cost,
                    }
                    if recovery
                    else {}
                ),
                **(
                    {"initialization_status": value["initialization_status"]}
                    if plan.get("protocol") in INITIALIZATION_PROTOCOLS
                    else {}
                ),
            }
            cards.append(
                f"<details><summary>{html.escape(row['task'])}: "
                f"{html.escape(row['scientific_status'])}</summary><pre>"
                f"{html.escape(json.dumps(details, indent=2))}</pre></details>"
            )
    groups = {}
    for arm in (
        ("actionable",)
        if plan.get("protocol") in {PROTOCOL_V2, *INITIALIZATION_PROTOCOLS}
        else ARMS
    ):
        selected = [r for r in rows if r["arm"] == arm]
        measured = [
            r["cost"]["observed_tokens"] for r in selected if r["status"] == "assessed"
        ]
        median = statistics.median(measured) if measured else None
        groups[arm] = {
            "status_counts": dict(Counter(r["status"] for r in selected)),
            "structural_eligible": sum(
                r["structural_eligible"] is True for r in selected
            ),
            "scientific_status_counts": dict(
                Counter(
                    "not_assessed"
                    if r["scientific_status"] is None
                    else r["scientific_status"]
                    for r in selected
                )
            ),
            "median_tokens": median,
            "unscaled_mad_tokens": statistics.median(abs(x - median) for x in measured)
            if measured
            else None,
        }
    summary = {
        "protocol": plan.get("protocol", PROTOCOL),
        "plan_sha256": plan["artifact_sha256"],
        "planned_episodes": len(rows),
        "groups": groups,
        "rows": rows,
        "physical_requests": sum(r["cost"]["physical_requests"] for r in rows),
        "observed_tokens": sum(r["cost"]["observed_tokens"] for r in rows),
        "unmeasured_requests": sum(
            r["cost"]["requests_with_unknown_usage"] for r in rows
        ),
        "optimizer_calls": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        **(
            {
                "cost_scope": "New retest attempts only; historical costs in rows.",
                "maximum_new_requests": MAX_CALLS * len(rows),
                "maximum_new_charged_tokens": MAX_TOKENS * len(rows),
            }
            if plan.get("protocol")
            in {
                PROTOCOL_INITIALIZATION_V4,
                PROTOCOL_INITIALIZATION_V5,
                PROTOCOL_INITIALIZATION_V6,
            }
            else {}
        ),
        **(
            {
                "provider_recovery_policy": plan["provider_recovery_policy"],
                "inherited_cost": cumulative_costs([r["inherited_cost"] for r in rows]),
                "new_cost": cumulative_costs([r["new_cost"] for r in rows]),
                "cumulative_cost": cumulative_costs([r["cost"] for r in rows]),
            }
            if recovery
            else {}
        ),
        **(
            {
                "initialization_complete": sum(
                    r["initialization_complete"] is True for r in rows
                ),
                "interaction_sources_available": sum(
                    r["interaction_source_available"] is True for r in rows
                ),
            }
            if plan.get("protocol") in INITIALIZATION_PROTOCOLS
            else {}
        ),
    }
    atomic_json(root / "summary.json", summary)
    (root / "SUMMARY.md").write_text("\n".join(parts) + "\n")
    (root / "REPORT.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Saved repair results</title>'
        "<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "details{border:1px solid #ccc;padding:15px;margin:12px}</style>"
        "<h1>Saved repair results</h1><p>Structural validity, conditional "
        "conflicts, and deferred evidence are distinct. No fitting or "
        "scientific recovery claim.</p>" + "\n".join(cards)
    )
    return summary
