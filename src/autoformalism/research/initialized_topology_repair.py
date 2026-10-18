"""Versioned live initializer handoff at the saved topology boundary."""

from __future__ import annotations

import difflib
import html
import json
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import initialization_handoff_audit as audit
from autoformalism.research.actionable_repair_status import (
    ALL_SEEN_WITNESSES,
    build_state,
)
from autoformalism.search import construction_initialized_repair as api
from autoformalism.search import topology_initialization as initial
from autoformalism.staged_topology import content_hash

PROTOCOL = "saved-topology-initialization-1"
PROTOCOL_V2 = "saved-topology-initialization-2"
PROTOCOL_V3 = "saved-topology-initialization-3"
PROTOCOL_V4 = "saved-topology-initialization-4"
PROTOCOL_V5 = "saved-topology-initialization-5"
PROTOCOL_V6 = "saved-topology-initialization-6"


def refine(source: Path, destination: Path, bindings: dict | None = None) -> dict:
    """Repeat the same pre-initialization starts with versioned runtime fixes.

    Bindings are reviewed public identities, never extracted by name or inferred
    from data. Historical results and plans are read-only.
    """
    from autoformalism.llm.construction import MEASURED_RESERVATION
    from autoformalism.research import saved_topology_repair as saved

    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("write the new plan outside its historical source")
    parent = sealed_read(source / "plan.json")
    if (
        parent.get("protocol") != PROTOCOL
        or parent.get("actionable_policy") != api.POLICY
        or parent.get("test_data_opened") is not False
        or parent["settings"]["maximum_requests"] != saved.MAX_CALLS
        or parent["settings"]["maximum_total_tokens"] != saved.MAX_TOKENS
    ):
        raise ValueError("requires the original public-only initializer plan")
    bindings = bindings or {}
    if set(bindings) - set(parent["starts"]):
        raise ValueError("measured binding refers to an unknown starting draft")
    starts = {}
    for task in parent["tasks"]:
        name = audit._name(task["source_task"])
        audit._name(task["task_id"])
        if name in starts:
            raise ValueError("one confirmation episode per starting draft required")
        original = parent["starts"][name]
        problem = saved.Problem(parent["cells"][task["benchmark_id"]], original["task"])
        draft = api.Draft.model_validate(original["draft"])
        if (
            problem.assess(draft) != original["assessment"]
            or problem.diagnose(draft) != original["diagnostics"]
        ):
            raise ValueError("historical starting evidence differs")
        contract = initial.MeasuredInitialInformationContract.model_validate(
            {
                **original["initial_information_contract"],
                "protocol": "recorded-t0-initial-information-2",
                "measured_initial_bindings": bindings.get(name, []),
            }
        )
        initial.validate_public_bindings(
            draft, problem.brief.scientific_context, contract
        )
        api.handoff_status(
            draft, problem.evidence, contract, initial.InitializerDependencyHandoff()
        )
        starts[name] = {
            **original,
            "initial_information_contract": contract.model_dump(mode="json"),
        }
    if set(starts) != set(parent["starts"]):
        raise ValueError("confirmation must retain every original starting draft")
    return sealed_write(
        destination / "plan.json",
        {
            **{k: v for k, v in parent.items() if k != "artifact_sha256"},
            "protocol": PROTOCOL_V2,
            "actionable_policy": api.POLICY_V2,
            "source_identity": saved.source_identity(),
            "historical_initialization_plan_sha256": parent["artifact_sha256"],
            "witness_policy": ALL_SEEN_WITNESSES,
            "budget_reservation_policy": MEASURED_RESERVATION,
            "starts": starts,
            "reviewed_measured_bindings_sha256": content_hash(bindings),
            "selection": "All original pre-initialization starts, seeds and maximum "
            "budgets, with explicit measured-initial bindings and runtime fixes. "
            "Historical final drafts are not used as new starts; no score selection.",
        },
    )


def freeze(source: Path, destination: Path) -> dict:
    """Use every latest v2 final draft, retaining its actual historical witnesses."""
    from autoformalism.research import saved_topology_repair as saved

    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("write the new plan outside its historical source")
    parent = audit._source(source)
    if parent["protocol"] != saved.PROTOCOL_V2:
        raise ValueError("requires the latest v2 saved repair results")
    starts, tasks = {}, []
    for task in parent["tasks"]:
        name = audit._name(task["task_id"])
        original = parent["starts"][task["source_task"]]
        result = sealed_read(source / "results" / name / "result.json")
        if (
            result["identity"] != saved.namespace(parent, task)
            or result["task"] != task
            or result["source_draft_sha256"] != content_hash(original["draft"])
            or result.get("test_data_opened") is not False
        ):
            raise ValueError("saved result identity differs")
        problem = saved.Problem(parent["cells"][task["benchmark_id"]], original["task"])
        draft = api.Draft.model_validate(result["draft"])
        starts[name] = {
            "task": original["task"],
            "draft": result["draft"],
            "assessment": problem.assess(draft),
            "diagnostics": problem.diagnose(draft),
            "previous_conflicts": audit._saved_conflicts(original, result),
            "source_result_sha256": result["artifact_sha256"],
            "initial_information_contract": audit.information_contract(
                problem.evidence, "causal_runtime"
            ).model_dump(mode="json"),
        }
        tasks.append({**task, "task_id": name + "_initialization", "source_task": name})
    return sealed_write(
        destination / "plan.json",
        {
            "protocol": PROTOCOL,
            "actionable_policy": api.POLICY,
            "source_identity": saved.source_identity(),
            "source_plan_sha256": parent["artifact_sha256"],
            "settings": parent["settings"],
            "cells": parent["cells"],
            "starts": starts,
            "tasks": tasks,
            "selection": "Every saved v2 final draft; no score selection. Latest "
            "historical states are predecessors, not equal-budget controls.",
            "maximum_calls_per_arm": saved.MAX_CALLS,
            "maximum_tokens_per_arm": saved.MAX_TOKENS,
            "initialization_calls_within_total_budget": 3,
            "automatic_followup": False,
            "test_data_opened": False,
            "optimizer_calls": 0,
        },
    )


class Session:
    """Keep the runtime contract, proposer choices and saved witnesses separate."""

    def __init__(self, problem, start, *, policy=api.POLICY):
        self.problem = problem
        if policy not in {
            api.POLICY,
            api.POLICY_V2,
            api.POLICY_V3,
            api.POLICY_V4,
            api.POLICY_V5,
            api.POLICY_V6,
        }:
            raise ValueError("unknown initializer session policy")
        self.policy = policy
        self.contract = initial.validate_information_contract(
            start["initial_information_contract"]
        )
        expected_contract = (
            "recorded-t0-initial-information-2"
            if policy
            in {
                api.POLICY_V2,
                api.POLICY_V3,
                api.POLICY_V4,
                api.POLICY_V5,
                api.POLICY_V6,
            }
            else "recorded-t0-initial-information-1"
        )
        if self.contract.protocol != expected_contract:
            raise ValueError("initializer policy and information contract differ")
        initial.validate_public_bindings(
            api.Draft.model_validate(start["draft"]),
            problem.brief.scientific_context,
            self.contract,
        )
        self.handoff = initial.InitializerDependencyHandoff()
        if policy in {api.POLICY_V3, api.POLICY_V4, api.POLICY_V5, api.POLICY_V6}:
            self.handoff = initial.InitializerDependencyHandoff.model_validate(
                start["initializer_handoff"]
            )
            api.handoff_status(
                api.Draft.model_validate(start["draft"]),
                problem.evidence,
                self.contract,
                self.handoff,
            )
        self.baseline = start.get("previous_conflicts", start["diagnostics"])
        self.initialization_calls = 0

    def diagnose(self, draft, tracked):
        """Retest old matched prefixes after every topology or initializer edit."""
        from autoformalism.research import construction_training as training

        questions = [
            *self.baseline.get(
                "tracked_conflicts", self.baseline.get("advisory_questions", ())
            ),
            *tracked.get("tracked_conflicts", tracked.get("advisory_questions", ())),
        ]
        unique = {content_hash(q): q for q in questions}
        return audit.qualified_feedback(
            draft,
            self.problem.evidence,
            self.contract,
            self.handoff,
            family=True,
            previous_questions=list(unique.values()),
            **(
                {"identity_policy": training.MATHEMATICAL_QUESTION_IDENTITY}
                if self.policy in {api.POLICY_V4, api.POLICY_V5, api.POLICY_V6}
                else {}
            ),
            **(
                {"initial_time_policy": initial.INITIAL_TIME_POLICY}
                if self.policy
                in {api.POLICY_V3, api.POLICY_V4, api.POLICY_V5, api.POLICY_V6}
                else {}
            ),
        )

    def status(self, draft):
        initial.validate_public_bindings(
            draft, self.problem.brief.scientific_context, self.contract
        )
        return api.handoff_status(
            draft, self.problem.evidence, self.contract, self.handoff
        )

    def state(self, draft, report, tracked, eligible, answers):
        return api.qualify_completion(
            build_state(
                report,
                tracked,
                eligible,
                answers,
                **(
                    {"witness_policy": ALL_SEEN_WITNESSES}
                    if self.policy
                    in {
                        api.POLICY_V2,
                        api.POLICY_V3,
                        api.POLICY_V4,
                        api.POLICY_V5,
                        api.POLICY_V6,
                    }
                    else {}
                ),
            ),
            self.status(draft),
        )

    def stage(self, draft):
        """Give dedicated choices three turns, then permit joint scientific repairs."""
        if not self.status(draft)["complete"] and self.initialization_calls < 3:
            return "initialization"
        return "repair"

    def packet_arguments(self):
        return {
            "evidence": self.problem.evidence,
            "initial_contract": self.contract,
            "handoff": self.handoff,
            **(
                {"policy": self.policy}
                if self.policy
                in {
                    api.POLICY_V2,
                    api.POLICY_V3,
                    api.POLICY_V4,
                    api.POLICY_V5,
                    api.POLICY_V6,
                }
                else {}
            ),
        }

    def result(self, draft, check):
        return {
            "initializer_handoff": self.handoff.model_dump(mode="json"),
            "initial_information_contract": self.contract.model_dump(mode="json"),
            "initialization_status": self.status(draft),
            "interaction_source": api.interaction_source(
                self.problem.brief,
                draft,
                check,
                self.problem.evidence,
                self.contract,
                self.handoff,
            ),
        }


def preview(plan: dict, output: Path) -> dict:
    """Show exact first requests and the joint repair contract, without generation."""
    from autoformalism.research import saved_topology_repair as saved
    from autoformalism.search import construction_actionable_repair as previous

    output.mkdir(parents=True, exist_ok=True)
    links = []
    for number, task in enumerate(plan["tasks"]):
        start = plan["starts"][task["source_task"]]
        focused = plan["protocol"] in {
            PROTOCOL_V3,
            PROTOCOL_V4,
            PROTOCOL_V5,
            PROTOCOL_V6,
        }
        problem = saved.Problem(plan["cells"][task["benchmark_id"]], start["task"])
        draft = api.Draft.model_validate(start["draft"])
        session = Session(problem, start, policy=plan["actionable_policy"])
        report = session.diagnose(draft, session.baseline)
        check = problem.assess(draft, qualified_memory=True)
        state = session.state(draft, report, session.baseline, check["eligible"], {})
        packets = {}
        for stage in {session.stage(draft), "repair"}:
            packets[stage] = saved.make_packet(
                api,
                problem,
                draft,
                start.get("last_edit") if focused else None,
                state,
                start.get("repetition_feedback") if focused else None,
                0,
                0,
                stage=stage,
                initialization=session,
            )
        old_state = build_state(
            start["diagnostics"], start["diagnostics"], check["eligible"]
        )
        old = saved.make_packet(
            previous,
            problem,
            draft,
            None,
            old_state,
            None,
            0,
            0,
            policy=previous.POLICY_V2,
        )
        if plan["protocol"] == PROTOCOL_V2:
            old_contract = {
                key: value
                for key, value in start["initial_information_contract"].items()
                if key != "measured_initial_bindings"
            }
            old_contract["protocol"] = "recorded-t0-initial-information-1"
            old_session = Session(
                problem, {**start, "initial_information_contract": old_contract}
            )
            old_report = old_session.diagnose(draft, old_session.baseline)
            old_state = old_session.state(
                draft, old_report, old_session.baseline, check["eligible"], {}
            )
            old = saved.make_packet(
                api,
                problem,
                draft,
                None,
                old_state,
                None,
                0,
                0,
                stage="repair",
                initialization=old_session,
            )
        if focused:
            old_session = Session(
                problem,
                start,
                policy=api.POLICY_V5
                if plan["protocol"] == PROTOCOL_V6
                else api.POLICY_V4
                if plan["protocol"] == PROTOCOL_V5
                else api.POLICY_V3
                if plan["protocol"] == PROTOCOL_V4
                else api.POLICY_V2,
            )
            old_session.handoff = session.handoff
            old_report = old_session.diagnose(draft, old_session.baseline)
            old_state = old_session.state(
                draft, old_report, old_session.baseline, check["eligible"], {}
            )
            old = saved.make_packet(
                api,
                problem,
                draft,
                start.get("last_edit"),
                old_state,
                start.get("repetition_feedback"),
                0,
                0,
                stage="repair",
                initialization=old_session,
            )
        diff = difflib.HtmlDiff(wrapcolumn=100).make_file(
            json.dumps(old, indent=2).splitlines(),
            json.dumps(packets["repair"], indent=2).splitlines(),
            fromdesc="Previous policy on the same saved model",
            todesc=plan["actionable_policy"],
            context=True,
            numlines=2,
        )
        (output / f"changes-{number:02d}.html").write_text(diff)
        content = [
            f"<h1>{html.escape(task['task_id'])}</h1>",
            f"<p>First live stage: {session.stage(draft)}. "
            "Repair preview uses the same "
            "saved model before any new reply; it is not a predicted response.</p>",
            "<h2>System</h2><pre>"
            + html.escape(saved.request_system(api, packets["repair"]))
            + "</pre>",
        ]
        for stage, packet in sorted(packets.items()):
            for label, value in (
                (stage, packet),
                (
                    stage + " response schema",
                    api.response_model(stage).model_json_schema(),
                ),
            ):
                content.append(
                    f"<details open><summary>{html.escape(label)}</summary><pre>"
                    f"{html.escape(json.dumps(value, indent=2))}</pre></details>"
                )
        (output / f"request-{number:02d}.html").write_text(
            '<!doctype html><meta charset="utf-8"><style>body{font:16px/1.5 system-ui;'
            "max-width:1100px;margin:35px auto}pre{white-space:pre-wrap;"
            "overflow-wrap:anywhere;"
            "background:#f1f5f9;padding:15px}details{margin:15px 0}</style>"
            + "\n".join(content)
        )
        (output / f"request-{number:02d}.json").write_text(
            json.dumps(
                {
                    "system": saved.request_system(api, packets["repair"]),
                    "first_stage": session.stage(draft),
                    "packets": packets,
                },
                indent=2,
            )
            + "\n"
        )
        links.append(
            f"<li>{html.escape(task['task_id'])}: "
            f'<a href="request-{number:02d}.html">requests</a> · '
            f'<a href="changes-{number:02d}.html">marked repair changes</a></li>'
        )
    (output / "PROMPTS.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        "<title>Initialization topology prompts</title>"
        "<style>body{font:16px/1.6 system-ui;max-width:1050px;margin:35px auto}</style>"
        "<h1>Initial dependencies at the topology boundary</h1>"
        + (
            "<p>V6 bookkeeping ACES retest: two saved final cases. Consistent "
            "runtime initializer records preserve numerical feedback. Explicit "
            "state removal or retyping automatically drops only inherited "
            "initializer records. Rejected attempts are compared by their "
            "assembled effects; truncated replies remain unavailable responses, "
            "not scientific stagnation. Marked changes compare V5 on the same "
            "saved starts. Historical results remain unchanged.</p>"
            if plan["protocol"] == PROTOCOL_V6
            else "<p>V5 evidence-feedback ACES retest: two saved final cases. "
            "Paired conditioning measurements and observed outputs are labeled "
            "explicitly. Unsupported initializer feedback names available "
            "information; equivalent unchanged initializers do not reset "
            "repetition. Historical policies retain exact replay.</p>"
            if plan["protocol"] == PROTOCOL_V5
            else "<p>V4 equality-feedback ACES retest: two explicitly selected saved "
            "cases. Current model equalities precede complete witnesses and "
            "assumptions. Description-only edits cannot reset scientific progress. "
            "Historical results remain unchanged; this is a separately budgeted "
            "diagnostic experiment.</p>"
            if plan["protocol"] == PROTOCOL_V4
            else "<p>V3 focused ACES retest: three explicitly selected saved cases. "
            "Feedback and equations precede complete public background. Repeated "
            "identical reply records are displayed with counts. Initial-time "
            "algebraic checks stop at differential states and use their declared "
            "initializers, rather than importing later ODE dependencies.</p>"
            if plan["protocol"] == PROTOCOL_V3
            else "<p>V2: repeated structural detail is displayed once; every distinct "
            "trajectory witness is retained. Reviewed direct initial measurements "
            "are fixed identities, not fitted values. Serving-token reservations "
            "replace byte estimates; all maximum budgets remain unchanged.</p>"
            if plan["protocol"] == PROTOCOL_V2
            else ""
        )
        + "<p>New: choose shared initial values or permitted initial-measurement "
        "dependencies "
        "before interaction formulas. Dependencies are distinct from ODE arguments. "
        "Missing choices remain unfinished. Joint repairs validate both records "
        "atomically. "
        "Previously matched trajectory prefixes remain in feedback after edits.</p>"
        "<p>Frozen saved drafts; no score selection. "
        "At most three initial-dependency calls within ten total calls per episode. "
        "No fitting, test access, or scientific-correctness claim. No generation "
        "performed "
        "to produce this preview.</p><ul>" + "\n".join(links) + "</ul>"
    )
    return {
        "requests_previewed": len(links),
        "llm_calls": 0,
        "path": str(output / "PROMPTS.html"),
    }
