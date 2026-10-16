"""Versioned live initializer handoff at the saved topology boundary."""

from __future__ import annotations

import difflib
import html
import json
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import initialization_handoff_audit as audit
from autoformalism.research.actionable_repair_status import build_state
from autoformalism.search import construction_initialized_repair as api
from autoformalism.search import topology_initialization as initial
from autoformalism.staged_topology import content_hash

PROTOCOL = "saved-topology-initialization-1"


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

    def __init__(self, problem, start):
        self.problem = problem
        self.contract = initial.InitialInformationContract.model_validate(
            start["initial_information_contract"]
        )
        self.handoff = initial.InitializerDependencyHandoff()
        self.baseline = start.get("previous_conflicts", start["diagnostics"])
        self.initialization_calls = 0

    def diagnose(self, draft, tracked):
        """Retest old matched prefixes after every topology or initializer edit."""
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
        )

    def status(self, draft):
        return api.handoff_status(
            draft, self.problem.evidence, self.contract, self.handoff
        )

    def state(self, draft, report, tracked, eligible, answers):
        return api.qualify_completion(
            build_state(report, tracked, eligible, answers), self.status(draft)
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
        problem = saved.Problem(plan["cells"][task["benchmark_id"]], start["task"])
        draft = api.Draft.model_validate(start["draft"])
        session = Session(problem, start)
        report = session.diagnose(draft, session.baseline)
        check = problem.assess(draft, qualified_memory=True)
        state = session.state(draft, report, session.baseline, check["eligible"], {})
        packets = {}
        for stage in {session.stage(draft), "repair"}:
            packets[stage] = saved.make_packet(
                api,
                problem,
                draft,
                None,
                state,
                None,
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
        diff = difflib.HtmlDiff(wrapcolumn=100).make_file(
            json.dumps(old, indent=2).splitlines(),
            json.dumps(packets["repair"], indent=2).splitlines(),
            fromdesc="Previous repair policy on the same saved model",
            todesc="Initialization-aware repair on the same saved model",
            context=True,
            numlines=2,
        )
        (output / f"changes-{number:02d}.html").write_text(diff)
        content = [
            f"<h1>{html.escape(task['task_id'])}</h1>",
            f"<p>First live stage: {session.stage(draft)}. "
            "Repair preview uses the same "
            "saved model before any new reply; it is not a predicted response.</p>",
            f"<h2>System</h2><pre>{html.escape(api.SYSTEM)}</pre>",
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
                    "system": api.SYSTEM,
                    "first_stage": session.stage(draft),
                    "packets": packets,
                },
                indent=2,
            )
            + "\n"
        )
        links.append(
            f'<li>{html.escape(task["task_id"])}: '
            f'<a href="request-{number:02d}.html">requests</a> · '
            f'<a href="changes-{number:02d}.html">marked repair changes</a></li>'
        )
    (output / "PROMPTS.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        '<title>Initialization topology prompts</title>'
        "<style>body{font:16px/1.6 system-ui;max-width:1050px;margin:35px auto}</style>"
        "<h1>Initial dependencies at the topology boundary</h1>"
        "<p>New: choose shared initial values or permitted initial-measurement "
        "dependencies "
        "before interaction formulas. Dependencies are distinct from ODE arguments. "
        "Missing choices remain unfinished. Joint repairs validate both records "
        "atomically. "
        "Previously matched trajectory prefixes remain in feedback after edits.</p>"
        "<p>Every saved v2 final draft; no score selection. "
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
