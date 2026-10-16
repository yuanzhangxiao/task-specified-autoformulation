#!/usr/bin/env python3
"""Freeze, preview, run and report a matched repair of saved topology drafts."""

from __future__ import annotations

import argparse
import difflib
import html
import json
import signal
import time
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research import saved_topology_repair as campaign
from autoformalism.research.actionable_repair_status import build_state
from autoformalism.search import construction_actionable_repair as actionable
from autoformalism.search import construction_evidence_repair as control
from scripts.jetstream_construction_campaign import compatibility, live_server
from scripts.jetstream_construction_probe import BASE_URL


def preview(root: Path, output: Path) -> dict:
    """Render the exact first request in each arm, including schema and feedback."""
    plan = campaign.verify(root)
    if plan["protocol"] in campaign.INITIALIZATION_PROTOCOLS:
        from autoformalism.research.initialized_topology_repair import preview as render

        return render(plan, output)
    output.mkdir(parents=True, exist_ok=True)
    rows, requests_by_source = [], {}
    v2 = plan.get("actionable_policy") == actionable.POLICY_V2
    for task in plan["tasks"]:
        start = plan["starts"][task["source_task"]]
        problem = campaign.Problem(plan["cells"][task["benchmark_id"]], start["task"])
        api = actionable if task["arm"] == "actionable" else control
        draft = control.Draft.model_validate(start["draft"])
        check = (
            problem.assess(draft, qualified_memory=True) if v2 else start["assessment"]
        )
        state = (
            build_state(
                start["diagnostics"],
                start["diagnostics"],
                check["eligible"],
            )
            if api is actionable
            else campaign.control_state(start["diagnostics"], {})
        )
        packet = campaign.make_packet(
            api,
            problem,
            draft,
            None,
            state,
            None,
            0,
            0,
            policy=plan.get("actionable_policy") if v2 else None,
        )
        request = {
            "system": api.SYSTEM,
            "user": packet,
            "response_schema": api.response_model("repair").model_json_schema(),
        }
        name = task["task_id"]
        requests_by_source.setdefault(task["source_task"], {})[task["arm"]] = request
        if v2:
            old_state = build_state(
                start["diagnostics"],
                start["diagnostics"],
                start["assessment"]["eligible"],
            )
            requests_by_source[task["source_task"]]["control"] = {
                **request,
                "user": campaign.make_packet(
                    actionable, problem, draft, None, old_state, None, 0, 0
                ),
            }
        (output / f"{name}.json").write_text(json.dumps(request, indent=2) + "\n")
        parts = [
            f"<h1>{html.escape(name)}</h1>",
            "<p>Actual first request from a frozen saved draft. "
            "No generation performed.</p>",
        ]
        for label, value in request.items():
            text = value if isinstance(value, str) else json.dumps(value, indent=2)
            parts.append(f"<h2>{html.escape(label)}</h2><pre>{html.escape(text)}</pre>")
        (output / f"{name}.html").write_text(
            '<!doctype html><meta charset="utf-8"><style>'
            "body{font:16px/1.5 system-ui;max-width:1100px;margin:40px auto}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;"
            "background:#f1f5f9;padding:18px}</style>" + "\n".join(parts)
        )
        rows.append(f'<li><a href="{name}.html">{name}</a></li>')
    comparisons = []
    for number, (source, arms) in enumerate(requests_by_source.items()):
        filename = f"changes-{number:02d}.html"
        marked = difflib.HtmlDiff(wrapcolumn=100).make_file(
            json.dumps(arms["control"], indent=2).splitlines(),
            json.dumps(arms["actionable"], indent=2).splitlines(),
            fromdesc="Actionable v1" if v2 else "Existing feedback",
            todesc="Actionable v2" if v2 else "Actionable repair",
            context=True,
            numlines=3,
        )
        (output / filename).write_text(marked)
        comparisons.append(f'<li><a href="{filename}">{html.escape(source)}</a></li>')
    introduction = (
        "V2 uses the same frozen starting drafts, public evidence, seeds and maximum "
        "budgets. Only the revised actionable arm runs; historical arms are retained "
        "separately. Marked comparisons show v1 versus v2 first requests. New feedback "
        "locates inconsistent mechanism records, explains unchanged closed-subsystem "
        "restrictions, and forbids completion as a substitute for unresolved repair. "
        "Required-memory paths are qualified against returning through their driver. "
        "Initial construction prompts are unchanged."
        if v2
        else "The control retains the old advisory completion rule. The actionable "
        "arm requests concrete corrections and treats unsupported assumption "
        "challenges as unresolved. Both begin with identical saved equations "
        "and evidence. Changed: clear conditional conflicts require edits; unsupported "
        "challenges remain unresolved; boundary references use exact handles. Initial "
        "construction prompts are unchanged."
    )
    (output / "PROMPTS.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Saved repair prompts</title>'
        "<style>body{font:16px/1.6 system-ui;max-width:1000px;"
        "margin:40px auto}</style><h1>Saved topology repair: "
        + ("v2 feedback" if v2 else "paired prompts")
        + "</h1><p>"
        + html.escape(introduction)
        + "</p><h2>Marked changes</h2><ul>"
        + "\n".join(comparisons)
        + "</ul><h2>Complete first requests</h2><ul>"
        + "\n".join(rows)
        + "</ul>"
    )
    return {
        "requests_previewed": len(rows),
        "llm_calls": 0,
        "path": str(output / "PROMPTS.html"),
    }


def run(inputs: Path, root: Path, work: Path, requests: Path) -> dict:
    """Reuse only the verified local server, with checkpoints and bounded resume."""
    plan = campaign.verify(inputs)
    provenance = compatibility(work, requests)
    live_server()
    sealed_write(
        root / "plan.json", {k: v for k, v in plan.items() if k != "artifact_sha256"}
    )
    sealed_write(
        root / "execution.json",
        {
            "protocol": plan["protocol"],
            "plan_sha256": plan["artifact_sha256"],
            "source_identity": plan["source_identity"],
            "compatibility": provenance,
            "base_url": BASE_URL,
            "test_data_opened": False,
            "automatic_followup": False,
        },
    )
    draining = False
    deadline = time.monotonic() + 10800

    def stop(signum, frame):
        nonlocal draining
        draining = True

    handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for task in plan["tasks"]:
            try:
                campaign.run_episode(
                    root,
                    plan,
                    task,
                    base_url=BASE_URL,
                    can_start=lambda: not draining
                    and time.monotonic() + 200 < deadline,
                )
            except DeferredCall:
                break
            campaign.report(root, plan)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        summary = campaign.report(root, plan)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "freeze",
            "refreeze",
            "initialize",
            "refine-initialization",
            "preview",
            "run",
            "report",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--measured-bindings", type=Path)
    args = parser.parse_args()
    with _lock(args.root / "campaign-lock"):
        if args.command in {
            "freeze",
            "refreeze",
            "initialize",
            "refine-initialization",
        }:
            if args.source is None:
                parser.error("freezing a plan requires --source")
            freezer = (
                campaign.refreeze if args.command == "refreeze" else campaign.freeze
            )
            if args.command == "initialize":
                from autoformalism.research.initialized_topology_repair import freeze

                freezer = freeze
            if args.command == "refine-initialization":
                from autoformalism.research.initialized_topology_repair import refine

                bindings = (
                    json.loads(args.measured_bindings.read_text())
                    if args.measured_bindings
                    else None
                )
                plan = refine(args.source, args.root, bindings)
            else:
                plan = freezer(args.source, args.root)
            value = {
                "identity": plan["artifact_sha256"],
                "saved_drafts": len(plan["starts"]),
                "repair_episodes": len(plan["tasks"]),
                "llm_calls": 0,
            }
        elif args.command == "preview":
            if args.output is None:
                parser.error("preview requires --output")
            value = preview(args.root, args.output)
        elif args.command == "run":
            if any(x is None for x in (args.inputs, args.work, args.requests)):
                parser.error("run requires --inputs, --work and --requests")
            value = run(args.inputs, args.root, args.work, args.requests)
        else:
            value = campaign.report(args.root, campaign.verify(args.root))
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
