"""Audit repair regressions and render marked prompts from verified saved calls."""

from __future__ import annotations

import difflib
import html
import json
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_comparison as campaign
from autoformalism.search import construction_deferred as deferred
from autoformalism.search import construction_ledger as ledger
from autoformalism.search.public_graph_obligations import PublicGraphContract
from autoformalism.staged_topology import content_hash


def passed_paths(check: dict) -> dict[str, dict]:
    """Identify executed successful predicates, not prose-based deletion judgments."""
    rows = []
    for row in check["public_structure_checks"]:
        if row["passed"]:
            rows.append({"kind": "public_path", "evidence": row})
    for row in check["reviewed_public_graph_checks"]:
        if row["passed"] is True:
            rows.append({"kind": "reviewed_graph", "evidence": row})
    for row in check["memory_binding_checks"]:
        if row["status"] == "passed":
            rows.append({"kind": "bound_memory", "evidence": row})
    return {
        content_hash(
            {
                "kind": row["kind"],
                **{
                    k: v
                    for k, v in row["evidence"].items()
                    if k in {"id", "requirement", "kind", "target", "driver"}
                },
            }
        ): row
        for row in rows
    }


def inspect(source: Path) -> tuple[dict, list[dict]]:
    """Compare actual predecessor/successor pairs; never simulate a fresh LLM chain."""
    plan = sealed_read(source / "plan.json")
    if (
        plan.get("protocol") != campaign.PROTOCOL
        or plan.get("test_data_opened") is not False
    ):
        raise ValueError("requires a public-only construction inspection")
    regressions, requests, no_overlap_stops = [], [], []
    for task in plan["tasks"]:
        name = task["task_id"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("invalid task directory")
        directory = source / "results" / name
        records = campaign.checked_records(
            directory,
            campaign.baseline.namespace(plan, task),
            task.get("starting_checkpoint", {}).get("draft"),
        )
        records = {r["request_hash"]: r for r in records}
        cell = plan["cells"][task["benchmark_id"]]
        brief = campaign.contract.visible_brief(cell, task)
        context = ValidationContext.model_validate(cell["context"])
        targets = campaign.contract.target_definitions(cell)
        contract = PublicGraphContract.model_validate(cell["public_graph_contract"])
        for path in sorted((directory / "construction/events").glob("*.json")):
            event = sealed_read(path)
            record = records[event["request_hash"]]
            body = record["request"]["body"]
            old = json.loads(body["messages"][1]["content"])
            draft = ledger.Draft.model_validate(event["before"])
            new = deferred.update_payload(old, brief, context, targets, draft, contract)
            requests.append(
                {
                    "task": name,
                    "event": event["index"],
                    "stage": event["stage"],
                    "request_hash": event["request_hash"],
                    "system": body["messages"][0]["content"],
                    "response_format": body.get("response_format"),
                    "before": old,
                    "after": new,
                }
            )
            if event["stage"] != "repair":
                continue
            before = deferred.assessment(brief, context, targets, draft, contract)
            after = deferred.assessment(
                brief,
                context,
                targets,
                ledger.Draft.model_validate(event["after"]),
                contract,
            )
            diagnostic = old["runtime_diagnostics"] or {}
            overlap_questions = diagnostic.get("clarification_requests", [])
            other_errors = diagnostic.get("structural_failures", [])
            if before["eligible"] and overlap_questions and not other_errors:
                no_overlap_stops.append(
                    {
                        "task": name,
                        "before_event": event["index"],
                        "request_hash": event["request_hash"],
                        "interpretation": "This saved predecessor would not require "
                        "overlap repair under the new rule. "
                        "No fresh response is inferred.",
                    }
                )
            if (
                before["graph_check_status"] != "assessed"
                or after["graph_check_status"] != "assessed"
            ):
                continue
            previous, following = passed_paths(before), passed_paths(after)
            lost = [previous[key] for key in previous.keys() - following.keys()]
            if lost:
                regressions.append(
                    {
                        "task": name,
                        "event": event["index"],
                        "request_hash": event["request_hash"],
                        "lost_predicates": sorted(lost, key=content_hash),
                        "overlap_question_count": len(overlap_questions),
                        "other_error_codes": [e["code"] for e in other_errors],
                        "overlap_only_feedback": bool(overlap_questions)
                        and not other_errors,
                    }
                )
    return {
        "policy": deferred.POLICY,
        "source_identity": plan["artifact_sha256"],
        "saved_requests": len(requests),
        "path_regressions": regressions,
        "regression_count": len(regressions),
        "overlap_only_regression_count": sum(
            r["overlap_only_feedback"] for r in regressions
        ),
        "saved_predecessors_eligible_without_overlap_gate": no_overlap_stops,
        "scope": "Observed predicate regressions and eligibility of saved "
        "predecessors, "
        "not a prediction of fresh runs or recovered scientific models.",
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }, requests


def marked_diff(before: str, after: str) -> str:
    """Escape all source text before highlighting line additions/removals."""
    result = []
    for line in difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile="saved prompt",
        tofile="new policy",
        lineterm="",
    ):
        tag = (
            "ins" if line.startswith("+") else "del" if line.startswith("-") else "span"
        )
        result.append(f"<{tag}>{html.escape(line)}</{tag}>")
    return "<pre class='diff'>" + "\n".join(result) + "</pre>"


def render(source: Path, output: Path) -> dict:
    """Save a self-contained HTML review and machine-readable audit, no live calls."""
    audit, requests = inspect(source)
    output.mkdir(parents=True, exist_ok=True)
    dump = lambda value: json.dumps(value, indent=2, ensure_ascii=False)  # noqa: E731
    (output / "AUDIT.json").write_text(dump(audit) + "\n")
    (output / "PROMPTS.json").write_text(dump(requests) + "\n")
    parts = [
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Minimal prompts — changes for review</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#203040;background:#f4f6f8;margin:0}
main{max-width:1100px;margin:auto;padding:32px}h1{font-size:32px;line-height:1.2}
h2{margin-top:32px}p{max-width:90ch}.card,details{background:white;border:1px solid
#ccd5de;border-radius:8px;padding:16px;margin:14px 0}
summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,monospace}
ins{background:#d7f2df;text-decoration:none}del{background:#ffe0df}
.badge{background:#e6edf8;padding:4px 10px;border-radius:5px}a{color:#154a91}
table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid
#d9dfe6;text-align:left;vertical-align:top}.muted{color:#536170}
</style><main><h1>Minimal prompts: changes for review</h1>
<p><span class="badge">Review only — no live calls</span></p>
<p>Green additions and red deletions compare the new policy with the actual saved
requests in the supplied inspection. Each preview uses its actual historical
predecessor. This is
not a simulated new conversation: some old repair calls would no longer occur.</p>
<div class="card"><h2>What changes</h2><ul>
<li>Matching dependencies no longer block topology or trigger an overlap question.</li>
<li>Visible requirement statuses distinguish declaration/path checks from deferred,
unavailable and unassessed scientific requirements.</li>
<li>Function overlap and signed-rate interpretation are saved for interaction
review.</li>
<li>No new deletion guard, automatic merging, scientific reassignment or benchmark
rule is added. Malformed receivers and missing encoded paths still block.</li></ul>
<p>The construction runner stops before functions. The interaction questions below
are a saved handoff, not an executed interaction-stage implementation.</p></div>"""
    ]
    if requests:
        parts.append(
            "<h2>Common system prompt — unchanged</h2><pre>"
            + html.escape(requests[0]["system"])
            + "</pre>"
        )
    parts.append(
        "<h2>Stage wording, in order</h2><p>The complete new stage text "
        "appears first. Expand the marked comparison to see the additions. "
        "Public inputs, current declarations, status tables and JSON schemas "
        "are shown for each actual request farther below.</p>"
    )
    for stage in ("variables", "shared_laws", "equations", "repair"):
        row = next((r for r in requests if r["stage"] == stage), None)
        if row is None:
            continue
        parts.append(
            f"<div class='card'><h3>{html.escape(stage)}</h3><pre>"
            + html.escape(row["after"]["stage_instructions"])
            + "</pre>"
        )
        parts.append(
            "<details><summary>Marked changes to this stage</summary>"
            + marked_diff(
                row["before"]["stage_instructions"], row["after"]["stage_instructions"]
            )
            + "</details>"
        )
        parts.append(
            "<details><summary>Editing rules and reply template</summary><pre>"
            + html.escape(row["after"].get("editing_rules", ""))
            + "\n"
            + html.escape(dump(row["after"]["response_template"]))
            + "</pre></details></div>"
        )
    parts.append(
        f"<h2>Repair audit</h2><p>{audit['regression_count']} repairs lost "
        "previously passing graph predicates. "
        f"{audit['overlap_only_regression_count']} "
        "followed overlap-only feedback.</p><table><tr><th>Construction</th>"
        "<th>Event</th><th>Preceding feedback</th></tr>"
    )
    for row in audit["path_regressions"]:
        reason = (
            "Overlap clarification only"
            if row["overlap_only_feedback"]
            else ", ".join(row["other_error_codes"])
        )
        parts.append(
            f"<tr><td>{html.escape(row['task'])}</td><td>{row['event']}</td>"
            f"<td>{html.escape(reason)}</td></tr>"
        )
    parts.append(
        "</table><p>Saved examples do not establish a general causal effect. "
        "A fresh run is needed to measure prospective improvement.</p>"
    )
    parts.append(
        "<h2>Deferred interaction question: signed quantities</h2><pre><ins>"
        + html.escape(deferred.SIGN_QUESTION)
        + "</ins></pre>"
    )
    parts.append(
        "<h2>All saved prompt locations</h2><p>Open a construction, then a "
        "stage. The first panel highlights changes; complete old and new "
        "requests remain available below it. Shared public context and "
        "the short system prompt are preserved.</p>"
    )
    for name in dict.fromkeys(r["task"] for r in requests):
        parts.append(f"<details><summary>{html.escape(name)}</summary>")
        for row in (r for r in requests if r["task"] == name):
            old, new = dump(row["before"]), dump(row["after"])
            parts.append(
                f"<details><summary>{row['event']:03d} — "
                f"{html.escape(row['stage'])}</summary>"
            )
            parts.append(marked_diff(old, new))
            for title, value in [
                ("System prompt (unchanged)", row["system"]),
                ("Complete new user prompt", new),
                ("Complete saved user prompt", old),
                ("Response JSON schema (unchanged)", dump(row["response_format"])),
            ]:
                parts.append(
                    f"<details><summary>{title}</summary><pre>{html.escape(value)}</pre></details>"
                )
            parts.append("</details>")
        parts.append("</details>")
    parts.append("</main></html>")
    (output / "PROMPTS.html").write_text("\n".join(parts))
    return {
        k: v
        for k, v in audit.items()
        if k
        not in {"path_regressions", "saved_predecessors_eligible_without_overlap_gate"}
    }
