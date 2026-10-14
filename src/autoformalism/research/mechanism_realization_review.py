"""Marked human review of the general mechanism bookkeeping contract."""

from __future__ import annotations

import html
import json
from pathlib import Path

from autoformalism.research.construction_deferred_review import marked_diff
from autoformalism.search import construction_protocol as previous
from autoformalism.search import construction_realizations as prompts


def _marked(title: str, old: str, new: str) -> str:
    """Show readable current text with an expandable literal wording diff."""
    from autoformalism.research.construction_focus_review import _block

    return (
        f"<h3>{html.escape(title)}</h3>"
        + _block(new)
        + "<details><summary>Changes from protocol-clarity prompts</summary>"
        + marked_diff(old, new)
        + "</details>"
    )


def render(source: Path, output: Path) -> dict:
    """Re-present saved predecessors; never invent scientific realization choices."""
    from autoformalism.research.construction_focus_review import _block, _page, inspect

    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("review must be outside the historical campaign")
    if output.exists() and any(output.iterdir()):
        raise ValueError("review output must be empty")
    summary, rows = inspect(source, prompts.POLICY)
    summary["legacy_preview_projection"] = (
        "Existing explicit memory claims become realization records citing only "
        "those already-assigned states. No other mechanism realization is inferred. "
        "Missing general records are shown as missing. Public evidence is unchanged."
    )
    output.mkdir(parents=True, exist_ok=True)
    links = []
    examples = {}
    for i, row in enumerate(rows):
        name = f"request-{i:03d}.html"
        packet = json.loads(row["after"]["messages"][1]["content"])
        lines = [
            item["topology"]
            for item in packet["model_view"]["processes"]
            + packet["model_view"]["balances"]
        ]
        content = (
            '<p><a href="PROMPTS.html">All prompts</a></p>'
            "<p>Actual saved predecessor, re-presented under the new contract. "
            "No response to this revised request has been collected.</p>"
            "<h2>Current stage</h2>"
            + _block(packet["stage_instructions"])
            + "<h2 id='model'>Model view</h2>"
            + _block("\n".join(lines) or "No topology declared yet.")
            + "<details><summary>Complete model view, as sent</summary>"
            + _block(packet["model_view"])
            + "</details><h2 id='checklist'>Requirement checklist</h2>"
            + _block(packet["requirements_checklist"])
            + "<h2 id='feedback'>Repair feedback</h2>"
            + _block(packet["feedback"])
            + "<h2>Previous edit receipt</h2>"
            + _block(packet["previous_edit"])
            + "<h2>Response template</h2>"
            + _block(packet["response_template"])
            + "<details><summary>Complete request and new output schema</summary>"
            + _block(row["after"])
            + "</details><details><summary>Marked full request changes</summary>"
            + marked_diff(
                json.dumps(row["before"], indent=2, sort_keys=True),
                json.dumps(row["after"], indent=2, sort_keys=True),
            )
            + "</details>"
        )
        (output / name).write_text(
            _page(f"{row['task']} · {row['stage']} · {row['event']}", content),
            encoding="utf-8",
        )
        for issue in packet["feedback"]["issues"]:
            code = issue.get("code", "")
            if "realization" in code or "memory" in code:
                examples.setdefault(code, name)
        links.append(
            f'<tr><td><a href="{name}">{html.escape(row["task"])} '
            f"/{row['event']:03d}</a></td><td>{html.escape(row['stage'])}</td>"
            f"<td>{row['saved_user_bytes']:,}</td>"
            f"<td>{row['revised_user_bytes']:,}</td></tr>"
        )
    stages = (
        ("variables", "1. Variables"),
        ("shared_laws", "2. Shared processes"),
        ("equations", "3. Remaining topology"),
        ("repair", "4. Global repair"),
    )
    content = (
        "<p><span class='badge'>New contract · review before experiment</span></p>"
        "<p>One mechanism record lists its proposed components and optional memory "
        "states. Components may be supplied quantities, generated algebraic or "
        "differential variables, or shared processes. For a local process, cite "
        "the variable whose equation contains it. The runtime derives types "
        "and checks references and applicable public paths. No new local-process "
        "identifiers or separate realization-type declaration are required.</p>"
        "<p><strong>Declared realization ≠ structural verification ≠ demonstrated "
        "behavior.</strong> Algebraic mechanisms need no artificial memory state. "
        "The public requirements, model, stage order, and ten-repair allowance "
        "are unchanged. The output schema and declaration checklist are updated. "
        "No fitting, LLM calls, or test-data access occurred in this review.</p>"
        "<h2>System and dictionary</h2>"
        + _marked("System instruction (each call)", previous.SYSTEM, prompts.SYSTEM)
        + _marked(
            "Mechanism records and memory",
            previous.MEMORY_RULE,
            prompts.REALIZATION_RULES,
        )
        + "<h2>All stage instructions</h2>"
        + "".join(
            _marked(title, previous.stage_text(stage), prompts.stage_text(stage))
            for stage, title in stages
        )
        + _marked("Editing contract", previous.EDITING, prompts.EDITING)
        + "<h2>New response formats</h2>"
        + "".join(
            f"<h3>{title}</h3>"
            + _block(
                prompts.response_model(stage)(stage_complete=False).model_dump(
                    mode="json"
                )
            )
            for stage, title in stages
        )
        + "<h2>Feedback examples from retained drafts</h2><ul>"
        + "".join(
            f'<li><a href="{name}#feedback">{html.escape(code)}</a></li>'
            for code, name in sorted(examples.items())
        )
        + "</ul><h2>Historical evidence and preview limits</h2>"
        + _block(summary["legacy_preview_projection"])
        + _block(summary["scope"])
        + "<h2>Every full request</h2><p>Sizes are UTF-8 user-message bytes, "
        "not model tokens. The full pages include the new response schema.</p>"
        "<table><tr><th>Task / turn</th><th>Stage</th><th>Saved bytes</th>"
        "<th>Revised bytes</th></tr>"
        + "".join(links)
        + "</table><h2>Offline audit</h2>"
        + _block(summary)
    )
    (output / "PROMPTS.html").write_text(
        _page("Mechanism realizations — revised prompts", content), encoding="utf-8"
    )
    (output / "PROMPTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
