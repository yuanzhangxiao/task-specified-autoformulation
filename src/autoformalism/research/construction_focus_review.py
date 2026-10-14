"""Preview focused requests on verified historical predecessors without LLM calls."""

from __future__ import annotations

import html
import json
from contextlib import suppress
from copy import deepcopy
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_comparison as campaign
from autoformalism.research.construction_deferred_review import marked_diff
from autoformalism.search import construction_compact as compact
from autoformalism.search import construction_completion as completion_feedback
from autoformalism.search import construction_coverage as coverage
from autoformalism.search import construction_decisions as decisions
from autoformalism.search import construction_edit_feedback as edit_feedback
from autoformalism.search import construction_focus as focused
from autoformalism.search import construction_handoff as handoff
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_ownership_feedback as ownership_feedback
from autoformalism.search import construction_readable as readable
from autoformalism.search.public_graph_obligations import PublicGraphContract


def preview(
    cell: dict,
    task: dict,
    event: dict,
    body: dict,
    policy: str = focused.POLICY,
    last: dict | None = None,
    budget_context: dict | None = None,
) -> dict:
    """Use the live projection on one actual historical predecessor, not a new chain."""
    if policy not in {
        focused.POLICY,
        readable.POLICY,
        completion_feedback.POLICY,
        coverage.POLICY,
        decisions.POLICY,
        edit_feedback.POLICY,
        ownership_feedback.POLICY,
    }:
        raise ValueError("unsupported preview policy")
    presentation = (
        ownership_feedback
        if policy == ownership_feedback.POLICY
        else edit_feedback
        if policy == edit_feedback.POLICY
        else decisions
        if policy == decisions.POLICY
        else coverage
        if policy == coverage.POLICY
        else completion_feedback
        if policy == completion_feedback.POLICY
        else readable
        if policy == readable.POLICY
        else focused
    )
    old = compact.expand_payload(json.loads(body["messages"][1]["content"]))
    brief = campaign.contract.visible_brief(cell, task)
    new = (
        old
        if event.get("bookkeeping_policy") == policy
        else presentation.payload(
            event["stage"],
            brief,
            ValidationContext.model_validate(cell["context"]),
            campaign.contract.target_definitions(cell),
            old["public_brief"],
            ledger.Draft.model_validate(event["before"]),
            PublicGraphContract.model_validate(cell["public_graph_contract"]),
            last or old.get("last_edit_result"),
            old.get("feedback", {}).get("continuation"),
        )
    )
    if policy == readable.POLICY and event["stage"] == "repair":
        readable.repair_budget(new, focused.REPAIR_REQUESTS, event["attempt"])
    if (
        policy
        in {
            completion_feedback.POLICY,
            coverage.POLICY,
            decisions.POLICY,
            edit_feedback.POLICY,
            ownership_feedback.POLICY,
        }
        and event["stage"] == "repair"
        and event.get("bookkeeping_policy") != policy
    ):
        if budget_context is None:
            raise ValueError("historical repair preview requires budget accounting")
        presentation.repair_budget(
            new,
            budget_context["maximum_calls"],
            event["attempt"],
            budget_context["ceiling"],
            budget_context["charged"],
            budget_context["repair_tokens"],
        )
    proposed = deepcopy(body)
    proposed["messages"] = [
        {"role": "system", "content": presentation.SYSTEM},
        {"role": "user", "content": json.dumps(new, sort_keys=True)},
    ]
    return {
        "task": task["task_id"],
        "event": event["index"],
        "stage": event["stage"],
        "source_request_sha256": event["request_hash"],
        "before": body,
        "after": proposed,
        "saved_user_bytes": len(body["messages"][1]["content"].encode()),
        "revised_user_bytes": len(proposed["messages"][1]["content"].encode()),
    }


def inspect(source: Path, policy: str = focused.POLICY) -> tuple[dict, list[dict]]:
    """Check sealed event chains and retain all displayed public evidence verbatim."""
    plan = sealed_read(source / "plan.json")
    if (
        plan.get("protocol") != campaign.PROTOCOL
        or plan.get("test_data_opened") is not False
    ):
        raise ValueError("requires a public-only construction inspection")
    requests = []
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
        by_hash = {r["request_hash"]: r for r in records}
        previous = None
        charged = 0
        for path in sorted((directory / "construction/events").glob("*.json")):
            event = sealed_read(path)
            record = by_hash[event["request_hash"]]
            requests.append(
                preview(
                    plan["cells"][task["benchmark_id"]],
                    task,
                    event,
                    record["request"]["body"],
                    policy,
                    previous,
                    {
                        "maximum_calls": plan["config"]["repair_requests"],
                        "ceiling": record["request"]["settings"][
                            "maximum_total_tokens"
                        ],
                        "charged": charged,
                        "repair_tokens": plan["config"]["repair_tokens"],
                    },
                )
            )
            charged += record["budget_charge"]
            previous = {
                "status": "accepted" if event["accepted"] else "rejected",
                "transaction_applied": event["accepted"],
                "error": event["error"],
                "edit_effects": handoff.edit_effects(
                    ledger.Draft.model_validate(event["before"]),
                    ledger.Draft.model_validate(event["after"]),
                )
                if event["accepted"]
                else None,
                "normalizations": event.get("normalizations", []),
                "name_conflicts": event.get("name_conflicts", []),
                "definition_conflicts": event.get("definition_conflicts", []),
                "rejected_reply": None,
            }
            if not event["accepted"]:
                # Unparseable delivery is preserved in the saved error.
                with suppress(ValueError, TypeError, KeyError):
                    previous["rejected_reply"] = visible_response(record)
    before = sum(r["saved_user_bytes"] for r in requests)
    after = sum(r["revised_user_bytes"] for r in requests)
    return {
        "policy": policy,
        "source_identity": plan["artifact_sha256"],
        "requests": len(requests),
        "tasks": len({r["task"] for r in requests}),
        "saved_user_bytes": before,
        "revised_user_bytes": after,
        "user_byte_reduction_fraction": 1 - after / before if before else None,
        "size_unit": "UTF-8 bytes, not model tokens; output schema unchanged",
        "scope": "Counterfactual requests on each actual saved predecessor. "
        "Historical replies are not replies to these prompts. "
        "No new construction chain or scientific improvement is inferred.",
        "new_schedule": {
            "additional_variable_recovery_calls": focused.LOCAL_RECOVERY_REQUESTS,
            "maximum_global_repair_calls": focused.REPAIR_REQUESTS,
            "stop_after_identical_no_progress_repairs": 2,
            "note": "Successful variable recovery resumes shared/topology stages; "
            "exhaustion stops incomplete. Success stops repair early. "
            "Existing overall request/token caps still apply.",
        },
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }, requests


def _block(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, indent=2)
    return "<pre>" + html.escape(text) + "</pre>"


def _page(title: str, content: str) -> str:
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{max-width:1080px;margin:2rem auto;padding:0 1rem;"
        "font:16px/1.55 system-ui;color:#18232c;background:#fbfaf6}"
        "a{color:#145d8a}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "padding:1rem;background:#f0f2f2;border-radius:6px;font:13px/1.5 monospace}"
        ".new{border-left:4px solid #268061;padding-left:1rem}"
        ".badge,ins{background:#d9f3e7;text-decoration:none}"
        "del{background:#ffe0dc}summary{cursor:pointer;padding:.5rem}"
        "table{border-collapse:collapse;width:100%}td,th{text-align:left;"
        "border-bottom:1px solid #cdd6d8;padding:.4rem;vertical-align:top}"
        "h2{margin-top:2rem}</style><body>"
        f"<h1>{html.escape(title)}</h1>{content}</body></html>"
    )


def _request_page(row: dict, policy: str = focused.POLICY) -> str:
    p = json.loads(row["after"]["messages"][1]["content"])
    view = p["model_view"]
    lines = [r["topology"] for r in view["processes"] + view["balances"]]
    decision_view = policy in {
        decisions.POLICY,
        edit_feedback.POLICY,
        ownership_feedback.POLICY,
    }
    body = (
        '<p><a href="PROMPTS.html">All stages and requests</a></p>'
        '<p><span class="badge">REVISED PREVIEW</span> Actual saved predecessor; '
        "no response to this revised request has been collected.</p>"
        "<h2>Current-stage instruction</h2>"
        + _block(p["stage_instructions"])
        + '<section class="new" id="model"><h2>New model view</h2>'
        + _block("\n".join(lines) or "No topology declared yet.")
        + "<p>The equations above are extracted from the single model_view "
        "in the actual request. The complete view below also gives types, "
        "scientific meanings and explicit assignments.</p>"
        + "<details><summary>Complete model view, as sent</summary>"
        + _block(view)
        + "</details>"
        + "</section>"
        + '<section class="new" id="feedback"><h2>New feedback</h2>'
        + _block(p["feedback"])
        + (
            "<h3>Memory assignments and structural checks</h3>"
            "<p>These checks do not establish mechanism coverage. Scientific "
            "review questions in the feedback are non-gating and require "
            "proposer judgment.</p>"
            if decision_view
            else "<h3>Requirement checklist</h3>"
        )
        + _block(p["requirements_checklist"])
        + "<h3>Eligible receivers</h3>"
        + _block(p["receiver_choices"])
        + "<h3>Previous edit receipt</h3>"
        + _block(p["previous_edit"])
        + (
            "<h3>Memory-assignment choices</h3>" + _block(p["memory_assignment_rules"])
            if decision_view
            else "<h3>Memory binding options</h3>"
        )
        + _block(p["memory_binding_options"])
        + "</section><h2>Complete revised request</h2>"
        + "<p>Includes the full public brief, messages, output JSON schema and "
        "saved generation settings. JSON escaping in this copy is transport "
        "serialization, not additional instruction text.</p>"
        + "<details><summary>System + user + output schema + settings</summary>"
        + _block(row["after"])
        + "</details>"
        + "<details><summary>Readable user payload in exact field order</summary>"
        + _block(p)
        + "</details>"
        + "<details><summary>Marked changes: red removed, green added</summary>"
        + marked_diff(
            json.dumps(
                json.loads(row["before"]["messages"][1]["content"]),
                indent=2,
                sort_keys=True,
            ),
            json.dumps(p, indent=2, sort_keys=True),
        )
        + "</details>"
    )
    return _page(f"{row['task']} · {row['stage']} · turn {row['event']}", body)


def render(source: Path, output: Path, policy: str = focused.POLICY) -> dict:
    """Write a fresh escaped HTML review; never overwrite a saved campaign."""
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("review must be outside the historical campaign")
    if output.exists() and any(output.iterdir()):
        raise ValueError("review output must be empty")
    summary, rows = inspect(source, policy)
    presentation = (
        ownership_feedback
        if policy == ownership_feedback.POLICY
        else edit_feedback
        if policy == edit_feedback.POLICY
        else decisions
        if policy == decisions.POLICY
        else coverage
        if policy == coverage.POLICY
        else completion_feedback
        if policy == completion_feedback.POLICY
        else readable
        if policy == readable.POLICY
        else focused
    )
    output.mkdir(parents=True, exist_ok=True)
    links, highlights = [], {}
    for i, row in enumerate(rows):
        name = f"request-{i:03d}.html"
        (output / name).write_text(_request_page(row, policy), encoding="utf-8")
        p = json.loads(row["after"]["messages"][1]["content"])
        issues = p["feedback"]["issues"]
        if policy in {
            decisions.POLICY,
            edit_feedback.POLICY,
            ownership_feedback.POLICY,
        }:
            for key, title in (
                ("memory_assignment_decisions", "Optional memory-assignment decision"),
                ("scientific_review", "Scientific review questions (non-gating)"),
            ):
                if key not in highlights and p["feedback"].get(key):
                    highlights[key] = f'<li><a href="{name}#feedback">{title}</a></li>'
        edit_examples = (
            (
                (
                    "undeclared_lhs_entries",
                    "Undeclared equation names: explicit choices",
                ),
                (
                    "process_edit_destination",
                    "Edit or remove a process in its own field",
                ),
                (
                    "process_replace_remove_conflict",
                    "Choose replacement or removal once",
                ),
                ("duplicate_receiver", "One signed use per process receiver"),
            )
            if policy in {edit_feedback.POLICY, ownership_feedback.POLICY}
            else ()
        )
        ownership_examples = (
            (("state_process_ownership", "Preserve state and process ownership"),)
            if policy == ownership_feedback.POLICY
            else ()
        )
        for code, title in (
            *ownership_examples,
            *edit_examples,
            ("confirm_memory_assignment", "Explicit memory-assignment confirmation"),
            ("duplicate_equation_entries", "One topology entry per quantity"),
            ("public_path", "Exact public pathway to repair"),
            ("equation_destination", "Exact quantity whose equation needs repair"),
            ("flow_classification_decision", "Boundary flow or internal transfer"),
            ("memory_driver_path", "Correct or explicitly withdraw a memory binding"),
            ("algebraic_dependency_cycle", "Algebraic cycle: conditional type repair"),
            ("conversion_role_decision", "Clarify a coefficient in conversion"),
            (
                "process_consumer_decision",
                "Additive contribution or composite function",
            ),
            (
                "memory_target_path",
                "Assigned memory: keep the required pathway visible",
            ),
            ("memory_assignment", "Missing memory assignment: explicit JSON example"),
            ("supplied_receiver", "Invalid receiver: preserve the valid effect"),
            ("compiler", "Incomplete RHS: show provisional paths too"),
        ):
            if code not in highlights and any(
                x["code"] == code
                and (code != "memory_assignment" or x.get("conditional_examples"))
                for x in issues
            ):
                highlights[code] = f'<li><a href="{name}#feedback">{title}</a></li>'
        links.append(
            f'<tr><td><a href="{name}">{html.escape(row["task"])} '
            f"/{row['event']:03d}</a></td><td>{html.escape(row['stage'])}</td>"
            f"<td>{row['saved_user_bytes']:,}</td>"
            f"<td>{row['revised_user_bytes']:,}</td></tr>"
        )
    body = (
        (
            '<section class="new"><h2>State/process ownership correction</h2><p>'
            "Review corrected ownership guidance against each actual saved "
            "predecessor and its previous edit receipt, including rejected edits. "
            "The feedback coordinates state and process naming choices without "
            "choosing the scientific representation. Stage instructions, model "
            "view, public evidence, schemas, checks and budgets retain the "
            "previous edit-feedback policy.</p></section>"
            if policy == ownership_feedback.POLICY
            else '<section class="new"><h2>Structural edit feedback</h2><p>'
            "Review concrete edit guidance against each actual saved predecessor "
            "and its previous edit receipt. Invalid replies remain rejected, and "
            "the runtime does not infer or apply scientific corrections. The "
            "memory-assignment rules, scientific questions, stage instructions, "
            "schemas and budgets retain the decisions policy.</p></section>"
            if policy == edit_feedback.POLICY
            else '<section class="new"><h2>Memory assignments '
            "and mechanism coverage</h2>"
            "<p>Memory assignments are explicit choices about dynamic storage; "
            "they do not certify that a public mechanism is represented. "
            "Review structural feedback alongside separate, non-gating scientific "
            "questions. Optional assignments with missing paths have a single "
            "coordinated decision; no edit or scientific interpretation is "
            "chosen by the runtime. Existing schemas, acceptance checks and "
            "budgets are retained.</p></section>"
            if policy == decisions.POLICY
            else '<section class="new"><h2>Public-coverage update</h2><p>'
            "Updated shared-process, remaining-topology and repair instructions. "
            "Variables, system instruction, editing contract and scientific checks "
            "are unchanged. Review physical direction across related balances; "
            "classification alone does not repair an incorrect effect. "
            "Feedback names exact missing paths, distinguishes supplied observations "
            "from modeled balances, and explains duplicate topology entries. "
            "This is a prompt change, not a new scientific certificate.</p></section>"
            if policy == coverage.POLICY
            else ""
        )
        + "<p>Review version — no live test launched. All public science and "
        "training evidence are retained verbatim; exact accepted declarations "
        "remain in the runtime log. This projection changes how they are "
        "presented, not the scientific acceptance rules.</p>"
        '<section class="new"><h2>Changes to review</h2><ul>'
        "<li>Policy: "
        + html.escape(policy)
        + ".</li>"
        + (
            "<li>Replace contradictory state/process correction advice with "
            "one coordinated ownership decision for each affected name.</li>"
            "<li>Qualify replacement/removal guidance using the actual retained "
            "declarations and previous rejected transaction.</li>"
            "<li>Retain complete compiler evidence, memory-assignment decisions "
            "and non-gating scientific questions.</li>"
            "<li>Stages, public evidence, schema, checks and budgets "
            "are unchanged.</li>"
            if policy == ownership_feedback.POLICY
            else "<li>Group undeclared equation names with their supplied roles and "
            "explicit editing choices.</li>"
            "<li>Use the previous rejected transaction to name process-owned "
            "definitions and replacement/removal conflicts.</li>"
            "<li>Clarify duplicate receiver edits while retaining the complete "
            "compiler error and other scientific feedback.</li>"
            "<li>Stages, public evidence, schema, checks and budgets "
            "are unchanged.</li>"
            if policy == edit_feedback.POLICY
            else "<li>Separate required memory choices from general mechanism "
            "coverage; "
            "show only required or already assigned memory options.</li>"
            "<li>Ask one explicit decision for each failing optional assignment; "
            "path repair is conditional on retaining that assignment.</li>"
            "<li>Ask about positive-rate meanings, signed uses and related "
            "balances across benchmarks, without inferring scientific edits.</li>"
            "<li>The existing model view, public evidence, schema, checks and "
            "call limits are preserved.</li>"
            if policy == decisions.POLICY
            else "<li>One model view with joint arguments, signs, conversions "
            "and meanings; "
            "no duplicate declaration JSON.</li><li>A compact edit receipt, "
            "one requirement checklist and specific feedback with conditional "
            "structured repair examples.</li><li>Internal bookkeeping labels and "
            "interaction handoff instructions removed from the proposer request.</li>"
            "<li>Memory assignment required is separate from whether a public "
            "mechanism is required.</li><li>Provisional dependency observations "
            "remain available when incomplete RHS entries block the final checks.</li>"
        )
        + "</ul></section><h2>Start with these feedback examples</h2><ul>"
        + "".join(highlights.values())
        + "</ul>"
        + "<h2>System instruction (every call)</h2>"
        + _block(presentation.SYSTEM)
        + (
            '<section class="new"><h2>Memory-assignment rule (every call)</h2>'
            + _block(decisions.MEMORY_RULES)
            + "</section>"
            if policy
            in {decisions.POLICY, edit_feedback.POLICY, ownership_feedback.POLICY}
            else ""
        )
        + "<h2>Stage instructions</h2>"
        + "".join(
            f"<h3>{title}</h3>" + _block(presentation.stage_text(stage))
            for stage, title in (
                ("variables", "1. Variables"),
                ("shared_laws", "2. Shared processes"),
                ("equations", "3. Remaining topology"),
                ("repair", "4. Global repair"),
            )
        )
        + "<h2>Editing contract</h2>"
        + _block(presentation.EDITING)
        + "<h2>Bounded continuation and scope</h2>"
        + _block(summary["new_schedule"])
        + "<p>"
        + html.escape(summary["scope"])
        + "</p>"
        + "<h2>Every historical request, re-presented</h2>"
        + "<p>Sizes are user-message UTF-8 bytes, not model-token estimates. "
        "The output JSON schema is unchanged. Every page includes the full "
        "request and marked differences.</p>"
        + "<table><tr><th>Case / turn</th><th>Stage</th>"
        "<th>Saved bytes</th><th>Revised bytes</th></tr>"
        + "".join(links)
        + "</table><h2>Offline audit</h2>"
        + _block(summary)
    )
    title = (
        "State/process ownership correction prompts"
        if policy == ownership_feedback.POLICY
        else "Structural edit feedback prompts"
        if policy == edit_feedback.POLICY
        else "Memory assignments and mechanism coverage prompts"
        if policy == decisions.POLICY
        else "Focused construction prompts"
    )
    (output / "PROMPTS.html").write_text(_page(title, body))
    (output / "PROMPTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
