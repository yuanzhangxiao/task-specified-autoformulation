#!/usr/bin/env python3
"""Compare repair feedback on actual saved predecessors, without new proposals.

Historical responses are not replayed into a hypothetical revised chain. Only
advisory answer identities are translated at the predecessor where each answer
was recorded, so unchanged diagnostic hypotheses can retain their dispositions.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter
from contextlib import suppress
from difflib import unified_diff
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import strict_provider_schema, visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_comparison as campaign
from autoformalism.research import construction_training as training
from autoformalism.search import construction_evidence as historical
from autoformalism.search import construction_evidence_repair as revised
from autoformalism.search import construction_stage_checks as stage_checks
from autoformalism.search import mechanism_realizations as realizations
from autoformalism.search.public_graph_obligations import PublicGraphContract
from autoformalism.staged_topology import content_hash

PROTOCOL = "saved-evidence-repair-feedback-audit-1"
SCOPE = (
    "Each request uses its actual saved predecessor. Historical responses are not "
    "responses to the revised prompts. Structural eligibility is separate from "
    "scientific consistency; answering a conditional question does not verify a "
    "model. No new construction chain or scientific improvement is inferred."
)


def entry_reason(assessment: dict) -> str:
    """Distinguish structural repair from newly requested scientific advice."""
    if not assessment["eligible"]:
        return "structural_obligation"
    if assessment.get("training_compatibility", {}).get("pending_questions"):
        return "conditional_scientific_question"
    return "completion_control"


def diagnostic_key(question: dict) -> tuple[str, str, str]:
    """Match the same displayed scientific question across identity policies."""
    return (
        question["target"],
        question["property"],
        json.dumps(question["witnesses"], sort_keys=True),
    )


def translate_decisions(
    old_questions: list[dict], new_questions: list[dict], decisions: list[dict]
) -> list[dict]:
    """Translate only answers to the same question at its actual predecessor."""
    old_by_id = {q["question_id"]: q for q in old_questions}
    new_by_key = {diagnostic_key(q): q for q in new_questions}
    if len(new_by_key) != len(new_questions):
        raise ValueError("ambiguous revised diagnostic identities")
    result = []
    for decision in decisions:
        old = old_by_id.get(decision["question_id"])
        if old is None:
            raise ValueError("saved advisory answer was not displayed")
        new = new_by_key.get(diagnostic_key(old))
        if new is None:
            raise ValueError("revised diagnostic changed scientific question")
        if any(
            old.get(key) != new.get(key)
            for key in (
                "scope",
                "verified",
                "assumptions",
                "conditioning",
                "question",
                "interpretation",
            )
        ):
            raise ValueError("revised diagnostic changed scientific question")
        result.append({**decision, "question_id": new["question_id"]})
    return result


def advisory_state(report: dict, answers: dict[str, dict]) -> dict:
    """Retain approximation/dispute as answers, never as scientific approvals."""
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


def full_assessment(api, brief, context, targets, graph, boundary, draft) -> dict:
    """Use the construction runner's structural checks on one frozen draft."""
    check = api.assess(
        brief,
        context,
        targets,
        draft,
        graph_contract=graph,
        clarify_overlaps=False,
        boundary_contract=boundary,
    )
    check["interaction_review_handoff"] = api.interaction_handoff(brief, draft)
    check["mechanism_realization_handoff"] = api.final_handoff(draft, check)
    values = realizations.variable_checklist(brief, draft, targets, graph)
    check["variable_checklist"] = values
    check["errors"].extend(
        {"code": "variable_declaration", "item": row}
        for row in values["items"]
        if row["status"] in {"missing", "inconsistent"}
    )
    check["eligible"] &= values["status"] == "ready"
    check["shared_process_checklist"] = stage_checks.shared_checklist(brief, draft)
    check["eligible"] &= check["shared_process_checklist"]["status"] == "ready"
    return check


def same_assessment(before: dict, after: dict) -> None:
    """Feedback changes must not silently change structural admission."""
    if before != after:
        changed = sorted(
            k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
        )
        raise ValueError(f"structural assessment changed: {changed}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_result(
    event: dict, record: dict, before_check: dict, after_check: dict
) -> dict:
    before = historical.Draft.model_validate(event["before"])
    after = historical.Draft.model_validate(event["after"])
    last = {
        "status": "accepted" if event["accepted"] else "rejected",
        "transaction_applied": event["accepted"],
        "error": event["error"],
        "stage_complete": event["stage_complete"],
        "compatibility_decisions": event.get("compatibility_decisions", []),
        "required_predicates_lost": revised.lost_required_predicates(
            before_check, after_check
        ),
        "edit_effects": historical.edit_effects(before, after)
        if event["accepted"]
        else None,
        "normalizations": event.get("normalizations", []),
        "name_conflicts": event.get("name_conflicts", []),
        "definition_conflicts": event.get("definition_conflicts", []),
        "rejected_reply": None,
    }
    if not event["accepted"]:
        with suppress(ValueError, TypeError, KeyError):
            last["rejected_reply"] = visible_response(record)
    return last


def audit(source: Path) -> tuple[dict, list[dict]]:
    """Validate all saved drafts and produce revised feedback without any calls."""
    plan = sealed_read(source / "plan.json")
    if (
        plan.get("protocol") != campaign.PROTOCOL
        or plan.get("test_data_opened") is not False
    ):
        raise ValueError("requires a sealed public-only construction plan")
    hashes = {"plan.json": _digest(source / "plan.json")}
    rows, previews = [], []
    counts: Counter[str] = Counter()
    for task in plan["tasks"]:
        name = task["task_id"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("invalid task directory")
        folder = source / "results" / name
        proposal = sealed_read(folder / "proposal.json")
        hashes[str((folder / "proposal.json").relative_to(source))] = _digest(
            folder / "proposal.json"
        )
        cell = plan["cells"][task["benchmark_id"]]
        brief = campaign.contract.visible_brief(cell, task)
        context = ValidationContext.model_validate(cell["context"])
        targets = campaign.contract.target_definitions(cell)
        graph = PublicGraphContract.model_validate(cell["public_graph_contract"])
        boundary = historical.BoundaryContract.model_validate(cell["boundary_contract"])
        evidence = training.load_evidence(cell["training_evidence"])
        records = campaign.checked_records(
            folder, campaign.baseline.namespace(plan, task)
        )
        by_hash = {r["request_hash"]: r for r in records}
        for request_hash in by_hash:
            path = folder / "calls" / f"{request_hash}.json"
            hashes[str(path.relative_to(source))] = _digest(path)
        answers, last = {}, None
        counts["tasks"] += 1
        row = {
            "task": name,
            "benchmark_id": task["benchmark_id"],
            "seed": task["seed"],
            "entry_reason": entry_reason(proposal["before_repair"]["assessment"]),
        }
        for checkpoint in ("before_repair", "final"):
            saved = proposal if checkpoint == "final" else proposal[checkpoint]
            draft = historical.Draft.model_validate(saved["draft"])
            old = full_assessment(
                historical, brief, context, targets, graph, boundary, draft
            )
            new = full_assessment(
                revised, brief, context, targets, graph, boundary, draft
            )
            same_assessment(
                old,
                {
                    k: v
                    for k, v in saved["assessment"].items()
                    if k != "training_compatibility"
                },
            )
            same_assessment(old, new)
            counts["saved_assessments_reproduced"] += 1
            row[checkpoint] = {
                "eligible": old["eligible"],
                "errors": old["errors"],
                "model_view": historical.model_view(draft),
                "training_compatibility": saved["assessment"]["training_compatibility"],
            }
        for path in sorted((folder / "construction/events").glob("*.json")):
            event = sealed_read(path)
            hashes[str(path.relative_to(source))] = _digest(path)
            record = by_hash[event["request_hash"]]
            before = historical.Draft.model_validate(event["before"])
            after = historical.Draft.model_validate(event["after"])
            before_check = full_assessment(
                historical, brief, context, targets, graph, boundary, before
            )
            after_check = full_assessment(
                historical, brief, context, targets, graph, boundary, after
            )
            if event["stage"] == "repair":
                old_advice = training.compatibility_feedback(
                    historical.project(before), evidence
                )
                new_advice = training.compatibility_feedback(
                    historical.project(before),
                    evidence,
                    identity_policy=training.SEMANTIC_QUESTION_IDENTITY,
                )
                same_assessment(
                    before_check,
                    full_assessment(
                        revised, brief, context, targets, graph, boundary, before
                    ),
                )
                same_assessment(
                    after_check,
                    full_assessment(
                        revised, brief, context, targets, graph, boundary, after
                    ),
                )
                counts["repair_assessments_unchanged"] += 2
                state = advisory_state(new_advice, answers)
                old_packet = json.loads(
                    record["request"]["body"]["messages"][1]["content"]
                )
                new_packet = revised.payload(
                    "repair",
                    brief,
                    context,
                    targets,
                    old_packet["public_brief"],
                    before,
                    graph,
                    last,
                    old_packet.get("feedback", {}).get("continuation"),
                    boundary_contract=boundary,
                    training_highlights=old_packet.get("training_highlights"),
                    compatibility_questions=state["pending_questions"],
                    compatibility_state=state,
                )
                # The historical budget is a frozen reporting fact, not a new run.
                budget = old_packet.get("feedback", {}).get("repair_budget")
                if budget is not None:
                    new_packet["feedback"]["repair_budget"] = budget
                previews.append(
                    {
                        "task": name,
                        "event": event["index"],
                        "source_request_sha256": event["request_hash"],
                        "source_before_sha256": content_hash(event["before"]),
                        "before_structural": before_check["eligible"],
                        "after_structural": after_check["eligible"],
                        "accepted_historical_reply": event["accepted"],
                        "historical_before": old_packet,
                        "revised_before": new_packet,
                        "historical_advice": old_advice,
                        "revised_advice": state,
                        "historical_after": historical.model_view(after),
                        "required_predicates_lost": revised.lost_required_predicates(
                            before_check, after_check
                        ),
                    }
                )
                if event["accepted"]:
                    translated = translate_decisions(
                        event.get("compatibility_questions", []),
                        new_advice["advisory_questions"],
                        event.get("compatibility_decisions", []),
                    )
                    answers.update({d["question_id"]: d for d in translated})
                counts["repair_requests"] += 1
            last = _last_result(event, record, before_check, after_check)
        rows.append(row)
    for relative, digest in hashes.items():
        if _digest(source / relative) != digest:
            raise ValueError(f"historical source changed during audit: {relative}")
    return {
        "protocol": PROTOCOL,
        "source_plan_sha256": plan["artifact_sha256"],
        "historical_policy": historical.POLICY,
        "revised_policy": revised.POLICY,
        "scope": SCOPE,
        "counts": dict(counts),
        "prompt_contract": {
            "system": revised.SYSTEM,
            "stages": {
                stage: {
                    "instructions": revised.stage_text(stage),
                    "instructions_changed": revised.stage_text(stage)
                    != historical.stage_text(stage),
                    "template": revised.response_model(stage)(
                        stage_complete=False
                    ).model_dump(mode="json"),
                    "provider_schema": strict_provider_schema(
                        revised.response_model(stage).model_json_schema()
                    ),
                }
                for stage in ("variables", "shared_laws", "equations", "repair")
            },
        },
        "rows": rows,
        "source_file_sha256": hashes,
        "historical_files_modified": False,
        "llm_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
    }, previews


def render(report: dict, previews: list[dict], destination: Path) -> None:
    """Write derived, escaped pages; preserve both actual and proposed payloads."""
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "requests.json").write_text(json.dumps(previews, indent=2) + "\n")
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Repair feedback audit</title>",
        "<style>body{max-width:1100px;margin:2rem auto;font:16px system-ui}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "td,th{padding:.5rem;text-align:left}table{border-collapse:collapse}"
        "tr{border-bottom:1px solid #ccc}</style>",
        "<h1>Evidence feedback v2: current prompts and saved repair examples</h1>",
        f"<p>{html.escape(SCOPE)}</p>",
        f"<pre>{html.escape(json.dumps(report['counts'], indent=2))}</pre>",
        "<p><a href='audit.json'>Audit and final scientific questions</a> · "
        "<a href='requests.json'>All request comparisons</a></p>",
        "<p><strong>A structural pass is not a scientific success.</strong> "
        "A model already conflicting with conditional trajectory evidence can "
        "appropriately enter scientific repair. A subsequent structural failure "
        "does not establish loss of a scientifically correct model.</p>",
    ]
    contract = report.get("prompt_contract", {})
    if contract:
        parts.extend(
            ("<h2>System instruction (unchanged)</h2>", _pre(contract["system"]))
        )
        for stage, details in contract["stages"].items():
            status = "updated" if details["instructions_changed"] else "unchanged"
            parts.extend(
                (
                    f"<h2>{html.escape(stage)} — {status}</h2>",
                    _pre(details["instructions"]),
                    "<details><summary>Reply template and strict schema</summary>",
                    _pre(details["template"]),
                    _pre(details["provider_schema"]),
                    "</details>",
                )
            )
    for row in report.get("rows", []):
        if (
            "dalla_man_t2_canonical_named_easy" in row["benchmark_id"]
            and row["seed"] == 2
        ):
            parts.extend(
                (
                    "<h2>T2-easy seed 2: clarify what failed</h2>",
                    "<p>Its glucose balance was not meal-only: it also depended on "
                    "U and EGP, with a delayed insulin pathway into U. The initial "
                    "questions instead identified an isolated scalar insulin "
                    "subsystem under identical forcing. Repair was scientifically "
                    "motivated, but did not produce a valid repaired model and "
                    "introduced algebraic cycles. This is an unsuccessful repair "
                    "of an already "
                    "limited model, not evidence that a scientifically correct "
                    "model became incorrect.</p>",
                    "<details><summary>Actual pre-repair model and witnesses</summary>",
                    _pre(row["before_repair"]["model_view"]),
                    _pre(
                        row["before_repair"]["training_compatibility"][
                            "advisory_questions"
                        ]
                    ),
                    "</details>",
                )
            )
    tasks = sorted({p["task"] for p in previews})
    parts.extend(
        (
            "<h2>Actual saved predecessors</h2>",
            "<label>Case/seed: <select id='task-filter'><option value=''>All</option>",
        )
    )
    parts.extend(
        f"<option value='{html.escape(t, quote=True)}'>{html.escape(t)}</option>"
        for t in tasks
    )
    parts.extend(
        (
            "</select></label>",
            "<table><thead><tr><th>Task/event</th>"
            "<th>Historical structural before → after</th>"
            "<th>Unanswered revised questions</th></tr></thead><tbody>",
        )
    )
    for index, preview in enumerate(previews):
        filename = f"request-{index:03d}.html"
        label = f"{preview['task']} / {preview['event']}"
        parts.append(
            f"<tr data-task='{html.escape(preview['task'], quote=True)}'>"
            f"<td><a href='{filename}'>{html.escape(label)}</a></td>"
            f"<td>{preview['before_structural']} → {preview['after_structural']}</td>"
            f"<td>{len(preview['revised_advice']['pending_questions'])}</td></tr>"
        )
        body = [
            "<!doctype html><meta charset='utf-8'>",
            f"<title>{html.escape(label)}</title>",
            "<style>body{max-width:1100px;margin:2rem auto;font:16px system-ui}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>",
            "<p><a href='PROMPTS.html'>Current prompts and all requests</a></p>",
            f"<h1>{html.escape(label)}</h1>",
            f"<p>{html.escape(SCOPE)}</p>",
        ]
        revised_packet = preview["revised_before"]
        body.extend(
            (
                "<h2>Current model view</h2>",
                _pre(revised_packet.get("model_view", {})),
                "<h2>Focused repair work</h2>",
                _pre(revised_packet.get("feedback", {}).get("repair_work", {})),
                "<h2>Current advisory answer state</h2>",
                _pre(preview["revised_advice"]),
                "<details><summary>Marked request changes "
                "(minus old, plus new)</summary>",
            )
        )
        difference = "\n".join(
            unified_diff(
                json.dumps(
                    preview["historical_before"], indent=2, sort_keys=True
                ).splitlines(),
                json.dumps(revised_packet, indent=2, sort_keys=True).splitlines(),
                fromfile="historical actual request",
                tofile="revised same predecessor",
                lineterm="",
            )
        )
        body.extend((_pre(difference), "</details>"))
        for key, title in (
            ("revised_before", "Revised request on the actual historical predecessor"),
            ("historical_before", "Original request"),
            ("historical_after", "Actual historical reply outcome"),
            (
                "required_predicates_lost",
                "Required structural predicates lost by the actual edit",
            ),
        ):
            body.extend(
                (
                    f"<details><summary>{title}</summary>",
                    f"<pre>{html.escape(json.dumps(preview[key], indent=2))}</pre>",
                    "</details>",
                )
            )
        (destination / filename).write_text("\n".join(body) + "\n")
    parts.append("</tbody></table>")
    parts.append(
        "<script>document.getElementById('task-filter').addEventListener('change',"
        "function(){document.querySelectorAll('tr[data-task]').forEach("
        "row=>{row.hidden=Boolean(this.value&&row.dataset.task!==this.value);});});"
        "</script>"
    )
    for name in ("REPORT.html", "PROMPTS.html"):
        (destination / name).write_text("\n".join(parts) + "\n")


def _pre(value: object) -> str:
    """Escape scientific text rather than interpreting it as HTML."""
    text = value if isinstance(value, str) else json.dumps(value, indent=2)
    return f"<pre>{html.escape(text)}</pre>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.output.resolve()
    if destination == source or source in destination.parents:
        parser.error("output must be outside the historical source directory")
    report, previews = audit(source)
    render(report, previews, destination)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"rows", "source_file_sha256", "prompt_contract"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
