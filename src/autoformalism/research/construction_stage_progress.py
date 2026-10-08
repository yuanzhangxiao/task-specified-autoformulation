"""Index saved staged transactions without treating acceptance as scientific success."""

from __future__ import annotations

import html
import json
from pathlib import Path

from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_comparison as campaign
from autoformalism.search import construction_ledger as ledger
from autoformalism.search.construction_feedback import variable_completion


def facts(brief, draft: dict) -> dict:
    """Report declaration and consumer facts, never a scientific correctness score."""
    value = ledger.Draft.model_validate(draft)
    required = {r.id for r in brief.requirements if r.requires_dynamic_memory}
    bound = {b.requirement_id for b in value.mechanism_bindings}
    return {
        "variables": draft["variables"],
        "missing_public_targets": variable_completion(brief, value)[
            "missing_public_targets"
        ],
        "missing_required_bindings": sorted(required - bound),
        "processes": draft["processes"],
        "multi_consumer_processes": [
            p.name for p in value.processes if len(p.uses) > 1
        ],
        "single_consumer_processes": [
            p.name for p in value.processes if len(p.uses) == 1
        ],
        "invalid_pairwise_transfers": [
            p.name
            for p in value.processes
            if p.kind == "transfer"
            and (
                len(p.uses) != 2 or {u.sign for u in p.uses} != {"positive", "negative"}
            )
        ],
        "equations": draft["equations"],
        "mechanism_bindings": draft["mechanism_bindings"],
        "feedback_bindings": draft["feedback_bindings"],
    }


def audit(source: Path) -> dict:
    """Keep first replies, stage exits and global repair separately attributable."""
    plan = sealed_read(source / "plan.json")
    if (
        plan.get("protocol") != campaign.PROTOCOL
        or plan.get("test_data_opened") is not False
    ):
        raise ValueError("requires a saved public construction comparison")
    rows = []
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
        events = [
            sealed_read(p)
            for p in sorted((directory / "construction/events").glob("*.json"))
        ]
        brief = campaign.contract.visible_brief(
            plan["cells"][task["benchmark_id"]], task
        )
        stages = {}
        for stage in dict.fromkeys(e["stage"] for e in events):
            group = [e for e in events if e["stage"] == stage]
            replies = []
            for event in group:
                record = by_hash[event["request_hash"]]
                try:
                    reply = visible_response(record)
                except (ValueError, TypeError, KeyError):
                    reply = None
                replies.append(
                    {
                        "event": event["index"],
                        "attempt": event["attempt"],
                        "accepted": event["accepted"],
                        "stage_complete": event["stage_complete"],
                        "changed_draft": event["before"] != event["after"],
                        "error": event["error"],
                        "reply": reply,
                        "call_file": str(
                            directory / "calls" / f"{event['request_hash']}.json"
                        ),
                        "event_file": str(
                            directory
                            / "construction/events"
                            / f"{event['index']:03d}.json"
                        ),
                    }
                )
            stages[stage] = {
                "replies": replies,
                "first_retained": facts(brief, group[0]["after"]),
                "stage_exit": facts(brief, group[-1]["after"]),
                "note": "Retained snapshots exclude rejected edits; inspect replies. "
                "Later calls may continue the stage or repair invalid edits, "
                "without scientific feedback.",
            }
        before_path, final_path = (
            directory / "construction/before_repair.json",
            directory / "proposal.json",
        )
        before = sealed_read(before_path) if before_path.exists() else None
        final = sealed_read(final_path) if final_path.exists() else None
        rows.append(
            {
                "task": task,
                "stages": stages,
                "pre_global": before,
                "final": {
                    "status": final["status"],
                    "facts": facts(brief, final["draft"]),
                    "assessment": final["assessment"],
                    "cost": final["cost"],
                }
                if final
                else None,
            }
        )
    return {
        "protocol": "saved-construction-stage-progress-1",
        "source": str(source.resolve()),
        "plan_sha256": plan["artifact_sha256"],
        "scope": "Recorded transactions and facts, not scientific correctness. "
        "Whole-topology feedback occurred at global repair; stage acceptance "
        "is not equivalent.",
        "rows": rows,
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }


def write_report(result: dict, output: Path) -> None:
    """Render escaped evidence outside the historical archive."""
    if output.resolve().is_relative_to(Path(result["source"]).resolve()):
        raise ValueError("output must be outside the source archive")
    output.mkdir(parents=True, exist_ok=True)
    (output / "stages.json").write_text(json.dumps(result, indent=2) + "\n")

    def block(value):
        return "<pre>" + html.escape(json.dumps(value, indent=2)) + "</pre>"

    parts = [
        "<!doctype html><meta charset='utf-8'>"
        "<title>Construction stage evidence</title>",
        "<style>body{max-width:1100px;margin:2em auto;font-family:system-ui}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f5f5f5;padding:1em}"
        "summary{cursor:pointer;padding:.6em}</style>",
        "<h1>Construction stage evidence</h1><p>"
        + html.escape(result["scope"])
        + "</p>",
    ]
    for row in result["rows"]:
        name = html.escape(row["task"]["task_id"])
        parts.append(f'<h2 id="{name}">{name}</h2>')
        for stage, data in row["stages"].items():
            parts.append(
                f"<details><summary>{html.escape(stage)}: "
                f"{len(data['replies'])} calls</summary>"
            )
            parts.append(block(data) + "</details>")
        parts.append(
            "<details><summary>Before global repair</summary>"
            + block(row["pre_global"])
            + "</details>"
        )
        parts.append(
            "<details><summary>Final retained model</summary>"
            + block(row["final"])
            + "</details>"
        )
    (output / "STAGES.html").write_text("\n".join(parts))
