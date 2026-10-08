#!/usr/bin/env python3
"""Replay saved replies against their displayed predecessors; no new model chain."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_comparison as campaign
from autoformalism.research import construction_contract as contracts
from autoformalism.search import construction_bookkeeping as bookkeeping
from autoformalism.search import construction_handoff as handoff
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_schedules as schedules
from autoformalism.search.public_graph_obligations import PublicGraphContract


def audit(
    source: Path, output: Path, *, bookkeeping_policy: bookkeeping.Policy = "legacy"
) -> dict:
    """Keep transaction acceptance separate from whole-topology eligibility."""
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("audit output must be outside historical source")
    plan = sealed_read(source / "plan.json")
    bookkeeping.validate_policy(
        bookkeeping_policy,
        "current" if plan.get("study") == "prompt_comparison" else None,
    )
    improved_bookkeeping = bookkeeping_policy == bookkeeping.POLICY
    if (
        plan.get("protocol")
        not in {
            "phase-c-construction-comparison-5",
            "phase-c-construction-comparison-6",
            "phase-c-construction-comparison-7",
            "phase-c-construction-comparison-8",
            "phase-c-construction-comparison-9",
            "phase-c-construction-comparison-10",
        }
        or plan.get("test_data_opened") is not False
    ):
        raise ValueError("requires a saved development construction plan")
    names = [t["task_id"] for t in plan["tasks"]]
    if len(names) != len(set(names)) or any(
        Path(n).name != n or n in {".", ".."} for n in names
    ):
        raise ValueError("invalid saved task IDs")
    rows, counts, missing = [], Counter(), []
    for task in plan["tasks"]:
        directory = source / "results" / task["task_id"]
        records = campaign.checked_records(
            directory, campaign.baseline.namespace(plan, task)
        )
        by_hash = {r["request_hash"]: r for r in records}
        if not records:
            missing.append(task["task_id"])
        cell = plan["cells"][task["benchmark_id"]]
        brief = contracts.visible_brief(cell, task)
        contract = PublicGraphContract.model_validate(cell["public_graph_contract"])
        for path in sorted((directory / "construction/events").glob("*.json")):
            event = sealed_read(path)
            record = by_hash[event["request_hash"]]
            payload = json.loads(record["request"]["body"]["messages"][1]["content"])
            before = ledger.Draft.model_validate(event["before"])
            accepted, error, check, log = False, None, None, []
            try:
                normalized, log = ledger.normalize_reply(
                    brief,
                    visible_response(record),
                    graph_contract=contract,
                    draft=before,
                    ignore_definition_description=improved_bookkeeping,
                )
                patch = ledger.DraftPatch.model_validate(normalized)
                schedules.validate_scope(
                    task["policy"],
                    event["stage"],
                    payload["selected_lhs"],
                    patch,
                    before,
                )
                after = ledger.apply_patch(brief, before, patch)
                after, extra = handoff.normalize_consumers(after, patch, before)
                log.extend(extra)
                accepted = True
                check = ledger.assess(
                    brief,
                    ValidationContext.model_validate(cell["context"]),
                    contracts.target_definitions(cell),
                    after,
                    graph_contract=contract,
                    clarify_overlaps=True,
                )
                if improved_bookkeeping:
                    check = bookkeeping.assessment_context(check, after)
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
            counts.update(x["code"] for x in log if accepted)
            rows.append(
                {
                    "task": task["task_id"],
                    "event": event["index"],
                    "request_hash": event["request_hash"],
                    "source_event_sha256": event["artifact_sha256"],
                    "historically_accepted": event["accepted"],
                    "currently_accepted": accepted,
                    "error": error,
                    "normalizations": log,
                    "counterfactual_eligible": check["eligible"] if check else None,
                    "counterfactual_errors": check["errors"] if check else None,
                    "edit_effects": handoff.edit_effects(before, after)
                    if accepted
                    else None,
                }
            )
    return sealed_write(
        output,
        {
            "protocol": "saved-construction-handoff-audit-1",
            "bookkeeping_policy": bookkeeping_policy,
            "source_plan_sha256": plan["artifact_sha256"],
            "audit_runtime": campaign.baseline.source_identity(),
            "audit_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "saved_attempts": len(rows),
            "planned_tasks": len(plan["tasks"]),
            "tasks_without_saved_calls": missing,
            "historically_accepted": sum(r["historically_accepted"] for r in rows),
            "currently_accepted": sum(r["currently_accepted"] for r in rows),
            "unexpected_acceptance_regressions": [
                {"task": r["task"], "event": r["event"], "error": r["error"]}
                for r in rows
                if r["historically_accepted"] and not r["currently_accepted"]
            ],
            "accepted_normalizations": dict(counts),
            "rows": rows,
            "scope": "Each reply uses its actual displayed predecessor, never a "
            "hypothetical new chain. Counts are transactions, not recovered models.",
            "llm_calls": 0,
            "optimizer_calls": 0,
            "test_data_opened": False,
            "historical_campaign_modified": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bookkeeping-policy", choices=bookkeeping.POLICIES, default="legacy"
    )
    args = parser.parse_args()
    result = audit(args.source, args.output, bookkeeping_policy=args.bookkeeping_policy)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"rows", "audit_runtime"}},
            indent=2,
        )
    )
