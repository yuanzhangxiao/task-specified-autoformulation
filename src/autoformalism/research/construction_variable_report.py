"""Variable-stage evidence, with readiness separate from scientific correctness."""

from collections import Counter
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.search import construction_checklist as checklist
from autoformalism.search import construction_ledger as ledger
from autoformalism.search.public_graph_obligations import PublicGraphContract


def report(root: Path, plan: dict) -> dict:
    """Summarize already-verified records; never synthesize a missing stage result."""
    from autoformalism.research.construction_contract import (
        target_definitions,
        visible_brief,
    )

    rows = []
    for task in plan["tasks"]:
        directory = root / "results" / task["task_id"]
        cell = plan["cells"][task["benchmark_id"]]
        brief = visible_brief(cell, task)
        events = [
            sealed_read(p)
            for p in sorted((directory / "construction/events").glob("*.json"))
        ]
        variables = [e for e in events if e["stage"] == "variables"]
        result_path = directory / "proposal.json"
        final = sealed_read(result_path) if result_path.exists() else None
        graph = PublicGraphContract.model_validate(cell["public_graph_contract"])
        definitions = target_definitions(cell)
        final_checks = (
            checklist.variable_checklist(
                brief, ledger.Draft.model_validate(final["draft"]), definitions, graph
            )
            if final
            else None
        )
        rows.append(
            {
                "task": task["task_id"],
                "variable_calls": len(variables),
                "first_reply_accepted": variables[0]["accepted"] if variables else None,
                "first_retained": variables[0]["variable_checklist_after"]
                if variables
                else None,
                "last_variable_snapshot": variables[-1]["variable_checklist_after"]
                if variables
                else None,
                "variable_stage_completed": variables[-1]["stage_complete"]
                if variables
                else None,
                "after_topology_and_global_repair": final_checks,
                "final_topology_status": final["status"] if final else "pending",
                "turns": [
                    {
                        "event": e["index"],
                        "request_hash": e["request_hash"],
                        "stage": e["stage"],
                        "accepted": e["accepted"],
                        "error": e["error"],
                        "stage_complete": e["stage_complete"],
                        "retained_checklist": e["variable_checklist_after"],
                    }
                    for e in events
                ],
            }
        )
    checkpoints = {}
    for key in (
        "first_retained",
        "last_variable_snapshot",
        "after_topology_and_global_repair",
    ):
        present = [r[key] for r in rows if r[key] is not None]
        checkpoints[key] = {
            "available": len(present),
            "declaration_ready": sum(r["status"] == "ready" for r in present),
            "all_public_targets_declared": sum(
                not r["missing_public_targets"] for r in present
            ),
            "blocking_items_by_code": dict(
                Counter(
                    item["code"]
                    for r in present
                    for item in r["items"]
                    if item["status"] in {"missing", "inconsistent"}
                )
            ),
        }
    value = {
        "policy": checklist.POLICY,
        "plan_sha256": plan["artifact_sha256"],
        "planned_tasks": len(rows),
        "checkpoints": checkpoints,
        "completed_variable_stages": sum(
            r["variable_stage_completed"] is True for r in rows
        ),
        "scope": "Retained declarations, not scientific correctness. First-reply "
        "acceptance is reported separately. A last variable snapshot may precede "
        "exhaustion/interruption rather than successful stage completion. "
        "Assignments do not certify pathways; topology checks remain separate.",
        "rows": rows,
        "report_llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }
    atomic_json(root / "VARIABLES.json", value)
    lines = [
        "# Variable-stage checklist evidence",
        "",
        value["scope"],
        "",
        "| Task | Calls | First retained | Last variable snapshot | Stage completed "
        "| After topology/repair |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]

    def status(v: dict | None) -> str:
        if v is None:
            return "pending"
        return "ready" if not v["blocking_items"] else "; ".join(v["blocking_items"])

    for row in rows:
        lines.append(
            f"| {row['task']} | {row['variable_calls']} | "
            f"{status(row['first_retained'])} | "
            f"{status(row['last_variable_snapshot'])} | "
            f"{row['variable_stage_completed']} | "
            f"{status(row['after_topology_and_global_repair'])} |"
        )
    (root / "VARIABLES.md").write_text("\n".join(lines) + "\n")
    return value
