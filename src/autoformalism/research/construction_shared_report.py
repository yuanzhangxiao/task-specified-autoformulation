"""Shared-process stage evidence, independent of eventual topology assembly."""

from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research.construction_contract import visible_brief
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_stage_checks as checks


def report(root: Path, plan: dict) -> dict:
    """Keep requested completion, local consistency and assembled success distinct."""
    rows = []
    for task in plan["tasks"]:
        directory = root / "results" / task["task_id"]
        brief = visible_brief(plan["cells"][task["benchmark_id"]], task)
        events = [
            sealed_read(p)
            for p in sorted((directory / "construction/events").glob("*.json"))
        ]
        shared = [e for e in events if e["stage"] == "shared_laws"]
        path = directory / "proposal.json"
        final = sealed_read(path) if path.exists() else None

        def snapshot(draft: dict | None, public=brief) -> dict | None:
            return (
                checks.shared_checklist(public, ledger.Draft.model_validate(draft))
                if draft is not None
                else None
            )

        rows.append(
            {
                "task": task["task_id"],
                "shared_calls": len(shared),
                "immediate_repair_calls": max(0, len(shared) - 1),
                "first_reply_accepted": shared[0]["accepted"] if shared else None,
                "stage_completed": shared[-1]["stage_complete"] if shared else None,
                "first_retained": snapshot(shared[0]["after"]) if shared else None,
                "last_shared_snapshot": snapshot(shared[-1]["after"])
                if shared
                else None,
                "before_global_repair": snapshot(final["before_repair"]["draft"])
                if final
                else None,
                "after_global_repair": snapshot(final["draft"]) if final else None,
                "final_topology_status": final["status"] if final else "pending",
                "turns": [
                    {
                        "event": e["index"],
                        "stage": e["stage"],
                        "request_hash": e["request_hash"],
                        "accepted": e["accepted"],
                        "error": e["error"],
                        "stage_complete": e["stage_complete"],
                        "retained_process_checks": e["shared_process_checklist_after"],
                    }
                    for e in events
                ],
            }
        )
    checkpoints = {}
    for name in (
        "first_retained",
        "last_shared_snapshot",
        "before_global_repair",
        "after_global_repair",
    ):
        available = [r[name] for r in rows if r[name] is not None]
        checkpoints[name] = {
            "available": len(available),
            "locally_consistent": sum(x["status"] == "ready" for x in available),
            "with_shared_declarations": sum(
                x["shared_declarations"] > 0 for x in available
            ),
            "with_local_declarations": sum(
                x["local_declarations"] > 0 for x in available
            ),
            "empty_decisions": sum(not x["processes"] for x in available),
        }
    value = {
        "policy": checks.POLICY,
        "plan_sha256": plan["artifact_sha256"],
        "scope": "Local declaration checks, not scientific correctness or verified "
        "assembly. Empty decisions are valid. Named local contributions are not "
        "shared laws. Saved partial edits do not establish stage completion. "
        "Unknown references and whole-graph checks remain deferred.",
        "planned_tasks": len(rows),
        "checkpoints": checkpoints,
        "completed_shared_stages": sum(r["stage_completed"] is True for r in rows),
        "rows": rows,
        "report_llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }
    atomic_json(root / "SHARED_PROCESSES.json", value)

    def display(v: dict | None) -> str:
        if v is None:
            return "pending"
        return (
            f"{v['status']}; shared={v['shared_declarations']}, "
            f"local={v['local_declarations']}"
        )

    lines = [
        "# Shared-process stage evidence",
        "",
        value["scope"],
        "",
        "| Task | Calls | First retained | Last shared snapshot | Stage completed "
        "| Before global repair | After global repair | Final topology |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['shared_calls']} | "
            f"{display(row['first_retained'])} | "
            f"{display(row['last_shared_snapshot'])} | {row['stage_completed']} | "
            f"{display(row['before_global_repair'])} | "
            f"{display(row['after_global_repair'])} | {row['final_topology_status']} |"
        )
    (root / "SHARED_PROCESSES.md").write_text("\n".join(lines) + "\n")
    return value
