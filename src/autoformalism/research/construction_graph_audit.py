"""Read-only counterfactual assessment of saved topology checkpoints."""

from collections import Counter
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_baseline as baseline
from autoformalism.research import construction_comparison as campaign
from autoformalism.research import construction_contract as contracts
from autoformalism.research.construction_obligations import reviewed_contract
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_ledger as ledger
from autoformalism.staged_topology import content_hash

PROTOCOL = "saved-public-graph-audit-1"


def audit(source: Path, output: Path) -> dict:
    """Use actual initial/final drafts; never simulate new replies or change sources."""
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("audit output must be separate from historical source")
    plan = sealed_read(source / "plan.json")
    if (
        plan.get("protocol")
        not in {f"phase-c-construction-comparison-{n}" for n in (1, 2, 3, 4)}
        or plan.get("test_data_opened") is not False
    ):
        raise ValueError("requires a saved public construction-comparison plan")
    ids = [t["task_id"] for t in plan["tasks"]]
    if len(ids) != len(set(ids)) or any(
        Path(n).name != n or n in {".", ".."} for n in ids
    ):
        raise ValueError("invalid saved task IDs")
    public_contracts = {
        name: reviewed_contract(name, PublicScientificBrief.model_validate(c["brief"]))
        for name, c in plan["cells"].items()
    }
    rows, inputs = [], {"plan.json": plan["artifact_sha256"]}
    for task in plan["tasks"]:
        directory = baseline.location(source, task)
        # Verify immutable calls and predecessor/successor chains using historical
        # identities, without accepting the old plan as a resumable current run.
        campaign.checked_records(directory, baseline.namespace(plan, task))
        cell = plan["cells"][task["benchmark_id"]]
        brief = contracts.visible_brief(cell, task)
        context = ValidationContext.model_validate(cell["context"])
        stages = {}
        for stage, path in (
            ("initial", directory / "construction/before_repair.json"),
            ("final", directory / "proposal.json"),
        ):
            if not path.exists():
                stages[stage] = {"status": "missing"}
                continue
            saved = sealed_read(path)
            inputs[str(path.relative_to(source))] = saved["artifact_sha256"]
            draft = ledger.Draft.model_validate(saved["draft"])
            base = ledger.assess(
                brief, context, contracts.target_definitions(cell), draft
            )
            new = ledger.assess(
                brief,
                context,
                contracts.target_definitions(cell),
                draft,
                graph_contract=public_contracts[task["benchmark_id"]],
            )
            stages[stage] = {
                "status": "audited",
                "draft_sha256": content_hash(saved["draft"]),
                "historical_eligible": saved["assessment"]["eligible"],
                "current_base_eligible": base["eligible"],
                "prospective_eligible": new["eligible"],
                "graph_check_status": new["graph_check_status"],
                "checks": new["reviewed_public_graph_checks"],
                "base_errors": base["errors"],
                "new_errors": [
                    e for e in new["errors"] if e["code"] == "reviewed_public_graph"
                ],
            }
        rows.append({**task, **stages})
    final = [r["final"] for r in rows if r["final"]["status"] == "audited"]
    result = {
        "protocol": PROTOCOL,
        "source_plan_sha256": plan["artifact_sha256"],
        "audit_runtime": baseline.source_identity(),
        "inputs": inputs,
        "public_contracts": {
            k: v.model_dump(mode="json") for k, v in public_contracts.items()
        },
        "planned_tasks": len(rows),
        "final_drafts_available": len(final),
        "historical_eligible": sum(r["historical_eligible"] for r in final),
        "current_base_eligible": sum(r["current_base_eligible"] for r in final),
        "prospective_eligible": sum(r["prospective_eligible"] for r in final),
        "newly_flagged_final_drafts": [
            r["task_id"]
            for r in rows
            if r["final"]["status"] == "audited"
            and r["final"]["current_base_eligible"]
            and not r["final"]["prospective_eligible"]
        ],
        "final_check_status_counts": dict(
            Counter(c["status"] for r in final for c in r["checks"])
        ),
        "rows": rows,
        "scope": (
            "Counterfactual predicates on actual saved drafts. The added contract was "
            "not displayed historically; this is not new live evidence or a "
            "science score."
        ),
        "llm_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "historical_campaign_modified": False,
    }
    return sealed_write(output, result)
