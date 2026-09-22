"""Replay saved repair decisions in their actual historical contexts, without calls."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ModelValidationError
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import basin_equation_audit as equations
from autoformalism.rebuttal import basin_repair_pilot as historical
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search import basin_model_repair_v2 as repair
from autoformalism.staged_topology import content_hash

PROTOCOL = "basin-repair-handoff-audit-1"
REPO = Path(__file__).resolve().parents[3]


def _files(source, plan):
    """Allowlist public repair records; no recursive scans or trajectory imports."""
    names = {"plan.json", "baseline.json"}
    task_ids = [t["task_id"] for t in plan["tasks"]]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate historical task IDs")
    for task in task_ids:
        if Path(task).name != task or task in {".", ".."}:
            raise ValueError("unsafe task ID")
        prefix = f"results/{task}"
        name = f"{prefix}/proposal.json"
        names.add(name)
        if (source / name).exists():
            p = sealed_read(source / name)
            for i, event in enumerate(p["events"]):
                digest = event["request_hash"]
                if len(digest) != 64 or any(
                    c not in "0123456789abcdef" for c in digest
                ):
                    raise ValueError("invalid request hash")
                names |= {f"{prefix}/attempt-{i}.json", f"{prefix}/calls/{digest}.json"}
    result = {}
    for name in sorted(names):
        path = source / name
        if not path.resolve().is_relative_to(source):
            raise ValueError("source symlink escapes campaign")
        result[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        )
    return result


def _outcome(bundle, parent, case, surveys):
    static = repair.assessment(bundle, case, surveys)
    failed = [c["code"] for c in static["checks"] if c["status"] == "fail"]
    return {
        "status": "static_repair_required"
        if failed
        else "unchanged_parent"
        if bundle["candidate"] == parent["candidate"]
        else "eligible_draft",
        "static_assessment": static,
        "candidate_sha256": content_hash(bundle["candidate"]),
    }


def _row(source, plan, baseline, task):
    """Evaluate replies independently; never splice in counterfactual predecessors."""
    name = task["task_id"]
    if not (source / "results" / name / "proposal.json").exists():
        return {
            "task": name,
            "status": "unavailable",
            "reason": "missing proposal",
            "attempts": [],
        }
    # Legacy replay verifies raw call hashes, exact displayed models, attempt seals,
    # transactions and terminal draft. It intentionally does not require old code
    # hashes to equal this newly versioned audit executable.
    saved = historical.checked_proposal(source, plan, task)
    parent = baseline["parents"][name]["bundle"]
    surveys = equations._surveys(plan["cells"][task["case"]])
    current, aliases, attempts = parent, {}, []
    for event in saved["events"]:
        row = {
            "attempt": event["attempt"],
            "request_hash": event["request_hash"],
            "historical_error": event["error"],
            "historical_decision": event["decision"],
            "context_sha256": content_hash(current),
            "counterfactual_only": True,
        }
        if event["raw"] is None:
            row.update(
                classification="delivery_unavailable",
                error="No visible saved JSON reply.",
            )
        else:
            try:
                outcome = repair.transition(
                    current,
                    parent,
                    event["raw"],
                    task["case"],
                    surveys,
                    aliases=aliases,
                )
                row.update(
                    classification=outcome["status"],
                    bundle=outcome["bundle"],
                    transaction=outcome["transaction"],
                    static_assessment=outcome["static_assessment"],
                )
            except (ValueError, ModelValidationError) as exc:
                row.update(classification="mechanically_blocked", error=str(exc))
        row["newly_mechanically_valid"] = bool(event["error"]) and row[
            "classification"
        ] not in {"delivery_unavailable", "mechanically_blocked"}
        row["unexpected_acceptance_regression"] = not event["error"] and row[
            "classification"
        ] in {"delivery_unavailable", "mechanically_blocked"}
        attempts.append(row)
        # Reconstruct ONLY the actual historical predecessor for the next reply.
        if event["transaction"]:
            current, _, _ = historical._transition(
                current, parent, event["raw"], task["case"], surveys
            )
            replacements = event["transaction"]["parameter_replacements"]
            aliases = {
                k: repair.old._substitute(v, replacements) for k, v in aliases.items()
            }
            aliases.update(replacements)
    if current != saved["bundle"]:
        raise ValueError("historical final context differs")
    final = _outcome(current, parent, task["case"], surveys)
    return {
        "task": name,
        "status": "audited",
        "historical_status": saved["status"],
        "attempts": attempts,
        "saved_final_draft": {**final, "bundle": current},
        "newly_eligible_saved_draft": saved["status"] == "unconfirmed_trial"
        and final["status"] == "eligible_draft",
    }


def audit(source: Path, output: Path):
    """Freeze file identities and checkpoint CPU analysis in a separate directory."""
    source, output = source.resolve(), output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("audit output must be separate from source")
    plan, baseline = (
        sealed_read(source / "plan.json"),
        sealed_read(source / "baseline.json"),
    )
    if (
        plan["protocol"] != historical.PROTOCOL
        or plan["policy"] != repair.old.POLICY
        or plan["baseline_sha256"] != baseline["artifact_sha256"]
        or plan["test_data_opened"]
        or plan["private_reference_opened"]
        or set(baseline["parents"]) != {t["task_id"] for t in plan["tasks"]}
    ):
        raise ValueError("historical repair identity differs")
    files = _files(source, plan)
    frozen = {
        "protocol": PROTOCOL,
        "policy": repair.POLICY,
        "source": str(source),
        "source_plan_sha256": plan["artifact_sha256"],
        "files": files,
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "entrypoints": {
            p: hashlib.sha256((REPO / p).read_bytes()).hexdigest()
            for p in (
                "scripts/audit_basin_repairs.py",
                "scripts/hpc/run_basin_repair_audit_aces.sh",
            )
        },
        "llm_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_promotion": False,
        "automatic_followup": False,
    }
    with public._lock(output):
        sealed_write(output / "freeze.json", frozen)
        rows = []
        for task in plan["tasks"]:
            path = output / "rows" / f"{task['task_id']}.json"
            row = _row(source, plan, baseline, task)
            sealed_write(path, row)
            rows.append(row)
        if _files(source, plan) != files:
            raise ValueError("historical source changed during audit")
        attempts = [a for r in rows for a in r["attempts"]]
        result = {
            **{k: v for k, v in frozen.items() if k not in {"files", "entrypoints"}},
            "identity": content_hash(frozen),
            "rows": rows,
            "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
            "classification_counts": dict(
                sorted(Counter(a["classification"] for a in attempts).items())
            ),
            "newly_mechanically_valid_attempts": sum(
                a["newly_mechanically_valid"] for a in attempts
            ),
            "newly_eligible_saved_drafts": [
                r["task"] for r in rows if r.get("newly_eligible_saved_draft")
            ],
            "unexpected_acceptance_regressions": [
                {"task": r["task"], **a}
                for r in rows
                for a in r["attempts"]
                if a["unexpected_acceptance_regression"]
            ],
            "unavailable_saved_attempts": sum(
                a["classification"] == "delivery_unavailable" for a in attempts
            ),
            "selection": "none; saved replies keep their actual historical contexts",
        }
        sealed_write(output / "summary.json", result)
        lines = [
            "# Saved basin repair handoff audit",
            "",
            "No LLM calls, optimization, rollouts or model promotion. "
            "Historical campaigns are unchanged.",
            "Replies are evaluated against their actual displayed predecessors, "
            "not a hypothetical new chain.",
            "Mechanical acceptance and static fit eligibility are separate; "
            "neither establishes accurate recovery.",
            "",
            f"Classification counts: {result['classification_counts']}",
            "Newly mechanically valid replies: "
            f"{result['newly_mechanically_valid_attempts']}",
            "Newly eligible saved final drafts: "
            f"{len(result['newly_eligible_saved_drafts'])}",
            "Unexpected mechanical regressions: "
            f"{len(result['unexpected_acceptance_regressions'])}",
            "",
            "| Task | Historical | Saved final draft | Newly valid replies |",
            "| --- | --- | --- | ---: |",
        ]
        for r in rows:
            lines.append(
                f"| {r['task']} | {r.get('historical_status', 'unavailable')} | "
                f"{r.get('saved_final_draft', {}).get('status', 'unavailable')} | "
                f"{sum(a['newly_mechanically_valid'] for a in r['attempts'])} |"
            )
        (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return result
