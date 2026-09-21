"""Fresh basin confirmation of the audited local function dependency policy."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import detention_process_pilot as pilot
from autoformalism.rebuttal import function_dependency_audit as offline
from autoformalism.rebuttal import process_handoff_confirmation as previous
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search import function_dependencies as dep
from autoformalism.staged_topology import content_hash

PROTOCOL = "function-dependency-confirmation-1"


def _files(source: Path, plan: dict) -> dict[str, str]:
    """Bind the audited replies plus all historical reporting inputs."""
    paths = set(offline.source_files(source, plan))
    for task in plan["tasks"]:
        for name in ("proposal.json", "result.json"):
            paths.add(source / "results" / task["task_id"] / name)
        paths.add(
            source
            / "construction/results"
            / task["construction_task"]["task_id"]
            / "proposal.json"
        )
    result = {}
    for path in sorted(paths):
        offline._read(path, source)  # Includes containment and existence checks.
        result[str(path.relative_to(source))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


def _repairs(root: Path, task: str) -> dict:
    """Count physical atomic-function requests once, retaining missing usage."""
    root = root.resolve()
    base = root / "construction/results" / task
    routes, calls, missing = [], {}, set()
    for route in ("review", "fallback"):
        path = base / route / "function_stage.json"
        if not path.exists():
            continue
        f = sealed_read(path)["result"]
        events = [e for e in f["events"] if e["step"].startswith("atomic_repair_")]
        for event in events:
            key = event["request_hash"]
            if not re.fullmatch(r"[0-9a-f]{64}", key):
                raise ValueError("invalid repair request hash")
            path = base / "calls" / f"{key}.json"
            if not path.exists():
                missing.add(key)
                continue
            record = offline._read(path, root)
            if record["request_hash"] != key or content_hash(record["request"]) != key:
                raise ValueError("repair request identity differs")
            calls[key] = record
        routes.append(
            {
                "route": route,
                "status": f["status"],
                "complete_model": f.get("complete_model", False),
                "policy": f.get("dependency_policy", "strict"),
                "batch_rejections": sum(
                    not b["batch_accepted"] for b in f["batch_term_audits"]
                ),
                "atomic_attempts": len(events),
                "atomic_rejections": sum(not e["accepted"] for e in events),
                "dependency_revisions": f.get("dependency_revisions", []),
                "connectivity": dep.connectivity(f.get("candidate")),
            }
        )
    usage = [r.get("observed_total_tokens") for r in calls.values()]
    return {
        "routes": routes,
        "physical_calls": len(calls),
        "observed_total_tokens": sum(v for v in usage if v is not None),
        "unmeasured_calls": sum(v is None for v in usage),
        "missing_call_records": len(missing),
        "scope": "atomic function repair only; common construction counted once",
    }


def construction_rows(root: Path, plan: dict) -> list[dict]:
    """Augment existing accounting without counting the two gain arms twice."""
    rows = previous._construction_rows(root, plan)
    for row in rows:
        row["function_repairs"] = _repairs(root, row["task"])
    return rows


def accounting(rows: list[dict]) -> dict:
    """Separate unavailable constructions, fallback use and observed repair cost."""
    result = previous._accounting(rows)
    result.update(
        fallback_constructions=sum(r["fallback_used"] is True for r in rows),
        review_models_retained=sum(
            r["status"] == "constructed" and r["fallback_used"] is False for r in rows
        ),
        atomic_function_repairs={
            key: sum(r["function_repairs"][key] for r in rows)
            for key in (
                "physical_calls",
                "observed_total_tokens",
                "unmeasured_calls",
                "missing_call_records",
            )
        },
    )
    return result


def model_row(root: Path, plan: dict, task: dict) -> dict:
    """Expose equation paths and fitted gains without inferring physical roles."""
    row = previous._row(root, plan, task)
    base = root / "results" / task["task_id"]
    p = sealed_read(base / "proposal.json") if (base / "proposal.json").exists() else {}
    r = sealed_read(base / "result.json") if (base / "result.json").exists() else {}
    bundle = p.get("bundle") or {}
    row["connectivity"] = dep.connectivity(bundle.get("candidate"))
    parameters = (r.get("fit") or {}).get("parameters")
    row["fitted_process_gains"] = (
        {k: v for k, v in parameters.items() if k.startswith("af_process_gain_")}
        if parameters is not None
        else None
    )
    return row


def freeze(source: Path, audit_root: Path, root: Path) -> dict:
    """Freeze matched public inputs in a fresh namespace after the audit passes."""
    source, audit_root, root = (p.resolve() for p in (source, audit_root, root))
    for other in (source, audit_root):
        if root.is_relative_to(other) or other.is_relative_to(root):
            raise ValueError("new campaign must be separate from source and audit")
    offline._read(source / "plan.json", source)
    old = sealed_read(source / "plan.json")
    config = pilot.PilotConfig.model_validate(old["config"])
    if (
        old["protocol"] != pilot.GAIN_PROTOCOL
        or config.protocol != pilot.GAIN_PROTOCOL
        or old["tasks"] != pilot.tasks(config)
        or (old.get("handoff_confirmation") or {}).get("protocol") != previous.PROTOCOL
        or old.get("function_dependency_policy", "strict") != "strict"
        or old.get("dependency_confirmation")
        or old["test_data_opened"]
        or old["private_reference_opened"]
    ):
        raise ValueError("requires the public v4 strict handoff confirmation")
    frozen = sealed_read(audit_root / "freeze.json")
    audit = sealed_read(audit_root / "summary.json")
    identity = {k: v for k, v in frozen.items() if k != "artifact_sha256"}
    names = {t["construction_task"]["task_id"] for t in old["tasks"]}
    if (
        frozen["protocol"] != offline.PROTOCOL
        or frozen["dependency_policy"] != dep.POLICY
        or audit["protocol"] != offline.PROTOCOL
        or audit["identity"] != content_hash(identity)
        or Path(frozen["source"]).resolve() != source
        or frozen["source_plan_sha256"] != old["artifact_sha256"]
        or audit["previously_accepted_now_blocked"]
        or audit["unavailable_saved_attempts"]
        or audit["llm_calls"]
        or audit["optimizer_calls"]
        or audit["test_data_opened"]
        or audit["whole_models_recovered"]
        or frozen["private_reference_opened"]
        or len(audit["constructions"]) != len(names)
        or {c["task"] for c in audit["constructions"]} != names
    ):
        raise ValueError("saved dependency audit gate failed or identity differs")
    audited = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in offline.source_files(source, old)
    }
    if audited != frozen["files"]:
        raise ValueError("historical campaign differs from audited source")
    files = _files(source, old)
    if old["runtime"] != public._runtime():
        raise ValueError("Python/package runtime differs from historical pilot")
    baseline = {
        "protocol": PROTOCOL,
        "plan_sha256": old["artifact_sha256"],
        "rows": [model_row(source, old, t) for t in old["tasks"]],
        "constructions": construction_rows(source, old),
    }
    if any(r["status"] == "missing" for r in baseline["rows"]):
        raise ValueError("historical results are incomplete")
    if any(r["status"] == "missing" for r in baseline["constructions"]):
        raise ValueError("historical constructions are incomplete")
    if _files(source, old) != files:
        raise ValueError("historical source changed during preparation")
    confirmation = {
        "protocol": PROTOCOL,
        "policy": dep.POLICY,
        "historical_plan_sha256": old["artifact_sha256"],
        "audit_freeze_sha256": frozen["artifact_sha256"],
        "audit_summary_sha256": audit["artifact_sha256"],
        "baseline_sha256": content_hash(baseline),
        "matched_inputs_sha256": content_hash(previous._matched(old)),
        "historical_source": str(source),
        "historical_audit": str(audit_root),
        "historical_files": files,
        "fresh_constructions": 8,
        "fit_arms": 16,
        "old_calls_or_models_reused": False,
        "automatic_followup": False,
    }
    plan = {
        **deepcopy(previous._matched(old)),
        "source_sha256": public._source_identity(),
        "launcher_sha256": pilot.launcher_hash(),
        "runtime": public._runtime(),
        "test_data_opened": False,
        "private_reference_opened": False,
        "function_dependency_policy": dep.POLICY,
        "dependency_confirmation": confirmation,
    }
    with public._lock(root):
        if not (root / "plan.json").exists() and any(
            (root / n).exists() for n in ("results", "construction", "scheduler")
        ):
            raise ValueError("new campaign already contains execution artifacts")
        sealed_write(root / "baseline.json", baseline)
        sealed_write(root / "plan.json", plan)
        return pilot.verify(root)


def verify_manifest(root: Path, plan: dict) -> None:
    """Workers verify the frozen new root without requiring historical mounts."""
    c = plan["dependency_confirmation"]
    baseline = sealed_read(root / "baseline.json")
    if (
        c["protocol"] != PROTOCOL
        or c["policy"] != dep.POLICY
        or plan.get("function_dependency_policy") != dep.POLICY
        or "handoff_confirmation" in plan
        or c["baseline_sha256"] != baseline["artifact_sha256"]
        or baseline["protocol"] != PROTOCOL
        or baseline["plan_sha256"] != c["historical_plan_sha256"]
        or [r["task"] for r in baseline["rows"]]
        != [t["task_id"] for t in plan["tasks"]]
        or c["matched_inputs_sha256"] != content_hash(previous._matched(plan))
        or plan["runtime"] != public._runtime()
        or len(plan["tasks"]) != c["fit_arms"]
        or c["fit_arms"] != 16
        or c["fresh_constructions"] != 8
        or len({t["construction_task"]["task_id"] for t in plan["tasks"]}) != 8
        or plan["test_data_opened"]
        or plan["private_reference_opened"]
    ):
        raise ValueError("dependency confirmation inputs, baseline or runtime differ")


def report(root: Path) -> dict:
    """Compare all planned arms; graph facts never certify scientific recovery."""
    plan = pilot.verify(root)
    if "dependency_confirmation" not in plan:
        raise ValueError("requires a frozen dependency confirmation")
    pilot.report(root)
    baseline = sealed_read(root / "baseline.json")
    current = [model_row(root, plan, t) for t in plan["tasks"]]
    constructions = construction_rows(root, plan)
    old = {r["task"]: r for r in baseline["rows"]}
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "historical_status_counts": dict(Counter(r["status"] for r in old.values())),
        "current_status_counts": dict(Counter(r["status"] for r in current)),
        "historical_accounting": accounting(baseline["constructions"]),
        "current_accounting": accounting(constructions),
        "construction_pairs": [
            {"task": a["task"], "historical": a, "current": b}
            for a, b in zip(baseline["constructions"], constructions, strict=True)
        ],
        "rows": [
            {"task": r["task"], "historical": old[r["task"]], "current": r}
            for r in current
        ],
        "scientific_compliance_certified": False,
        "test_data_opened": False,
        "automatic_followup": False,
        "comparison_scope": (
            "Historical strict v4 versus fresh dependency-policy construction. "
            "Matched inventories, data, seeds, model settings and budgets. "
            "Changed prompts can change sampled models. No new-variable stage, "
            "optimizer change, model promotion or automatic follow-up."
        ),
    }
    public._write(root / "comparison.json", summary)
    lines = [
        "# Fresh function dependency confirmation",
        "",
        "Eight fresh constructions, sixteen fits. " + summary["comparison_scope"],
        "",
        f"Historical status: {summary['historical_status_counts']}",
        f"Current status: {summary['current_status_counts']}",
        "",
        "| Construction | Before | After | Fallback before/after | "
        "Total tokens before/after | Atomic repair tokens before/after |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pair in summary["construction_pairs"]:
        a, b = pair["historical"], pair["current"]
        total = [(r["usage"] or {}).get("observed_total_tokens") for r in (a, b)]
        repair = [r["function_repairs"]["observed_total_tokens"] for r in (a, b)]
        lines.append(
            f"| {pair['task']} | {a['status']} | {b['status']} | "
            f"{a['fallback_used']} / {b['fallback_used']} | "
            f"{total[0]} / {total[1]} | {repair[0]} / {repair[1]} |"
        )
    lines += [
        "",
        "| Arm | Before | After | Train before/after | "
        "Validation before/after | Replay after |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pair in summary["rows"]:
        a, b = pair["historical"], pair["current"]
        scores = [
            " / ".join(str((r.get(s) or {}).get("normalized_mse")) for r in (a, b))
            for s in ("training", "validation")
        ]
        lines.append(
            f"| {pair['task']} | {a['status']} | {b['status']} | "
            f"{scores[0]} | {scores[1]} | "
            f"{(b['replay'] or {}).get('replay_agreement')} |"
        )
    lines += [
        "",
        "Repair accounting and dependency ledgers are in comparison.json. "
        "Token sums cover observed usage; missing/unmeasured calls are explicit. "
        "Connectivity is syntactic, not fitted activity or physical conservation. "
        "Replay is numerical consistency, not accurate prediction. "
        "Failures and unavailable metrics remain in the denominators.",
    ]
    (root / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    return summary
