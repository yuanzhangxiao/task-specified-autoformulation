"""Fresh matched basin constructions after the saved assembly audit passes."""

from __future__ import annotations

import hashlib
from collections import Counter
from copy import deepcopy
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import detention_process_pilot as pilot
from autoformalism.rebuttal import function_dependency_confirmation as previous
from autoformalism.rebuttal import process_assembly_confirmation as parent
from autoformalism.rebuttal import process_revision_audit as offline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.process_handoff_confirmation import _matched
from autoformalism.search import function_delivery as delivery
from autoformalism.search import function_dependencies as dep
from autoformalism.search import process_assembly_contract as assembly
from autoformalism.search import process_assembly_revision as revision
from autoformalism.staged_topology import content_hash

PROTOCOL = "process-revision-confirmation-1"


def construction_rows(root: Path, plan: dict) -> list[dict]:
    """Report legacy decisions separately from identified delivery transactions."""
    rows = previous.construction_rows(root, plan)
    for row in rows:
        stages = []
        for route in ("review", "fallback"):
            path = (
                root
                / "construction/results"
                / row["task"]
                / route
                / "function_stage.json"
            )
            if not path.exists():
                continue
            f = sealed_read(path)["result"]
            stage = f.get("function_delivery")
            if stage is None:
                stages.append({"route": route, "policy": "legacy", "counters": None})
                continue
            batches = [e["result"] for e in stage["ledger"] if e["kind"] == "batch"]
            repairs = [e["result"] for e in stage["ledger"] if e["kind"] == "repair"]
            outcomes = [o for b in batches for o in b["outcomes"]]
            txs = [o["transaction"] for o in outcomes if o["transaction"]]
            txs += [r["transaction"] for r in repairs if r["transaction"]]
            stages.append(
                {
                    "route": route,
                    "policy": delivery.POLICY,
                    "counters": {
                        "batch_calls": len(batches),
                        "delivery_diagnostics": sum(
                            len(b["delivery_errors"]) for b in batches
                        ),
                        "batch_slots_accepted": sum(
                            o["transaction"] is not None for o in outcomes
                        ),
                        "batch_slots_rejected": sum(
                            o["transaction"] is None for o in outcomes
                        ),
                        "repair_calls": sum(len(r["events"]) for r in repairs),
                        "topology_scope_calls": sum(
                            e["scope"] == "topology"
                            for r in repairs
                            for e in r["events"]
                        ),
                        "assignment_labels_removed": sum(
                            len(o["delivery_normalizations"]) for o in outcomes
                        )
                        + sum(len(r["delivery_normalizations"]) for r in repairs),
                        "conversion_edits": sum(
                            len(t["conversion_changes"]) for t in txs
                        ),
                        "retained_overlap_transactions": sum(
                            bool(t["remaining_conversion_overlap"]) for t in txs
                        ),
                        "companion_law_revisions": sum(
                            len(t["reply"]["companion_laws"]) for t in txs
                        ),
                    },
                }
            )
        row["identified_delivery"] = stages
    return rows


def accounting(rows: list[dict]) -> dict:
    """Retain complete physical-call accounting, including fallback and failures."""
    result = previous.accounting(rows)
    counters = Counter()
    stages = [s for r in rows for s in r["identified_delivery"]]
    for stage in stages:
        if stage["counters"] is not None:
            counters.update(stage["counters"])
    result["identified_delivery"] = {
        "policies": dict(Counter(s["policy"] for s in stages)),
        "counters": dict(counters)
        if any(s["counters"] is not None for s in stages)
        else None,
        "scope": "Slots and attempts, including fallback; not recovered models.",
    }
    return result


def _audited_files(source: Path, plan: dict) -> dict[str, str]:
    """Verify containment before hashing the exact metadata read by the audit."""
    hashes = {}
    for path in offline.source_files(source, plan):
        offline._read(path, source)
        hashes[str(path.relative_to(source))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return hashes


def _files(source: Path, plan: dict) -> dict[str, str]:
    """Include the v6 baseline used by its manifest verifier in the snapshot."""
    files = previous._files(source, plan)
    path = source / "baseline.json"
    offline._read(path, source)
    files["baseline.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def freeze(source: Path, audit_root: Path, root: Path) -> dict:
    """Import identical v6 public inputs; historical outcomes are report-only."""
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
        or (old.get("assembly_confirmation") or {}).get("protocol") != parent.PROTOCOL
        or old.get("function_dependency_policy") != dep.POLICY
        or old.get("process_assembly_policy") != assembly.POLICY
        or old.get("revision_confirmation")
        or old["test_data_opened"]
        or old["private_reference_opened"]
    ):
        raise ValueError("requires the completed public v6 assembly confirmation")
    # Validate the old manifest without imposing today's source/launcher hashes.
    parent.verify_manifest(source, old)
    frozen = sealed_read(audit_root / "freeze.json")
    audit = sealed_read(audit_root / "summary.json")
    identity = {k: v for k, v in frozen.items() if k != "artifact_sha256"}
    names = {t["construction_task"]["task_id"] for t in old["tasks"]}
    if (
        frozen["protocol"] != offline.PROTOCOL
        or frozen["policy"] != revision.POLICY
        or audit["protocol"] != offline.PROTOCOL
        or audit["identity"] != content_hash(identity)
        or Path(frozen["source"]).resolve() != source
        or frozen["source_plan_sha256"] != old["artifact_sha256"]
        or audit["unexpected_acceptance_regressions"]
        or audit["unavailable_saved_attempts"]
        or audit["missing_constructions"]
        or any(
            audit[k]
            for k in (
                "llm_calls",
                "optimizer_calls",
                "test_data_opened",
                "whole_models_recovered",
                "automatic_followup",
            )
        )
        or any(
            frozen[k]
            for k in (
                "llm_calls",
                "optimizer_calls",
                "trajectory_values_used",
                "test_data_opened",
                "private_reference_opened",
            )
        )
        or len(audit["constructions"]) != len(names)
        or {c["task"] for c in audit["constructions"]} != names
        or any(not c["routes"] for c in audit["constructions"])
        or any(
            not r["dependency_replay"]["exact_ledger_and_final_state_verified"]
            for c in audit["constructions"]
            for r in c["routes"]
        )
    ):
        raise ValueError("saved assembly audit gate failed or identity differs")
    if _audited_files(source, old) != frozen["files"]:
        raise ValueError("historical campaign differs from audited source")
    files = _files(source, old)
    baseline = {
        "protocol": PROTOCOL,
        "plan_sha256": old["artifact_sha256"],
        "rows": [previous.model_row(source, old, t) for t in old["tasks"]],
        "constructions": construction_rows(source, old),
    }
    if any(
        r["status"] == "missing" for r in baseline["rows"] + baseline["constructions"]
    ):
        raise ValueError("historical results or constructions are incomplete")
    if _files(source, old) != files:
        raise ValueError("historical source changed during preparation")
    matched = _matched(old)
    confirmation = {
        "protocol": PROTOCOL,
        "policy": delivery.POLICY,
        "historical_plan_sha256": old["artifact_sha256"],
        "audit_freeze_sha256": frozen["artifact_sha256"],
        "audit_summary_sha256": audit["artifact_sha256"],
        "audit_classification_counts": audit["classification_counts"],
        "baseline_sha256": content_hash(baseline),
        "matched_inputs_sha256": content_hash(matched),
        "historical_source": str(source),
        "historical_audit": str(audit_root),
        "historical_files": files,
        "fresh_constructions": 8,
        "fit_arms": 16,
        "old_calls_or_models_reused": False,
        "automatic_followup": False,
    }
    plan = {
        **deepcopy(matched),
        "source_sha256": public._source_identity(),
        "launcher_sha256": pilot.launcher_hash(),
        "runtime": public._runtime(),
        "test_data_opened": False,
        "private_reference_opened": False,
        "function_dependency_policy": dep.POLICY,
        "process_assembly_policy": assembly.POLICY,
        "revision_confirmation": confirmation,
        "function_delivery_policy": delivery.POLICY,
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
    """Workers verify frozen inputs without rereading historical directories."""
    c = plan["revision_confirmation"]
    baseline = sealed_read(root / "baseline.json")
    if (
        c["protocol"] != PROTOCOL
        or c["policy"] != delivery.POLICY
        or plan.get("function_delivery_policy") != delivery.POLICY
        or plan.get("process_assembly_policy") != assembly.POLICY
        or plan.get("function_dependency_policy") != dep.POLICY
        or "handoff_confirmation" in plan
        or "dependency_confirmation" in plan
        or "assembly_confirmation" in plan
        or c["baseline_sha256"] != baseline["artifact_sha256"]
        or baseline["protocol"] != PROTOCOL
        or baseline["plan_sha256"] != c["historical_plan_sha256"]
        or [r["task"] for r in baseline["rows"]]
        != [t["task_id"] for t in plan["tasks"]]
        or c["matched_inputs_sha256"] != content_hash(_matched(plan))
        or plan["runtime"] != public._runtime()
        or len(plan["tasks"]) != c["fit_arms"]
        or c["fit_arms"] != 16
        or c["fresh_constructions"] != 8
        or len({t["construction_task"]["task_id"] for t in plan["tasks"]}) != 8
        or c["old_calls_or_models_reused"]
        or c["automatic_followup"]
        or plan["test_data_opened"]
        or plan["private_reference_opened"]
    ):
        raise ValueError("assembly confirmation inputs, baseline or runtime differ")


def report(root: Path) -> dict:
    """Compare construction and fitting separately; retain every planned arm."""
    plan = pilot.verify(root)
    if "revision_confirmation" not in plan:
        raise ValueError("requires a frozen assembly confirmation")
    summary = previous.comparison_report(
        root,
        plan,
        protocol=PROTOCOL,
        construction_reader=construction_rows,
        accounting_reader=accounting,
        title="Fresh identified-function and process-repair confirmation",
        scope="Historical v6 versus fresh identified-function delivery and repairs. "
        "Same saved "
        "inventories, data, seeds, model settings, gain arms and budgets. "
        "Changed prompts can change sampled models; this is a combined interface "
        "comparison. No variable-selection rerun, fitter change, model promotion "
        "or automatic follow-up.",
    )
    with (root / "COMPARISON.md").open("a") as stream:
        stream.write(
            "\nAssembly diagnostics count saved slots/attempts once per construction, "
            "including fallback. Legacy decisions were not recorded.\n\n"
        )
        for label in ("historical", "current"):
            stream.write(
                f"{label}: {summary[label + '_accounting']['identified_delivery']}\n\n"
            )
    return summary
