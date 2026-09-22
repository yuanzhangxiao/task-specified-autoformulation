"""Fresh matched basin constructions after the saved assembly audit passes."""

from __future__ import annotations

import hashlib
from collections import Counter
from copy import deepcopy
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import detention_process_pilot as pilot
from autoformalism.rebuttal import function_dependency_confirmation as previous
from autoformalism.rebuttal import process_assembly_audit as offline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.process_handoff_confirmation import _matched
from autoformalism.search import function_dependencies as dep
from autoformalism.search import process_assembly_contract as assembly
from autoformalism.staged_topology import content_hash

PROTOCOL = "process-assembly-confirmation-1"


def construction_rows(root: Path, plan: dict) -> list[dict]:
    """Count common construction decisions once, including rejected attempts."""
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
            failures = [
                {
                    "stage": "batch",
                    "interaction_id": b["interaction_id"],
                    "error": b["batch_error"],
                }
                for b in f["batch_term_audits"]
                if b.get("batch_error")
            ] + [
                {"stage": "atomic", **e}
                for e in f["events"]
                if e["step"].startswith("atomic_repair_") and not e["accepted"]
            ]
            stages.append(
                {
                    "route": route,
                    "status": f["status"],
                    "policy": f.get("assembly_policy", "legacy"),
                    "committed_slot_decisions": f.get("assembly_decisions", []),
                    "rejected_attempts": failures,
                }
            )
        row["assembly"] = stages
    return rows


def accounting(rows: list[dict]) -> dict:
    """Report observed records separately from missing constructions and costs."""
    result = previous.accounting(rows)
    stages = [s for r in rows for s in r["assembly"]]
    decisions = [d for s in stages for d in s["committed_slot_decisions"]]
    errors = [str(e.get("error", "")) for s in stages for e in s["rejected_attempts"]]
    result["assembly"] = {
        "function_stages_recorded": len(stages),
        "policies": dict(Counter(s["policy"] for s in stages)),
        "committed_slot_decisions": len(decisions),
        "outer_sign_normalized_slots": sum(
            bool(d["sign_normalizations"]) for d in decisions
        ),
        "intrinsic_factor_confirmations": sum(
            d["intrinsic_factor_confirmed"] for d in decisions
        ),
        "conversion_rejected_attempts": sum(
            e.startswith("CONSUMER_CONVERSION_OVERLAP") for e in errors
        ),
        "target_path_rejected_attempts": sum(
            e.startswith("PROCESS_TARGET_PATH_LOST") for e in errors
        ),
        "scope": "Observed records include fallback. Slots/attempts are not models. "
        "Legacy decisions were not recorded and are not retrospective zeros.",
    }
    if not any(s["policy"] == assembly.POLICY for s in stages):
        for key in (
            "committed_slot_decisions",
            "outer_sign_normalized_slots",
            "intrinsic_factor_confirmations",
            "conversion_rejected_attempts",
            "target_path_rejected_attempts",
        ):
            result["assembly"][key] = None
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
    """Include the v5 baseline used by its manifest verifier in the snapshot."""
    files = previous._files(source, plan)
    path = source / "baseline.json"
    offline._read(path, source)
    files["baseline.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def freeze(source: Path, audit_root: Path, root: Path) -> dict:
    """Import identical v5 public inputs; historical outcomes are report-only."""
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
        or (old.get("dependency_confirmation") or {}).get("protocol")
        != previous.PROTOCOL
        or old.get("function_dependency_policy") != dep.POLICY
        or old.get("process_assembly_policy", "legacy") != "legacy"
        or old.get("assembly_confirmation")
        or old["test_data_opened"]
        or old["private_reference_opened"]
    ):
        raise ValueError("requires the completed public v5 dependency confirmation")
    # Validate the old manifest without imposing today's source/launcher hashes.
    previous.verify_manifest(source, old)
    frozen = sealed_read(audit_root / "freeze.json")
    audit = sealed_read(audit_root / "summary.json")
    identity = {k: v for k, v in frozen.items() if k != "artifact_sha256"}
    names = {t["construction_task"]["task_id"] for t in old["tasks"]}
    if (
        frozen["protocol"] != offline.PROTOCOL
        or frozen["assembly_policy"] != assembly.POLICY
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
        "policy": assembly.POLICY,
        "historical_plan_sha256": old["artifact_sha256"],
        "audit_freeze_sha256": frozen["artifact_sha256"],
        "audit_summary_sha256": audit["artifact_sha256"],
        "audit_classification_counts": audit["classification_counts"],
        "historical_alias_replays": audit["historical_alias_replays"],
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
        "assembly_confirmation": confirmation,
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
    c = plan["assembly_confirmation"]
    baseline = sealed_read(root / "baseline.json")
    if (
        c["protocol"] != PROTOCOL
        or c["policy"] != assembly.POLICY
        or plan.get("process_assembly_policy") != assembly.POLICY
        or plan.get("function_dependency_policy") != dep.POLICY
        or "handoff_confirmation" in plan
        or "dependency_confirmation" in plan
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
    if "assembly_confirmation" not in plan:
        raise ValueError("requires a frozen assembly confirmation")
    summary = previous.comparison_report(
        root,
        plan,
        protocol=PROTOCOL,
        construction_reader=construction_rows,
        accounting_reader=accounting,
        title="Fresh process assembly confirmation",
        scope="Historical v5 versus fresh assembly-policy construction. Same saved "
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
            stream.write(f"{label}: {summary[label + '_accounting']['assembly']}\n\n")
    return summary
