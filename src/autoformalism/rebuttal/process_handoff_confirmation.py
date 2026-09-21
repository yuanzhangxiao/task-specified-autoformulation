"""Fresh matched signed-process pilot after a successful saved-response audit."""

from __future__ import annotations

import hashlib
from collections import Counter
from copy import deepcopy
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import detention_process_pilot as pilot
from autoformalism.rebuttal import process_handoff_audit as offline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.process_equation_diagnostics import diagnose
from autoformalism.staged_topology import content_hash

PROTOCOL = "process-handoff-confirmation-1"
MATCHED_FIELDS = (
    "protocol",
    "config",
    "source_plan_sha256",
    "source_public_files",
    "cells",
    "tasks",
    "paired_inputs",
)


def _matched(plan: dict) -> dict:
    """The complete scientific inputs, matrix and configured budgets."""
    return {key: plan[key] for key in MATCHED_FIELDS}


def _source_unchanged(source: Path, plan: dict, files: dict) -> None:
    paths = offline.source_files(source, plan)
    current = {}
    for path in paths:
        offline._read(path, source)
        current[str(path.relative_to(source))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    if current != files:
        raise ValueError("historical campaign differs from the audited source")


def _row(root: Path, plan: dict, task: dict) -> dict:
    """Read one published arm without evaluating its model or fitting parameters."""
    directory = root / "results" / task["task_id"]
    proposal = (
        sealed_read(directory / "proposal.json")
        if (directory / "proposal.json").exists()
        else {}
    )
    result = (
        sealed_read(directory / "result.json")
        if (directory / "result.json").exists()
        else {}
    )
    if proposal and proposal["task"] != task:
        raise ValueError("proposal task differs from the matched matrix")
    if result and result["proposal_sha256"] != proposal.get("artifact_sha256"):
        raise ValueError("result and proposal differ")
    fit = result.get("fit") or {}
    bundle = proposal.get("bundle")
    return {
        "task": task["task_id"],
        "status": result.get("status", "missing"),
        "proposal_status": proposal.get("status", "missing"),
        "fallback_used": proposal.get("fallback_used"),
        "common_proposal_sha256": proposal.get("common_proposal_sha256"),
        "training": fit.get("training"),
        "validation": fit.get("validation"),
        "replay": result.get("replay"),
        "complexity": (proposal.get("equation_inventory") or {}).get("counts"),
        "structure": proposal.get("structural"),
        "fit": {
            k: fit.get(k)
            for k in (
                "message",
                "actual_residual_calls",
                "budget_exhausted",
                "native_optimizer_converged",
            )
        },
        "equation_diagnostics": diagnose(
            bundle, fit.get("parameters"), plan["cells"][task["case"]]
        )
        if bundle
        else None,
    }


def _construction_rows(root: Path, plan: dict) -> list[dict]:
    names = sorted({t["construction_task"]["task_id"] for t in plan["tasks"]})
    rows = []
    for name in names:
        path = root / "construction/results" / name / "proposal.json"
        p = sealed_read(path) if path.exists() else {}
        rows.append(
            {
                "task": name,
                "status": p.get("status", "missing"),
                "fallback_used": p.get("fallback_used"),
                "usage": p.get("usage"),
                "attempts": [
                    {
                        "route": a["route"],
                        "error": a.get("error"),
                        "process_status": (a.get("process_review") or {}).get("status"),
                    }
                    for a in p.get("attempts", [])
                ],
            }
        )
    return rows


def _accounting(rows: list[dict]) -> dict:
    """Count common constructions once; missing usage is explicitly unavailable."""
    measured = [r["usage"] for r in rows if r["usage"] is not None]
    return {
        "planned_constructions": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "constructions_with_usage": len(measured),
        "constructions_without_usage": len(rows) - len(measured),
        "physical_calls": sum(u["physical_calls"] for u in measured),
        "observed_total_tokens": sum(u["observed_total_tokens"] for u in measured),
        "unmeasured_calls": sum(u["unmeasured_calls"] for u in measured),
        "historical_variable_selection_cost_included": False,
    }


def freeze(source: Path, audit_root: Path, root: Path) -> dict:
    """Import frozen public inputs only; retain old outcomes as reporting metadata."""
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
        or old.get("handoff_confirmation") is not None
        or old["test_data_opened"]
        or old["private_reference_opened"]
    ):
        raise ValueError("requires the original public gain-policy pilot")
    frozen = sealed_read(audit_root / "freeze.json")
    audit = sealed_read(audit_root / "summary.json")
    identity = {k: v for k, v in frozen.items() if k != "artifact_sha256"}
    if (
        frozen["protocol"] != offline.PROTOCOL
        or audit["protocol"] != offline.PROTOCOL
        or audit["identity"] != content_hash(identity)
        or Path(frozen["source"]).resolve() != source
        or frozen["source_plan_sha256"] != old["artifact_sha256"]
        or audit["previously_accepted_now_blocked"]
        or audit["unavailable_saved_attempts"]
        or audit["llm_calls"] != 0
        or audit["optimizer_calls"] != 0
        or audit["test_data_opened"]
    ):
        raise ValueError("saved-response audit gate failed or identity differs")
    _source_unchanged(source, old, frozen["files"])
    if old["runtime"] != public._runtime():
        raise ValueError("Python/package runtime differs from the historical pilot")
    if {r["task"] for r in audit["models"]} != {t["task_id"] for t in old["tasks"]}:
        raise ValueError("audit matrix differs")
    baseline = {
        "protocol": PROTOCOL,
        "plan_sha256": old["artifact_sha256"],
        "rows": [_row(source, old, task) for task in old["tasks"]],
        "constructions": _construction_rows(source, old),
    }
    counts = dict(Counter(r["status"] for r in baseline["rows"]))
    if counts != audit["historical_status_counts"] or counts.get("missing"):
        raise ValueError("historical results are incomplete or differ from audit")
    _source_unchanged(source, old, frozen["files"])
    confirmation = {
        "protocol": PROTOCOL,
        "handoff": "signed-process-handoff-2",
        "historical_plan_sha256": old["artifact_sha256"],
        "audit_freeze_sha256": frozen["artifact_sha256"],
        "audit_summary_sha256": audit["artifact_sha256"],
        "baseline_sha256": content_hash(baseline),
        "matched_inputs_sha256": content_hash(_matched(old)),
        "historical_source": str(source),
        "historical_audit": str(audit_root),
        "fresh_constructions": 8,
        "fit_arms": 16,
        "old_calls_or_models_reused": False,
        "automatic_followup": False,
    }
    plan = {
        **deepcopy(_matched(old)),
        "source_sha256": public._source_identity(),
        "launcher_sha256": pilot.launcher_hash(),
        "runtime": public._runtime(),
        "test_data_opened": False,
        "private_reference_opened": False,
        "handoff_confirmation": confirmation,
    }
    with public._lock(root):
        if not (root / "plan.json").exists() and any(
            (root / name).exists() for name in ("results", "construction", "scheduler")
        ):
            raise ValueError("new campaign already contains execution artifacts")
        # Identical partial preparation is resumable; old outcomes never enter results/.
        sealed_write(root / "baseline.json", baseline)
        sealed_write(root / "plan.json", plan)
        return pilot.verify(root)


def verify_manifest(root: Path, plan: dict) -> None:
    """Workers need only the new frozen root, not mutable historical directories."""
    c = plan["handoff_confirmation"]
    baseline = sealed_read(root / "baseline.json")
    if (
        c["protocol"] != PROTOCOL
        or c["handoff"] != "signed-process-handoff-2"
        or c["baseline_sha256"] != baseline["artifact_sha256"]
        or baseline["protocol"] != PROTOCOL
        or baseline["plan_sha256"] != c["historical_plan_sha256"]
        or [r["task"] for r in baseline["rows"]]
        != [t["task_id"] for t in plan["tasks"]]
        or c["matched_inputs_sha256"] != content_hash(_matched(plan))
        or plan["runtime"] != public._runtime()
        or len(plan["tasks"]) != c["fit_arms"]
        or len({t["construction_task"]["task_id"] for t in plan["tasks"]})
        != c["fresh_constructions"]
    ):
        raise ValueError("confirmation inputs, baseline or runtime differ")


def report(root: Path) -> dict:
    """Keep all arms and scientific diagnostics separate from predictive accuracy."""
    plan = pilot.verify(root)
    if "handoff_confirmation" not in plan:
        raise ValueError("requires a frozen handoff confirmation")
    pilot.report(root)
    baseline = sealed_read(root / "baseline.json")
    old = {r["task"]: r for r in baseline["rows"]}
    current = [_row(root, plan, task) for task in plan["tasks"]]
    constructions = _construction_rows(root, plan)
    comparison = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "historical_status_counts": dict(Counter(r["status"] for r in old.values())),
        "current_status_counts": dict(Counter(r["status"] for r in current)),
        "historical_accounting": _accounting(baseline["constructions"]),
        "current_accounting": _accounting(constructions),
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
            "Historical vs corrected construction package, not an isolated effect "
            "of empty-term validation. New gain arms share each fresh construction. "
            "Old and new sampled models can differ. No model selection or promotion."
        ),
    }
    public._write(root / "comparison.json", comparison)
    lines = [
        "# Fresh process-handoff confirmation",
        "",
        "Eight fresh constructions, sixteen fits; same saved inventories, data, "
        "seeds, model settings and budgets as v3. Fitting is unchanged.",
        "",
        comparison["comparison_scope"],
        "",
        f"Historical arms: {comparison['historical_status_counts']}",
        f"Current arms: {comparison['current_status_counts']}",
        "",
        "| Construction | Historical | Corrected | Historical tokens | New tokens |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in comparison["construction_pairs"]:
        a, b = row["historical"], row["current"]
        values = [
            row["task"],
            a["status"],
            b["status"],
            (a["usage"] or {}).get("observed_total_tokens"),
            (b["usage"] or {}).get("observed_total_tokens"),
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += [
        "",
        "| Arm | Before | After | Train before | Train after | "
        "Validation before | Validation after |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["rows"]:
        a, b = row["historical"], row["current"]
        values = [row["task"], a["status"], b["status"]] + [
            (r.get(split) or {}).get("normalized_mse")
            for split in ("training", "validation")
            for r in (a, b)
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += [
        "",
        "Numerical replay is solver consistency, not prediction accuracy. "
        "Equation facts and declared-transfer cancellation in comparison.json "
        "are advisory, not scientific certification. Missing results are not "
        "zeros. No automatic follow-up or held-out test access.",
    ]
    (root / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    return comparison
