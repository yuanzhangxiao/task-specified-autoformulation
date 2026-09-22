"""One frozen-profile fit per eligible saved repair, without new proposer calls."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.rebuttal import basin_repair_audit as audit
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.review_deadline_pipeline import request_for
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.search import basin_model_repair_v3 as repair
from autoformalism.staged_topology import content_hash

PROTOCOL = "basin-saved-repair-fit-1"
REPO = Path(__file__).resolve().parents[3]
SELECTION = (
    "latest eligible saved reply in its actual context; otherwise eligible actual "
    "final draft; historical confirmed attempts retained"
)


def launcher_hash() -> str:
    """Pin worker, submission and CLI code in addition to package source."""
    return content_hash(
        {
            p: hashlib.sha256((REPO / p).read_bytes()).hexdigest()
            for p in (
                "scripts/basin_saved_fit.py",
                "scripts/submit_basin_saved_fit.py",
                "scripts/hpc/run_basin_saved_fit_aces.sh",
                "scripts/submit_review_continuation.py",
                "scripts/submit_basin_repair_pilot.py",
            )
        }
    )


def choose(row: dict) -> dict:
    """Choose by recorded chronology, before fitting and without numerical scores."""
    if row["status"] != "audited":
        return {"status": "unavailable", "origin": None, "bundle": None}
    if row["historical_status"] == "confirmed":
        return {
            "status": "historical_fit_retained",
            "origin": "historical_confirmed",
            "bundle": row["saved_final_draft"]["bundle"],
        }
    eligible = [a for a in row["attempts"] if a["classification"] == "eligible_for_fit"]
    if eligible:
        a = max(eligible, key=lambda x: x["attempt"])
        return {
            "status": "ready",
            "origin": "saved_reply",
            "attempt": a["attempt"],
            "request_hash": a["request_hash"],
            "context_sha256": a["context_sha256"],
            "bundle": a["bundle"],
        }
    final = row["saved_final_draft"]
    if final["status"] == "eligible_draft":
        return {
            "status": "ready",
            "origin": "saved_final_draft",
            "bundle": final["bundle"],
        }
    return {"status": "no_eligible_child", "origin": final["status"], "bundle": None}


def _snapshot(source: Path, plan: dict) -> dict:
    files = audit._files(source, plan)
    for task in plan["tasks"]:
        name = f"results/{task['task_id']}/result.json"
        path = source / name
        if not path.resolve().is_relative_to(source):
            raise ValueError("result symlink escapes source")
        files[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        )
    return files


def freeze(source: Path, gate: Path, root: Path) -> dict:
    """Verify the prior audit, rebuild raw replies, and seal choices before fitting."""
    source, gate, root = (p.resolve() for p in (source, gate, root))
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in (source, gate)):
        raise ValueError("output must be separate from historical source and audit")
    old = sealed_read(source / "plan.json")
    baseline = sealed_read(source / "baseline.json")
    frozen, summary = (
        sealed_read(gate / "freeze.json"),
        sealed_read(gate / "summary.json"),
    )
    files = _snapshot(source, old)
    if (
        old["protocol"] != audit.historical.PROTOCOL
        or old["policy"] != repair.old.POLICY
        or old["baseline_sha256"] != baseline["artifact_sha256"]
        or old["config"]["fit_profile"] != "collocation-single-target-v2"
        or old["test_data_opened"]
        or old["private_reference_opened"]
        or frozen["protocol"] != audit.PROTOCOL
        or frozen["policy"] != repair.previous.POLICY
        or Path(frozen["source"]).resolve() != source
        or frozen["source_plan_sha256"] != old["artifact_sha256"]
        or frozen["files"] != audit._files(source, old)
        or summary["identity"] != frozen["artifact_sha256"]
        or summary["unexpected_acceptance_regressions"]
        or any(
            summary[k]
            for k in (
                "llm_calls",
                "optimizer_calls",
                "solver_rollouts",
                "test_data_opened",
                "private_reference_opened",
            )
        )
    ):
        raise ValueError("historical audit gate differs")
    rows, historical = [], {}
    # The old policy remains executable: its audit must reproduce exactly.
    old_rows = [audit._row(source, old, baseline, t) for t in old["tasks"]]
    if old_rows != summary["rows"]:
        raise ValueError("historical audit rows differ on independent replay")
    for task in old["tasks"]:
        row = audit._row(
            source, old, baseline, task, policy=repair, include_context=True
        )
        if any(a["unexpected_acceptance_regression"] for a in row["attempts"]):
            raise ValueError("new binding policy regressed a historical acceptance")
        rows.append(row)
        path = source / "results" / task["task_id"] / "result.json"
        result = sealed_read(path) if path.exists() else None
        if result:
            proposal = sealed_read(path.with_name("proposal.json"))
            if (
                result["task"] != task
                or result["proposal_sha256"] != proposal["artifact_sha256"]
            ):
                raise ValueError("historical result belongs to another proposal")
        historical[task["task_id"]] = result
    if _snapshot(source, old) != files:
        raise ValueError("historical artifacts changed during freeze")
    inputs = {
        "parents": baseline["parents"],
        "rows": rows,
        "historical_results": historical,
    }
    selections = {r["task"]: choose(r) for r in rows}
    plan = {
        "protocol": PROTOCOL,
        "policy": repair.POLICY,
        "selection_rule": SELECTION,
        "config": old["config"],
        "tasks": old["tasks"],
        "cells": old["cells"],
        "inputs_sha256": content_hash(inputs),
        "selections": selections,
        "source": str(source),
        "source_files": files,
        "source_plan_sha256": old["artifact_sha256"],
        "gate_sha256": summary["artifact_sha256"],
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "launcher_sha256": launcher_hash(),
        "llm_calls": 0,
        "automatic_followup": False,
        "automatic_promotion": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "maximum_new_fits": sum(s["status"] == "ready" for s in selections.values()),
    }
    with public._lock(root):
        sealed_write(root / "inputs.json", inputs)
        sealed_write(root / "plan.json", plan)
    return verify(root)


def verify(root: Path) -> dict:
    """Verify copied inputs, deterministic choice and executable policy on resume."""
    plan, inputs = sealed_read(root / "plan.json"), sealed_read(root / "inputs.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["policy"] != repair.POLICY
        or plan["selection_rule"] != SELECTION
        or plan["inputs_sha256"] != inputs["artifact_sha256"]
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or plan["launcher_sha256"] != launcher_hash()
        or plan["config"]["fit_profile"] != "collocation-single-target-v2"
        or any(
            plan[k]
            for k in (
                "llm_calls",
                "automatic_followup",
                "automatic_promotion",
                "test_data_opened",
                "private_reference_opened",
            )
        )
        or plan["selections"] != {r["task"]: choose(r) for r in inputs["rows"]}
        or set(plan["selections"]) != {t["task_id"] for t in plan["tasks"]}
    ):
        raise ValueError("saved fitting plan, code, inputs or selection differ")
    return plan


def fit_task(root: Path, index: int) -> dict:
    """Use the established warm-start fitter and its consumed-attempt semantics."""
    plan = verify(root)
    if index not in range(len(plan["tasks"])):
        raise ValueError("task index outside plan")
    task = plan["tasks"][index]
    name = task["task_id"]
    selected = plan["selections"][name]
    directory = root / "results" / name
    with public._lock(directory):
        if (directory / "result.json").exists():
            return checked_result(root, plan, task)
        result = {
            "plan_sha256": plan["artifact_sha256"],
            "task": task,
            "selection_sha256": content_hash(selected),
            "status": selected["status"],
            "fit": None,
            "assessment": None,
            "replay": None,
            "new_fit": False,
        }
        if selected["status"] == "ready":
            inputs = sealed_read(root / "inputs.json")
            parent = inputs["parents"][name]
            cell = plan["cells"][task["case"]]
            bundle = selected["bundle"]
            static = repair.assessment(
                bundle, task["case"], audit.equations._surveys(cell)
            )
            if any(c["status"] == "fail" for c in static["checks"]):
                raise ValueError("selected child has a static failure")
            request = request_for(bundle, plan, task, 1)
            sibling_fit.prepare_child_fit(
                request_for(parent["bundle"], plan, task, 0),
                request,
                parent["result"]["fit"]["parameters"],
                PublicSplit.model_validate(cell["training"]),
                PublicSplit.model_validate(cell["validation"]),
                directory / "fit",
                lineage={
                    "plan": plan["artifact_sha256"],
                    "selection": content_hash(selected),
                },
            )
            fit = sibling_fit.execute_child_fit(directory / "fit")
            result.update(
                status=fit.status, fit=fit.model_dump(mode="json"), new_fit=True
            )
            if fit.status == "complete":
                result["assessment"] = repair.assessment(
                    bundle,
                    task["case"],
                    audit.equations._surveys(cell),
                    dict(fit.parameters),
                )
                marker = directory / "replay_started.json"
                if marker.exists():
                    result["replay"] = {
                        "status": "interrupted",
                        "replay_agreement": False,
                    }
                else:
                    sealed_write(marker, {"fit_sha256": content_hash(result["fit"])})
                    try:
                        result["replay"] = audit.historical.prior.replay(
                            request, dict(fit.parameters), cell
                        )
                    except (ValueError, RuntimeError, TimeoutError) as exc:
                        result["replay"] = {
                            "status": "failed",
                            "error": str(exc),
                            "replay_agreement": False,
                        }
        sealed_write(directory / "result.json", result)
        return checked_result(root, plan, task)


def checked_result(root: Path, plan: dict, task: dict) -> dict:
    """Reports verify numerical artifacts read-only, never restarting fitting."""
    directory = root / "results" / task["task_id"]
    result = sealed_read(directory / "result.json")
    if (
        result["plan_sha256"] != plan["artifact_sha256"]
        or result["task"] != task
        or result["selection_sha256"]
        != content_hash(plan["selections"][task["task_id"]])
    ):
        raise ValueError("saved fit result identity differs")
    if result["new_fit"]:
        frozen = public._read(directory / "fit" / "freeze.json")
        selected = plan["selections"][task["task_id"]]
        observed = sibling_fit.inspect_child_fit(directory / "fit")["result"]
        if (
            selected["status"] != "ready"
            or frozen["lineage"]
            != {
                "plan": plan["artifact_sha256"],
                "selection": content_hash(selected),
            }
            or frozen["public_fit"]["request"]
            != request_for(selected["bundle"], plan, task, 1).model_dump(mode="json")
            or observed != result["fit"]
            or result["status"] != observed["status"]
        ):
            raise ValueError("saved numerical evidence differs")
    elif result["status"] != plan["selections"][task["task_id"]]["status"]:
        raise ValueError("non-fitting result differs from the selected disposition")
    return result


def report(root: Path) -> dict:
    """Report historical and new attempts separately, with every check denominator."""
    plan = verify(root)
    inputs = sealed_read(root / "inputs.json")
    rows = []
    for task, audited in zip(plan["tasks"], inputs["rows"], strict=True):
        name = task["task_id"]
        selected = plan["selections"][name]
        path = root / "results" / name / "result.json"
        result = checked_result(root, plan, task) if path.exists() else None
        history = inputs["historical_results"][name]
        evidence = (
            history if selected["status"] == "historical_fit_retained" else result
        )
        fit = (evidence or {}).get("fit") or {}
        static = (
            repair.assessment(
                selected["bundle"],
                task["case"],
                audit.equations._surveys(plan["cells"][task["case"]]),
            )
            if selected["bundle"]
            else None
        )
        rows.append(
            {
                "task": name,
                "selection_status": selected["status"],
                "origin": selected["origin"],
                "selected_attempt": selected.get("attempt"),
                "status": (result or {}).get(
                    "status",
                    "pending" if selected["status"] == "ready" else selected["status"],
                ),
                "training": fit.get("training"),
                "validation": fit.get("validation"),
                "replay": (evidence or {}).get("replay"),
                "assessment": (evidence or {}).get("assessment"),
                "static_assessment": static,
                "historical_result": history,
                "parent_training": inputs["parents"][name]["result"]["fit"].get(
                    "training"
                ),
                "parent_validation": inputs["parents"][name]["result"]["fit"].get(
                    "validation"
                ),
                "new_fit": bool((result or {}).get("new_fit")),
                "historical_status": audited.get("historical_status"),
            }
        )
    attempts = [a for row in inputs["rows"] for a in row["attempts"]]
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "rows": rows,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "selection_counts": dict(Counter(r["selection_status"] for r in rows)),
        "classification_counts": dict(Counter(a["classification"] for a in attempts)),
        "newly_mechanically_valid_attempts": sum(
            a["newly_mechanically_valid"] for a in attempts
        ),
        "new_fit_results": sum(r["new_fit"] for r in rows),
        "maximum_new_fits": plan["maximum_new_fits"],
        "llm_calls": 0,
        "test_data_opened": False,
        "automatic_promotion": False,
        "automatic_followup": False,
        "new_fitted_checks": audit.historical._check_counts(
            [r["assessment"] for r in rows if r["new_fit"] and r["assessment"]]
        ),
    }
    public._write(root / "summary.json", summary)
    lines = [
        "# Saved-reply basin fitting handoff",
        "",
        "Latest eligible reply in its actual historical context; "
        "no score-based child selection.",
        "No new LLM calls. Frozen fitter and compatible parent warm starts. "
        "Historical fits are retained separately.",
        "Replay is numerical agreement, not predictive accuracy. "
        "Scoped checks are not a science score.",
        "Historical parents have different total budgets. "
        "No automatic promotion or follow-up.",
        "",
        "| Task | Origin | Status | Train NMSE | Validation NMSE "
        "| Failed checks | Replay |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        values = [
            r["task"],
            r["origin"],
            r["status"],
            (r["training"] or {}).get("normalized_mse"),
            (r["validation"] or {}).get("normalized_mse"),
            sum(c["status"] == "fail" for c in r["assessment"]["checks"])
            if r["assessment"]
            else None,
            (r["replay"] or {}).get("replay_agreement"),
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    equations = [
        "# Frozen saved children",
        "",
        "Choices were sealed before fitting. "
        "Aliases and normalizations are in inputs.json.",
    ]
    for task in plan["tasks"]:
        s = plan["selections"][task["task_id"]]
        if not s["bundle"]:
            continue
        base = s["bundle"]["initialization"]["base_candidate"]
        equations += ["", "## " + task["task_id"], "", s["status"], "", "```text"]
        equations += [p["name"] + " = " + p["expression"] for p in base["processes"]]
        equations += [e["state"] + "' = " + e["rhs"] for e in base["state_equations"]]
        equations += ["```"]
    (root / "EQUATIONS.md").write_text("\n".join(equations) + "\n")
    return summary
