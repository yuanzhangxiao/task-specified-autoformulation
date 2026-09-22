"""One saved-model repair episode, explicit preview, and unchanged fitting profile."""

from __future__ import annotations

import hashlib
import json
import re
import signal
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import monotonic

from autoformalism.expressions import ModelValidationError
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.llm.staged_topology import DeferredCall, visible_response
from autoformalism.rebuttal import basin_equation_audit as audit
from autoformalism.rebuttal import detention_process_pilot as prior
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.rebuttal.review_deadline_pipeline import request_for
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.search import basin_model_repair as repair
from autoformalism.staged_topology import content_hash

PROTOCOL = "basin-equation-repair-1"
REPO = Path(__file__).resolve().parents[3]


def launcher_hash() -> str:
    """Bind Python and shell entry points as well as package source."""
    paths = [
        "scripts/basin_repair_pilot.py",
        "scripts/submit_basin_repair_pilot.py",
        "scripts/hpc/run_basin_repair_pilot_aces.sh",
        "scripts/hpc/submit_basin_repair_pilot_aces.sh",
        "scripts/hpc/run_staged_topology_server.sh",
    ]
    return content_hash(
        {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in paths}
    )


def freeze(source: Path, audit_root: Path, root: Path) -> dict:
    """Verify the existing audit, independently reconstruct, then copy public inputs."""
    source, audit_root, root = (p.resolve() for p in (source, audit_root, root))
    if any(
        root.is_relative_to(p) or p.is_relative_to(root) for p in (source, audit_root)
    ):
        raise ValueError("new output must be separate from source and audit")
    old = audit._read(source, "plan.json")
    config = prior.PilotConfig.model_validate(old["config"])
    frozen = sealed_read(audit_root / "freeze.json")
    summary = sealed_read(audit_root / "summary.json")
    records, hashes = audit._snapshot(source, audit._paths(old))
    if (
        old.get("function_delivery_policy") != "identified-function-delivery-1"
        or old["tasks"] != prior.tasks(config)
        or old.get("test_data_opened")
        or old.get("private_reference_opened")
        or frozen["protocol"] != audit.PROTOCOL
        or frozen["policy"] != repair.checks.POLICY
        or summary["identity"]
        != content_hash({k: v for k, v in frozen.items() if k != "artifact_sha256"})
        or summary["protocol"] != audit.PROTOCOL
        or frozen["source_plan_sha256"] != old["artifact_sha256"]
        or Path(frozen["source"]).resolve() != source
        or hashes != frozen["files"]
        or any(
            summary[k]
            for k in (
                "llm_calls",
                "optimizer_calls",
                "test_data_opened",
                "private_reference_opened",
            )
        )
    ):
        raise ValueError("source or equation audit gate differs")
    for case, cell in old["cells"].items():
        if (
            cell["brief"]["scientific_context"].strip()
            != (REPO / f"configs/detention_prompts/{case}.md").read_text().strip()
        ):
            raise ValueError("public rule profile differs")
    rows, verified = [], set()
    for task in old["tasks"]:
        row = audit._row(task, old["cells"][task["case"]], records, verified)
        rows.append(row)
    if rows != summary["rows"] or any(r["status"] != "assessed" for r in rows):
        raise ValueError("requires all independently reconstructed audit rows")
    parents = {}
    for task, row in zip(old["tasks"], rows, strict=True):
        name = task["task_id"]
        result = records[f"results/{name}/result.json"]
        if result is None or (result.get("fit") or {}).get("parameters") is None:
            raise ValueError(
                "requires completed saved fits, including parameter-free fits"
            )
        parents[name] = {
            "bundle": records[f"results/{name}/proposal.json"]["bundle"],
            "audit": row,
            "result": result,
        }
    if audit._snapshot(source, audit._paths(old))[1] != hashes:
        raise ValueError("source changed while preparing")
    baseline = {
        "parents": parents,
        "source_plan_sha256": old["artifact_sha256"],
        "audit_summary_sha256": summary["artifact_sha256"],
    }
    new_config = deepcopy(old["config"])
    new_config["protocol"] = PROTOCOL
    plan = {
        "protocol": PROTOCOL,
        "policy": repair.POLICY,
        "config": new_config,
        "tasks": old["tasks"],
        "cells": old["cells"],
        "baseline_sha256": content_hash(baseline),
        "source_files": hashes,
        "source_plan_sha256": old["artifact_sha256"],
        "audit_summary_sha256": summary["artifact_sha256"],
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "launcher_sha256": launcher_hash(),
        "max_repair_calls": config.model_settings.attempts_per_step,
        "saved_gain_policy_labels": (
            "parent provenance only; child equations are directly revised"
        ),
        "automatic_followup": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "scientific_judge": "off",
        "selection": "none_diagnostic_children_only",
    }
    with public._lock(root):
        if not (root / "plan.json").exists() and any(
            (root / n).exists() for n in ("results", "scheduler")
        ):
            raise ValueError("output already contains execution artifacts")
        sealed_write(root / "baseline.json", baseline)
        sealed_write(root / "plan.json", plan)
    return verify(root)


def verify(root: Path) -> dict:
    """Workers do not need historical mounts; frozen code and inputs must match."""
    plan = sealed_read(root / "plan.json")
    baseline = sealed_read(root / "baseline.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["policy"] != repair.POLICY
        or plan["source_sha256"] != public._source_identity()
        or plan["launcher_sha256"] != launcher_hash()
        or plan["runtime"] != public._runtime()
        or plan["baseline_sha256"] != baseline["artifact_sha256"]
        or plan["automatic_followup"]
        or plan["test_data_opened"]
        or plan["private_reference_opened"]
        or plan["config"]["fit_profile"] != "collocation-single-target-v2"
        or plan["max_repair_calls"]
        != plan["config"]["model_settings"]["attempts_per_step"]
        or set(baseline["parents"]) != {t["task_id"] for t in plan["tasks"]}
    ):
        raise ValueError("frozen repair inputs, runtime or policy differ")
    return plan


def _transition(current, parent, raw, case, surveys):
    """Only explicit acceptance of a previously displayed child admits fitting."""
    reply = repair.ModelRepair.model_validate(raw)
    if reply.accept_displayed:
        if current["candidate"] == parent["candidate"]:
            return current, None, "retained"
        static = repair.assessment(current, case, surveys)
        failed = [c["code"] for c in static["checks"] if c["status"] == "fail"]
        if failed:
            raise ValueError(
                f"current assembled model still has static violations: {failed}"
            )
        return current, None, "confirmed"
    if not reply.has_edits():
        raise ValueError(
            "empty edit is not acceptance; use accept_displayed or supply a patch"
        )
    tx = repair.apply(current, raw, case)
    return tx["bundle"], {k: v for k, v in tx.items() if k != "bundle"}, None


def run_one(root: Path, plan: dict, task: dict, client) -> dict:
    """Cache every call, replay every transaction, preserve unconfirmed attempts."""
    parent = sealed_read(root / "baseline.json")["parents"][task["task_id"]]
    original = parent["bundle"]
    current = original
    events = []
    previous = None
    status = None
    directory = root / "results" / task["task_id"]
    surveys = audit._surveys(plan["cells"][task["case"]])
    with public._lock(directory):
        if (directory / "proposal.json").exists():
            return checked_proposal(root, plan, task)
        for i in range(plan["max_repair_calls"]):
            payload = repair.payload(
                current,
                parent["audit"]["assessment"],
                repair.assessment(current, task["case"], surveys),
                plan["max_repair_calls"] - i,
                previous,
            )
            try:
                record = client.call(
                    system=repair.SYSTEM,
                    user=json.dumps(payload, sort_keys=True),
                    response_model=repair.ModelRepair,
                    step="assembled_model_repair",
                    attempt=i,
                )
            except RepairBudgetExceeded as exc:
                status = "request_budget_exhausted"
                previous = {"error": str(exc)}
                break
            raw = None
            error = None
            transaction = None
            before = content_hash(current)
            try:
                raw = visible_response(record)
                current, transaction, status = _transition(
                    current, original, raw, task["case"], surveys
                )
            except (ValueError, ModelValidationError) as exc:
                error = str(exc)
            event = {
                "attempt": i,
                "request_hash": record["request_hash"],
                "payload_sha256": content_hash(payload),
                "before_sha256": before,
                "after_sha256": content_hash(current),
                "raw": raw,
                "error": error,
                "transaction": transaction,
                "decision": status,
            }
            sealed_write(directory / f"attempt-{i}.json", event)
            events.append(event)
            previous = {"error": error, "raw": raw, "transaction": transaction}
            if status is not None:
                break
        status = status or "unconfirmed_trial"
        proposal = {
            "plan_sha256": plan["artifact_sha256"],
            "task": task,
            "status": status,
            "parent_candidate_sha256": content_hash(original["candidate"]),
            "events": events,
            "last_feedback": previous,
            "bundle": current,
            "static_assessment": repair.assessment(current, task["case"], surveys),
            "historical_models_unchanged": True,
            "automatic_promotion": False,
        }
        sealed_write(directory / "proposal.json", proposal)
        return checked_proposal(root, plan, task)


def checked_proposal(root: Path, plan: dict, task: dict) -> dict:
    """Independent replay from raw cached replies before admitting a fitted child."""
    directory = root / "results" / task["task_id"]
    p = sealed_read(directory / "proposal.json")
    parent = sealed_read(root / "baseline.json")["parents"][task["task_id"]]
    original = parent["bundle"]
    current = original
    previous = None
    status = None
    surveys = audit._surveys(plan["cells"][task["case"]])
    if (
        p["plan_sha256"] != plan["artifact_sha256"]
        or p["task"] != task
        or p["parent_candidate_sha256"] != content_hash(original["candidate"])
        or p["automatic_promotion"] is not False
        or p["historical_models_unchanged"] is not True
        or len(p["events"]) > plan["max_repair_calls"]
    ):
        raise ValueError("repair proposal identity differs")
    for i, event in enumerate(p["events"]):
        if status is not None:
            raise ValueError("extra calls after acceptance")
        if not re.fullmatch(r"[a-f0-9]{64}", event["request_hash"]):
            raise ValueError("invalid request digest")
        payload = repair.payload(
            current,
            parent["audit"]["assessment"],
            repair.assessment(current, task["case"], surveys),
            plan["max_repair_calls"] - i,
            previous,
        )
        path = directory / "calls" / f"{event['request_hash']}.json"
        record = public._read(path)
        if (
            content_hash(record["request"]) != event["request_hash"]
            or record["request_hash"] != event["request_hash"]
            or record["attempt"] != i
            or record["step"] != "assembled_model_repair"
            or record["request"]["settings"] != plan["config"]["model_settings"]
            or record["request"]["namespace"]
            != content_hash([plan["artifact_sha256"], task])
        ):
            raise ValueError("cached repair call identity differs")
        messages = record["request"]["body"]["messages"]
        if messages != [
            {"role": "system", "content": repair.SYSTEM},
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        ]:
            raise ValueError("repair payload differs from displayed model")
        before = content_hash(current)
        raw = None
        error = None
        tx = None
        try:
            raw = visible_response(record)
            current, tx, status = _transition(
                current, original, raw, task["case"], surveys
            )
        except (ValueError, ModelValidationError) as exc:
            error = str(exc)
        expected = {
            "attempt": i,
            "request_hash": event["request_hash"],
            "payload_sha256": content_hash(payload),
            "before_sha256": before,
            "after_sha256": content_hash(current),
            "raw": raw,
            "error": error,
            "transaction": tx,
            "decision": status,
        }
        if event != expected or sealed_read(directory / f"attempt-{i}.json") != {
            **expected,
            "artifact_sha256": content_hash(expected),
        }:
            raise ValueError("repair transaction ledger differs")
        previous = {"error": error, "raw": raw, "transaction": tx}
    if p["bundle"] != current or p["static_assessment"] != repair.assessment(
        current, task["case"], surveys
    ):
        raise ValueError("assembled repair preview differs")
    if p["status"] not in (
        {status} if status else {"unconfirmed_trial", "request_budget_exhausted"}
    ):
        raise ValueError("repair acceptance differs")
    if p["status"] != "request_budget_exhausted" and p["last_feedback"] != previous:
        raise ValueError("repair feedback differs")
    return p


def run_proposals(root: Path, base_url: str, wall_seconds: float) -> None:
    """One local server, bounded repair calls per saved arm, no new constructions."""
    plan = verify(root)
    deadline = monotonic() + wall_seconds - plan["config"]["shutdown_margin_seconds"]
    draining = False

    def drain(*_):
        nonlocal draining
        draining = True

    signal.signal(signal.SIGTERM, drain)
    signal.signal(signal.SIGINT, drain)
    # Reuse the historical budgeted/cached transport without changing its protocol.
    client_plan = {
        **plan,
        "config": {**plan["config"], "protocol": prior.GAIN_PROTOCOL},
    }
    for task in plan["tasks"]:
        client = prior.make_client(
            root,
            client_plan,
            task,
            base_url,
            can_start=lambda: not draining and monotonic() < deadline,
        )
        try:
            run_one(root, plan, task, client)
        except DeferredCall:
            return


def fit_task(root: Path, index: int) -> dict:
    """One frozen-fitter attempt only after explicit review of the complete child."""
    plan = verify(root)
    if index not in range(len(plan["tasks"])):
        raise ValueError("task index outside plan")
    task = plan["tasks"][index]
    directory = root / "results" / task["task_id"]
    with public._lock(directory):
        if (directory / "result.json").exists():
            return sealed_read(directory / "result.json")
        if not (directory / "proposal.json").exists():
            return {"status": "proposal_missing"}
        proposal = checked_proposal(root, plan, task)
        result = {
            "task": task,
            "proposal_sha256": proposal["artifact_sha256"],
            "status": proposal["status"],
            "fit": None,
            "replay": None,
            "assessment": None,
        }
        if proposal["status"] == "confirmed":
            request = request_for(proposal["bundle"], plan, task, 1)
            cell = plan["cells"][task["case"]]
            parent = sealed_read(root / "baseline.json")["parents"][task["task_id"]]
            sibling_fit.prepare_child_fit(
                request_for(parent["bundle"], plan, task, 0),
                request,
                parent["result"]["fit"]["parameters"],
                PublicSplit.model_validate(cell["training"]),
                PublicSplit.model_validate(cell["validation"]),
                directory / "fit",
                lineage={
                    "plan": plan["artifact_sha256"],
                    "proposal": proposal["artifact_sha256"],
                },
            )
            fit = sibling_fit.execute_child_fit(directory / "fit")
            result.update(status=fit.status, fit=fit.model_dump(mode="json"))
            if fit.status == "complete":
                result["assessment"] = repair.assessment(
                    proposal["bundle"],
                    task["case"],
                    audit._surveys(cell),
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
                        result["replay"] = prior.replay(
                            request, dict(fit.parameters), cell
                        )
                    except (ValueError, RuntimeError, TimeoutError) as exc:
                        result["replay"] = {
                            "status": "failed",
                            "error": str(exc),
                            "replay_agreement": False,
                        }
        return sealed_write(directory / "result.json", result)


def report(root: Path) -> dict:
    """No winner selection: report fit quality and each scoped physics check apart."""
    plan = verify(root)
    parents = sealed_read(root / "baseline.json")["parents"]
    rows = []
    for task in plan["tasks"]:
        name = task["task_id"]
        directory = root / "results" / name
        proposal = (
            checked_proposal(root, plan, task)
            if (directory / "proposal.json").exists()
            else None
        )
        result = (
            sealed_read(directory / "result.json")
            if (directory / "result.json").exists()
            else None
        )
        if result and (
            not proposal or result["proposal_sha256"] != proposal["artifact_sha256"]
        ):
            raise ValueError("fit belongs to another proposal")
        records = [public._read(p) for p in (directory / "calls").glob("*.json")]
        measured = [
            r["observed_total_tokens"]
            for r in records
            if r.get("observed_total_tokens") is not None
        ]
        old = parents[name]
        rows.append(
            {
                "task": name,
                "case": task["case"],
                "gain_parent": task["gain_policy"],
                "status": (result or {}).get("status", "missing"),
                "repair_status": (proposal or {}).get("status", "missing"),
                "parent_training": old["result"]["fit"].get("training"),
                "parent_validation": old["result"]["fit"].get("validation"),
                "training": ((result or {}).get("fit") or {}).get("training"),
                "validation": ((result or {}).get("fit") or {}).get("validation"),
                "replay": (result or {}).get("replay"),
                "before": old["audit"]["assessment"],
                "after": (result or {}).get("assessment"),
                "static_after": (proposal or {}).get("static_assessment"),
                "physical_calls": len(records),
                "observed_total_tokens": sum(measured),
                "unmeasured_calls": len(records) - len(measured),
            }
        )
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "rows": rows,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "repair_status_counts": dict(Counter(r["repair_status"] for r in rows)),
        "physical_calls": sum(r["physical_calls"] for r in rows),
        "observed_total_tokens": sum(r["observed_total_tokens"] for r in rows),
        "unmeasured_calls": sum(r["unmeasured_calls"] for r in rows),
        "scientific_compliance_score": None,
        "automatic_followup": False,
        "automatic_promotion": False,
        "test_data_opened": False,
        "critic_calls": 0,
    }
    summary["checks_before"] = _check_counts([r["before"] for r in rows])
    summary["checks_after_fitting"] = _check_counts(
        [r["after"] for r in rows if r["after"]]
    )
    summary["fitted_assessments_available"] = sum(r["after"] is not None for r in rows)
    public._write(root / "summary.json", summary)
    lines = [
        "# Saved basin model repair",
        "",
        "One bounded episode per saved arm; paired arms share constructions.",
        "Fixed/shared declarations propagate; rebuilt models are reviewed.",
        "Frozen fitter. Historical parents are not equal-budget controls.",
        "No automatic promotion/follow-up. Scoped checks are not a science score.",
        "",
        "| Task | Repair | Result | Old val NMSE | New val NMSE "
        "| Old failed checks | New failed checks | Tokens |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def score(value):
        return (value or {}).get("normalized_mse")

    def failed(value):
        return sum(c["status"] == "fail" for c in value["checks"]) if value else None

    for row in rows:
        values = [
            row["task"],
            row["repair_status"],
            row["status"],
            score(row["parent_validation"]),
            score(row["validation"]),
            failed(row["before"]),
            failed(row["after"]),
            row["observed_total_tokens"],
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    equations = [
        "# Rebuilt basin equations",
        "",
        "Raw scientific decisions and mechanical rewrites are in each proposal.json.",
        "Unconfirmed trials are diagnostic drafts, not accepted models.",
    ]
    for task in plan["tasks"]:
        path = root / "results" / task["task_id"] / "proposal.json"
        if not path.exists():
            continue
        proposal = sealed_read(path)
        base = proposal["bundle"]["initialization"]["base_candidate"]
        equations += [
            "",
            "## " + task["task_id"],
            "",
            "Status: " + proposal["status"],
            "",
            "```text",
        ]
        equations += [p["name"] + " = " + p["expression"] for p in base["processes"]]
        equations += [e["state"] + "' = " + e["rhs"] for e in base["state_equations"]]
        equations += [
            "```",
            "",
            "Fitted parameters: " + ", ".join(p["name"] for p in base["parameters"]),
        ]
    (root / "EQUATIONS.md").write_text("\n".join(equations) + "\n")
    return summary


def _check_counts(assessments: list[dict]) -> dict:
    """Keep every scoped check denominator; unavailable results are not zeros."""
    result = {}
    for value in assessments:
        for check in value["checks"]:
            result.setdefault(check["code"], Counter())[check["status"]] += 1
    return {k: dict(v) for k, v in sorted(result.items())}
