#!/usr/bin/env python3
"""Collect or report a portable, read-only census of a frozen component campaign.

Standard library only. This does not import the modeling runtime, compile a
candidate, evaluate a scientific predicate, or authorize a retry or test access.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

if __package__:
    from . import component_scheduler_audit as scheduler
    from .component_audit_io import Snapshot, diagnostic, digest, identifier, label
    from .component_stage_inventory import inventory
else:
    import component_scheduler_audit as scheduler
    from component_audit_io import Snapshot, diagnostic, digest, identifier, label
    from component_stage_inventory import inventory

PROTOCOL = "component-foundation-audit-1"
CAMPAIGN = "final-component-campaign-1"
DELIVERY_FAILURES = {"provider_request_failed", "request_preflight_failed"}
# Mirrors the frozen nine-case campaign, not a discovery of new benchmark cases.
SECONDARY_CELLS = {
    f"phase_b_dalla_man_{target}_canonical_named_{difficulty}"
    for target in ("t1", "t2")
    for difficulty in ("easy", "hard")
}


def counts(values) -> dict:
    """Stable counts preserve missing/unknown categories instead of dropping rows."""
    return dict(sorted(Counter(values).items()))


def metric(fit: dict | None, split: str) -> float | None:
    """Project a finite recorded rollout NMSE without recomputing predictions."""
    value = (fit or {}).get(split) or {}
    number = value.get("normalized_mse")
    return (
        float(number)
        if type(number) in {int, float} and math.isfinite(number)
        else None
    )


def fit_summary(fit: dict | None) -> dict:
    """Do not equate a terminal receipt with a usable numerical model."""
    fit = fit or {}
    scores = {s: metric(fit, s) for s in ("training", "validation")}
    usable = (
        fit.get("status") == "complete"
        and fit.get("parameters") is not None
        and all(
            scores[s] is not None
            and (fit.get(s) or {}).get("available") is True
            and not (fit.get(s) or {}).get("failed_trajectories")
            for s in scores
        )
    )
    return {
        "status": label(fit.get("status")),
        "finite_complete_fit": usable,
        **{f"{s}_nmse": v for s, v in scores.items()},
    }


def bound(reader: Snapshot, path: str, expected: dict) -> dict | None:
    """Check campaign seals and exact task/predecessor bindings when available."""
    value = reader.read(path, sealed=True)
    if value is not None and any(value.get(k) != v for k, v in expected.items()):
        reader.issue(path, "artifact_binding_differs")
        return None
    return value


def receipt(reader: Snapshot, path: str) -> dict | None:
    """Validate the fitter's distinct compact-hash wrapper, without execution."""
    value = reader.read(path)
    if value is None:
        return None
    try:
        valid = isinstance(value.get("result"), dict) and value.get("sha256") == digest(
            value["result"], compact=True
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        reader.issue(path, "fit_receipt_hash_differs")
        return None
    return value


def selected_summary(
    reader: Snapshot, selected: dict | None, task: dict, index: int, path: str
) -> dict | None:
    """Join retained models to their origin receipt; never export their equations."""
    if not selected:
        return None
    origin = selected.get("origin_round")
    if (
        selected.get("origin_task") != task["task_id"]
        or type(origin) is not int
        or not 0 <= origin <= index
    ):
        reader.issue(path, "selected_origin_differs")
        return {"origin_round": None, "status": "invalid_origin"}
    saved = receipt(
        reader, f"results/{task['task_id']}/round_{origin:02d}/fit/result.json"
    )
    fit = selected.get("fit") or {}
    if (
        saved is None
        or saved.get("result") != fit
        or selected.get("fit_result_sha256") != digest(saved, compact=True)
    ):
        reader.issue(path, "selected_fit_receipt_differs")
    if fit.get("request_sha256") != digest(selected.get("request"), compact=True):
        reader.issue(path, "selected_request_hash_differs")
    return {
        "origin_round": origin,
        "fit_result_sha256": selected.get("fit_result_sha256"),
        **fit_summary(fit),
    }


def round_record(
    reader: Snapshot, plan: dict, task: dict, index: int, parent: dict | None
) -> tuple[dict, dict | None, list, list]:
    """Collect exact planned paths, preserving interruptions and missing stages."""
    base = f"results/{task['task_id']}/round_{index:02d}"
    expected = {"task": task, "round": index}
    proposal = bound(
        reader,
        f"{base}/proposal.json",
        {**expected, "parent_sha256": (parent or {}).get("artifact_sha256")},
    )
    result = bound(reader, f"{base}/result.json", expected)
    if index and parent is None and (proposal or result):
        reader.issue(base, "missing_predecessor_result")
    if result:
        links = {
            "parent_sha256": (parent or {}).get("artifact_sha256"),
            "proposal_sha256": (proposal or {}).get("artifact_sha256"),
        }
        if any(k in result and result[k] != v for k, v in links.items()):
            reader.issue(f"{base}/result.json", "result_predecessor_differs")
    critic_path = f"{base}/critic.json"
    critic = None
    if task["critic"] or reader.exists(critic_path):
        critic = bound(
            reader,
            critic_path,
            {
                "identity": plan["artifact_sha256"],
                "task": task,
                "round": index,
                "proposal_sha256": (proposal or {}).get("artifact_sha256"),
            },
        )
    fit = receipt(reader, f"{base}/fit/result.json")
    worker_started = reader.exists(f"{base}/worker_started.json")
    fit_started = reader.exists(f"{base}/fit/started.json")
    row = {
        "task": task["task_id"],
        "round": index,
        "status": label((result or {}).get("status")) or "missing",
        "result_sha256": (result or {}).get("artifact_sha256"),
        "proposal_status": label((proposal or {}).get("status")),
        "proposal_sha256": (proposal or {}).get("artifact_sha256"),
        "critic_status": label((critic or {}).get("status")),
        "critic_receipt": critic is not None,
        "worker_started": worker_started,
        "fit_started": fit_started,
        "fit_receipt": fit is not None,
        "fit": fit_summary((fit or {}).get("result")),
        "fit_trigger": label((result or {}).get("fit_trigger")),
        "selected": selected_summary(
            reader, (result or {}).get("selected"), task, index, f"{base}/result.json"
        ),
        "fallback_used": (proposal or {}).get("fallback_used")
        if type((proposal or {}).get("fallback_used")) is bool
        else None,
        **diagnostic((result or {}).get("error")),
    }
    stage_rows, attempts = inventory(reader, base, task["task_id"], index, proposal)
    return row, result, stage_rows, attempts


def pruning_record(reader: Snapshot, plan: dict, task: dict, last: dict | None) -> dict:
    """Record terminal pruning and final-critic coverage, not re-certification."""
    base = f"pruning/{task['task_id']}"
    expected = {"identity": plan["artifact_sha256"], "task": task}
    result = bound(reader, f"{base}/result.json", expected)
    if result:
        binding = bound(
            reader,
            f"{base}/parent.json",
            {
                **expected,
                "source_result_sha256": (last or {}).get("artifact_sha256"),
                "parent_sha256": digest((last or {}).get("selected"), compact=True),
            },
        )
        if binding is None:
            reader.issue(f"{base}/parent.json", "missing_or_invalid_pruning_parent")
        if last is None:
            reader.issue(f"{base}/result.json", "pruning_without_final_round")
        if "choice" in result:
            choice = bound(
                reader, f"{base}/choice.json", {"identity": plan["artifact_sha256"]}
            )
            if choice != result["choice"]:
                reader.issue(f"{base}/result.json", "pruning_choice_differs")
    fits = {}
    for arm in ("control", "pruned"):
        saved = receipt(reader, f"{base}/{arm}/result.json")
        fitted = (saved or {}).get("result")
        recorded = (result or {}).get("fits", {}).get(arm)
        if recorded is not None and recorded != fitted:
            reader.issue(f"{base}/result.json", "pruning_fit_differs")
        fits[arm] = {
            **fit_summary(fitted),
            "started": reader.exists(f"{base}/{arm}/started.json"),
        }
    review_path = f"{base}/critic.json"
    review = None
    if task["critic"] or reader.exists(review_path):
        review = bound(
            reader,
            review_path,
            {
                "identity": plan["artifact_sha256"],
                "pruning_sha256": (result or {}).get("artifact_sha256"),
            },
        )
    parent = (last or {}).get("selected")
    missing_review = bool(task["critic"] and parent and result and review is None)
    selection = (result or {}).get("selection") or {}
    final = {
        "endpoint": "final",
        "status": "missing"
        if result is None or missing_review
        else label(result.get("status")) or "unrecorded",
        "fit": fit_summary(selection.get("fit") or (parent or {}).get("fit")),
    }
    endpoints = [final]
    if task["cell"] in SECONDARY_CELLS and all(
        task[k] for k in ("critic", "scientific_verifier", "shared_processes")
    ):
        control = (result or {}).get("fits", {}).get("control")
        endpoints.extend(
            [
                {
                    "endpoint": "no_pruning",
                    "status": "complete"
                    if parent
                    else "model_unavailable"
                    if last
                    else "missing",
                    "fit": fit_summary((parent or {}).get("fit")),
                },
                {
                    "endpoint": "unchanged_refit_control",
                    "status": "complete"
                    if control
                    else "model_unavailable"
                    if result
                    else "missing",
                    "fit": fit_summary(control),
                },
            ]
        )
    return {
        "status": label((result or {}).get("status")) or "missing",
        "choice_status": label(((result or {}).get("choice") or {}).get("status")),
        "selected": label(selection.get("selected")),
        "fits": fits,
        "critic_status": label((review or {}).get("status")),
        "missing_final_critic": missing_review,
        "endpoints": endpoints,
    }


def frontier(rows: list[dict], pruning: dict, task: dict, authorized: bool) -> dict:
    """Advisory first missing stage; never reset budgets or infer active jobs."""
    for index, row in enumerate(rows):
        if row["result_sha256"] is not None:
            continue
        action = "inspect_saved_state"
        if index and rows[index - 1]["result_sha256"] is None:
            action = "wait_for_predecessor"
        elif index and rows[index - 1]["proposal_status"] in DELIVERY_FAILURES:
            action = "inspect_delivery_failure"
        elif row["worker_started"] or row["fit_started"] or row["fit_receipt"]:
            action = "inspect_consumed_fit_attempt"
        elif row["proposal_sha256"] is None:
            if task["critic"] and not authorized:
                action = "inspect_critic_authorization"
            elif (
                task["critic"]
                and index
                and rows[index - 1]["selected"]
                and not rows[rows[index - 1]["selected"].get("origin_round") or 0][
                    "critic_receipt"
                ]
            ):
                action = "critic_for_retained_parent"
            else:
                action = "propose"
        elif task["critic"] and not row["critic_receipt"]:
            action = "critic" if authorized else "inspect_critic_authorization"
        else:
            action = "fit"
        return {"round": index, "action": action}
    if pruning["status"] == "missing":
        return {"round": len(rows) - 1, "action": "prune"}
    if pruning["missing_final_critic"]:
        return {
            "round": len(rows),
            "action": "final_critic" if authorized else "inspect_critic_authorization",
        }
    return {"round": None, "action": "recorded_endpoint_complete"}


def scheduler_inventory(reader: Snapshot, plan: dict) -> tuple[list[dict], list[dict]]:
    """Project worker starts and submission IDs; these are not scheduler states."""
    workers, submissions = [], []
    for name in reader.children("workers"):
        identifier(name)
        value = bound(
            reader,
            f"workers/{name}/started.json",
            {"identity": plan["artifact_sha256"]},
        )
        if value:
            finished = bound(
                reader,
                f"workers/{name}/result.json",
                {k: value.get(k) for k in ("identity", "worker", "stage", "job_id")},
            )
            operations = (finished or {}).get("completed_operations")
            workers.append(
                {
                    "worker": name,
                    "stage": label(value.get("stage")),
                    "job_id": label(value.get("job_id")),
                    "finish_receipt": finished is not None,
                    "completed_operations": operations
                    if type(operations) is int and operations >= 0
                    else None,
                }
            )
    for name in reader.children("submissions"):
        identifier(name)
        path = f"submissions/{name}/manifest.json"
        entry = {"wave": name, "jobs": {}, "unconfirmed_stages": []}
        if reader.exists(path):
            value = reader.read(path) or {}
            if value.get("plan_sha256") != plan["artifact_sha256"]:
                reader.issue(path, "submission_plan_differs")
            else:
                entry["jobs"] = {
                    label(k): str(v)
                    for k, v in (value.get("jobs") or {}).items()
                    if label(k) and re.fullmatch(r"\d+", str(v))
                }
        # An interrupted/rejected sbatch may never produce a complete manifest.
        for stage in ("propose", "critic", "fit", "prune"):
            path = f"submissions/{name}/{stage}.reply.json"
            if stage in entry["jobs"]:
                continue
            if reader.exists(path):
                reply = reader.read(path) or {}
                stdout = str(reply.get("stdout", "")).strip()
                if reply.get("returncode") == 0 and re.fullmatch(
                    r"\d+(;[\w.-]+)?", stdout
                ):
                    entry["jobs"][stage] = stdout.split(";", 1)[0]
                else:
                    entry["unconfirmed_stages"].append(stage)
            elif reader.exists(f"submissions/{name}/{stage}.intent.json"):
                entry["unconfirmed_stages"].append(stage)
        submissions.append(entry)
    return workers, submissions


def collect(root: Path) -> dict:
    """Produce a complete planned-row census from observed, explicitly named files."""
    reader = Snapshot(root)
    plan = reader.read("plan.json", sealed=True)
    if not plan or plan.get("protocol") != CAMPAIGN:
        raise ValueError("sealed plan tagged final-component-campaign-1 required")
    total = plan["config"]["rounds"]
    tasks = plan["tasks"]
    if (
        type(total) is not int
        or not 1 <= total <= 1000
        or not tasks
        or len(tasks) > 10000
    ):
        raise ValueError("invalid or unbounded frozen task matrix")
    names = [identifier(t["task_id"]) for t in tasks]
    if len(names) != len(set(names)):
        raise ValueError("duplicate task identity in frozen plan")
    auth = bound(
        reader,
        "critic_authorization.json",
        {"identity": plan["artifact_sha256"], "known_case_gate_passed": True},
    )
    frozen = reader.exists("evaluation_freeze.json")
    lineages, rows, stages, attempts = [], [], [], []
    for number, task in enumerate(tasks, 1):
        parent, history = None, []
        for index in range(total):
            row, parent, stage_rows, stage_attempts = round_record(
                reader, plan, task, index, parent
            )
            rows.append(row)
            history.append(row)
            stages.extend(stage_rows)
            attempts.extend(stage_attempts)
        pruning = pruning_record(reader, plan, task, parent)
        action = frontier(history, pruning, task, auth is not None)
        if frozen:
            action = {"round": None, "action": "frozen_no_search_actions"}
        lineages.append(
            {
                "task": task["task_id"],
                "factors": {
                    k: task.get(k)
                    for k in (
                        "cell",
                        "arm",
                        "seed",
                        "critic",
                        "scientific_verifier",
                        "shared_processes",
                        "block",
                    )
                },
                "round_status_counts": counts(r["status"] for r in history),
                "last_recorded_round": next(
                    (r["round"] for r in reversed(history) if r["result_sha256"]), None
                ),
                "last_retained": next(
                    (r["selected"] for r in reversed(history) if r["selected"]), None
                ),
                "pruning": pruning,
                "frontier": action,
            }
        )
        print(f"Inspected lineage {number}/{len(tasks)}: {task['task_id']}", flush=True)
    workers, submissions = scheduler_inventory(reader, plan)
    reader.finish()
    payload = {
        "protocol": PROTOCOL,
        "campaign_protocol": CAMPAIGN,
        "plan_sha256": plan["artifact_sha256"],
        "planned_rounds": total,
        "matrix_contract_revalidated": False,
        "lineages": lineages,
        "rounds": rows,
        "stages": stages,
        "attempts": attempts,
        "workers": workers,
        "submissions": submissions,
        "files": sorted(reader.files.values(), key=lambda x: x["path"]),
        "issues": sorted(reader.issues, key=lambda x: (x["path"], x["code"])),
        "evaluation_freeze_present": frozen,
        "scope": (
            "Recorded artifacts only. No compiler or scientific predicates were rerun. "
            "Counts follow the declared plan; its full matrix contract was not "
            "revalidated. "
            "Stage rejections are not necessarily model failures. "
            "Scheduler state unknown. Plan may embed public train/validation data; "
            "projections exclude them."
        ),
        "llm_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "campaign_changes": 0,
        "test_data_opened": False,
    }
    return {**payload, "artifact_sha256": digest(payload)}


def report(value: dict, scheduler_text: str | None = None) -> dict[str, str]:
    """Reproduce portable summary tables without source campaign or dependencies."""
    if value.get("protocol") != PROTOCOL or value.get("artifact_sha256") != digest(
        {k: v for k, v in value.items() if k != "artifact_sha256"}
    ):
        raise ValueError("audit bundle seal or protocol differs")
    lineages, rows, attempts = value["lineages"], value["rounds"], value["attempts"]
    summary = {
        "planned_lineages": len(lineages),
        "planned_round_rows": len(rows),
        "round_status_counts": counts(r["status"] for r in rows),
        "endpoint_status_counts": counts(
            e["status"] for t in lineages for e in t["pruning"]["endpoints"]
        ),
        "frontier_counts": counts(t["frontier"]["action"] for t in lineages),
        "artifact_issues": len(value["issues"]),
        "recorded_stage_events": len(attempts),
        "proposal_status_counts": counts(
            r["proposal_status"] or "missing" for r in rows
        ),
        "critic_status_counts": counts(
            r["critic_status"] or "no_receipt" for r in rows
        ),
        "stage_status_counts": {
            stage: counts(s["status"] for s in value["stages"] if s["stage"] == stage)
            for stage in sorted({s["stage"] for s in value["stages"]})
        },
        "integrity_review_required": bool(value["issues"]),
        "scheduler_state": "saved_observation_in_scheduler_summary"
        if scheduler_text is not None
        else "not_collected",
        "matrix_contract_revalidated": value["matrix_contract_revalidated"],
        "unconfirmed_submission_stages": sum(
            len(s["unconfirmed_stages"]) for s in value["submissions"]
        ),
        "test_data_opened": False,
    }
    text = [
        "# Component campaign foundation audit",
        "",
        value["scope"],
        "",
        "This audit does not authorize resume, promotion or test evaluation. "
        "A portable report verifies the audit bundle seal, "
        "not possession of the original source files.",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## First unresolved stage per lineage",
        "",
        "Actions are diagnostic hints, not scheduler instructions. "
        "Resolve artifact issues and confirm no active writer before recovery. "
        "Consumed attempts must retain their budgets.",
        "",
        "| Task | Last recorded round | Next round | Recorded frontier |",
        "|---|---:|---:|---|",
    ]
    for t in lineages:
        text.append(
            f"| {t['task']} | {t['last_recorded_round']} | "
            f"{t['frontier']['round']} | {t['frontier']['action']} |"
        )
    stage_text = [
        "# Saved stage evidence",
        "",
        "Counts are saved validator events, not physical requests, model success "
        "rates or independent scientific judgments. Missing records are unknown. "
        "Partial acceptance is retained in audit.json. "
        "Repeated repair attempts can belong to one model.",
        "",
        "| Stage | Accepted | Rejected | Unrecorded | Task/round/route groups |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in sorted({a["stage"] for a in attempts}):
        items = [a for a in attempts if a["stage"] == stage]
        c = Counter(a["outcome"] for a in items)
        groups = {(a["task"], a["round"], a["route"]) for a in items}
        stage_text.append(
            f"| {stage} | {c['accepted']} | {c['rejected']} | "
            f"{c['unrecorded']} | {len(groups)} |"
        )
    stage_text.extend(
        [
            "",
            "Recorded diagnostic categories (heuristic text classification; "
            "inspect original artifacts before assigning causes):",
            "",
            "```json",
            json.dumps(
                counts(a["category"] for a in attempts if a["category"]), indent=2
            ),
            "```",
            "",
        ]
    )
    jobs = sorted(
        {w["job_id"] for w in value["workers"] if w["job_id"]}
        | {j for s in value["submissions"] for j in s["jobs"].values() if j}
    )
    outputs = {
        "SUMMARY.md": "\n".join(text) + "\n",
        "STAGES.md": "\n".join(stage_text),
        "summary.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "job_ids.txt": "\n".join(jobs) + ("\n" if jobs else ""),
        "issues.json": json.dumps(value["issues"], indent=2) + "\n",
    }
    if scheduler_text is not None:
        outputs.update(scheduler.outputs(value, scheduler_text))
    return outputs


def write_outputs(directory: Path, outputs: dict[str, str]) -> None:
    """Idempotent for identical output; never overwrite another audit snapshot."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        path = directory / name
        if path.is_symlink() or (path.exists() and path.read_text() != content):
            raise ValueError(
                "output differs or is symlinked; use a fresh output directory"
            )
    for name, content in outputs.items():
        path = directory / name
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--root", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--input", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.add_argument("--scheduler", type=Path)
    args = parser.parse_args()
    if args.command == "collect":
        if args.output.resolve().is_relative_to(args.root.resolve()):
            raise ValueError("audit output must be outside the campaign root")
        value = collect(args.root)
    else:
        value = json.loads(args.input.read_text())
    scheduler_path = getattr(args, "scheduler", None)
    outputs = report(
        value, scheduler_path.read_bytes().decode("utf-8") if scheduler_path else None
    )
    outputs["audit.json"] = json.dumps(value, indent=2, sort_keys=True) + "\n"
    write_outputs(args.output, outputs)
    print(outputs["summary.json"], end="")


if __name__ == "__main__":
    main()
