"""Audit and rejudge saved M4 pairs under versioned, equivalent sign extraction."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.judging.hybrid import FACTOR_SIGN_POLICY, build_atomic_evidence_plan
from autoformalism.llm.exceptions import LLMError
from autoformalism.rebuttal import repair_scientific_judge as judge
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.process_pruning import history
from autoformalism.rebuttal.repair_evidence import model_hash
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = "saved-judge-sign-recheck-1"
REQUEST_KEYS = {
    "parent",
    "candidate",
    "context",
    "public_prompt",
    "seed",
    "model_revision",
    "protocol",
}


def evidence_changes(request: dict) -> list[dict]:
    """Align additive occurrences, including processes, without editing equations."""
    models = [
        CandidateModel.model_validate(request[k]) for k in ("parent", "candidate")
    ]
    old = build_atomic_evidence_plan(*models)
    new = build_atomic_evidence_plan(*models, sign_policy=FACTOR_SIGN_POLICY)
    changes = []
    for before, after in zip(old.occurrences, new.occurrences, strict=True):
        if (
            before.candidate_side,
            before.equation_location,
            before.governed_quantity,
        ) != (after.candidate_side, after.equation_location, after.governed_quantity):
            raise ValueError("sign audit occurrence alignment differs")
        if before != after:
            changes.append(
                {
                    "before": {**asdict(before), "symbols": list(before.symbols)},
                    "after": {**asdict(after), "symbols": list(after.symbols)},
                }
            )
    return changes


def corrected_request(request: dict) -> dict:
    """Change only versioned evidence semantics, preserving all scientific inputs."""
    if set(request) != REQUEST_KEYS or request["protocol"] != judge.judge_protocol():
        raise ValueError("requires the closed, historical scientific judge request")
    ValidationContext.model_validate(request["context"])
    for key in ("parent", "candidate"):
        CandidateModel.model_validate(request[key])
    return {**request, "protocol": judge.judge_protocol(sign_policy=FACTOR_SIGN_POLICY)}


def snapshot(source: Path) -> dict:
    """Verify saved request/receipt/stage linkage; import only symbolic requests."""
    history.require_open(source)
    old = sealed_read(source / "plan.json")
    if old["protocol"] != "general-advisory-critic-1" or old["test_data_opened"]:
        raise ValueError("requires the public M4 critic campaign")
    revision = old["judge_revision"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("judge model must have an immutable revision")
    tasks = [row["task"]["task_id"] for row in old["rows"]]
    if (
        not tasks
        or len(tasks) != len(set(tasks))
        or any(not re.fullmatch(r"[A-Za-z0-9_-]+", task) for task in tasks)
    ):
        raise ValueError("invalid or duplicate source tasks")
    reviews = {}
    placements = []
    for task in tasks:
        for stage in ("parent", "child"):
            saved = sealed_read(source / "results" / task / f"{stage}_review.json")
            key = saved["request_sha256"]
            if not re.fullmatch(r"[a-f0-9]{64}", key):
                raise ValueError("invalid saved request identity")
            cache = source / "judge_cache" / key
            request = sealed_read(cache / "request.json")["request"]
            receipt = sealed_read(cache / "receipt.json")
            corrected_request(request)
            if (
                saved["identity"] != old["artifact_sha256"]
                or key != content_hash(request)
                or receipt["request"] != request
                or receipt["review"] != saved["review"]
                or receipt["artifact_sha256"] != saved["receipt_sha256"]
                or receipt["review"]["request_sha256"] != key
                or request["model_revision"] != revision
                or saved["candidate_sha256"]
                != model_hash(CandidateModel.model_validate(request["candidate"]))
            ):
                raise ValueError("saved review provenance differs")
            if key not in reviews:
                changes = evidence_changes(request)
                reviews[key] = {
                    "historical_request_sha256": key,
                    "historical_receipt_sha256": receipt["artifact_sha256"],
                    "request": request,
                    "historical_review": receipt["review"],
                    "affected": bool(changes),
                    "changed_occurrences": changes,
                }
            placements.append(
                {
                    "task": task,
                    "stage": stage,
                    "request_sha256": key,
                    "source_stage_sha256": saved["artifact_sha256"],
                }
            )
    return {
        "source_plan_sha256": old["artifact_sha256"],
        "judge_revision": revision,
        "serving_image_sha256": old["serving_image_sha256"],
        "reviews": list(reviews.values()),
        "placements": placements,
    }


def launcher_hash() -> str:
    """Pin entry points alongside the source and runtime."""
    names = (
        "scripts/judge_sign_recheck.py",
        "scripts/submit_judge_sign_recheck.py",
        "scripts/hpc/run_judge_sign_recheck_aces.sh",
        "scripts/smoke_judge_sign_recheck.py",
        "scripts/submit_shared_process_pilot.py",
        "scripts/submit_review_continuation.py",
    )
    return content_hash(
        {n: hashlib.sha256((REPO / n).read_bytes()).hexdigest() for n in names}
    )


def freeze(source: Path, root: Path) -> dict:
    """Import once; repeated freezes verify source receipts without new reviews."""
    source, root = source.resolve(), root.resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("source and output must be separate")
    imported = snapshot(source)
    with public._lock(root):
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "source": str(source),
                **imported,
                "sign_policy": FACTOR_SIGN_POLICY,
                "judge_protocol": judge.judge_protocol(sign_policy=FACTOR_SIGN_POLICY),
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "launcher_sha256": launcher_hash(),
                "optimizer_calls": 0,
                "proposer_calls": 0,
                "solver_rollouts": 0,
                "model_changes": 0,
                "selection_changes": 0,
                "test_data_opened": False,
                "automatic_followup": False,
            },
        )


def verify(root: Path) -> dict:
    """Do not resume a frozen pilot with different evidence or execution code."""
    history.require_open(root)
    plan = sealed_read(root / "plan.json")
    history.require_open(Path(plan["source"]))
    if (
        plan["protocol"] != PROTOCOL
        or plan["sign_policy"] != FACTOR_SIGN_POLICY
        or plan["judge_protocol"]
        != judge.judge_protocol(sign_policy=FACTOR_SIGN_POLICY)
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or plan["launcher_sha256"] != launcher_hash()
    ):
        raise ValueError("frozen sign recheck identity differs")
    return plan


def run_one(root: Path, index: int, base_url: str) -> dict:
    """Rejudge both stages/orientations once, only when symbolic evidence changed."""
    plan = verify(root)
    row = plan["reviews"][index]
    work = root / "reviews" / row["historical_request_sha256"]
    with public._lock(work):
        path = work / "result.json"
        if path.exists():
            result = sealed_read(path)
            if result["identity"] != plan["artifact_sha256"]:
                raise ValueError("review belongs to another recheck")
            return result
        common = {
            "identity": plan["artifact_sha256"],
            "historical_request_sha256": row["historical_request_sha256"],
        }
        if not row["affected"]:
            return sealed_write(
                path,
                {
                    **common,
                    "status": "unchanged_evidence",
                    "new_review": None,
                    "new_request_sha256": None,
                },
            )
        request = corrected_request(row["request"])
        key = content_hash(request)
        cache = work / "judge"
        sealed_write(cache / "request.json", {"request": request})
        try:
            review = judge.perform_review(request, cache, base_url)
        except (OSError, TimeoutError, LLMError) as error:
            review = {
                "request_sha256": key,
                "status": "unavailable",
                "findings": [],
                "error": f"{type(error).__name__}: {error}",
                "cost": judge.review_cost(cache),
            }
        if review["request_sha256"] != key:
            raise ValueError("recheck result names another model pair")
        return sealed_write(
            path,
            {
                **common,
                "status": review["status"],
                "new_review": review,
                "new_request_sha256": key,
            },
        )


def report(root: Path) -> dict:
    """Separate affected-review availability, findings, and new costs from history."""
    plan = verify(root)
    by_request = {}
    costs = []
    for row in plan["reviews"]:
        key = row["historical_request_sha256"]
        path = root / "reviews" / key / "result.json"
        result = sealed_read(path) if path.exists() else None
        if result and result["identity"] != plan["artifact_sha256"]:
            raise ValueError("result belongs to another campaign")
        new = result["new_review"] if result else None
        if row["affected"]:
            # Include partial calls even if the process died before publication.
            costs.append(judge.review_cost(root / "reviews" / key / "judge"))
        by_request[key] = {
            "request_sha256": key,
            "affected": row["affected"],
            "status": result["status"]
            if result
            else ("pending" if row["affected"] else "unchanged_evidence"),
            "changed_occurrences": row["changed_occurrences"],
            "historical_status": row["historical_review"]["status"],
            "historical_findings": row["historical_review"].get("findings", []),
            "new_findings": new.get("findings", []) if new else None,
            "new_availability": judge.review_status(new),
        }
    value = {
        "identity": plan["artifact_sha256"],
        "protocol": PROTOCOL,
        "unique_reviews": len(by_request),
        "affected_unique_reviews": sum(r["affected"] for r in by_request.values()),
        "status_counts": dict(Counter(r["status"] for r in by_request.values())),
        "cost": {
            "physical_requests": sum(c["physical_requests"] for c in costs),
            "observed_total_tokens": sum(c["observed_total_tokens"] for c in costs),
            "usage_missing_events": sum(c["usage_missing_events"] for c in costs),
            "usage_complete": all(c["usage_complete"] for c in costs),
            "scope": "New recorded calls only; interrupted usage may be unobserved.",
        },
        "reviews": list(by_request.values()),
        "rows": [{**p, **by_request[p["request_sha256"]]} for p in plan["placements"]],
        "optimizer_calls": 0,
        "proposer_calls": 0,
        "solver_rollouts": 0,
        "model_changes": 0,
        "selection_changes": 0,
        "test_data_opened": False,
        "automatic_followup": False,
    }
    public._write(root / "summary.json", value)
    lines = [
        "# Saved scientific-judge sign recheck",
        "",
        "Same symbolic pairs, public prompts, model revision, seeds and scoring.",
        "Only affected reviews rerun: both stages and both orientations.",
        "Historical models, fits, reviews and selections remain unchanged.",
        "Outer polarity is syntax, not a value's sign or scientific certification.",
        "This correction does not establish fresh judge calibration or critic benefit.",
        "",
        f"Unique reviews: {len(by_request)}; "
        f"affected: {value['affected_unique_reviews']}.",
        f"Status counts: {value['status_counts']}.",
        "",
        "| Task | Stage | Affected | Status | Old findings | New findings |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in value["rows"]:
        count = len(row["new_findings"]) if row["new_findings"] is not None else "—"
        lines.append(
            f"| {row['task']} | {row['stage']} | {row['affected']} | {row['status']} "
            f"| {len(row['historical_findings'])} | {count} |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return value
