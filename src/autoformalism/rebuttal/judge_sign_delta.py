"""Portable symbolic sign rechecks with a separate, pinned Delta execution identity."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from autoformalism.rebuttal import judge_sign_recheck as original
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write

PROTOCOL = "saved-judge-sign-recheck-delta-1"
REPO = original.REPO
ZERO_FIELDS = (
    "optimizer_calls",
    "proposer_calls",
    "solver_rollouts",
    "model_changes",
    "selection_changes",
)
FLAG_FIELDS = ("test_data_opened", "automatic_followup")
SNAPSHOT_FIELDS = (
    "source_plan_sha256",
    "judge_revision",
    "serving_image_sha256",
    "reviews",
    "placements",
    "sign_policy",
    "judge_protocol",
)


def file_hash(path: Path) -> str:
    """Hash large container images without loading them into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def imported_plan(path: Path) -> dict:
    """Recheck a self-sealed ACES plan without accessing its remote source paths."""
    plan = sealed_read(path)
    allowed = {
        "protocol",
        "source",
        "source_sha256",
        "runtime",
        "launcher_sha256",
        "artifact_sha256",
        *ZERO_FIELDS,
        *FLAG_FIELDS,
        *SNAPSHOT_FIELDS,
    }
    if (
        set(plan) != allowed
        or plan["protocol"] != original.PROTOCOL
        or any(plan[k] != 0 for k in ZERO_FIELDS)
        or any(plan[k] is not False for k in FLAG_FIELDS)
        or plan["sign_policy"] != original.FACTOR_SIGN_POLICY
        or plan["judge_protocol"]
        != original.judge.judge_protocol(sign_policy=original.FACTOR_SIGN_POLICY)
    ):
        raise ValueError("requires a closed public sign-recheck plan")
    for key, length in (("judge_revision", 40), ("serving_image_sha256", 64)):
        if not re.fullmatch(rf"[a-f0-9]{{{length}}}", plan[key]):
            raise ValueError(f"invalid frozen {key}")
    keys = set()
    for row in plan["reviews"]:
        request = row["request"]
        original.corrected_request(request)
        key = original.content_hash(request)
        changes = original.evidence_changes(request)
        if (
            key in keys
            or row["historical_request_sha256"] != key
            or row["historical_review"]["request_sha256"] != key
            or request["model_revision"] != plan["judge_revision"]
            or row["affected"] is not bool(changes)
            or row["changed_occurrences"] != changes
        ):
            raise ValueError("portable request/evidence provenance differs")
        keys.add(key)
    placements = plan["placements"]
    if (
        not keys
        or {p["request_sha256"] for p in placements} != keys
        or len({(p["task"], p["stage"]) for p in placements}) != len(placements)
        or any(p["stage"] not in {"parent", "child"} for p in placements)
    ):
        raise ValueError("portable placements differ")
    return plan


def launcher_hash() -> str:
    """Pin portable entry points and reused scheduler helpers."""
    paths = (
        "scripts/judge_sign_delta.py",
        "scripts/submit_judge_sign_delta.py",
        "scripts/hpc/run_judge_sign_recheck_delta.sh",
        "scripts/smoke_judge_sign_delta.py",
        "scripts/prepare_vllm_image.py",
        "scripts/submit_shared_process_pilot.py",
        "scripts/submit_review_continuation.py",
    )
    return original.content_hash({p: file_hash(REPO / p) for p in paths})


def freeze(source: Path, root: Path, image_sha256: str) -> dict:
    """Bind exact historical requests to a new execution, never resume ACES in place."""
    if source.resolve().is_relative_to(root.resolve()):
        raise ValueError("source plan and output must be separate")
    if not re.fullmatch(r"[a-f0-9]{64}", image_sha256):
        raise ValueError("invalid Delta image digest")
    imported = imported_plan(source)
    with original.public._lock(root):
        sealed_write(
            root / "origin_plan.json",
            {k: v for k, v in imported.items() if k != "artifact_sha256"},
        )
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                **{k: imported[k] for k in SNAPSHOT_FIELDS},
                "origin_plan_sha256": imported["artifact_sha256"],
                "execution": {
                    "platform": "delta-gpuA40x4",
                    "tensor_parallel_size": 4,
                    "model": "openai/gpt-oss-120b",
                    "vllm_version": "0.27.1",
                    "serving_image_sha256": image_sha256,
                    "origin_serving_image_sha256": imported["serving_image_sha256"],
                    "same_image_bytes": image_sha256
                    == imported["serving_image_sha256"],
                    "origin_runtime": imported["runtime"],
                    "runtime": original.public._runtime(),
                },
                "source_sha256": original.public._source_identity(),
                "launcher_sha256": launcher_hash(),
                **dict.fromkeys(ZERO_FIELDS, 0),
                **dict.fromkeys(FLAG_FIELDS, False),
            },
        )


def verify(root: Path) -> dict:
    """Verify the local pin and the copied origin, without cross-cluster file access."""
    original.history.require_open(root)
    plan = sealed_read(root / "plan.json")
    origin = imported_plan(root / "origin_plan.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["origin_plan_sha256"] != origin["artifact_sha256"]
        or any(plan[k] != origin[k] for k in SNAPSHOT_FIELDS)
        or plan["execution"]["runtime"] != original.public._runtime()
        or plan["execution"]["tensor_parallel_size"] != 4
        or plan["source_sha256"] != original.public._source_identity()
        or plan["launcher_sha256"] != launcher_hash()
    ):
        raise ValueError("frozen Delta sign recheck identity differs")
    return plan


def run_one(root: Path, index: int, base_url: str) -> dict:
    """Reuse the established cached, bounded paired-review implementation."""
    return original.run_verified_review(root, verify(root), index, base_url)


def report(root: Path) -> dict:
    """Reuse unchanged availability/cost accounting, adding execution provenance."""
    return original.report_verified_plan(root, verify(root))
