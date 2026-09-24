"""Concurrent, resumable per-lineage workers sharing one campaign filesystem."""

from __future__ import annotations

import fcntl
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal import component_campaign as campaign
from autoformalism.rebuttal import component_critic as critic
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.search import scientific_verification

STAGES = ("propose", "critic", "fit", "prune")
DELIVERY_FAILURES = {"provider_request_failed", "request_preflight_failed"}


@contextmanager
def claim(path: Path):
    """An abandoned OS lock is released automatically, without budget reset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def ready(root: Path, plan: dict, task: dict, index: int, stage: str) -> bool:
    """Advisory reviews precede fitting; scientific decisions never depend on them."""
    directory = io.round_path(root, task, index)
    if index >= plan["config"]["rounds"]:
        if stage != "critic" or not task["critic"]:
            return False
        last = io.read_round(root, task, index - 1)
        return bool(
            (last or {}).get("selected")
            and (root / "pruning" / task["task_id"] / "result.json").exists()
            and not (root / "pruning" / task["task_id"] / "critic.json").exists()
        )
    if stage == "prune":
        return (
            index == plan["config"]["rounds"] - 1
            and (directory / "result.json").exists()
            and not (root / "pruning" / task["task_id"] / "result.json").exists()
        )
    if index:
        parent = io.read_round(root, task, index - 1)
        if not parent or parent.get("proposal_status") in DELIVERY_FAILURES:
            return False
    else:
        parent = None
    if stage == "propose":
        if (directory / "proposal.json").exists():
            return False
        if task["critic"] and not (root / "critic_authorization.json").exists():
            return False
        if task["critic"] and (parent or {}).get("selected"):
            origin = parent["selected"]["origin_round"]
            return (io.round_path(root, task, origin) / "critic.json").exists()
        return True
    if not (directory / "proposal.json").exists():
        return False
    if stage == "critic":
        return (
            task["critic"]
            and not (directory / "critic.json").exists()
            and (root / "critic_authorization.json").exists()
        )
    if stage == "fit":
        return not (directory / "result.json").exists() and (
            not task["critic"] or (directory / "critic.json").exists()
        )
    raise ValueError("unknown worker stage")


def pending(root: Path, plan: dict, stage: str) -> bool:
    """Do not exit while another service can still supply a prerequisite."""
    for task in plan["tasks"]:
        if stage == "prune":
            if not (root / "pruning" / task["task_id"] / "result.json").exists():
                return True
        elif stage != "critic" or task["critic"]:
            if stage == "critic":
                last = io.read_round(root, task, plan["config"]["rounds"] - 1)
                if last is None or (
                    last.get("selected")
                    and not (
                        root / "pruning" / task["task_id"] / "critic.json"
                    ).exists()
                ):
                    return True
            filename = {
                "propose": "proposal.json",
                "critic": "critic.json",
                "fit": "result.json",
            }[stage]
            if any(
                not (io.round_path(root, task, i) / filename).exists()
                for i in range(plan["config"]["rounds"])
            ):
                return True
    return False


def perform(root, plan, task, index, stage, base_url, can_start, key_supplier):
    """Reuse existing proposal, fit supervisor and pruning implementations."""
    if stage == "propose":
        client = pipeline._client(root, plan, task, index, base_url, can_start)
        with scientific_verification.scope(task["scientific_verifier"]):
            return pipeline.propose_one(root, plan, task, index, client)
    if stage == "critic":
        if index == plan["config"]["rounds"]:
            return critic.review_final(root, plan, task, key_supplier)
        return critic.review_one(root, plan, task, index, key_supplier)
    if stage == "fit":
        # The established supervisor seals interruptions and never fits twice.
        command = [
            sys.executable,
            str(io.REPO / "scripts/review_deadline.py"),
            "fit-task",
            "--root",
            str(root),
            "--round",
            str(index),
            "--task-index",
            str(task["index"]),
        ]
        subprocess.run(command, check=True)
        return io.read_round(root, task, index)
    return campaign.prune_one(root, plan, task)


def run(
    root: Path,
    stage: str,
    *,
    seconds=21600,
    base_url=None,
    once=False,
    key_supplier=None,
    performer=None,
) -> dict:
    """Poll ready work within an allocation; restarting consumes remaining budgets."""
    if stage not in STAGES or seconds <= 0:
        raise ValueError("valid stage and positive worker duration required")
    if stage == "propose" and not base_url:
        raise ValueError("proposer requires its pinned local server URL")
    plan = campaign.verify(root)
    io.require_open(root)
    if stage == "critic":
        critic.checked_authorization(root, plan)
        if key_supplier is None:
            raise ValueError("critic key supplier required")
    worker = uuid.uuid4().hex
    record = {
        "identity": plan["artifact_sha256"],
        "worker": worker,
        "stage": stage,
        "host": platform.node(),
        "platform": platform.platform(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "seconds": seconds,
        "started_unix": time.time(),
    }
    receipt = root / "workers" / worker / "started.json"
    sealed_write(receipt, record)
    deadline = time.monotonic() + seconds
    count = 0
    # Leave room for one bounded operation and its checkpoint before walltime.
    margin = {
        "propose": 300,
        "critic": 3600,
        "fit": plan["config"]["fit_worker_seconds"] + 30,
        "prune": 3900,
    }[stage]

    def can_start():
        return time.monotonic() + (0 if once else margin) < deadline

    while can_start():
        found = False
        # Round-first order prevents fast lineages consuming every GPU slot.
        for index in range(plan["config"]["rounds"] + 1):
            for task in plan["tasks"]:
                if not can_start():
                    break
                if not ready(root, plan, task, index, stage):
                    continue
                lock = root / "claims" / task["task_id"] / f"{index}-{stage}.lock"
                with claim(lock) as held:
                    if not held or not ready(root, plan, task, index, stage):
                        continue
                    with io.execution_lease(root):
                        try:
                            value = (performer or perform)(
                                root,
                                plan,
                                task,
                                index,
                                stage,
                                base_url,
                                can_start,
                                key_supplier,
                            )
                        except DeferredCall:
                            return {"status": "deferred", "completed_operations": count}
                    count += 1
                    found = True
                    print(
                        json.dumps(
                            {
                                "stage": stage,
                                "task": task["task_id"],
                                "round": index,
                                "status": (value or {}).get("status"),
                            }
                        ),
                        flush=True,
                    )
                    if once:
                        return {
                            "status": "one_operation",
                            "completed_operations": count,
                        }
        if once or not pending(root, plan, stage):
            break
        if not found:
            time.sleep(min(5, max(0, deadline - time.monotonic())))
    value = {**record, "finished_unix": time.time(), "completed_operations": count}
    sealed_write(receipt.with_name("result.json"), value)
    return {"status": "worker_finished", "completed_operations": count}
