"""Queue the November stress campaign's disjoint final 33 blocks behind wave eight."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import runpy
import subprocess
from pathlib import Path
from statistics import median

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import component_critic as critic
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts.iclr_stress_solver_archive import PLAN, STORE, compact, digest
from scripts.submit_review_continuation import submit_job

FIRST, STOP = 24, 57
PREVIOUS, WAVE = "2356070", "wave9"
HELPER = "0613b9c6c251aae52abf3a6fb54e48235259216b9090623c6837f05de1b20629"


def roster(root: Path) -> dict:
    plan = sealed_read(root / "plan.json")
    if (
        plan["artifact_sha256"] != PLAN
        or len(plan["blocks"]) != STOP
        or plan["seeds"] != [0, 1, 2]
        or plan["arms"] != ["full", "fixed_gamma"]
        or plan["visits"] != 15
        or not plan["pruning_enabled"]
        or not plan["critic_enabled"]
    ):
        raise ValueError("wrong stress execution plan")
    if [(b["block"], b["cell"], b["seed"]) for b in plan["blocks"][21:24]] != [
        (21 + i, "sensor_03", i) for i in range(3)
    ]:
        raise ValueError("wave eight must own blocks 21-23")
    return plan


def array_blocks(spec: str) -> set[int]:
    """Read numeric Slurm ranges with one optional concurrency suffix."""
    ids = set()
    for part in spec.split("%")[0].split(","):
        start, sep, end = part.partition("-")
        ids.update(range(int(start), int(end) + 1 if sep else int(start) + 1))
    return ids


def previous_waves(root: Path) -> None:
    """Ensure all earlier submissions cover exactly blocks 0..23 with no overlap."""
    seen = set()
    for i in range(1, 9):
        folder = root / "submissions" / f"wave{i}"
        jid = (folder / "array.id").read_text().strip()
        intent = json.loads((folder / "array.intent.json").read_bytes())["argv"]
        (spec,) = [v.split("=", 1)[1] for v in intent if v.startswith("--array=")]
        blocks = array_blocks(spec)
        if (
            not jid.isdecimal()
            or blocks != set(range(3 * (i - 1), 3 * i))
            or seen & blocks
            or (i == 8 and jid != PREVIOUS)
        ):
            raise ValueError("prior Slurm allocation differs from the expected roster")
        seen |= blocks
    if seen != set(range(FIRST)):
        raise ValueError("previous waves do not cover 0..23")
    for f in (root / "submissions").glob("*/array.intent.json"):
        if f.parent.name not in {f"wave{i}" for i in range(1, 9)}:
            raise ValueError(
                f"reconcile the existing or uncertain submission {f.parent.name}"
            )


def wave8_dependency() -> list[str]:
    """Wait for the live parent array; a retired parent cannot be a Slurm dependency."""
    active = subprocess.check_output(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%F"], text=True
    ).split()
    if PREVIOUS in active:
        return [f"--dependency=afterany:{PREVIOUS}"]
    states = subprocess.check_output(
        [
            "sacct",
            "--array",
            "-X",
            "-n",
            "-P",
            "-j",
            PREVIOUS,
            "-o",
            "JobID%45,State%30",
        ],
        text=True,
    )
    terminal_states = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
        "SPECIAL_EXIT",
    }
    finished = set()
    for line in states.splitlines():
        job, _, state = line.strip().partition("|")
        if state and state.split()[0].rstrip("+") in terminal_states:
            finished.add(job)
    if not {f"{PREVIOUS}_{i}" for i in range(21, 24)} <= finished:
        raise ValueError(
            "wave 8 left the queue and accounting is incomplete; retry shortly"
        )
    return []


def group_free() -> int:
    """Read the hard project file quota and usage reported by Lustre."""
    fields = subprocess.check_output(
        ["lfs", "quota", "-q", "-p", "456144", "/scratch"], text=True
    ).split()
    return int(fields[7].rstrip("*")) - int(fields[5].rstrip("*"))


def capacity(root: Path, mirror: Path, *, after_wave8: bool = False) -> dict:
    """Estimate loose entries from complete runs, plus room for two active tasks."""
    store = root / STORE
    records = []
    for i in range(21):
        rec = json.loads((store / f"block{i:04d}.closed.json").read_bytes())
        if (
            rec["binding"]["plan"] != PLAN
            or rec["binding"]["block"] != i
            or Path(rec["backup"]).parent != mirror
            or not Path(rec["backup"]).is_file()
            or not (store / rec["archive"]).is_file()
        ):
            raise ValueError("archived receipts or mirror missing")
        records.append(rec)
    loose = []
    for rec in records:
        count = 0
        for r in rec["binding"]["endpoints"]:
            for area in ("results", "pruning"):
                path = root / area / r["source_id"]
                count += sum(1 + len(files) for _, _, files in os.walk(path))
        # Shared critic-cache entries are accounted for separately below.
        loose.append(count)
    judge = sum(1 + len(f) for _, _, f in os.walk(root / "judge_cache"))
    expected_persistent = math.ceil(1.25 * (sum(loose) + judge) / len(loose)) + 5
    live_pair = 2 * max(n + r["entries"] for n, r in zip(loose, records, strict=True))
    # Wave eight finishes before wave nine starts. The first new worker compacts
    # its three terminal blocks, then measures available space before any search.
    wave8_loose = 0
    for b in range(21, FIRST):
        for arm in ("full", "fixed_gamma"):
            for area in ("results", "pruning"):
                path = root / area / f"block{b:04d}_{arm}"
                wave8_loose += sum(1 + len(f) for _, _, f in os.walk(path))
    wave8_growth = max(
        0, 3 * (expected_persistent + max(r["entries"] for r in records)) - wave8_loose
    )
    future_peak = expected_persistent * (STOP - FIRST) + live_pair
    budget = (
        future_peak
        if after_wave8
        else max(wave8_growth, future_peak + 3 * expected_persistent)
    )
    return {
        "measured_blocks": len(loose),
        "compacted_entries": sum(r["entries"] for r in records),
        "retained_per_pair_median": median(loose),
        "max_observed_live_pair_entries": live_pair // 2,
        "reserved_for_wave8_growth": wave8_growth,
        "estimated_entries_with_headroom": budget,
        "free_group_entries": group_free(),
        "remaining_blocks": STOP - FIRST,
    }


def prepare_workers(root: Path, mirror: Path) -> dict:
    """Finish wave-eight storage once; serialize the first two arriving workers."""
    store = root / STORE
    receipt = store / "wave8-closeout.json"
    with (store / ".wave8-closeout.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.is_file():
            saved = sealed_read(receipt)
            if saved["plan"] != PLAN or saved["mirror"] != str(mirror):
                raise ValueError("wrong wave-eight archive closeout")
            return {"wave8_checked": True}
        # Submission attaches afterany to a live parent or checks its terminal
        # accounting. The compute node can therefore work even if sacct is delayed.
        archived = [compact(root, mirror, i) for i in range(21, FIRST)]
        stats = capacity(root, mirror, after_wave8=True)
        if stats["free_group_entries"] < stats["estimated_entries_with_headroom"]:
            raise ValueError(f"insufficient free entries after wave 8: {stats}")
        record = {
            "plan": PLAN,
            "mirror": str(mirror),
            "blocks": archived,
            "capacity": stats,
        }
        return sealed_write(receipt, record)


def run(root: Path, ops: Path, mirror: Path, account: str, action: str):
    plan = roster(root)
    critic.checked_authorization(root, plan)
    if action == "worker-start":
        io.require_open(root)
        return prepare_workers(root, mirror)
    sub = root / "submissions"
    folder = sub / WAVE
    with public._lock(sub):
        if (folder / "array.id").is_file():
            job = (folder / "array.id").read_text().strip()
            intent = json.loads((folder / "array.intent.json").read_bytes())["argv"]
            if (
                not job.isdecimal()
                or "--array=24-56%2" not in intent
                or [v for v in intent if v.startswith("--dependency=")]
                not in ([], [f"--dependency=afterany:{PREVIOUS}"])
                or str(ops / "worker.sh") not in intent
            ):
                raise ValueError("wave9 already names a different submission")
            return {"already_submitted": job}
        io.require_open(root)
        previous_waves(root)
        for b in plan["blocks"][FIRST:]:
            if any((root / "results" / t["task_id"]).exists() for t in b["tasks"]):
                raise ValueError(
                    "future block already started outside the combined submission"
                )
        stats = capacity(root, mirror)
        if stats["free_group_entries"] < stats["estimated_entries_with_headroom"]:
            raise ValueError(
                f"insufficient measured headroom for combined launch: {stats}"
            )
        dependency = wave8_dependency()
        suffix = [
            f"--account={account}",
            "--partition=gpu",
            "--array=24-56%2",
            *dependency,
            "--export=ALL",
            f"--output={root}/logs/{WAVE}-%A_%a.out",
            f"--error={root}/logs/{WAVE}-%A_%a.err",
        ]
        if action == "check":
            return {
                **stats,
                "array": "24-56%2",
                "dependency": dependency,
                "tasks": 2 * (STOP - FIRST),
                "wave": WAVE,
            }
        if action == "dry-run":
            tested = subprocess.run(
                ["sbatch", "--test-only", *suffix, str(ops / "worker.sh"), "run", "0"],
                check=True,
                capture_output=True,
                text=True,
            )
            reply = (tested.stdout + tested.stderr).strip()
            if "to start" not in reply.lower():
                raise ValueError(f"Slurm did not confirm the scheduling test: {reply}")
            return {**stats, "scheduler_test": reply, "dependency": dependency}
        helper = root.parent / "stress-ops-2337694/iclr_aces_pilot.py"
        if digest(helper) != HELPER:
            raise ValueError("critic submission helper changed")
        auth = runpy.run_path(str(helper))
        key = auth["checked_critic_key"](
            os.environ.get("AF_JETSTREAM_API_KEY") or auth["read_critic_key"]()
        )
        auth["verify_critic_access"](key)
        os.environ.update(
            AF_JETSTREAM_API_KEY=key,
            AF_SEARCH_ROOT=str(root),
            AF_REMAINDER_OPS=str(ops),
            AF_SOLVER_BACKUP=str(mirror),
        )
        folder.mkdir(exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        job = submit_job(folder, "array", suffix, ops / "worker.sh", "run", 0)
        return {**stats, "job": job, "array": "24-56%2", "dependency": dependency}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("check", "dry-run", "submit", "worker-start"))
    for n in ("root", "ops", "mirror"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--account", required=True)
    a = p.parse_args()
    print(
        json.dumps(
            run(
                a.root.resolve(),
                a.ops.resolve(),
                a.mirror.resolve(),
                a.account,
                a.action,
            ),
            indent=2,
        )
    )
