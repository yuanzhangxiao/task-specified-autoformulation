#!/usr/bin/env python3
"""Submit one optional-review basin campaign; no automatic subsequent rounds."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import detention_process_pilot as io

_spec = importlib.util.spec_from_file_location(
    "shared_pilot_support", Path(__file__).with_name("submit_review_continuation.py")
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
submit_job = _module.submit_job


def source_commit(repo: Path) -> str:
    """Support a pinned git archive to avoid another repository on quota-limited HPC."""
    marker = repo / "SOURCE_COMMIT"
    if marker.exists():
        commit = marker.read_text().strip()
    else:
        if subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        ).strip():
            raise ValueError("use a clean pinned checkout or git archive")
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("expected full commit identity")
    return commit


def submit(root: Path, index: int = 0) -> dict:
    """Preserve intent, uncertain scheduler replies, and consumed fit attempts."""
    root = root.resolve()
    plan = io.verify(root)
    if (
        plan["protocol"] not in {io.PROTOCOL, io.BOUND_PROTOCOL, io.GAIN_PROTOCOL}
        or index != 0
    ):
        raise ValueError("requires optional basin review pilot")
    repo = io.REPO
    commit = source_commit(repo)
    if os.environ.get("AF_COMMIT", commit) != commit:
        raise ValueError("pinned commit changed")
    for key in (
        "AF_PYTHON",
        "AF_VLLM_IMAGE",
        "AF_HF_HOME",
        "AF_COMPUTE_CACHE_ROOT",
        "AF_IPC_TMP_ROOT",
    ):
        if not os.environ.get(key):
            raise ValueError(f"set {key}")
    for key in ("AF_PYTHON", "AF_VLLM_IMAGE"):
        if not Path(os.environ[key]).is_file():
            raise ValueError(f"missing {key}")
    os.environ.update(
        AF_REPO_ROOT=str(repo),
        AF_OUTPUT_ROOT=str(root),
        AF_COMMIT=commit,
        PYTHONDONTWRITEBYTECODE="1",
    )
    with public._lock(root / "scheduler"):
        manifest = root / f"submission-round-{index}.json"
        if manifest.exists():
            value = public._read(manifest)
            if (
                value["commit"] != commit
                or value["plan_sha256"] != plan["artifact_sha256"]
            ):
                raise ValueError("submission identity differs")
            return value
        previous = {}
        directory = root / "submission-intent" / f"round-{index}"
        directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir()
        except FileExistsError:
            raise ValueError(
                "partial/uncertain submission exists; inspect jobs and receipts"
            ) from None
        public._write(directory / "prerequisites.json", {"results": previous})
        (root / "logs").mkdir(exist_ok=True)
        worker = repo / "scripts/hpc/run_detention_process_pilot_aces.sh"
        jobs = {}

        def queue(stage, options):
            key = f"{stage}-{index}"
            jobs[key] = submit_job(
                directory,
                key,
                [
                    "--kill-on-invalid-dep=yes",
                    "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
                    "--nodes=1",
                    "--ntasks=1",
                    "--export=ALL",
                    f"--job-name=basin-review-{key}",
                    f"--output={root}/logs/{key}-%A_%a.out",
                    f"--error={root}/logs/{key}-%A_%a.err",
                    *options,
                ],
                worker,
                stage,
                index,
            )
            return jobs[key]

        gate = queue(
            "prepare",
            ["--partition=cpu", "--cpus-per-task=1", "--mem=16G", "--time=00:30:00"],
        )
        proposer = queue(
            "propose",
            [
                f"--dependency=afterok:{gate}",
                "--partition=gpu",
                "--gres=gpu:h100:1",
                "--cpus-per-task=8",
                "--mem=64G",
                "--time=06:30:00",
                "--signal=B:TERM@300",
            ],
        )
        fit = queue(
            "fit",
            [
                f"--dependency=afterok:{proposer}",
                "--partition=cpu",
                f"--array=0-{len(plan['tasks']) - 1}%4",
                "--cpus-per-task=1",
                "--mem=16G",
                "--time=00:40:00",
                "--signal=B:TERM@120",
            ],
        )
        queue(
            "report",
            [
                f"--dependency=afterany:{fit}",
                "--partition=cpu",
                "--cpus-per-task=1",
                "--mem=4G",
                "--time=00:15:00",
            ],
        )
        value = {
            "protocol": plan["protocol"],
            "commit": commit,
            "plan_sha256": plan["artifact_sha256"],
            "round": index,
            "jobs": jobs,
            "tasks": len(plan["tasks"]),
            "automatic_next_round": False,
            "automatic_test_access": False,
        }
        if "handoff_confirmation" in plan:
            value["experiment"] = plan["handoff_confirmation"]["protocol"]
            value["fresh_constructions"] = plan["handoff_confirmation"][
                "fresh_constructions"
            ]
        public._write(manifest, value)
        public._write(root / "submission_manifest.json", value)
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(0,), default=0)
    args = parser.parse_args()
    print(json.dumps(submit(args.root, args.round), indent=2))
