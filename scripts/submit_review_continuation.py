#!/usr/bin/env python3
"""Submit one continuation visit and a bounded next-visit dispatcher on ACES."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import review_deadline_io as io


def submit_job(
    directory: Path, key: str, options: list[str], worker: Path, stage: str, index: int
) -> str:
    """Use a real script through the site wrapper; preserve uncertain replies."""
    argv = ["sbatch", "--parsable", *options, str(worker), stage, str(index)]
    public._write(directory / f"{key}.intent.json", {"argv": argv})
    try:
        reply = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        public._write(directory / f"{key}.reply.json", {"status": "uncertain_timeout"})
        raise ValueError(
            "uncertain scheduler timeout; inspect intent and jobs before recovery"
        ) from None
    receipt = {
        "returncode": reply.returncode,
        "stdout": reply.stdout,
        "stderr": reply.stderr,
    }
    public._write(directory / f"{key}.reply.json", receipt)
    job = reply.stdout.strip().split(";")[0]
    if reply.returncode or not job.isdecimal():
        raise ValueError(f"scheduler reply for {key} unconfirmed; inspect {directory}")
    (directory / f"{key}.id").write_text(job + "\n")
    return job


def submit(root: Path, index: int) -> dict:
    """Never resubmit a partially recorded visit or depend on retired source jobs."""
    root = root.resolve()
    io.require_open(root)
    plan = io.verify(root)
    if plan["protocol"] not in io.CONTINUATION_PROTOCOLS:
        raise ValueError("requires an imported continuation campaign")
    if not 1 <= index < plan["config"]["rounds"]:
        raise ValueError("visit outside frozen continuation budget")
    repo = io.REPO
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("use a clean pinned checkout")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if os.environ.get("AF_COMMIT", commit) != commit:
        raise ValueError("pinned commit changed")
    for name in (
        "AF_PYTHON",
        "AF_VLLM_IMAGE",
        "AF_HF_HOME",
        "AF_COMPUTE_CACHE_ROOT",
        "AF_IPC_TMP_ROOT",
    ):
        if not os.environ.get(name):
            raise ValueError(f"set {name} before submission")
    if (
        not Path(os.environ["AF_PYTHON"]).is_file()
        or not Path(os.environ["AF_VLLM_IMAGE"]).is_file()
    ):
        raise ValueError("Python or serving image is missing")
    os.environ.update(
        AF_REPO_ROOT=str(repo), AF_OUTPUT_ROOT=str(root), AF_COMMIT=commit
    )
    with public._lock(root / "scheduler"):
        manifest = root / f"submission-round-{index}.json"
        if manifest.exists():
            value = public._read(manifest)
            if (
                value["plan_sha256"] != plan["artifact_sha256"]
                or value["commit"] != commit
            ):
                raise ValueError("submission manifest identity differs")
            return value
        if index > 1 and not (root / f"submission-round-{index - 1}.json").exists():
            raise ValueError("previous visit submission is incomplete")
        prior = {}
        for task in plan["tasks"]:
            result = io.read_round(root, task, index - 1)
            if result is None or result["task"] != task or result["round"] != index - 1:
                raise ValueError(
                    f"previous visit incomplete for {task['task_id']}; "
                    "no jobs submitted"
                )
            prior[task["task_id"]] = result["artifact_sha256"]
        directory = root / "submission-intent" / f"round-{index}"
        directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir()
        except FileExistsError:
            raise ValueError(
                f"partial/uncertain submission intent exists: {directory}; inspect jobs"
            ) from None
        public._write(
            directory / "prerequisites.json",
            {"plan": plan["artifact_sha256"], "results": prior},
        )
        (root / "logs").mkdir(exist_ok=True)
        worker = repo / "scripts/hpc/run_review_continuation_aces.sh"
        base = [
            "--kill-on-invalid-dep=yes",
            "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
            "--nodes=1",
            "--ntasks=1",
            "--export=ALL",
        ]
        jobs = {}

        def queue(stage, at, options):
            key = f"{stage}-{at}"
            job = submit_job(
                directory,
                key,
                [
                    *base,
                    f"--job-name=review-v{plan['protocol'].rsplit('-', 1)[-1]}-{key}",
                    f"--output={root}/logs/{key}-%A_%a.out",
                    f"--error={root}/logs/{key}-%A_%a.err",
                    *options,
                ],
                worker,
                stage,
                at,
            )
            jobs[key] = job
            return job

        gpu = [
            "--partition=gpu",
            "--gres=gpu:h100:1",
            "--cpus-per-task=8",
            "--mem=64G",
            "--time=06:30:00",
            "--signal=B:TERM@300",
        ]
        if index == 1:
            prep = queue(
                "prepare",
                0,
                [
                    "--partition=cpu",
                    "--cpus-per-task=1",
                    "--mem=16G",
                    "--time=01:00:00",
                ],
            )
            gpu.append(f"--dependency=afterok:{prep}")
        proposer = queue("propose", index, gpu)
        fit = queue(
            "fit",
            index,
            [
                f"--dependency=afterany:{proposer}",
                "--partition=cpu",
                f"--array=0-{len(plan['tasks']) - 1}%16",
                "--cpus-per-task=1",
                "--mem=16G",
                "--time=00:40:00",
                "--signal=B:TERM@120",
            ],
        )
        finish = queue(
            "finish",
            index,
            [
                f"--dependency=afterany:{fit}",
                "--partition=cpu",
                "--cpus-per-task=1",
                "--mem=4G",
                "--time=00:15:00",
            ],
        )
        if index + 1 < plan["config"]["rounds"]:
            queue(
                "dispatch",
                index + 1,
                [
                    f"--dependency=afterany:{finish}",
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
            "phase_round": index,
            "global_round": plan["continuation"]["source_round"] + index,
            "additional_visits": plan["continuation"]["additional_visits"],
            "jobs": {
                p.stem: p.read_text().strip()
                for p in sorted((root / "submission-intent").glob("round-*/*.id"))
            },
            "all_rounds_submitted": index + 1 == plan["config"]["rounds"],
            "automatic_test_access": False,
        }
        public._write(manifest, value)
        public._write(root / "submission_manifest.json", value)
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(submit(args.root, args.round), indent=2))
