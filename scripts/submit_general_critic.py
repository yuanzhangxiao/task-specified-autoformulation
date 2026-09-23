#!/usr/bin/env python3
"""Submit one bounded critic/revision/fit episode with durable scheduler receipts."""

import argparse
import json
import os
import re
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import general_critic_io as io
from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.submit_shared_process_pilot import source_commit, submit_job


def submit(source: Path, root: Path) -> dict:
    """Pin six phases up front; repeating submission never duplicates existing jobs."""
    source, root = source.resolve(), root.resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("output must be separate")
    commit = source_commit(io.REPO)
    if os.environ.get("AF_COMMIT", commit) != commit:
        raise ValueError("pinned commit differs")
    for name in ("AF_PYTHON", "AF_VLLM_IMAGE"):
        if not Path(os.environ[name]).is_file():
            raise ValueError(f"missing {name}")
    revision = os.environ["AF_JUDGE_REVISION"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("set immutable AF_JUDGE_REVISION")
    if not os.environ.get("AF_HF_HOME"):
        raise ValueError("set AF_HF_HOME")
    io.pruning.history.require_open(source)
    old = sealed_read(source / "plan.json")
    if old["protocol"] != io.pruning.PROTOCOL:
        raise ValueError("requires Milestone 3 pruning source")
    # Fail before allocating GPU jobs if the prerequisite is not complete.
    for row in old["rows"]:
        result = sealed_read(
            source / "results" / row["task"]["task_id"] / "result.json"
        )
        if (
            result["identity"] != old["artifact_sha256"]
            or result["status"] != "complete"
        ):
            raise ValueError("pruning prerequisites are incomplete")
    identity = {
        "commit": commit,
        "source": str(source),
        "root": str(root),
        "source_plan_sha256": old["artifact_sha256"],
        "judge_revision": revision,
        "python": os.environ["AF_PYTHON"],
        "image": os.environ["AF_VLLM_IMAGE"],
        "hf_home": os.environ["AF_HF_HOME"],
    }
    os.environ.update(
        AF_REPO_ROOT=str(io.REPO),
        AF_SOURCE_ROOT=str(source),
        AF_OUTPUT_ROOT=str(root),
        AF_COMMIT=commit,
        PYTHONDONTWRITEBYTECODE="1",
    )
    with public._lock(root / "scheduler"):
        manifest = root / "submission_manifest.json"
        if manifest.exists():
            saved = public._read(manifest)
            if saved["identity"] != identity:
                raise ValueError("submission identity differs")
            return saved
        intent = root / "submission-intent"
        try:
            intent.mkdir()
        except FileExistsError:
            raise ValueError(
                "partial/uncertain submission: inspect receipts and jobs; "
                "do not delete intent"
            ) from None
        public._write(intent / "identity.json", identity)
        (root / "logs").mkdir(exist_ok=True)
        jobs = {}

        def queue(stage, extra):
            jobs[stage] = submit_job(
                intent,
                stage,
                [
                    "--kill-on-invalid-dep=yes",
                    "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
                    "--nodes=1",
                    "--ntasks=1",
                    "--export=ALL",
                    f"--job-name=critic-{stage}",
                    f"--output={root}/logs/{stage}-%A_%a.out",
                    f"--error={root}/logs/{stage}-%A_%a.err",
                    *extra,
                ],
                io.REPO / "scripts/hpc/run_general_critic_aces.sh",
                stage,
                0,
            )
            return jobs[stage]

        prepare = queue(
            "prepare",
            ["--partition=cpu", "--cpus-per-task=4", "--mem=16G", "--time=02:00:00"],
        )
        prior = prepare
        for stage, gpus, hours in (
            ("parent-review", 2, "04:00:00"),
            ("propose", 1, "01:00:00"),
            ("child-review", 2, "04:00:00"),
        ):
            prior = queue(
                stage,
                [
                    f"--dependency=afterok:{prior}",
                    "--partition=" + os.environ.get("AF_PARTITION", "gpu"),
                    f"--gres=gpu:h100:{gpus}",
                    "--cpus-per-task=8",
                    "--mem=128G",
                    f"--time={hours}",
                ],
            )
        fit = queue(
            "fit",
            [
                f"--dependency=afterok:{prior}",
                "--partition=cpu",
                "--cpus-per-task=1",
                "--mem=16G",
                "--time=00:30:00",
                f"--array=0-{len(old['rows']) - 1}%4",
            ],
        )
        queue(
            "report",
            [
                f"--dependency=afterany:{fit}",
                "--partition=cpu",
                "--cpus-per-task=1",
                "--mem=4G",
                "--time=00:10:00",
            ],
        )
        saved = {
            "protocol": io.PROTOCOL,
            "identity": identity,
            "jobs": jobs,
            "tasks": len(old["rows"]),
            "maximum_benchmark_fits": len(old["rows"]),
            "automatic_followup": False,
        }
        public._write(manifest, saved)
        return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(submit(args.source, args.root), indent=2))
