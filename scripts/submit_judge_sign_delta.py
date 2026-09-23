#!/usr/bin/env python3
"""Submit an isolated Delta recheck from one exported ACES sign-plan JSON file."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal import judge_sign_delta as campaign
from scripts.submit_shared_process_pilot import source_commit, submit_job


def submit(source: Path, root: Path) -> dict:
    """Record submission intents; never silently repeat an uncertain queue operation."""
    source, root = source.resolve(), root.resolve()
    if source.is_relative_to(root):
        raise ValueError("source plan and output must be separate")
    imported = campaign.imported_plan(source)
    commit = source_commit(campaign.REPO)
    if os.environ.get("AF_COMMIT", commit) != commit:
        raise ValueError("pinned commit differs")
    for name in ("AF_PYTHON", "AF_VLLM_IMAGE"):
        if not Path(os.environ[name]).is_file():
            raise ValueError(f"missing {name}")
    if not os.environ.get("AF_HF_HOME"):
        raise ValueError("set AF_HF_HOME")
    identity = {
        "commit": commit,
        "source_plan": str(source),
        "root": str(root),
        "origin_plan_sha256": imported["artifact_sha256"],
        "python": os.environ["AF_PYTHON"],
        "image": os.environ["AF_VLLM_IMAGE"],
        "hf_home": os.environ["AF_HF_HOME"],
        "cpu_account": os.environ.get("AF_CPU_ACCOUNT", "bibo-delta-cpu"),
        "gpu_account": os.environ.get("AF_GPU_ACCOUNT", "bibo-delta-gpu"),
    }
    os.environ.update(
        AF_REPO_ROOT=str(campaign.REPO),
        AF_SOURCE_PLAN=str(source),
        AF_OUTPUT_ROOT=str(root),
        AF_COMMIT=commit,
        AF_SIGN_ORIGIN_SHA256=imported["artifact_sha256"],
        PYTHONDONTWRITEBYTECODE="1",
    )
    public = campaign.original.public
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
                "partial submission: inspect receipts/jobs; retain intent"
            ) from None
        public._write(intent / "identity.json", identity)
        (root / "logs").mkdir(exist_ok=True)
        jobs = {}

        def queue(stage: str, extra: list[str]) -> str:
            gpu = stage == "review"
            jobs[stage] = submit_job(
                intent,
                stage,
                [
                    "--kill-on-invalid-dep=yes",
                    "--nodes=1",
                    "--ntasks=1",
                    "--export=ALL",
                    "--account=" + identity["gpu_account" if gpu else "cpu_account"],
                    "--partition=" + ("gpuA40x4" if gpu else "cpu"),
                    f"--job-name=sign-delta-{stage}",
                    f"--output={root}/logs/{stage}-%A_%a.out",
                    f"--error={root}/logs/{stage}-%A_%a.err",
                    *extra,
                ],
                campaign.REPO / "scripts/hpc/run_judge_sign_recheck_delta.sh",
                stage,
                0,
            )
            return jobs[stage]

        prior = queue(
            "prepare",
            [
                "--cpus-per-task=4",
                "--mem=16G",
                "--time=01:00:00",
            ],
        )
        affected = sum(r["affected"] for r in imported["reviews"])
        if affected:
            prior = queue(
                "review",
                [
                    f"--dependency=afterok:{prior}",
                    "--gpus-per-node=4",
                    "--cpus-per-task=16",
                    "--mem=128G",
                    "--time=03:00:00",
                ],
            )
        queue(
            "report",
            [
                f"--dependency=afterany:{prior}",
                "--cpus-per-task=1",
                "--mem=4G",
                "--time=00:10:00",
            ],
        )
        value = {
            "protocol": campaign.PROTOCOL,
            "identity": identity,
            "jobs": jobs,
            "maximum_new_paired_reviews": affected,
            "optimizer_calls": 0,
            "proposer_calls": 0,
            "automatic_followup": False,
        }
        public._write(manifest, value)
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(submit(args.source_plan, args.root), indent=2))
