#!/usr/bin/env python3
"""Submit a CPU evidence audit, affected-only judge job, and final report."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal import judge_sign_recheck as campaign
from scripts.submit_shared_process_pilot import source_commit, submit_job


def submit(source: Path, root: Path) -> dict:
    """Keep scheduler intents durable and all new work outside the saved campaign."""
    source, root = source.resolve(), root.resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("source and output must be separate")
    commit = source_commit(campaign.REPO)
    if os.environ.get("AF_COMMIT", commit) != commit:
        raise ValueError("pinned commit differs")
    for name in ("AF_PYTHON", "AF_VLLM_IMAGE"):
        if not Path(os.environ[name]).is_file():
            raise ValueError(f"missing {name}")
    if not os.environ.get("AF_HF_HOME"):
        raise ValueError("set AF_HF_HOME")
    # Fail before allocating resources if any stage or symbolic receipt is missing.
    imported = campaign.snapshot(source)
    identity = {
        "commit": commit,
        "source": str(source),
        "root": str(root),
        "snapshot_sha256": campaign.content_hash(imported),
        "python": os.environ["AF_PYTHON"],
        "image": os.environ["AF_VLLM_IMAGE"],
        "hf_home": os.environ["AF_HF_HOME"],
    }
    os.environ.update(
        AF_REPO_ROOT=str(campaign.REPO),
        AF_SOURCE_ROOT=str(source),
        AF_OUTPUT_ROOT=str(root),
        AF_COMMIT=commit,
        AF_SIGN_SNAPSHOT_SHA256=identity["snapshot_sha256"],
        PYTHONDONTWRITEBYTECODE="1",
    )
    with campaign.public._lock(root / "scheduler"):
        manifest = root / "submission_manifest.json"
        if manifest.exists():
            saved = campaign.public._read(manifest)
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
        campaign.public._write(intent / "identity.json", identity)
        (root / "logs").mkdir(exist_ok=True)
        jobs = {}

        def queue(stage: str, extra: list[str]) -> str:
            jobs[stage] = submit_job(
                intent,
                stage,
                [
                    "--kill-on-invalid-dep=yes",
                    "--nodes=1",
                    "--ntasks=1",
                    "--export=ALL",
                    "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
                    f"--job-name=sign-recheck-{stage}",
                    f"--output={root}/logs/{stage}-%A_%a.out",
                    f"--error={root}/logs/{stage}-%A_%a.err",
                    *extra,
                ],
                campaign.REPO / "scripts/hpc/run_judge_sign_recheck_aces.sh",
                stage,
                0,
            )
            return jobs[stage]

        prepare = queue(
            "prepare",
            [
                "--partition=cpu",
                "--cpus-per-task=4",
                "--mem=16G",
                "--time=01:00:00",
            ],
        )
        affected = sum(r["affected"] for r in imported["reviews"])
        prior = prepare
        if affected:
            prior = queue(
                "review",
                [
                    f"--dependency=afterok:{prepare}",
                    "--partition=" + os.environ.get("AF_PARTITION", "gpu"),
                    "--gres=gpu:h100:2",
                    "--cpus-per-task=8",
                    "--mem=128G",
                    "--time=08:00:00",
                ],
            )
        queue(
            "report",
            [
                f"--dependency=afterany:{prior}",
                "--partition=cpu",
                "--cpus-per-task=1",
                "--mem=4G",
                "--time=00:10:00",
            ],
        )
        saved = {
            "protocol": campaign.PROTOCOL,
            "identity": identity,
            "jobs": jobs,
            "maximum_new_paired_reviews": affected,
            "optimizer_calls": 0,
            "proposer_calls": 0,
            "automatic_followup": False,
        }
        campaign.public._write(manifest, saved)
        return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(submit(args.source, args.root), indent=2))
