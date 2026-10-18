#!/usr/bin/env python3
"""Submit one bounded initializer-choice GPU job with idempotent Slurm receipts."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.research.construction_comparison import check_storage
from scripts import initializer_choice_aces as campaign
from scripts.submit_review_continuation import submit_job


def submit(root: Path, wave: str) -> dict:
    """Reuse a confirmed receipt; never retry an uncertain sbatch submission."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", wave):
        raise ValueError("wave must be a simple identifier")
    plan, execution = campaign.verify(root)
    for key in ("AF_PYTHON", "AF_REPO_ROOT", "AF_VLLM_IMAGE", "AF_HF_HOME"):
        if not os.environ.get(key):
            raise ValueError(f"set {key}")
    if Path(os.environ["AF_REPO_ROOT"]).resolve() != campaign.REPO:
        raise ValueError("repository path differs from the executing source")
    if (
        campaign.file_hash(Path(os.environ["AF_VLLM_IMAGE"]))
        != execution["image_sha256"]
    ):
        raise ValueError("serving image differs")
    os.environ["AF_OUTPUT_ROOT"] = str(root)
    identity = {
        "plan_sha256": plan["artifact_sha256"],
        "execution_sha256": execution["artifact_sha256"],
        "wave": wave,
    }
    directory = root / "submissions" / wave
    with public._lock(directory):
        manifest = directory / "manifest.json"
        if manifest.exists():
            saved = public._read(manifest)
            if any(saved.get(k) != v for k, v in identity.items()):
                raise ValueError("submission identity differs")
            return saved
        intent = directory / "propose.intent.json"
        if intent.exists():
            raise ValueError(f"submission outcome needs inspection: {directory}")
        # Group quotas can exhaust files long before bytes: test both before queueing.
        check_storage(root, probe_bytes=16 * 1024 * 1024)
        with tempfile.TemporaryDirectory(dir=root, prefix=".inode-preflight-") as probe:
            for index in range(512):
                (Path(probe) / str(index)).touch()
        (root / "logs").mkdir(exist_ok=True)
        public._write(directory / "identity.json", identity)
        options = [
            "--kill-on-invalid-dep=yes",
            "--nodes=1",
            "--ntasks=1",
            "--export=ALL",
            "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
            "--partition=gpu",
            "--gres=gpu:h100:1",
            "--cpus-per-task=8",
            "--mem=64G",
            "--time=01:30:00",
            "--signal=B:TERM@300",
            "--exclude=ac042",
            "--job-name=phasec-initializer-choice",
            f"--output={root}/logs/{wave}-%j.out",
            f"--error={root}/logs/{wave}-%j.err",
        ]
        job = submit_job(
            directory,
            "propose",
            options,
            campaign.REPO / "scripts/hpc/run_initializer_choice_aces.sh",
            "propose",
            0,
        )
        result = {
            **identity,
            "jobs": {"propose": job},
            "maximum_cases": len(plan["cases"]),
            "maximum_generation_requests": campaign.MAXIMUM_CALLS,
            "maximum_charged_tokens": campaign.MAXIMUM_TOKENS,
            "automatic_followup": False,
            "test_data_opened": False,
        }
        public._write(manifest, result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--wave", default="choice-1")
    args = parser.parse_args()
    print(json.dumps(submit(args.root.resolve(), args.wave), indent=2))
