#!/usr/bin/env python3
"""Submit one bounded five-seed initializer retest on the existing H100 launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import construction_initialization_repair as campaign
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]


def submit(root: Path, wave: str = "repair-1") -> dict:
    """Never duplicate a confirmed or uncertain submission, even after a restart."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", wave):
        raise ValueError("wave must be a simple identifier")
    plan = sealed_read(root / "plan.json")
    campaign.verify(root)
    for key in ("AF_PYTHON", "AF_VLLM_IMAGE", "AF_HF_HOME"):
        if not os.environ.get(key):
            raise ValueError(f"set {key}")
    image = Path(os.environ["AF_VLLM_IMAGE"])
    with image.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    execution = sealed_write(
        root / "execution.json",
        {
            "protocol": "initialization-repair-aces-execution-1",
            "plan_sha256": plan["artifact_sha256"],
            "image_sha256": digest,
            "launcher_sha256": hashlib.sha256(
                (REPO / "scripts/hpc/run_initializer_choice_aces.sh").read_bytes()
            ).hexdigest(),
            "automatic_followup": False,
        },
    )
    os.environ.update(
        AF_REPO_ROOT=str(REPO),
        AF_OUTPUT_ROOT=str(root),
        AF_INITIALIZATION_WORKER="graph-repair",
    )
    directory = root / "submissions" / wave
    identity = {"execution_sha256": execution["artifact_sha256"], "wave": wave}
    with public._lock(directory):
        manifest = directory / "manifest.json"
        if manifest.exists():
            result = json.loads(manifest.read_text())
            if result.get("execution_sha256") != identity["execution_sha256"]:
                raise ValueError("submission identity differs")
            return result
        if (directory / "propose.intent.json").exists():
            raise ValueError(
                f"inspect uncertain submission before retrying: {directory}"
            )
        (root / "logs").mkdir(exist_ok=True)
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
            "--time=01:00:00",
            "--signal=B:TERM@300",
            "--exclude=ac042",
            "--job-name=phasec-init-repair",
            f"--output={root}/logs/repair-%j.out",
            f"--error={root}/logs/repair-%j.err",
        ]
        job = submit_job(
            directory,
            "propose",
            options,
            REPO / "scripts/hpc/run_initializer_choice_aces.sh",
            "propose",
            0,
        )
        result = {**identity, "job": job, "automatic_followup": False}
        public._write(manifest, result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wave", default="repair-1")
    args = parser.parse_args()
    print(json.dumps(submit(args.root.resolve(), args.wave), indent=2))
