"""Exercise the uploaded bundle entry point without Slurm or model calls."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_shorten_entry_uses_new_bundle_and_existing_campaign(tmp_path):
    if not shutil.which("sha256sum"):
        pytest.skip("bundle entry requires sha256sum")
    repo = Path(__file__).resolve().parents[1]
    bundle = tmp_path / "new bundle"
    script = bundle / "scripts/hpc/iclr_targeted_perturbations_aces.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(repo / "scripts/hpc/iclr_targeted_perturbations_aces.sh", script)
    (bundle / "SHA256SUMS").write_text(
        f"{hashlib.sha256(script.read_bytes()).hexdigest()}  "
        "scripts/hpc/iclr_targeted_perturbations_aces.sh\n"
    )
    activation = tmp_path / "activate.sh"
    activation.write_text("module() { return 0; }\nroot=wrong\naction=wrong\n")
    interpreter = tmp_path / "python"
    interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$AF_CAPTURE"\n')
    interpreter.chmod(0o755)
    captured = tmp_path / "arguments"
    root, old = tmp_path / "saved campaign", tmp_path / "old bundle"
    environment = os.environ | {
        "AF_ANCHOR_ACTIVATION": str(activation),
        "AF_TARGETED_ROOT": str(root),
        "AF_PYTHON": str(interpreter),
        "AF_CAPTURE": str(captured),
    }
    subprocess.run(
        ["bash", str(script), "shorten", "2376062", str(old)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert captured.read_text().splitlines() == [
        "scripts/shorten_iclr_targeted_job.py",
        "--root",
        str(root),
        "--job",
        "2376062",
        "--code",
        str(old),
        "--replacement",
        str(bundle),
        "--apply",
    ]
    captured.unlink()
    result = subprocess.run(
        ["bash", str(script), "shorten", "2376062"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "Use: shorten JOB OLD_CODE" in result.stderr
    assert not captured.exists()
