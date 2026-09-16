"""Exercise real submission shell with an offline scheduler and frozen-plan stub."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return str(path)


def test_submission_ids_dependencies_and_no_duplicate(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "plan.json").write_text(
        json.dumps(
            {
                "config": {"rounds": 3},
                "tasks": [
                    {"index": 0, "arm": "full"},
                    {"index": 1, "arm": "refit_only"},
                ],
            }
        )
    )
    log = tmp_path / "scheduler.jsonl"
    executable(
        binaries / "git",
        '#!/bin/sh\ncase "$*" in *status*) exit 0;; *) echo abc123;; esac\n',
    )
    executable(binaries / "module", "#!/bin/sh\nexit 0\n")
    executable(
        binaries / "sbatch",
        """#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
log=Path(os.environ['FAKE_SCHEDULER'])
old=log.read_text().splitlines() if log.exists() else []
with log.open('a') as stream:stream.write(json.dumps(sys.argv[1:])+'\\n')
print(9000+len(old))
""",
    )
    python = executable(
        binaries / "project-python",
        """#!/bin/sh
case "$1" in *review_deadline.py) exit 0;; esac
exec python3 "$@"
""",
    )
    image = tmp_path / "image.sif"
    image.touch()
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(ROOT),
        "AF_PYTHON": python,
        "AF_OUTPUT_ROOT": str(root),
        "AF_PUBLIC_ROOT": str(tmp_path / "public"),
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": str(tmp_path / "hf"),
        "FAKE_SCHEDULER": str(log),
    }
    command = ["bash", str(ROOT / "scripts/hpc/submit_review_deadline_aces.sh")]
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 11  # prepare, demo, three proposer/fit/barrier trios
    assert "--gres=gpu:h100:1" in rows[2]
    assert "--dependency=afterok:9000" in rows[2]
    assert "--array=0%16" in rows[3]
    assert "--array=0,1%16" in rows[6]
    assert "--dependency=afterany:9003" in rows[4]
    assert "--dependency=afterok:9000,afterany:9004" in rows[5]
    manifest = json.loads((root / "submission_manifest.json").read_text())
    assert not manifest["automatic_test_access"]
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    assert len(log.read_text().splitlines()) == 11


def test_worker_and_transfer_shell_syntax():
    for name in (
        "run_review_deadline_aces.sh",
        "submit_review_deadline_aces.sh",
        "stage_review_deadline_public.sh",
    ):
        subprocess.run(["bash", "-n", str(ROOT / "scripts/hpc" / name)], check=True)
