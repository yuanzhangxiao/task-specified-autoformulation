"""Exercise real submission shell with an offline scheduler and frozen-plan stub."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize(
    "protocol,partial_failure",
    [
        ("review-deadline-1", False),
        ("review-deadline-2", False),
        ("review-deadline-2", True),
    ],
)
def test_submission_ids_dependencies_and_no_duplicate(
    tmp_path, protocol, partial_failure
):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "plan.json").write_text(
        json.dumps(
            {
                "config": {"rounds": 3, "protocol": protocol},
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
if os.environ.get('FAKE_FAIL_AT') == str(len(old)):
    print('sbatch: error: QOSMaxSubmitJobPerUserLimit', file=sys.stderr)
    sys.exit(1)
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
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"protocol": protocol, "platform": "aces-h100x1", "public_cells": ["cell"]}
        )
    )
    public_cell = tmp_path / "public/phase_b_v1/cell"
    public_cell.mkdir(parents=True)
    for name in ("manifest.json", "proposer_prompt.txt", "train.csv", "validation.csv"):
        (public_cell / name).write_text("fixture")
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(ROOT),
        "AF_CONFIG": str(config),
        "AF_PYTHON": python,
        "AF_OUTPUT_ROOT": str(root),
        "AF_PUBLIC_ROOT": str(tmp_path / "public"),
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": str(tmp_path / "hf"),
        "FAKE_SCHEDULER": str(log),
    }
    launcher = (
        "submit_review_deadline_v2_aces.sh"
        if protocol == "review-deadline-2"
        else "submit_review_deadline_aces.sh"
    )
    command = ["bash", str(ROOT / "scripts/hpc" / launcher)]
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert "--gres=gpu:h100:1" in rows[2]
    assert "--dependency=afterok:9000" in rows[2]
    assert "--array=0%16" in rows[3]
    assert "--dependency=afterany:9003" in rows[4]
    if protocol == "review-deadline-2":
        # Only visit zero is queued. Later visits are submitted by a CPU
        # dispatcher after the previous fitting array and finish job end.
        assert len(rows) == 6
        assert rows[5][-2:] == ["submit-next", "1"]
        assert "--dependency=afterany:9004" in rows[5]
        initial_manifest = json.loads((root / "submission_manifest.json").read_text())
        assert initial_manifest["submitted_through_round"] == 0
        assert not initial_manifest["all_rounds_submitted"]
        assert not initial_manifest["submission_complete"]
        assert initial_manifest["round_submission_complete"]
        subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        assert len(log.read_text().splitlines()) == 6
        if partial_failure:
            rejected = subprocess.run(
                [*command, "--round", "1"],
                env={**env, "FAKE_FAIL_AT": "7"},
                capture_output=True,
                text=True,
            )
            assert rejected.returncode != 0
            assert "QOSMaxSubmitJobPerUserLimit" in rejected.stderr
            assert (root / "submission-intent/propose-1.id").read_text().strip() == (
                "9006"
            )
            assert not (root / "submission-round-1.json").exists()
            again = subprocess.run(
                [*command, "--round", "1"], env=env, capture_output=True, text=True
            )
            assert again.returncode == 2
            assert "submission intent exists" in again.stderr
            assert len(log.read_text().splitlines()) == 8
            return
        for index in (1, 2):
            subprocess.run(
                [*command, "--round", str(index)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(rows) == 13
        assert "--dependency=afterok:9000,afterany:9004" in rows[6]
        assert "--array=0,1%16" in rows[7]
        assert rows[9][-2:] == ["submit-next", "2"]
        assert "--dependency=afterany:9008" in rows[9]
        assert "--dependency=afterok:9000,afterany:9008" in rows[10]
    else:
        assert len(rows) == 11
        assert "--array=0,1%16" in rows[6]
        assert "--dependency=afterok:9000,afterany:9004" in rows[5]
    manifest = json.loads((root / "submission_manifest.json").read_text())
    assert not manifest["automatic_test_access"]
    if protocol == "review-deadline-2":
        assert manifest["all_rounds_submitted"]
        assert manifest["submitted_through_round"] == 2
        for index in (1, 2):
            subprocess.run(
                [*command, "--round", str(index)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    assert len(log.read_text().splitlines()) == len(rows)


def test_worker_and_transfer_shell_syntax():
    for name in (
        "run_review_deadline_aces.sh",
        "submit_review_deadline_aces.sh",
        "stage_review_deadline_public.sh",
    ):
        subprocess.run(["bash", "-n", str(ROOT / "scripts/hpc" / name)], check=True)
