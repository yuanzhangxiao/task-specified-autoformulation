"""Exercise real submission shell with an offline scheduler and frozen-plan stub."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write

ROOT = Path(__file__).resolve().parents[1]


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return str(path)


def submission_fixture(tmp_path, protocol="review-deadline-2"):
    """Run the real shell with isolated accounting, submission and plan fixtures."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "plan.json").write_text(
        json.dumps(
            {
                "config": {"rounds": 3, "protocol": protocol},
                "tasks": [
                    {"index": 0, "arm": "full", "task_id": "full"},
                    {"index": 1, "arm": "refit_only", "task_id": "refit"},
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
import json,os,re,sys
from pathlib import Path
log=Path(os.environ['FAKE_SCHEDULER'])
old=log.read_text().splitlines() if log.exists() else []
with log.open('a') as stream:stream.write(json.dumps(sys.argv[1:])+'\\n')
retired=int(os.environ.get('FAKE_RETIRED_THROUGH', '0'))
for arg in sys.argv[1:]:
    if arg.startswith('--dependency='):
        if any(int(n) <= retired for n in re.findall(r'\\d+', arg)):
            print('sbatch: error: Job dependency problem', file=sys.stderr)
            sys.exit(1)
if os.environ.get('FAKE_FAIL_AT') == str(len(old)):
    print('sbatch: error: QOSMaxSubmitJobPerUserLimit', file=sys.stderr)
    sys.exit(1)
print(9000+len(old))
""",
    )
    executable(
        binaries / "sacct",
        """#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
if 'FAKE_ACCOUNTING_STDOUT' in os.environ:
    print(os.environ['FAKE_ACCOUNTING_STDOUT'])
    sys.exit(int(os.environ.get('FAKE_ACCOUNTING_EXIT', '0')))
lines=Path(os.environ['FAKE_SCHEDULER']).read_text().splitlines()
rows=[json.loads(line) for line in lines]
ids=next(a.split('=',1)[1] for a in sys.argv[1:] if a.startswith('--jobs='))
assert '--allocations' in sys.argv and '--starttime=1970-01-01' in sys.argv
for job in ids.split(','):
    args=rows[int(job)-9000]
    name=next(a.split('=',1)[1] for a in args if a.startswith('--job-name='))
    print(f'{job}|{name}|COMPLETED|0:0')
""",
    )
    python = executable(
        binaries / "project-python",
        """#!/bin/sh
case "$1" in *review_deadline.py) exit 0;; esac
exec """
        + shlex.quote(sys.executable)
        + ' "$@"\n',
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
    return root, env, command, log


def completed_round(root, index):
    """Failed fits still produce terminal, sealed results and allow progression."""
    plan = json.loads((root / "plan.json").read_text())
    for task in plan["tasks"]:
        sealed_write(
            root / "results" / task["task_id"] / f"round_{index:02d}" / "result.json",
            {"task": task, "round": index, "status": "fit_failed"},
        )


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
    root, env, command, log = submission_fixture(tmp_path, protocol)
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
        completed_round(root, 0)
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
            completed_round(root, index - 1)
            subprocess.run(
                [*command, "--round", str(index)],
                env={**env, "FAKE_RETIRED_THROUGH": "9005" if index == 1 else "9009"},
                check=True,
                capture_output=True,
                text=True,
            )
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(rows) == 13
        assert not any(a.startswith("--dependency=") for a in rows[6])
        assert "--array=0,1%16" in rows[7]
        assert rows[9][-2:] == ["submit-next", "2"]
        assert "--dependency=afterany:9008" in rows[9]
        assert not any(a.startswith("--dependency=") for a in rows[10])
        prerequisite = json.loads(
            (root / "submission-intent/round-2/prerequisites.json").read_text()
        )
        assert prerequisite["previous_round"] == 1
        assert set(prerequisite["results"]) == {"full", "refit"}
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
                env={**env, "FAKE_ACCOUNTING_STDOUT": "", "FAKE_ACCOUNTING_EXIT": "1"},
                check=True,
                capture_output=True,
                text=True,
            )
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    assert len(log.read_text().splitlines()) == len(rows)


@pytest.mark.parametrize(
    "accounting,exit_code",
    [
        ("", "0"),
        (
            "9000|review-v2-prepare-0|FAILED|1:0\n9004|review-v2-finish-0|COMPLETED|0:0",
            "0",
        ),
        (
            "9000|review-v2-prepare-0|COMPLETED|0:0\n9004|review-v2-finish-0|RUNNING|0:0",
            "0",
        ),
        (
            "9000|review-v2-prepare-0|COMPLETED|0:0\n9004|review-v2-finish-0|FAILED|1:0",
            "0",
        ),
        (
            "9000|review-v2-prepare-0|COMPLETED|1:0\n9004|review-v2-finish-0|COMPLETED|0:0",
            "0",
        ),
        (
            "9000|unrelated-job|COMPLETED|0:0\n9004|review-v2-finish-0|COMPLETED|0:0",
            "0",
        ),
        (
            "9000.batch|review-v2-prepare-0|COMPLETED|0:0\n9004|review-v2-finish-0|COMPLETED|0:0",
            "0",
        ),
        (
            "9000|review-v2-prepare-0|COMPLETED|0:0\n9000|review-v2-prepare-0|COMPLETED|0:0\n9004|review-v2-finish-0|COMPLETED|0:0",
            "0",
        ),
        (
            "9000|review-v2-prepare-0|COMPLETED|0:0\n9004|review-v2-finish-0|COMPLETED|0:0",
            "1",
        ),
    ],
)
def test_uncertain_accounting_submits_nothing_and_can_be_rechecked(
    tmp_path, accounting, exit_code
):
    root, env, command, log = submission_fixture(tmp_path)
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    completed_round(root, 0)
    rejected = subprocess.run(
        [*command, "--round", "1"],
        env={
            **env,
            "FAKE_ACCOUNTING_STDOUT": accounting,
            "FAKE_ACCOUNTING_EXIT": exit_code,
        },
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "prerequisites not confirmed" in rejected.stderr
    assert not (root / "submission-intent/round-1").exists()
    assert len(log.read_text().splitlines()) == 6
    subprocess.run(
        [*command, "--round", "1"], env=env, check=True, capture_output=True, text=True
    )
    assert len(log.read_text().splitlines()) == 10


@pytest.mark.parametrize(
    "fault", ["missing", "digest", "task", "round", "job_id", "commit", "incomplete"]
)
def test_prior_round_artifacts_are_checked_before_submission_intent(tmp_path, fault):
    root, env, command, log = submission_fixture(tmp_path)
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    completed_round(root, 0)
    path = root / "results/full/round_00/result.json"
    if fault == "missing":
        path.unlink()
    elif fault in {"digest", "task", "round"}:
        value = json.loads(path.read_text())
        if fault == "digest":
            value["status"] = "tampered"
            path.write_text(json.dumps(value))
        else:
            path.unlink()
            value.pop("artifact_sha256")
            value.update(
                {"task": {"task_id": "other"}} if fault == "task" else {"round": 9}
            )
            sealed_write(path, value)
    elif fault == "job_id":
        (root / "submission-intent/prepare-0.id").write_text("12345\n")
    else:
        path = root / "submission-round-0.json"
        value = json.loads(path.read_text())
        value.update(
            {"commit": "other"}
            if fault == "commit"
            else {"round_submission_complete": False}
        )
        path.write_text(json.dumps(value))
    rejected = subprocess.run(
        [*command, "--round", "1"], env=env, capture_output=True, text=True
    )
    assert rejected.returncode == 2
    assert "prerequisites not confirmed" in rejected.stderr
    assert not (root / "submission-intent/round-1").exists()
    assert len(log.read_text().splitlines()) == 6


def test_worker_and_transfer_shell_syntax():
    for name in (
        "run_review_deadline_aces.sh",
        "submit_review_deadline_aces.sh",
        "stage_review_deadline_public.sh",
    ):
        subprocess.run(["bash", "-n", str(ROOT / "scripts/hpc" / name)], check=True)
