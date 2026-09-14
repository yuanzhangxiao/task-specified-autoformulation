"""Offline scheduler checks for the fitting-free two-job experiment."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def launcher(tmp_path):
    if not all(shutil.which(c) for c in ("bash", "jq", "sha256sum")):
        pytest.skip("shell tools unavailable")
    repo = Path(__file__).resolve().parents[1]
    bins, root, source = (tmp_path / name for name in ("bin", "output", "source"))
    for path in (bins, root, source):
        path.mkdir()
    (source / "plan.json").write_text("{}")
    image = tmp_path / "image.sif"
    image.write_bytes(b"fixture")
    log = tmp_path / "scheduler.json"
    scripts = {
        "module": "#!/bin/sh\nexit 0\n",
        "git": (
            '#!/bin/sh\ncase "$*" in *status*) exit 0 ;; '
            '*) echo "${REVISION:-abc123}" ;; esac\n'
        ),
        "squeue": '#!/bin/sh\n[ "${QUEUED:-0}" = 0 ] || echo 102\nexit 0\n',
        "sacct": '#!/bin/sh\necho "${ACCOUNTING_STATE:-COMPLETED}"\n',
        "fakepython": f"""#!{sys.executable}
import json, os, pathlib, sys
if 'summary' in sys.argv:
    print(json.dumps({{'status': os.environ.get('SUMMARY_STATUS', 'partial')}}))
elif len(sys.argv) > 1 and sys.argv[1] == '-':
    destination = pathlib.Path(sys.argv[2]).resolve()
    source = pathlib.Path(sys.argv[3]).resolve()
    if destination.is_relative_to(source):
        sys.exit(2)
""",
        "sbatch": f"""#!{sys.executable}
import json, os, pathlib, sys
path = pathlib.Path({str(log)!r})
calls = json.loads(path.read_text()) if path.exists() else []
calls.append(sys.argv[1:]); path.write_text(json.dumps(calls))
assert '--nodes=1' in sys.argv and '--ntasks=1' in sys.argv
print(100 + len(calls))
if os.environ.get('FAIL_STAGE') == sys.argv[-1]: sys.exit(1)
""",
    }
    for name, script in scripts.items():
        path = bins / name
        path.write_text(script)
        path.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(bins) + os.pathsep + os.environ["PATH"],
        AF_REPO_ROOT=str(repo),
        AF_PYTHON=str(bins / "fakepython"),
        AF_OUTPUT_ROOT=str(root),
        AF_SOURCE_ROOT=str(source),
        AF_VLLM_IMAGE=str(image),
        AF_HF_HOME=str(tmp_path / "hf"),
        AF_RESUME="0",
    )

    def run(**overrides):
        return subprocess.run(
            ["bash", str(repo / "scripts/hpc/submit_prefit_feedback_aces.sh")],
            env={**env, **overrides},
            capture_output=True,
            text=True,
        )

    return run, log, root


def test_only_cpu_replay_and_gpu_repair_are_submitted(launcher):
    run, log, root = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    calls = json.loads(log.read_text())
    assert [c[-1] for c in calls] == ["prepare", "repair"]
    assert "--partition=cpu" in calls[0]
    assert "--gres=gpu:h100:1" in calls[1]
    assert "--dependency=afterok:101" in calls[1]
    assert run().returncode == 0
    assert json.loads(log.read_text()) == calls
    assert (root / "submission_manifest.json").exists()


@pytest.mark.parametrize("stage, count", [("prepare", 1), ("repair", 2)])
def test_uncertain_submission_leaves_intent_and_cannot_duplicate(
    launcher, stage, count
):
    run, log, root = launcher
    assert run(FAIL_STAGE=stage).returncode != 0
    assert len(json.loads(log.read_text())) == count
    assert (root / "submissions/initial").exists()
    assert run().returncode != 0
    assert len(json.loads(log.read_text())) == count


def test_resume_only_gpu_and_terminal_campaign_has_no_new_job(launcher):
    run, log, root = launcher
    assert run().returncode == 0
    (root / "plan.json").write_text("{}")
    result = run(AF_RESUME="1")
    assert result.returncode == 0, result.stderr
    calls = json.loads(log.read_text())
    assert [c[-1] for c in calls] == ["prepare", "repair", "repair"]
    assert run(AF_RESUME="1", SUMMARY_STATUS="complete").returncode == 0
    assert len(json.loads(log.read_text())) == 3


@pytest.mark.parametrize(
    "overrides",
    [{"QUEUED": "1"}, {"ACCOUNTING_STATE": "UNKNOWN"}, {"REVISION": "different"}],
)
def test_resume_rejects_active_uncertain_or_changed_runs(launcher, overrides):
    run, log, root = launcher
    assert run().returncode == 0
    (root / "plan.json").write_text("{}")
    assert run(AF_RESUME="1", **overrides).returncode != 0
    assert len(json.loads(log.read_text())) == 2


def test_shared_server_selects_fit_free_cli():
    result = subprocess.run(
        [
            "bash",
            "scripts/hpc/run_staged_topology_server.sh",
            "--check-config",
            "configs/prefit_matched_feedback_v1.json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "prefit_feedback_campaign.py"


def test_worker_surfaces_preflight_failure_in_job_stderr(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    bins = tmp_path / "bin"
    bins.mkdir()
    module = bins / "module"
    module.write_text("#!/bin/sh\nexit 0\n")
    module.chmod(0o755)
    python = bins / "failed-preflight"
    python.write_text("#!/bin/sh\necho 'fixture dependency error'\nexit 2\n")
    python.chmod(0o755)
    root = tmp_path / "output"
    result = subprocess.run(
        ["bash", str(repo / "scripts/hpc/run_prefit_feedback_aces.sh"), "prepare"],
        env={
            **os.environ,
            "PATH": str(bins) + os.pathsep + os.environ["PATH"],
            "AF_REPO_ROOT": str(repo),
            "AF_PYTHON": str(python),
            "AF_OUTPUT_ROOT": str(root),
            "AF_SOURCE_ROOT": str(tmp_path / "source"),
            "AF_CONFIG": str(repo / "configs/prefit_matched_feedback_v1.json"),
            "AF_VLLM_IMAGE": str(tmp_path / "unused-image"),
            "AF_HF_HOME": str(tmp_path / "unused-hf"),
            "SLURM_JOB_ID": "12345",
        },
        capture_output=True,
        text=True,
    )
    log = root / "runtime/preflight-12345.log"
    assert result.returncode == 2
    assert str(log) in result.stderr and "fixture dependency error" in result.stderr
    assert log.read_text() == "fixture dependency error\n"
    assert not (root / "replay.json").exists()
