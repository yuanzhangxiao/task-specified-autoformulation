"""Offline ACES submission, queued identity checks and uncertain-outcome guards."""

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    if not all(shutil.which(c) for c in ("bash", "jq", "sha256sum")):
        pytest.skip("shell tools unavailable")
    bins = tmp_path / "bin"
    bins.mkdir()
    parent, continuation = tmp_path / "parent/fit", tmp_path / "prior/continuation"
    for root in (parent, continuation):
        root.mkdir(parents=True)
        for name in ("freeze.json", "result.json", "backend_result.json"):
            (root / name).write_text("{}")
    (parent.parent / "handoff.json").write_text("{}")
    source, construction = tmp_path / "requirements", tmp_path / "construction"
    for root in (source, construction):
        root.mkdir()
        (root / "plan.json").write_text("{}")
    config = tmp_path / "config.json"
    config.write_bytes((REPO / "configs/prefit_numerical_sibling_v1.json").read_bytes())
    image = tmp_path / "vllm.sif"
    image.write_bytes(b"test-only image")
    log = tmp_path / "scheduler.json"
    scripts = {
        "module": "#!/bin/sh\nexit 0\n",
        "git": (
            '#!/bin/sh\ncase "$*" in *status*) [ "${DIRTY:-0}" = 0 ] '
            '|| echo modified ;; *) echo "${REVISION:-abc123}" ;; esac\n'
        ),
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
    root = tmp_path / "output"
    env = dict(
        {k: v for k, v in os.environ.items() if not k.startswith("AF_")},
        PATH=str(bins) + os.pathsep + os.environ["PATH"],
        AF_REPO_ROOT=str(REPO),
        AF_OUTPUT_ROOT=str(root),
        AF_PYTHON=sys.executable,
        AF_PARENT_FIT=str(parent),
        AF_CONTINUATION_FIT=str(continuation),
        AF_SOURCE_ROOT=str(source),
        AF_CONSTRUCTION_ROOT=str(construction),
        AF_CONFIG=str(config),
        AF_VLLM_IMAGE=str(image),
        AF_HF_HOME=str(tmp_path / "hf"),
        SLURM_JOB_ID="999",
    )

    def run(*, worker=False, **overrides):
        script = "run" if worker else "submit"
        return subprocess.run(
            [
                "bash",
                str(REPO / f"scripts/hpc/{script}_prefit_numerical_sibling_aces.sh"),
                *(["fit"] if worker else []),
            ],
            env={**env, **overrides},
            capture_output=True,
            text=True,
        )

    return run, log, root, env


def test_cpu_gpu_cpu_dependency_chain_and_idempotence(launcher):
    run, log, root, _ = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    calls = json.loads(log.read_text())
    assert [c[-1] for c in calls] == ["prepare", "propose", "fit"]
    assert "--cpus-per-task=1" in calls[0] and "--partition=cpu" in calls[0]
    assert "--gres=gpu:h100:1" in calls[1]
    assert "--dependency=afterok:101" in calls[1]
    assert "--kill-on-invalid-dep=yes" in calls[1]
    assert "--dependency=afterany:102" in calls[2]
    assert "--cpus-per-task=1" in calls[2]
    assert run().returncode == 0
    assert json.loads(log.read_text()) == calls
    assert (root / "submission-intent/identity.json").exists()


@pytest.mark.parametrize("stage, count", [("prepare", 1), ("propose", 2), ("fit", 3)])
def test_uncertain_submission_cannot_duplicate(launcher, stage, count):
    run, log, root, _ = launcher
    assert run(FAIL_STAGE=stage).returncode != 0
    assert len(json.loads(log.read_text())) == count
    assert (root / "submission-intent/identity.json").exists()
    assert run().returncode != 0
    assert len(json.loads(log.read_text())) == count


def test_concurrent_invocations_reserve_one_chain(launcher):
    run, log, _, _ = launcher
    with ThreadPoolExecutor(2) as pool:
        outcomes = list(pool.map(lambda _: run(), range(2)))
    assert any(r.returncode == 0 for r in outcomes)
    assert len(json.loads(log.read_text())) == 3


@pytest.mark.parametrize(
    "name",
    ["AF_PARENT_FIT", "AF_CONTINUATION_FIT", "AF_SOURCE_ROOT", "AF_CONSTRUCTION_ROOT"],
)
def test_all_historical_experiments_are_protected(launcher, name):
    run, log, _, env = launcher
    protected = Path(env[name])
    if name.endswith("FIT"):
        protected = protected.parent
    destination = protected / "new-output"
    result = run(AF_OUTPUT_ROOT=str(destination))
    assert result.returncode != 0 and "separate" in result.stderr
    assert not destination.exists() and not log.exists()


@pytest.mark.parametrize(
    "override",
    [
        {"REVISION": "different"},
        {"DIRTY": "1"},
        {"AF_SOURCE_ROOT": "/different-source"},
    ],
)
def test_queued_identity_drift_rejected_before_work(launcher, override):
    run, log, root, _ = launcher
    assert run().returncode == 0
    result = run(worker=True, **override)
    assert result.returncode != 0 and "Queued" in result.stderr
    assert not (root / "runtime").exists()
    assert len(json.loads(log.read_text())) == 3


@pytest.mark.parametrize(
    "name, filename",
    [
        ("AF_CONFIG", None),
        ("AF_PARENT_FIT", "backend_result.json"),
        ("AF_CONTINUATION_FIT", "result.json"),
        ("AF_SOURCE_ROOT", "plan.json"),
    ],
)
def test_queued_content_drift_rejected(launcher, name, filename):
    run, _, root, env = launcher
    assert run().returncode == 0
    changed = Path(env[name])
    if filename:
        changed /= filename
    changed.write_text('{"changed":true}')
    result = run(worker=True)
    assert result.returncode != 0 and "content changed" in result.stderr
    assert not (root / "runtime").exists()
    assert run().returncode != 0


def test_final_stage_reports_missing_preparation(launcher):
    run, _, root, _ = launcher
    assert run().returncode == 0
    result = run(worker=True)
    assert result.returncode != 0
    summary = json.loads((root / "summary.json").read_text())
    assert summary["status"] == "preparation_missing"


def test_shared_server_dispatches_new_protocol():
    result = subprocess.run(
        [
            "bash",
            str(REPO / "scripts/hpc/run_staged_topology_server.sh"),
            "--check-config",
            str(REPO / "configs/prefit_numerical_sibling_v1.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "prefit_numerical_sibling.py"
