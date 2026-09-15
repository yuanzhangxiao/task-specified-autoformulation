"""One-CPU submission is idempotent and never silently duplicates uncertain work."""

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
    bins, source, parent, public = (
        tmp_path / n for n in ("bin", "source", "parent", "public")
    )
    for path in (bins, source, parent, public):
        path.mkdir()
    for path in (source, parent):
        (path / "plan.json").write_text("{}")
    log = tmp_path / "scheduler.json"
    scripts = {
        "module": "#!/bin/sh\nexit 0\n",
        "git": (
            '#!/bin/sh\ncase "$*" in *status*) '
            '[ "${DIRTY:-0}" = 0 ] || echo " M src/changed.py"; exit 0 ;; '
            '*) echo "${REVISION:-abc123}" ;; esac\n'
        ),
        "sbatch": f"""#!{sys.executable}
import json, os, pathlib, sys
path = pathlib.Path({str(log)!r})
calls = json.loads(path.read_text()) if path.exists() else []
calls.append(sys.argv[1:]); path.write_text(json.dumps(calls))
print('501')
sys.exit(int(os.environ.get('FAIL_SUBMIT', '0')))
""",
    }
    for name, script in scripts.items():
        path = bins / name
        path.write_text(script)
        path.chmod(0o755)
    root = tmp_path / "output"
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("AF_")},
        "PATH": str(bins) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(repo),
        "AF_OUTPUT_ROOT": str(root),
        "AF_SOURCE_ROOT": str(source),
        "AF_CONSTRUCTION_ROOT": str(parent),
        "AF_PUBLIC_ROOT": str(public),
        "AF_PYTHON": sys.executable,
    }

    def run(**updates):
        return subprocess.run(
            ["bash", str(repo / "scripts/hpc/submit_prefit_fit_handoff_aces.sh")],
            env={**env, **updates},
            capture_output=True,
            text=True,
        )

    return run, log, root, env, bins


def test_one_cpu_no_gpu_and_repeated_submission_is_read_only(launcher):
    run, log, root, _, _ = launcher
    first = run()
    assert first.returncode == 0, first.stderr
    calls = json.loads(log.read_text())
    assert len(calls) == 1
    assert "--partition=cpu" in calls[0] and "--cpus-per-task=1" in calls[0]
    assert not any("--gres" in arg for arg in calls[0])
    assert run().stdout == first.stdout
    assert json.loads(log.read_text()) == calls
    assert (
        json.loads((root / "submission_manifest.json").read_text())["job_id"] == "501"
    )


def test_uncertain_submission_keeps_intent_and_cannot_duplicate(launcher):
    run, log, root, _, _ = launcher
    assert run(FAIL_SUBMIT="1").returncode != 0
    assert (root / "submission-intent/identity.json").exists()
    assert run().returncode != 0
    assert len(json.loads(log.read_text())) == 1


def test_changed_revision_cannot_reuse_submission(launcher):
    run, log, _, _, _ = launcher
    assert run().returncode == 0
    assert run(REVISION="different").returncode != 0
    assert len(json.loads(log.read_text())) == 1


def test_worker_surfaces_preflight_failure(launcher):
    run, _, root, env, bins = launcher
    assert run().returncode == 0
    python = bins / "fail-python"
    python.write_text("#!/bin/sh\necho 'fixture dependency failure'\nexit 2\n")
    python.chmod(0o755)
    repo = Path(env["AF_REPO_ROOT"])
    result = subprocess.run(
        ["bash", str(repo / "scripts/hpc/run_prefit_fit_handoff_aces.sh")],
        env={
            **env,
            "AF_PYTHON": str(python),
            "AF_SELECTION": str(repo / "configs/prefit_public_fit_handoff_v1.json"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "fixture dependency failure" in result.stderr
    assert not (root / "fit/started.json").exists()


@pytest.mark.parametrize("drift", ["commit", "dirty", "selection"])
def test_worker_rejects_queued_checkout_drift_before_preflight(launcher, drift):
    run, _, root, env, _ = launcher
    assert run().returncode == 0
    repo = Path(env["AF_REPO_ROOT"])
    selection = repo / "configs/prefit_public_fit_handoff_v1.json"
    updates = {}
    if drift == "commit":
        updates["REVISION"] = "different"
    elif drift == "dirty":
        updates["DIRTY"] = "1"
    else:
        changed = root / "changed-selection.json"
        changed.write_text(selection.read_text() + "\n")
        selection = changed
    result = subprocess.run(
        ["bash", str(repo / "scripts/hpc/run_prefit_fit_handoff_aces.sh")],
        env={**env, **updates, "AF_SELECTION": str(selection)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "changed after submission" in result.stderr
    assert not (root / "runtime/preflight-local.log").exists()
    assert not (root / "fit/started.json").exists()


def test_fast_job_can_start_from_identity_before_final_job_manifest(launcher):
    run, _, root, env, bins = launcher
    assert run().returncode == 0
    (root / "submission_manifest.json").unlink()
    python = bins / "reached-preflight"
    python.write_text("#!/bin/sh\necho 'preflight reached'\nexit 3\n")
    python.chmod(0o755)
    repo = Path(env["AF_REPO_ROOT"])
    result = subprocess.run(
        ["bash", str(repo / "scripts/hpc/run_prefit_fit_handoff_aces.sh")],
        env={
            **env,
            "AF_PYTHON": str(python),
            "AF_SELECTION": str(repo / "configs/prefit_public_fit_handoff_v1.json"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "preflight reached" in result.stderr
