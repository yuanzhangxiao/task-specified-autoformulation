"""Bounded ACES chain and idempotence using an offline scheduler fixture."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(params=("no_job_config", "construction_job_config"))
def launcher(tmp_path, monkeypatch, request):
    if not all(shutil.which(c) for c in ("bash", "jq")):
        pytest.skip("shell tools unavailable")
    repo = Path(__file__).resolve().parents[1]
    # ACES preflight inherits this selector through sbatch --export=ALL.
    # Exercise that environment even when pytest runs on a clean workstation.
    if request.param == "construction_job_config":
        monkeypatch.setenv(
            "AF_CONFIG", str(repo / "configs/prefit_construction_audit_v1.json")
        )
    else:
        monkeypatch.delenv("AF_CONFIG", raising=False)
    bins = tmp_path / "bin"
    bins.mkdir()
    root = tmp_path / "output"
    root.mkdir()
    image = tmp_path / "image.sif"
    image.write_bytes(b"synthetic fixture")
    log = tmp_path / "scheduler.json"
    scripts = {
        "module": "#!/bin/sh\nexit 0\n",
        "git": '#!/bin/sh\ncase "$*" in *status*) exit 0 ;; *) echo abc123 ;; esac\n',
        "squeue": '#!/bin/sh\n[ "${QUEUED:-0}" = 0 ] || echo 102\nexit 0\n',
        "sacct": '#!/bin/sh\necho "${ACCOUNTING_STATE:-COMPLETED}"\n',
        "fakepython": f"""#!{sys.executable}
import json, os, pathlib, sys
root = pathlib.Path({str(root)!r})
if 'freeze' in sys.argv:
    path = pathlib.Path(sys.argv[sys.argv.index('--config')+1])
    config = json.loads(path.read_text())
    plan = {{'artifact_sha256':'frozen', 'config': config}}
    (root/'plan.json').write_text(json.dumps(plan))
elif 'summary' in sys.argv:
    print(json.dumps({{'status':os.environ.get('SUMMARY_STATUS','partial'),
        'construction_complete':os.environ.get('BUILT','0') == '1'}}))
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
    # Only fixture settings and explicit run() overrides may select a workflow.
    env = dict(
        {key: value for key, value in os.environ.items() if not key.startswith("AF_")},
        PATH=str(bins) + os.pathsep + os.environ["PATH"],
        AF_REPO_ROOT=str(repo),
        AF_PYTHON=str(bins / "fakepython"),
        AF_OUTPUT_ROOT=str(root),
        AF_VLLM_IMAGE=str(image),
        AF_HF_HOME=str(tmp_path / "hf"),
        AF_RESUME="0",
    )

    def run(**overrides):
        return subprocess.run(
            ["bash", str(repo / "scripts/hpc/submit_prefit_construction_aces.sh")],
            env={**env, **overrides},
            capture_output=True,
            text=True,
        )

    return run, log, root


def test_cpu_gpu_cpu_chain_and_repeated_invocation(launcher):
    run, log, root = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    calls = json.loads(log.read_text())
    assert [c[-1] for c in calls] == ["prepare", "construct", "fit"]
    assert "--gres=gpu:h100:1" in calls[1]
    assert "--time=04:00:00" in calls[1]
    assert "--dependency=afterok:101" in calls[1]
    assert "--dependency=afterany:102" in calls[2]
    assert "--partition=cpu" in calls[0] and "--partition=cpu" in calls[2]
    assert run().returncode == 0
    assert json.loads(log.read_text()) == calls
    assert (
        json.loads((root / "submission_manifest.json").read_text())["fit_job"] == "103"
    )


@pytest.mark.parametrize("stage,count", [("prepare", 1), ("construct", 2), ("fit", 3)])
def test_ambiguous_submission_preserves_intent_and_never_duplicates(
    launcher, stage, count
):
    run, log, root = launcher
    assert run(FAIL_STAGE=stage).returncode != 0
    assert len(json.loads(log.read_text())) == count
    assert (root / "submissions/initial").is_dir()
    assert not (root / "submission_manifest.json").exists()
    assert run().returncode != 0
    assert len(json.loads(log.read_text())) == count


@pytest.mark.parametrize("built,additional", [("0", 2), ("1", 1)])
def test_resume_only_needed_stages(launcher, built, additional):
    run, log, _ = launcher
    assert run().returncode == 0
    result = run(AF_RESUME="1", BUILT=built)
    assert result.returncode == 0, result.stderr
    calls = json.loads(log.read_text())
    assert len(calls) == 3 + additional
    assert calls[-1][-1] == "fit"
    if built == "1":
        assert not any(c.startswith("--dependency") for c in calls[-1])


@pytest.mark.parametrize(
    "overrides", [{"QUEUED": "1"}, {"ACCOUNTING_STATE": "UNKNOWN"}]
)
def test_resume_refuses_active_or_unknown_jobs(launcher, overrides):
    run, log, _ = launcher
    assert run().returncode == 0
    assert run(AF_RESUME="1", **overrides).returncode != 0
    assert len(json.loads(log.read_text())) == 3


def test_complete_campaign_requires_no_new_allocation(launcher):
    run, log, _ = launcher
    assert run().returncode == 0
    assert run(AF_RESUME="1", SUMMARY_STATUS="complete").returncode == 0
    assert len(json.loads(log.read_text())) == 3


@pytest.mark.parametrize("built,additional", [("0", 2), ("1", 1)])
def test_construction_only_schedules_audit_and_never_fitting(
    launcher, built, additional
):
    run, log, root = launcher
    config = str(
        Path(__file__).resolve().parents[1]
        / "configs/prefit_construction_audit_v1.json"
    )
    result = run(AF_CONFIG=config)
    assert result.returncode == 0, result.stderr
    calls = json.loads(log.read_text())
    assert [c[-1] for c in calls] == ["prepare", "construct", "audit"]
    assert "--dependency=afterany:102" in calls[-1]
    assert "--time=00:30:00" in calls[-1]
    manifest = json.loads((root / "submission_manifest.json").read_text())
    assert manifest["audit_job"] == "103" and "fit_job" not in manifest
    assert run(AF_CONFIG=config).returncode == 0
    assert json.loads(log.read_text()) == calls
    result = run(AF_RESUME="1", AF_CONFIG=config, BUILT=built)
    assert result.returncode == 0, result.stderr
    calls = json.loads(log.read_text())
    assert len(calls) == 3 + additional
    assert calls[-1][-1] == "audit" and all(c[-1] != "fit" for c in calls)
    result = run(AF_RESUME="1", AF_CONFIG=config, SUMMARY_STATUS="complete")
    assert result.returncode == 0
    assert len(json.loads(log.read_text())) == len(calls)
