"""Recovery never chooses retries from scores or silently resets native budgets."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/recover_fitter_outage.py"
SPEC = importlib.util.spec_from_file_location("outage_recovery", SCRIPT)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def fixture(tmp_path, campaign="piecewise", cases=2):
    source = tmp_path / "original"
    frozen = {
        "plan": {
            "protocol": "fitter-rate-stopping-1",
            "guard_seconds": 120,
            "replay_seconds": 90,
        },
        "runtime": {},
        "launcher": "test",
    }
    if campaign == "piecewise":
        frozen.update(
            cases=[{"index": i, "label": f"case{i}"} for i in range(cases)], assets={}
        )
        key = "identity"
    else:
        frozen.update(
            tasks=[
                {"name": f"pair_{i}", "source_task": f"fit_{i}", "kind": "pair"}
                for i in range(cases)
            ],
            source_manifest={},
        )
        key = "freeze_sha256"
    frozen[key] = recovery.content_hash(frozen)
    recovery.write(source / "freeze.json", frozen)
    args = argparse.Namespace(
        output=tmp_path / "recovered",
        source=source,
        campaign=campaign,
        legacy_repo=tmp_path / "legacy",
        python=Path(sys.executable),
        dependencies=tmp_path / "deps",
    )
    return args, frozen


def result(args, frozen, index, arm, status):
    root = args.source / f"results/case_{index:03d}" / arm
    mesh, method = arm.split("_", 1)
    recovery.write(
        root / "result.json",
        {
            "identity": recovery.content_hash(
                [frozen["identity"], index, int(mesh[-1]), method]
            ),
            "status": status,
            "fit": {"training": {"normalized_mse": 1e6}},
        },
    )


def test_standard_library_bootstrap_supports_old_system_python():
    ast.parse(SCRIPT.read_text(), feature_version=(3, 6))
    parsed = ast.parse(SCRIPT.read_text())
    top_imports = [n for n in parsed.body if isinstance(n, ast.ImportFrom)]
    assert all(n.module == "pathlib" for n in top_imports)


def test_piecewise_recovers_only_unavailable_and_preserves_bad_results(tmp_path):
    args, frozen = fixture(tmp_path)
    arms = recovery.ARMS["piecewise"]
    result(args, frozen, 0, arms[0], "fit_failed")
    result(args, frozen, 0, arms[1], "replay_unverified")
    root = args.source / "results/case_000"
    recovery.write(root / arms[2] / "fit.json", {"parameters": {"k": 2}})
    recovery.write(root / arms[3] / "initializer.json", {"parameters": {"k": 1}})
    recovery.write(
        root / arms[3] / "poll_checkpoint.json",
        {"elapsed": 97, "history": [{"cost": 123}], "iteration": 10},
    )
    original = recovery.inventory(args.source)
    plan = recovery.prepare(args)
    assert [a["mode"] for a in plan["tasks"][0]["actions"]] == [
        "preserve_terminal",
        "preserve_terminal",
        "reuse_fit",
        "resume_poll",
    ]
    checkpoint = f"results/case_000/{arms[3]}/poll_checkpoint.json"
    assert (
        recovery.digest(args.output / "snapshot" / checkpoint) == original[checkpoint]
    )
    assert recovery.inventory(args.source) == original
    assert recovery.prepare(args) == plan
    report = recovery.summarize(args.output)
    assert len(report["rows"]) == 8
    assert report["counts"] == {
        "fit_failed": 1,
        "replay_unverified": 1,
        "unavailable": 6,
    }


def test_native_restart_is_explicit_and_reuses_finished_initializer(tmp_path):
    args, _ = fixture(tmp_path)
    path = "results/case_000/mesh1_branch_sensitivity"
    for name in (
        "fit_started.json",
        "initializer_started.json",
        "initializer.json",
        "collocation/progress.json",
    ):
        recovery.write(args.source / path / name, {"tag": name})
    plan = recovery.prepare(args)
    assert plan["tasks"][0]["actions"][0]["mode"] == "restart_interrupted_native"
    assert (args.output / "snapshot" / path / "initializer.json").exists()
    assert not (args.output / "snapshot" / path / "fit_started.json").exists()
    assert (args.source / path / "fit_started.json").exists()
    assert path + "/fit_started.json" in plan["excluded_from_attempt"]
    # A second invocation must not delete checkpoints created by this attempt.
    recovery.write(args.output / "snapshot" / path / "fit_started.json", {"new": True})
    recovery.prepare(args)
    assert recovery.read(args.output / "snapshot" / path / "fit_started.json") == {
        "new": True
    }
    recovery.write(args.output / "smoke/passed.json", {"identity": plan["identity"]})
    recovery.write(args.output / "attempts/task_000/started.json", {})
    with pytest.raises(ValueError, match="native fit was interrupted"):
        recovery.run(args.output, index=0)


def test_stopping_preserves_terminal_and_reuses_saved_fit(tmp_path):
    args, frozen = fixture(tmp_path, "stopping")
    terminal = {
        "identity": recovery.content_hash(
            [frozen["freeze_sha256"], frozen["tasks"][0]]
        ),
        "status": "pair_unverified",
        "arms": {
            "rates_default": {"status": "fit_failed"},
            "rates_no_ftol": {"status": "replay_unverified"},
        },
    }
    recovery.write(args.source / "results/pair_0/result.json", terminal)
    root = args.source / "results/pair_1"
    recovery.write(root / "result.json", {"status": "timeout"})
    recovery.write(root / "rates_default/fit.json", {"fit": {"parameters": {"k": 1}}})
    recovery.write(root / "rates_no_ftol/fit_started.json", {"budget_seconds": 123})
    replay = root / "rates_default/replays/Radau_tight/train"
    recovery.write(replay / "trajectory-0.json", {"status": "complete"})
    recovery.write(
        replay / "trajectory-1.json",
        {"status": "failed", "message": "replay wall-clock limit reached"},
    )
    recovery.write(
        replay / "trajectory-2.json", {"status": "failed", "message": "nonfinite RHS"}
    )
    recovery.write(replay / "result.json", {"status": "incomplete"})
    original = recovery.inventory(args.source)
    plan = recovery.prepare(args)
    assert not plan["tasks"][0]["selected"]
    assert [a["mode"] for a in plan["tasks"][1]["actions"]] == [
        "reuse_fit",
        "restart_interrupted_native",
    ]
    assert recovery.inventory(args.source) == original
    copied = (
        args.output / "snapshot/results/pair_1/rates_default/replays/Radau_tight/train"
    )
    assert (copied / "trajectory-0.json").exists()
    assert not (copied / "trajectory-1.json").exists()
    assert (copied / "trajectory-2.json").exists()
    assert not (copied / "result.json").exists()
    assert not (args.output / "snapshot/results/pair_1/result.json").exists()


@pytest.mark.parametrize("fault", ["traversal", "symlink", "hash", "overlap"])
def test_prepare_fails_closed(tmp_path, fault):
    args, frozen = fixture(tmp_path)
    if fault == "overlap":
        args.output = args.source / "retry"
    elif fault == "symlink":
        (args.source / "results").symlink_to(tmp_path)
    else:
        frozen["assets"] = {
            "../escaped" if fault == "traversal" else "asset.json": "wrong"
        }
        (args.source / "asset.json").write_text("{}")
        frozen["identity"] = recovery.content_hash(
            {k: v for k, v in frozen.items() if k != "identity"}
        )
        recovery.write(args.source / "freeze.json", frozen)
    with pytest.raises(ValueError):
        recovery.prepare(args)


def test_partial_prepare_resumes_and_refuses_changed_source(tmp_path, monkeypatch):
    args, frozen = fixture(tmp_path)
    result(args, frozen, 0, recovery.ARMS["piecewise"][0], "complete")
    copy = recovery.shutil.copyfile
    count = 0

    def fail_once(*a, **kw):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("simulated filesystem outage")
        return copy(*a, **kw)

    monkeypatch.setattr(recovery.shutil, "copyfile", fail_once)
    with pytest.raises(OSError, match="outage"):
        recovery.prepare(args)
    assert not (args.output / "prepared.json").exists()
    monkeypatch.setattr(recovery.shutil, "copyfile", copy)
    plan = recovery.prepare(args)
    assert recovery.load_plan(args.output) == plan
    retained = (
        args.output / "snapshot/results/case_000/mesh1_branch_sensitivity/result.json"
    )
    bad = recovery.read(retained)
    bad["status"] = "fit_failed"
    recovery.write(retained, bad)
    with pytest.raises(ValueError, match="retained"):
        recovery.summarize(args.output)


def test_poll_missing_initializer_is_not_silently_restarted(tmp_path):
    args, _ = fixture(tmp_path)
    recovery.write(
        args.source / "results/case_000/mesh1_directional_poll/poll_checkpoint.json", {}
    )
    with pytest.raises(ValueError, match="no saved initializer"):
        recovery.prepare(args)


def test_changed_copied_freeze_is_rejected_before_worker_or_summary(tmp_path):
    args, _ = fixture(tmp_path)
    recovery.prepare(args)
    recovery.write(args.output / "snapshot/freeze.json", {"changed": True})
    with pytest.raises(ValueError, match="original freeze changed"):
        recovery.summarize(args.output)


def test_stopping_report_separates_initializers_and_stress(tmp_path):
    args, frozen = fixture(tmp_path, "stopping", cases=3)
    for task, name in zip(
        frozen["tasks"],
        [
            "fit_moderate_collocation_sensitivity",
            "fit_moderate_alternating_collocation_sensitivity",
            "fit_stress_joint_collocation_sensitivity",
        ],
        strict=True,
    ):
        task["source_task"] = name
    frozen["freeze_sha256"] = recovery.content_hash(
        {k: v for k, v in frozen.items() if k != "freeze_sha256"}
    )
    recovery.write(args.source / "freeze.json", frozen)
    recovery.prepare(args)
    groups = recovery.summarize(args.output)["by_arm"]
    assert len(groups) == 6
    assert sum(g["total"] for g in groups) == 6
    assert {(g["initializer"], g["population"]) for g in groups} == {
        ("C+S", "ordinary"),
        ("A+S", "ordinary"),
        ("J+S", "stress"),
    }


def test_supervisor_bounds_hanging_startup_and_preserves_diagnostics(tmp_path):
    start = time.monotonic()
    record = recovery.supervise(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path,
        seconds=0.1,
        grace=0.1,
    )
    assert time.monotonic() - start < 5
    assert record["status"] == "timeout" and record["child_reaped"]
    stages = [
        json.loads(line)["stage"]
        for line in (tmp_path / "stages.jsonl").read_text().splitlines()
    ]
    assert stages == ["spawn", "spawned", "supervisor_finished"]


def test_supervisor_survives_worker_failure(tmp_path):
    record = recovery.supervise(
        [sys.executable, "-c", "raise SystemExit(17)"], tmp_path, 5
    )
    assert record == {"status": "worker_failed", "exit_code": 17}


def test_teardown_remains_bounded_when_child_cannot_be_reaped(tmp_path, monkeypatch):
    class Unreapable:
        pid = 1234567

        def wait(self, timeout):
            assert timeout <= 1
            raise subprocess.TimeoutExpired("blocked I/O", timeout)

        def poll(self):
            return None

    signals = []
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *a, **kw: Unreapable())
    monkeypatch.setattr(recovery.os, "killpg", lambda pid, sig: signals.append(sig))
    record = recovery.supervise(["unreapable"], tmp_path, 1, grace=0.1)
    assert record["status"] == "timeout" and not record["child_reaped"]
    assert len(signals) == 2
    assert recovery.read(tmp_path / "supervisor.json") == record


def test_stopping_summary_checks_replay_array_hash(tmp_path):
    args, frozen = fixture(tmp_path, "stopping", cases=1)
    root = args.source / "results/pair_0"
    array = root / "rates_default/replays/Radau_tight/train/trajectory-0.npz"
    array.parent.mkdir(parents=True)
    array.write_bytes(b"frozen array")
    recovery.write(
        root / "result.json",
        {
            "identity": recovery.content_hash(
                [frozen["freeze_sha256"], frozen["tasks"][0]]
            ),
            "status": "complete",
            "arms": {
                "rates_default": {
                    "status": "complete",
                    "replays": {
                        "Radau_tight/train": {
                            "trajectories": [
                                {
                                    "status": "complete",
                                    "array": array.name,
                                    "array_sha256": recovery.digest(array),
                                }
                            ]
                        }
                    },
                }
            },
        },
    )
    recovery.prepare(args)
    (args.output / "snapshot" / array.relative_to(args.source)).write_bytes(
        b"corrupted"
    )
    with pytest.raises(ValueError, match="replay array"):
        recovery.summarize(args.output)


def test_smoke_gate_and_task_lock(tmp_path):
    args, _ = fixture(tmp_path)
    recovery.prepare(args)
    with pytest.raises(ValueError, match="smoke"):
        recovery.run(args.output, index=0)
    with (
        recovery.locked(tmp_path / "test.lock"),
        pytest.raises(BlockingIOError),
        recovery.locked(tmp_path / "test.lock"),
    ):
        pass


def test_submission_gates_sparse_array_and_prevents_duplicate_jobs(tmp_path):
    args, frozen = fixture(tmp_path)
    for arm in recovery.ARMS["piecewise"]:
        result(args, frozen, 1, arm, "fit_failed")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "submissions"
    for name, code in {
        "git": "#!/bin/sh\nexit 0\n",
        "sbatch": (
            '#!/bin/sh\necho "$*" >> "$SUBMIT_LOG"\n'
            "awk 'END {print NR}' \"$SUBMIT_LOG\"\n"
        ),
    }.items():
        path = bin_dir / name
        path.write_text(code)
        path.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        SUBMIT_LOG=str(log),
        AF_BOOTSTRAP_PYTHON=sys.executable,
        AF_NUMERICAL_PYTHON=sys.executable,
        AF_CASADI_ROOT=str(args.dependencies),
        AF_ORIGINAL_OUTPUT=str(args.source),
        AF_LEGACY_REPO=str(args.legacy_repo),
        AF_RECOVERY_OUTPUT=str(args.output),
    )
    command = [
        "bash",
        str(SCRIPT.parent / "hpc/submit_fitter_outage_recovery_delta.sh"),
        "piecewise",
    ]
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    lines = log.read_text().splitlines()
    assert len(lines) == 3
    assert "--dependency=afterok:1" in lines[1] and "--array=0%1" in lines[1]
    assert "--dependency=afterany:2" in lines[2]
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    assert log.read_text().splitlines() == lines
