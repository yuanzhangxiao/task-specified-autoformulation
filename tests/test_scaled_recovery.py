"""Recovery must preserve budgets, isolate bad checkpoints, and report startup."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("casadi")

REPO = Path(__file__).resolve().parents[1]


def module(name):
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / (name + ".py")
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def prepared(tmp_path):
    smoke, io = module("smoke_scaled_recovery"), module("scaled_recovery_io")
    source, local = tmp_path / "source", tmp_path / "local"
    smoke.control(source)
    frozen = io.prepare(source, local, REPO)
    return source, local, frozen


def test_source_identity_and_zero_prior_calls_required(prepared, tmp_path):
    source, local, frozen = prepared
    io = module("scaled_recovery_io")
    assert io.verify(local, REPO) == frozen
    source_before = {str(p): io.sha(p) for p in source.rglob("*") if p.is_file()}
    assert io.prepare(source, local, REPO) == frozen
    assert source_before == {
        str(p): io.sha(p) for p in source.rglob("*") if p.is_file()
    }
    (source / "results/task_000/screen/calls").mkdir()
    with pytest.raises(ValueError, match="not begun"):
        io.prepare(source, tmp_path / "invalid", REPO)
    problem = io.read(local / "problem.json")
    problem["start"]["r"] = 3
    io.write(local / "problem.json", problem)
    with pytest.raises(ValueError, match="frozen asset"):
        io.verify(local, REPO)


def test_native_recovery_smoke_and_immutable_resume(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/smoke_scaled_recovery.py"),
            "--output",
            str(tmp_path),
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO / "src"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-5000:]
    assert json.loads(result.stdout.splitlines()[-1])["pass"]


def test_corrupt_saved_array_does_not_block_ordinary(prepared, tmp_path):
    source, _, _ = prepared
    io, runner = module("scaled_recovery_io"), module("run_scaled_recovery")
    point = io.read(source / "results/task_000/initializer/native/checkpoint.json")
    (source / "results/task_000/initializer/native" / point["array"]).write_bytes(
        b"broken"
    )
    output = tmp_path / "corrupt"
    io.prepare(source, output, REPO)
    result = runner.run(output, tmp_path / "durable", 0)
    assert result["strict"]
    assert result["stages"]["screen-ordinary"]["valid_calls"] == 1
    assert result["stages"]["screen-checkpoint"]["status"] == "startup_failed"
    assert "array digest" in result["stages"]["screen-checkpoint"]["message"]


class NoopPublisher:
    def flush(self):
        pass

    def tick(self):
        pass


def test_interrupted_stage_retains_best_without_new_worker(tmp_path, monkeypatch):
    runner, io = module("run_scaled_recovery"), module("scaled_recovery_io")
    root = tmp_path / "refinement"
    io.write(
        root / "started.json", {"identity": "i", "stage": "refinement", "task_index": 0}
    )
    best = {"parameters": {"x": 1}, "cost": 0.1}
    io.write(root / "calls/best_evaluated.json", best)
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("must not relaunch")
    )
    result = runner.stage(
        tmp_path, tmp_path, 0, "refinement", 1, {"identity": "i"}, NoopPublisher()
    )
    assert result["status"] == "interrupted" and result["best"] == best


@pytest.mark.parametrize(
    "ready,expected", [(False, "startup_timeout"), (True, "numerical_timeout")]
)
def test_hard_timeouts_distinguish_startup_and_numerics(
    tmp_path, monkeypatch, ready, expected
):
    runner = module("run_scaled_recovery")
    fake = tmp_path / "fake.py"
    fake.write_text(
        "import time,json,pathlib\n"
        + (
            f"p=pathlib.Path({str(tmp_path / 'screen-ordinary/ready.json')!r}); "
            "p.write_text(json.dumps({'identity':'i',"
            "'monotonic':time.monotonic()}))\n"
            if ready
            else ""
        )
        + "time.sleep(10)\n"
    )
    monkeypatch.setattr(runner, "__file__", str(fake))
    result = runner.stage(
        tmp_path,
        tmp_path,
        0,
        "screen-ordinary",
        0.15,
        {"identity": "i", "startup_seconds": 0.2},
        NoopPublisher(),
    )
    assert result["status"] == expected
    assert result["worker_wall_seconds"] < 2
    assert not (tmp_path / "screen-ordinary/calls").exists()


def test_best_screen_survives_worse_or_failed_refinement():
    runner = module("run_scaled_recovery")
    best = {"cost": 0.1, "parameters": {"x": 1}}
    assert runner.select(
        {"screen": {"best": best}, "refinement": {"best": {"cost": 2}}}
    ) == (best, "screen")
    assert runner.select({"screen": {"best": best}, "refinement": {"best": None}}) == (
        best,
        "screen",
    )


def test_publisher_timeout_is_bounded(tmp_path, monkeypatch):
    runner = module("run_scaled_recovery")
    fake = tmp_path / "stuck.py"
    fake.write_text("import time; time.sleep(10)\n")
    monkeypatch.setattr(runner, "__file__", str(fake))
    publisher = runner.Publisher(tmp_path / "local", tmp_path / "durable", 0.1)
    with pytest.raises(RuntimeError, match="publication"):
        publisher.flush()


def test_checkpoint_q_is_checked_without_reading_latent_nodes(prepared, monkeypatch):
    _, output, frozen = prepared
    io = module("scaled_recovery_io")
    import numpy as np

    from autoformalism.rebuttal.scaled_recovery_worker import checkpoint_parameters

    original = np.load

    class OnlyQ:
        def __init__(self, path, **kw):
            self.arrays = original(path, **kw)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.arrays.close()

        def __getitem__(self, key):
            assert key == "q"
            return self.arrays[key]

    monkeypatch.setattr(np, "load", OnlyQ)
    assert checkpoint_parameters(output, frozen, 0, io.read(output / "common.json"))


def test_no_screened_point_skips_refinement_and_replay_workers(
    prepared, tmp_path, monkeypatch
):
    _, output, _ = prepared
    runner = module("run_scaled_recovery")
    called = []

    def failed_stage(output, task, index, name, seconds, frozen, publisher):
        called.append(name)
        return {"status": "startup_timeout", "best": None, "calls": 0}

    monkeypatch.setattr(runner, "stage", failed_stage)
    result = runner.run(output, tmp_path / "durable", 0)
    assert called == ["screen-ordinary", "screen-checkpoint"]
    assert result["status"] == "execution_failed"
    assert result["selected_source"] is None
    assert result["stages"]["refinement"]["calls"] == 0


def test_launcher_submits_only_preparation_pair_and_summary(tmp_path):
    repo, binpath, output = [tmp_path / p for p in ("repo", "bin", "output")]
    repo.mkdir()
    binpath.mkdir()
    for name, body in {
        "git": 'if [[ "$1" == rev-parse ]]; then echo pinned; fi',
        "sbatch": 'echo "$*" >> "$JOB_LOG"; echo 12345',
    }.items():
        path = binpath / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binpath) + ":" + os.environ["PATH"],
        "AF_REPO_ROOT": str(repo),
        "AF_PYTHON": sys.executable,
        "AF_OUTPUT_ROOT": str(output),
        "JOB_LOG": str(tmp_path / "jobs"),
    }
    command = ["bash", str(REPO / "scripts/hpc/submit_scaled_recovery_delta.sh")]
    for _ in range(2):
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=20
        )
        assert result.returncode == 0, result.stderr
    jobs = (tmp_path / "jobs").read_text().splitlines()
    assert len(jobs) == 3
    assert "--array=0-1%2" in jobs[1] and "afterok:12345" in jobs[1]
    manifest = json.loads((output / "submission.json").read_text())
    assert manifest["submission_complete"] and manifest["collocation_reruns"] == 0
    # A failed submission with an intent but no manifest cannot silently retry.
    (output / "submission.json").unlink()
    result = subprocess.run(command, env=env, capture_output=True, timeout=20)
    assert (
        result.returncode != 0
        and len((tmp_path / "jobs").read_text().splitlines()) == 3
    )


def test_old_numerical_source_drift_is_rejected(tmp_path, monkeypatch):
    import hashlib
    import io
    import tarfile

    staging, helper = (
        module("stage_scaled_recovery_runtime"),
        module("scaled_recovery_io"),
    )
    files = {
        "src/autoformalism/fitting/probe.py": b"frozen numerical code\n",
        "scripts/run_scaled_alternating.py": b"frozen runner\n",
    }
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        for name, data in files.items():
            header = tarfile.TarInfo(name)
            header.size = len(data)
            bundle.addfile(header, io.BytesIO(data))
    monkeypatch.setattr(
        staging.subprocess, "check_output", lambda *a, **kw: archive.getvalue()
    )
    helper.write(tmp_path / "source/submission.json", {"commit": "a" * 40})
    helper.write(
        tmp_path / "source/freeze.json",
        {
            "code": helper.digest(
                {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
            )
        },
    )
    code = tmp_path / "repo/src/autoformalism/fitting/probe.py"
    code.parent.mkdir(parents=True)
    code.write_bytes(files["src/autoformalism/fitting/probe.py"])
    assert staging.audit_numerical_sources(tmp_path / "repo", tmp_path / "source")
    code.write_text("modified")
    with pytest.raises(ValueError, match="numerical source changed"):
        staging.audit_numerical_sources(tmp_path / "repo", tmp_path / "source")


def test_local_environment_uses_no_shared_python_paths(tmp_path):
    staging = module("stage_scaled_recovery_runtime")
    env = staging.local_environment(tmp_path)
    assert env["PYTHONHOME"] == str(tmp_path / "runtime/python")
    assert all(
        Path(p).is_relative_to(tmp_path) for p in env["PYTHONPATH"].split(os.pathsep)
    )
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONPATH"].index("extras") < env["PYTHONPATH"].index("site")


def installed_wheel(root, name, version, files, requires=()):
    """Make a real installed-distribution inventory for packaging tests."""
    dist = root / f"{name.replace('-', '_')}-{version}.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        + "".join(f"Requires-Dist: {value}\n" for value in requires)
    )
    for file, data in files.items():
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    records = [*files, f"{dist.name}/METADATA", f"{dist.name}/RECORD"]
    (dist / "RECORD").write_text("".join(f"{file},,\n" for file in records))
    return dist


def test_dependency_copy_preserves_precedence_native_files_and_imports(tmp_path):
    deps, staging = (
        module("scaled_runtime_dependencies"),
        module("stage_scaled_recovery_runtime"),
    )
    extras, site = tmp_path / "extras", tmp_path / "site"
    installed_wheel(site, "probe", "1.0", {"probe.py": "VALUE = 1\n"})
    installed_wheel(
        extras,
        "probe",
        "2.0",
        {"probe.py": "from child import VALUE\n"},
        ["child>=1", 'absent; extra == "gpu"', 'absent; python_version < "3"'],
    )
    installed_wheel(
        site,
        "child",
        "1.0",
        {
            "child.py": "VALUE = 7\n",
            "child.libs/library.so": "native-support",
            "child_data/resource.txt": "resource",
            "redirect.pth": "unwanted side effect",
            "__pycache__/child.pyc": "unneeded",
        },
    )
    installed_wheel(
        site, "unrelated-large-package", "1.0", {"huge.bin": "never copied"}
    )
    selection = deps.select([extras, site], ("probe",))
    assert set(selection["packages"]) == {"probe", "child"}
    assert selection["packages"]["probe"]["version"] == "2.0"
    targets = []
    for index, root in enumerate((extras, site)):
        files = selection["files"][str(root)]
        inventory = tmp_path / f"files-{index}.txt"
        inventory.write_text("\n".join(files) + "\n")
        target = tmp_path / f"relocated-{index}"
        target.mkdir()
        staging.command(
            [
                "rsync",
                "-aL",
                "--files-from=" + str(inventory),
                str(root) + "/",
                str(target) + "/",
            ],
            tmp_path,
            f"copy-{index}",
        )
        targets.append(target)
    assert not (targets[1] / "huge.bin").exists()
    assert not (targets[1] / "redirect.pth").exists()
    assert (targets[1] / "child.libs/library.so").read_text() == "native-support"
    result = subprocess.run(
        [sys.executable, "-S", "-c", "import probe; print(probe.VALUE)"],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, targets))},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0 and result.stdout.strip() == "7", result.stderr


def test_dependency_selection_fails_closed_on_missing_or_conflicting_metadata(tmp_path):
    deps = module("scaled_runtime_dependencies")
    dist = installed_wheel(tmp_path, "probe", "1.0", {"probe.py": ""}, ["missing"])
    with pytest.raises(ValueError, match="missing installed dependency"):
        deps.select([tmp_path], ("probe",))
    with pytest.raises(ValueError, match="does not satisfy"):
        deps.select([tmp_path], ("probe>=2",))
    (dist / "RECORD").unlink()
    with pytest.raises(ValueError, match="refusing whole-site copy"):
        deps.select([tmp_path], ("probe",))


def test_preparation_timeout_publishes_stage_and_summary_without_freeze(tmp_path):
    staging, io = module("stage_scaled_recovery_runtime"), module("scaled_recovery_io")
    work, output = tmp_path / "work", tmp_path / "output"
    logs = work / "preparation"
    logs.mkdir(parents=True)
    with pytest.raises(subprocess.TimeoutExpired):
        staging.command(
            [
                sys.executable,
                "-u",
                "-c",
                "print('copying child.libs'); import time; time.sleep(10)",
            ],
            logs,
            "copy_site",
            seconds=0.2,
        )
    staging.publish_preparation(output, work, 124)
    assert io.read(output / "preparation/copy_site.json")["status"] == "timeout"
    assert "copying child.libs" in (output / "preparation/copy_site.log").read_text()
    report = io.report(output)
    assert report["status"] == "preparation_interrupted"
    assert report["preparation"]["last_stage"]["stage"] == "copy_site"
    assert all(row["status"] == "blocked_preparation" for row in report["rows"])
    assert "not a numerical fitting result" in (output / "SUMMARY.md").read_text()
    assert not (output / "freeze.json").exists()


def test_failed_preparation_and_absent_status_are_reportable(tmp_path):
    io = module("scaled_recovery_io")
    assert io.report(tmp_path)["status"] == "preparation_not_ready"
    io.write(
        tmp_path / "preparation/result.json",
        {"status": "failed", "message": "dependency missing"},
    )
    assert io.report(tmp_path)["preparation"]["message"] == "dependency missing"
    # A malformed freeze is still an error once preparation is available.
    io.write(tmp_path / "freeze.json", {"identity": "wrong"})
    with pytest.raises(ValueError, match="content identity"):
        io.report(tmp_path)


def test_build_cli_preserves_exception_before_exit_publication(tmp_path):
    io = module("scaled_recovery_io")
    base = [
        sys.executable,
        "-S",
        str(REPO / "scripts/stage_scaled_recovery_runtime.py"),
    ]
    work, output = tmp_path / "work", tmp_path / "output"
    result = subprocess.run(
        [
            *base,
            "build",
            "--repo",
            str(REPO),
            "--source",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--work",
            str(work),
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0
    published = subprocess.run(
        [
            *base,
            "publish-preparation",
            "--output",
            str(output),
            "--work",
            str(work),
            "--exit-code",
            str(result.returncode),
        ],
        capture_output=True,
        timeout=10,
    )
    assert published.returncode == 0, published.stderr
    report = io.report(output)
    assert report["status"] == "preparation_failed"
    assert report["preparation"]["error_type"] == "FileNotFoundError"
    assert report["preparation"]["last_stage"]["stage"] == "numerical_source_audit"
