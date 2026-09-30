"""No duplicate allocations or hidden search workers in the operational wrappers."""

from types import SimpleNamespace as NS

import pytest

from autoformalism.fitting import public_fitting as public
from scripts import submit_component_closeout as closeout
from scripts import submit_phase_c_baseline as pilot


def setup(monkeypatch, tmp_path):
    plan = {
        "artifact_sha256": "plan",
        "config": {
            "platform": "aces-h100x1",
            "rounds": 15,
            "wall_seconds": 21600,
        },
        "tasks": [{"task_id": f"task{i}"} for i in range(32)],
    }
    python = tmp_path / "python"
    python.touch()
    for key, value in {
        "AF_PYTHON": str(python),
        "AF_COMMIT": "a" * 40,
        "AF_JETSTREAM_API_KEY": "test-only",
        "AF_VLLM_IMAGE": "image",
        "AF_HF_HOME": "cache",
        "AF_COMPUTE_CACHE_ROOT": "compiler",
        "AF_IPC_TMP_ROOT": "ipc",
        "USER": "tester",
    }.items():
        monkeypatch.setenv(key, value)
    calls = []

    def submit_job(directory, stage, opts, worker, command, index):
        assert not (directory / f"{stage}.intent.json").exists()
        calls.append((stage, opts))
        public._write(
            directory / f"{stage}.intent.json",
            {"argv": ["sbatch", "--parsable", *opts, str(worker), command, str(index)]},
        )
        job = str(100 + len(calls))
        (directory / f"{stage}.id").write_text(job)
        return job

    support = NS(source_commit=lambda repo: "a" * 40, submit_job=submit_job)
    runtime = NS(
        campaign=NS(verify=lambda root: plan),
        io=NS(
            REPO=tmp_path / "frozen",
            require_open=lambda root: None,
            read_round=lambda *args: {"recorded": True},
        ),
        checked_authorization=lambda *args: None,
        support=lambda name: support,
        public=public,
        account=lambda *args: "cpu-account",
        partition=lambda *args: "cpu",
    )
    monkeypatch.setattr(closeout.subprocess, "check_output", lambda *args, **kw: "")
    monkeypatch.setattr(pilot.baseline, "verify", lambda root: plan)
    monkeypatch.setattr(pilot, "support", lambda name: support)
    return plan, runtime, calls


def test_cpu_closeout_and_repeat(monkeypatch, tmp_path):
    _, runtime, calls = setup(monkeypatch, tmp_path)
    root = tmp_path / "campaign"
    result = closeout.submit(runtime, root, "closeout")
    assert list(result["jobs"]) == ["prune", "critic"]
    assert [x for _, opts in calls for x in opts if x.startswith("--array")] == [
        "--array=0-3",
        "--array=0-0",
    ]
    assert all(not any("gpu" in x for x in opts) for _, opts in calls)
    assert closeout.submit(runtime, root, "closeout") == result
    assert len(calls) == 2


def test_closeout_refuses_unfinished_search_and_existing_workers(monkeypatch, tmp_path):
    _, runtime, calls = setup(monkeypatch, tmp_path)
    runtime.io.read_round = lambda *args: None
    with pytest.raises(ValueError, match="incomplete"):
        closeout.submit(runtime, tmp_path / "campaign", "closeout")
    runtime.io.read_round = lambda *args: {}
    monkeypatch.setattr(
        closeout.subprocess, "check_output", lambda *args, **kw: "component-fit\n"
    )
    with pytest.raises(ValueError, match="still queued"):
        closeout.submit(runtime, tmp_path / "campaign", "closeout")
    assert not calls


@pytest.mark.parametrize("which", ["closeout", "pilot"])
def test_ambiguous_receipts_never_resubmit(monkeypatch, tmp_path, which):
    _, runtime, calls = setup(monkeypatch, tmp_path)
    root = tmp_path / "campaign"
    run = (
        (lambda: closeout.submit(runtime, root, "wave"))
        if which == "closeout"
        else (lambda: pilot.submit(root, "wave"))
    )
    result = run()
    count = len(calls)
    directory = root / "submissions/wave"
    (directory / "manifest.json").unlink()
    stage = next(iter(result["jobs"]))
    (directory / f"{stage}.id").unlink()
    with pytest.raises(ValueError, match="uncertain"):
        run()
    assert len(calls) == count


def test_pilot_dependencies_and_idempotence(monkeypatch, tmp_path):
    _, _, calls = setup(monkeypatch, tmp_path)
    root = tmp_path / "campaign"
    result = pilot.submit(root, "pilot")
    assert pilot.submit(root, "pilot") == result and len(calls) == 3
    assert [s for s, _ in calls] == ["propose", "fit-assess", "report"]
    assert "--gres=gpu:h100:1" in calls[0][1]
    assert "--array=0-31%4" in calls[1][1]
    assert "32-fit-assess-tasks" in result["resources"]
    assert "--dependency=afterany:101" in calls[1][1]
    assert "--dependency=afterany:102" in calls[2][1]
