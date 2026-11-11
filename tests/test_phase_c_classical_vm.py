"""The classical Phase C tasks run on a Jetstream2 VM as Delta's worker runs them.

No baseline runs here: the subprocess that would start scripts/run_baseline.py
is replaced, and what is checked is what it would have been handed.
"""

import hashlib
import json

import pytest

from autoformalism.rebuttal import phase_c_baseline_plan as plan_module
from autoformalism.rebuttal import phase_c_classical_vm as vm
from scripts.run_baseline import build_parser
from tests.test_phase_c_baselines import CELL, JETSTREAM, SHIPPED, _release

SINDY, PYSR = 0, 2  # method-major order: sindy seeds 0 and 1, then pysr


def _plan(tmp_path, release, source=JETSTREAM):
    payload = json.loads(source.read_text())
    prompt = release / "public" / CELL / "proposer_prompt.txt"
    payload["release_summary_sha256"] = hashlib.sha256(
        (release / "summary.json").read_bytes()
    ).hexdigest()
    payload["cells"] = [
        {
            "benchmark_id": CELL,
            "tier": "fixed",
            "public_prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        }
    ]
    path = tmp_path / f"plan-{source.stem}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def frozen(tmp_path):
    """A frozen one-cell Jetstream2 plan with PySR's runtime recorded."""
    release = _release(tmp_path / "release")
    root = tmp_path / "run"
    vm.prepare(_plan(tmp_path, release), release, root)
    runtime = root / "runtime"
    runtime.mkdir()
    (runtime / "pysr_runtime.json").write_text('{"status": "ready"}\n')
    digest = hashlib.sha256((runtime / "pysr_runtime.json").read_bytes()).hexdigest()
    (runtime / "pysr_runtime.json.sha256").write_text(f"{digest}  pysr_runtime.json\n")
    return root, release


@pytest.fixture
def started(monkeypatch):
    """Record each baseline the runner would start, and start none."""
    calls = []

    class _Done:
        returncode = 0

    def fake_run(command, *, cwd, env, check):
        calls.append({"command": command, "cwd": cwd, "env": env, "check": check})
        return _Done()

    monkeypatch.setattr(vm.subprocess, "run", fake_run)
    return calls


def _run(root, release, index, tmp_path):
    return vm.run(
        root,
        index,
        release=release,
        julia_depot=tmp_path / "depot",
        base={"PATH": "/usr/bin", "OMP_NUM_THREADS": "64"},
    )


def test_prepare_freezes_a_jetstream_plan_and_says_what_each_task_needs(tmp_path):
    release = _release(tmp_path / "release")
    summary = vm.prepare(_plan(tmp_path, release), release, tmp_path / "run")
    assert summary["expected"] == 4
    assert summary["cpus_per_task"] == 16
    assert summary["tasks_by_method"] == {"sindy": 2, "pysr": 2}
    # Preparing again for a relaunch is the same freeze, not a second one.
    assert vm.prepare(_plan(tmp_path, release), release, tmp_path / "run") == summary


def test_prepare_refuses_a_plan_for_delta(tmp_path):
    release = _release(tmp_path / "release")
    with pytest.raises(ValueError, match="not a Jetstream2 VM"):
        vm.prepare(_plan(tmp_path, release, SHIPPED), release, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("index", [SINDY, PYSR])
def test_a_task_gets_the_delta_workers_arguments_and_threads(
    frozen, started, tmp_path, index
):
    root, release = frozen
    assert _run(root, release, index, tmp_path) == 0
    [call] = started
    assert call["command"][1].endswith("scripts/run_baseline.py")
    assert call["check"] is False
    task = vm.load_task(root.resolve(), index)
    args = build_parser().parse_args(call["command"][2:])
    assert args.phase_c_release == release.resolve()
    assert (args.benchmark_id, args.method, args.seed) == (
        CELL,
        task.method,
        task.repetition,
    )
    assert args.output_root == root.resolve() / "runs"
    assert args.wall_timeout_seconds == task.wall_timeout_seconds
    assert args.development_only
    if task.method == "sindy":
        assert tuple(args.sindy_thresholds) == task.sindy_thresholds
    else:
        assert (args.pysr_iterations, args.maximum_expression_size) == (40, 30)
    environment = call["env"]
    for name in vm.THREAD_VARIABLES:
        assert environment[name] == "16"  # the task's CPUs, not the VM's
    assert environment["JULIA_DEPOT_PATH"] == str(tmp_path / "depot")
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["PYTHONPATH"] == str(vm.REPO / "src")
    assert environment["PATH"] == "/usr/bin"


@pytest.mark.parametrize("status", sorted(vm.OUTCOMES))
def test_a_task_that_recorded_an_outcome_is_not_run_again(
    frozen, started, tmp_path, status
):
    root, release = frozen
    directory = vm.run_directory(root.resolve(), vm.load_task(root.resolve(), SINDY))
    directory.mkdir(parents=True)
    (directory / "run_status.json").write_text(json.dumps({"status": status}))
    assert _run(root, release, SINDY, tmp_path) == 0
    assert started == []


def test_an_interrupted_task_starts_again(frozen, started, tmp_path):
    root, release = frozen
    directory = vm.run_directory(root.resolve(), vm.load_task(root.resolve(), SINDY))
    directory.mkdir(parents=True)  # a run that was killed before it recorded anything
    assert _run(root, release, SINDY, tmp_path) == 0
    assert len(started) == 1


def test_a_changed_ledger_is_refused(frozen, started, tmp_path):
    root, release = frozen
    ledger = root / "frozen" / "task_plan.jsonl"
    ledger.write_text(ledger.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its recorded digest"):
        _run(root, release, SINDY, tmp_path)
    assert started == []


def test_pysr_needs_its_recorded_runtime(frozen, started, tmp_path):
    root, release = frozen
    (root / "runtime" / "pysr_runtime.json").write_text('{"status": "other"}\n')
    with pytest.raises(ValueError, match="differs from its recorded digest"):
        _run(root, release, PYSR, tmp_path)
    (root / "runtime" / "pysr_runtime.json.sha256").unlink()
    with pytest.raises(ValueError, match="recorded digest is missing"):
        _run(root, release, PYSR, tmp_path)
    assert started == []


def test_a_different_release_is_refused(frozen, started, tmp_path):
    root, _ = frozen
    other = _release(tmp_path / "other", prompt="A different prompt.")
    with pytest.raises(ValueError, match="release receipt differs"):
        _run(root, other, SINDY, tmp_path)
    assert started == []


def test_a_task_outside_the_plan_or_for_delta_is_refused(tmp_path, started):
    release = _release(tmp_path / "release")
    root = tmp_path / "run"
    plan_module.freeze_phase_c_baseline_plan(
        _plan(tmp_path, release, SHIPPED), root / "frozen", release
    )
    with pytest.raises(ValueError, match="not a classical Jetstream2 CPU task"):
        _run(root, release, SINDY, tmp_path)
    with pytest.raises(ValueError, match="has no task 9"):
        _run(root, release, 9, tmp_path)
    assert started == []
