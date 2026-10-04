"""Checkpoint policy isolation, bounded replay failures and exact resume."""

import sys
from contextlib import suppress
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import checkpoint_diagnostic as d
from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import qualification as q
from autoformalism.fitting import transcription_campaign as c
from autoformalism.fitting import transcription_fit as fit
from autoformalism.fitting import transcription_solver as solver
from tests.test_fitting_strategies import small_payload


@pytest.fixture(scope="module")
def inputs():
    return controls.make_inputs()


def test_compact_native_keeps_parameters_but_defers_trajectory_work(
    inputs, tmp_path, monkeypatch
):
    original = solver.mesh_tools.collocation_indicators
    calls = []

    def indicators(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(solver.mesh_tools, "collocation_indicators", indicators)
    counts = {}
    for mode in ("legacy", "compact"):
        folder = tmp_path / mode
        folder.mkdir()
        payload = small_payload(inputs, "collocation")
        payload.update(checkpoint_mode=mode, seconds=30)
        before = len(calls)
        result = solver.solve(payload, folder)
        assert result["native_success"]
        counts[mode] = len(calls) - before
        pool = public._read(folder / "checkpoints.json")["pool"]
        assert 1 <= len(pool) <= 8
        assert all(bool(p["parameters"]) for p in pool)
        assert all(("trajectories" in p) == (mode == "legacy") for p in pool)
        if mode == "compact":
            detail = public._read(folder / "final_checkpoint_diagnostics.json")
            assert detail["trajectories"]
            assert detail["parameters"] in [p["parameters"] for p in pool]
            assert counts[mode] == len(payload["training"]["rows"])
    assert counts["compact"] < counts["legacy"]


def test_compact_checkpoint_survives_native_interruption(inputs, tmp_path, monkeypatch):
    write = public._write

    def interrupted(path, value):
        write(path, value)
        if path.name == "checkpoints.json":
            raise KeyboardInterrupt

    monkeypatch.setattr(public, "_write", interrupted)
    payload = small_payload(inputs, "collocation")
    payload["checkpoint_mode"] = "compact"
    # CasADi may translate callback exceptions into a native RuntimeError;
    # the disk checkpoint must survive either exit path.
    with suppress(KeyboardInterrupt):
        solver.solve(payload, tmp_path)
    pool = public._read(tmp_path / "checkpoints.json")["pool"]
    assert pool and pool[0]["parameters"]


def test_invalid_checkpoint_modes(inputs, tmp_path):
    payload = small_payload(inputs, "collocation")
    for extra in (
        {"checkpoint_mode": "invalid"},
        {"checkpoint_mode": "compact", "reuse_chunks": 2},
    ):
        with pytest.raises(ValueError, match="checkpoint"):
            solver.build_problem(payload | extra, tmp_path)


def config():
    return c.CampaignConfig(
        starts=1,
        include_cstr=False,
        cases=("linear",),
        numerical_diagnostic={},
        checkpoint_diagnostic=True,
    )


def test_frozen_pairs_differ_only_in_checkpoint_mode(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(c.controls, "make_inputs", lambda: deepcopy(inputs))
    c.prepare(tmp_path, config())
    plan, data = c.verify(tmp_path)
    assert plan["protocol"] == d.PROTOCOL and len(plan["tasks"]) == 6
    for i in range(0, 6, 2):
        a, b = [c.worker_payload(plan, data, t) for t in plan["tasks"][i : i + 2]]
        assert a.pop("checkpoint_mode") == "legacy"
        assert b.pop("checkpoint_mode") == "compact"
        assert a == b
        assert a["arm"] in d.BASE_ARMS
        assert not {"reference_parameters", "validation"} & set(a)
    c.prepare(tmp_path, config())
    with pytest.raises(ValueError, match="requires"):
        c.CampaignConfig(checkpoint_diagnostic=True)


def test_m8_scheduler_uses_cpu_resources_and_reuses_receipts(
    inputs, tmp_path, monkeypatch
):
    from scripts import submit_phase_c_fitting_strategies as submit

    monkeypatch.setattr(c.controls, "make_inputs", lambda: deepcopy(inputs))
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(submit.subprocess, "check_output", lambda *a, **k: "frozen\n")
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        assert "--partition=cpu" in argv and "--mem=16G" in argv
        assert not any("--gres" in v or "--gpus" in v for v in argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submit.subprocess, "run", sbatch)
    cfg = tmp_path / "config.json"
    cfg.write_text(config().model_dump_json())
    root = tmp_path / "campaign"
    args = {"account": "test-cpu", "concurrency": 2}
    result = submit.submit(root, cfg, **args)
    assert len(calls) == 3 and "--time=00:25:00" in calls[1]
    assert "--array=0-5%2" in calls[1]
    assert submit.submit(root, cfg, **args) == result and len(calls) == 3


def replay_arguments(inputs):
    case = inputs["cases"]["linear"]
    req = c.case_request("linear", case, 0)
    return [
        req,
        public._lower(req)[1],
        q.PublicSplit.model_validate(case["training"]),
        q.PublicSplit.model_validate(case["validation"]),
        30,
    ]


@pytest.mark.parametrize("error", [TimeoutError, RuntimeError])
def test_failed_replay_preserves_coverage_and_does_not_retry(
    inputs, tmp_path, monkeypatch, error
):
    calls = []

    def simulate(model, trajectory, *args, **kwargs):
        calls.append(1)
        if len(calls) == 4:
            raise error("injected integration failure")
        return SimpleNamespace(success=True, predictions=trajectory.targets)

    monkeypatch.setattr(q, "simulate_trajectory", simulate)
    args = replay_arguments(inputs)
    path = tmp_path / "replay_progress.json"
    result = q.replay(*args, journal=path)
    assert not result["complete"] and result["metrics"]["train"] is None
    assert result["rows"][0]["complete"]
    assert result["rows"][1]["solver_scores"]["Radau"]
    assert len(result["rows"]) == sum(len(s.rows) for s in args[2:4])
    if error is TimeoutError:
        assert len(calls) == 4 and result["budget_exhausted"]
    monkeypatch.setattr(
        q, "simulate_trajectory", lambda *a, **k: pytest.fail("budget reset")
    )
    assert q.replay(*args, journal=path) == result
    with pytest.raises(ValueError, match="identity"):
        q.replay(*args[:-1], 31, journal=path)


def test_interrupted_replay_closes_without_more_integration(
    inputs, tmp_path, monkeypatch
):
    calls = []

    def simulate(model, trajectory, *args, **kwargs):
        calls.append(1)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return SimpleNamespace(success=True, predictions=trajectory.targets)

    monkeypatch.setattr(q, "simulate_trajectory", simulate)
    args = replay_arguments(inputs)
    path = tmp_path / "replay_progress.json"
    with pytest.raises(KeyboardInterrupt):
        q.replay(*args, journal=path)
    monkeypatch.setattr(
        q, "simulate_trajectory", lambda *a, **k: pytest.fail("no retry")
    )
    result = q.replay(*args, journal=path)
    assert result["stop_reason"] == "replay_interrupted"
    assert result["rows"][0]["complete"]
    assert result["rows"][1]["solver_scores"]["Radau"]
    assert not result["complete"] and not result["budget_restarted"]


def test_campaign_failed_replay_is_terminal_and_fitting_not_repeated(
    inputs, tmp_path, monkeypatch
):
    monkeypatch.setattr(c.controls, "make_inputs", lambda: deepcopy(inputs))
    c.prepare(tmp_path, config())
    plan, _ = c.verify(tmp_path)
    seal(
        tmp_path / "qualification/result.json",
        {"passed": True, "plan_sha256": public.content_sha256(plan)},
    )

    def run(payload, directory):
        return {
            "parameters": public._lower(
                q.PublicFitRequest.model_validate(payload["request"])
            )[1],
            "worker_payload_sha256": public.content_sha256(payload),
            "total_seconds": 1,
            "stop_reason": "fixture",
            "budget_exhausted": False,
        }

    def timeout(*args, **kwargs):
        raise TimeoutError("injected replay timeout")

    monkeypatch.setattr(c, "run", run)
    monkeypatch.setattr(q, "simulate_trajectory", timeout)
    result = c.run_task(tmp_path, 0)
    assert result["status"] == "no_complete_replay"
    assert result["evaluation_stop_reason"] == "replay_wall_budget_exhausted"
    assert result["parameters"] and not result["accuracy_passed"]
    monkeypatch.setattr(c, "run", lambda *a: pytest.fail("no duplicate fitting"))
    assert c.run_task(tmp_path, 0) == result
    assert c.report(tmp_path)["recorded"] == 1


def test_partial_rollout_decision_and_completed_counts_survive(tmp_path, monkeypatch):
    for name in ("refinement-0", "refinement-1"):
        (tmp_path / name).mkdir()
    public._write(
        tmp_path / "diagnostic.json",
        {"decision": {"triggered": True, "used": True}, "stages": [{"phase": 0}]},
    )
    public._write(tmp_path / "refinement-0/training_history.json", [{"call": 1}])
    public._write(
        tmp_path / "refinement-1/training_history.json", [{"call": 1}, {"call": 2}]
    )
    public._write(
        tmp_path / "refinement-0/best.json",
        {"parameters": {"a": 1}, "training_nmse": 0.1},
    )
    monkeypatch.setattr(
        fit, "invoke", lambda *a: {"wall_timeout": True, "elapsed_seconds": 60}
    )
    result = fit.run(
        {
            "policy": {"seconds": 60},
            "arm": "rollout_restart",
            "numerical_diagnostic": {},
        },
        tmp_path,
    )
    assert result["decision"]["used"] and result["stages"] == [{"phase": 0}]
    assert result["completed_training_calls_by_phase"] == {"0": 1, "1": 2}
    assert result["actual_residual_calls"] is None


def test_partial_native_metadata_survives_interrupted_screening(tmp_path):
    folder = tmp_path / "mesh-0"
    folder.mkdir()
    public._write(folder / "layout.json", {"variables": 42})
    public._write(folder / "solver_progress.json", {"iteration": 7})
    result = d.recover_metadata(tmp_path)
    stage = result["stages"][0]
    assert stage["level"] == 0 and stage["partial_metadata"]
    assert stage["diagnostics"]["layout"] == {"variables": 42}
    assert stage["diagnostics"]["solver_progress"] == {"iteration": 7}
    assert stage["diagnostics"]["native"] is None
    assert result["actual_residual_calls"] is None


def test_historical_timeout_closeout_is_separate_and_rejects_wrong_log(
    inputs, tmp_path, monkeypatch
):
    from scripts.closeout_fitting_replay_timeout import closeout

    monkeypatch.setattr(c.controls, "make_inputs", lambda: deepcopy(inputs))
    c.prepare(tmp_path, config())
    plan, data = c.verify(tmp_path)
    task = plan["tasks"][0]
    folder = tmp_path / "results" / task["task_id"]
    folder.mkdir(parents=True)
    backend = {
        "worker_payload_sha256": public.content_sha256(
            c.worker_payload(plan, data, task)
        ),
        "training_nmse": 0.5,
    }
    seal(folder / "backend.json", backend)
    public._write(
        tmp_path / "submission_manifest.json",
        {
            "identity": {"plan_sha256": public.content_sha256(plan)},
            "jobs": {"fit": "123"},
        },
    )
    (tmp_path / "logs").mkdir()
    log = tmp_path / "logs/fit-123_0.err"
    log.write_text("unrelated worker failure")
    with pytest.raises(ValueError, match="log"):
        closeout(tmp_path, 0)
    log.write_text(
        "checked = replay(\n fitting/qualification.py\n"
        "TimeoutError: fitting wall-clock limit reached"
    )
    frozen = (tmp_path / "summary.json").read_bytes()
    result = closeout(tmp_path, 0)
    assert closeout(tmp_path, 0) == result
    assert (tmp_path / "summary.json").read_bytes() == frozen
    assert not (folder / "result.json").exists()
    assert result["training_nmse"] is None and not result["new_replay_performed"]
    summary = public._read(tmp_path / "summary-closeout.json")
    assert summary["recorded"] == 1
    assert summary["status_counts"]["no_complete_replay"] == 1
