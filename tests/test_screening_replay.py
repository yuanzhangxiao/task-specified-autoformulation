"""M10 provenance, calibrated bounds, telemetry and no-budget-reset behavior."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import bounded_screening as bounded
from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import screening_replay as replay
from autoformalism.fitting import screening_replay_export as exporter
from autoformalism.fitting import transcription_campaign as source
from tests.test_fitting_strategies import small_payload


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """Tiny saved M9 records; synthetic unit-test scores are not experiment evidence."""
    inputs = controls.make_inputs()
    p = small_payload(inputs, "collocation")
    truth = controls.TRUTHS["linear"] | {"init_z_value": 0.4}
    case = deepcopy(inputs["cases"]["linear"])
    case["training"] = p["training"]
    common = {k: p[k] for k in ("request", "coordinates", "nodes")}
    common.update(
        case="linear",
        seed=0,
        assisted_start={
            "eligible": True,
            "parameters": truth,
            "training_nmse": 1e-14,
            "training_verified": True,
            "source_seconds": 9,
            "source_backend_sha256": "source-fit",
            "source_residual_calls": 3,
        },
    )
    config = source.CampaignConfig(
        starts=1,
        include_cstr=False,
        cases=("linear",),
        numerical_diagnostic={},
        screening_diagnostic={},
    )
    arms = [
        "fixed_dense_collocation_bounded_rk45",
        "fixed_dense_collocation_bounded_radau",
        "fixed_dense_collocation_assisted",
        "fixed_reduced_collocation_assisted",
    ]
    plan = {
        "protocol": source.screening.PROTOCOL,
        "source_sha256": "old-code",
        "config": config.model_dump(mode="json"),
        "commons": {"linear_s0": common},
        "tasks": [
            {"task_id": f"linear_s0_{arm}", "arm": arm, "common": "linear_s0"}
            for arm in arms
        ],
        "test_data_opened": False,
        "inputs_sha256": "old-inputs",
    }
    inputs = {"cases": {"linear": case}}
    source_root = tmp_path / "source"
    seal(
        source_root / "qualification/result.json",
        {"passed": True, "plan_sha256": public.content_sha256(plan)},
    )
    for task in plan["tasks"][:2]:
        worker = source.worker_payload(plan, inputs, task)
        folder = source_root / "results" / task["task_id"]
        attempts = []
        for index, parameters in enumerate((p["start"], truth)):
            directory = folder / "fit/screens/phase-0" / f"{index:03d}"
            directory.mkdir(parents=True)
            payload = {k: worker[k] for k in ("training", "request", "screening")}
            payload.update(parameters=parameters, seconds=20)
            public._write(directory / "payload.json", payload)
            attempt = {
                "phase": "phase-0",
                "index": index,
                "parameters_sha256": public.content_sha256(parameters),
                "status": "complete" if index == 0 else "timeout",
                "process": {"elapsed_seconds": 5},
            }
            if index == 0:
                attempt["training_nmse"] = 0.7
                public._write(
                    directory / "result.json",
                    {
                        "status": "complete",
                        "parameters": parameters,
                        "training_nmse": 0.7,
                        "maximum_trajectory_nmse": 0.7,
                        "payload_sha256": public.content_sha256(payload),
                    },
                )
            attempts.append(attempt)
        seal(
            folder / "backend.json",
            {
                "worker_payload_sha256": public.content_sha256(worker),
                "screening": {"attempts": attempts},
                "total_seconds": 30,
                "parameters": p["start"],
            },
        )
    monkeypatch.setattr(source, "verify", lambda *a, **k: (plan, inputs))
    path = tmp_path / "bundle.json"
    exporter.export(source_root, path, case_name="linear")
    # Freeze the test's identity independently of concurrent checkout edits.
    monkeypatch.setattr(public, "_source_identity", lambda: "unit-test-source")
    return source_root, path, p, truth


def test_export_all_vectors_and_matching_cached_scores(saved, tmp_path):
    root, path, _, _ = saved
    data = read_seal(path)
    assert len(data["pools"]) == 2 and len(data["assisted"]) == 2
    assert len(data["calibration_points"]) == 2
    for pool in data["pools"]:
        assert len(pool["points"]) == 2
        assert sum(p["cached"] is not None for p in pool["points"]) == 1
        assert "payload" not in pool
    target = next(root.glob("results/*/fit/screens/*/000/payload.json"))
    bad = public._read(target)
    bad["training"]["name"] = "val"
    public._write(target, bad)
    with pytest.raises(ValueError, match="contract"):
        exporter.export(root, tmp_path / "bad.json", case_name="linear")


def test_calibrated_limit_has_headroom_and_requires_good_start_rechecks():
    policy = replay.ReplayPolicy()
    rows = [
        {
            "required_good_start": True,
            "status": "complete",
            "training_nmse": 1e-14,
            "seconds": 44,
        },
        {"required_good_start": False, "status": "timeout", "seconds": 120},
    ]
    assert replay.calibrated_limit(rows, policy)["point_seconds"] == 98
    rows[0]["status"] = "timeout"
    assert not replay.calibrated_limit(rows, policy)["ready"]
    rows[0].update(status="complete", seconds=160)
    assert not replay.calibrated_limit(rows, policy)["ready"]
    with pytest.raises(ValueError, match="inverted"):
        replay.ReplayPolicy(minimum_point_seconds=120, maximum_point_seconds=60)


def test_point_progress_complete_and_failed_partial(tmp_path, saved, monkeypatch):
    _, _, p, truth = saved
    payload = {k: p[k] for k in ("request", "training")}
    payload.update(
        parameters=truth, seconds=30, screening={"method": "RK45", "point_seconds": 30}
    )
    assert bounded.evaluate(payload, tmp_path)["status"] == "complete"
    progress = public._read(tmp_path / "progress.json")
    assert progress["completed_trajectories"] == 1
    assert progress["trajectories"][0]["solver_counts"]["nfev"] > 0
    assert progress["setup_seconds"] >= 0

    from autoformalism.fitting import sensitivity_probe

    def timeout(*args, **kwargs):
        assert (
            public._read(tmp_path / "progress.json")["status"] == "trajectory_started"
        )
        raise TimeoutError("deliberate test timeout")

    monkeypatch.setattr(sensitivity_probe, "symbolic_rollout", timeout)
    result = bounded.evaluate(payload, tmp_path)
    assert result["status"] == "unavailable" and "training_nmse" not in result
    assert result["timing"]["completed_trajectories"] == 0


def ready(root, plan, method="Radau", value=True):
    seal(
        root / "calibration" / method / "result.json",
        {
            "plan_sha256": public.content_sha256(plan),
            "method": method,
            "ready": value,
            "status": "complete",
            "point_seconds": 30,
        },
    )


def test_blocked_and_interrupted_never_launch_fit(saved, tmp_path, monkeypatch):
    _, path, _, _ = saved
    root = tmp_path / "campaign"
    replay.prepare(root, path, replay.ReplayPolicy())
    plan, _ = replay.verify(root)
    ready(root, plan, value=False)
    monkeypatch.setattr(
        replay.transcription_fit, "run", lambda *a: pytest.fail("launched")
    )
    result = replay.run_task(root, 0)
    assert result["status"] == "calibration_blocked"
    assert replay.run_task(root, 0) == result
    other = tmp_path / "other"
    replay.prepare(other, path, replay.ReplayPolicy())
    plan, _ = replay.verify(other)
    ready(other, plan)
    calibration = read_seal(other / "calibration/Radau/result.json")
    task = plan["tasks"][0]
    seal(
        other / "results" / task["task_id"] / "started.json",
        {
            "plan_sha256": public.content_sha256(plan),
            "calibration_sha256": public.content_sha256(calibration),
            "task": task,
        },
    )
    assert replay.run_task(other, 0)["status"] == "interrupted"
    assert replay.report(other)["recorded"] == 1


def test_assisted_payload_and_completed_resume(saved, tmp_path, monkeypatch):
    _, path, _, truth = saved
    root = tmp_path / "campaign"
    replay.prepare(root, path, replay.ReplayPolicy())
    plan, data = replay.verify(root)
    ready(root, plan)
    payload = replay._assisted_payload(
        data, data["assisted"][0], 90, replay.ReplayPolicy()
    )
    assert payload["final_screen_reserve_seconds"] == 95
    assert payload["screening_diagnostic"]["point_seconds"] == 90
    assert not {"validation", "reference_parameters"} & payload.keys()
    calls = []

    def run(p, folder):
        calls.append(p)
        return {"parameters": truth, "total_seconds": 1, "stop_reason": "unit-test"}

    monkeypatch.setattr(replay.transcription_fit, "run", run)
    monkeypatch.setattr(replay, "_evaluation", lambda *a: {"status": "complete"})
    result = replay.run_task(root, 0)
    assert replay.run_task(root, 0) == result and len(calls) == 1
    assert replay.report(root)["status"] == "incomplete"


def test_cached_incumbent_survives_all_timeouts(saved, tmp_path, monkeypatch):
    _, path, p, _ = saved
    data = read_seal(path)
    calls = []

    def timeout(payload, folder, seconds):
        calls.append(payload)
        assert set(payload) == {
            "request",
            "training",
            "screening",
            "parameters",
            "seconds",
        }
        return {"wall_timeout": True, "elapsed_seconds": 0.1, "returncode": -9}

    monkeypatch.setattr(bounded, "invoke_point", timeout)
    result = replay._pool(
        data, data["pools"][0], 60, replay.ReplayPolicy(), tmp_path / "pool"
    )
    assert result["parameters"] == p["start"]
    assert result["reused_complete_screens"] == 1 and len(calls) == 1
    assert result["screening"]["best"]["training_nmse"] == 0.7


def test_calibration_resume_does_not_repeat_calls(saved, tmp_path, monkeypatch):
    _, path, _, _ = saved
    root = tmp_path / "campaign"
    replay.prepare(root, path, replay.ReplayPolicy())
    calls = []

    def evaluate(payload, folder, seconds):
        calls.append(payload)
        folder.mkdir(parents=True, exist_ok=True)
        public._write(
            folder / "result.json",
            {
                "status": "complete",
                "training_nmse": 1e-14,
                "maximum_trajectory_nmse": 1e-14,
                "parameters": payload["parameters"],
                "payload_sha256": public.content_sha256(payload),
            },
        )
        return {"wall_timeout": False, "elapsed_seconds": 35, "returncode": 0}

    monkeypatch.setattr(bounded, "invoke_point", evaluate)
    result = replay.calibrate(root, 1)
    assert result["ready"] and result["point_seconds"] == 80
    assert replay.calibrate(root, 1) == result and len(calls) == 2


def test_scheduler_cpu_arrays_and_idempotent_receipts(saved, tmp_path, monkeypatch):
    from scripts import submit_phase_c_screening_replay as submit

    _, inputs, _, _ = saved
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(submit.subprocess, "check_output", lambda *a, **k: "revision\n")
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        assert "--partition=cpu" in argv and "--mem=16G" in argv
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submit.subprocess, "run", sbatch)
    config = tmp_path / "config.json"
    config.write_text("{}")
    root = tmp_path / "submit"
    result = submit.submit(root, config, account="test", concurrency=2, inputs=inputs)
    assert "--array=0-1%2" in calls[1]
    assert "--array=0-3%2" in calls[2]
    assert "--dependency=afterok:102" in calls[2]
    assert (
        submit.submit(root, config, account="test", concurrency=2, inputs=inputs)
        == result
    )
    assert len(calls) == 4
