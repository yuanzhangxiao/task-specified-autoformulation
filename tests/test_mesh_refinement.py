"""Mesh convergence diagnostic: transfer, safety, provenance and exact resume."""

import subprocess
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import mesh_refinement as campaign
from autoformalism.fitting import mesh_refinement_fit as fitting
from autoformalism.fitting import mesh_refinement_process as process
from autoformalism.fitting import mesh_refinement_worker as worker
from autoformalism.fitting import public_fitting as public
from tests.test_screening_replay import saved as saved_fixture


@pytest.fixture
def saved(tmp_path, monkeypatch):
    return saved_fixture.__wrapped__(tmp_path, monkeypatch)


def test_transfer_preserves_quadratic_and_internal_nodes():
    old = {
        "trajectories": {
            "t": {
                "mesh": [0.0, 1.0, 2.0],
                "nodes": [[0.0], [1.0], [4.0]],
                "inner_nodes": [[1 / 9], [16 / 9]],
            }
        }
    }
    new = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    got = worker.transfer(old, {"t": new.tolist()})
    assert np.allclose(np.asarray(got["nodes"]["t"]).ravel(), new**2)
    assert np.allclose(
        np.asarray(got["inner_nodes"]["t"]).ravel(), (new[:-1] + np.diff(new) / 3) ** 2
    )
    with pytest.raises(ValueError, match="boundaries"):
        worker.transfer(old, {"t": [0.0, 0.5, 2.0]})
    bad = deepcopy(old)
    bad["trajectories"]["t"]["inner_nodes"] = [[float("nan")], [0]]
    with pytest.raises(ValueError, match="nodes"):
        worker.transfer(bad, {"t": new.tolist()})


def test_cleanup_failure_is_recorded_and_never_relaunched(tmp_path, monkeypatch):
    calls = []

    def wait(**kwargs):
        raise subprocess.TimeoutExpired("worker", kwargs["timeout"])

    def popen(*a, **k):
        calls.append(a)
        return SimpleNamespace(pid=123, returncode=None, wait=wait)

    monkeypatch.setattr(process.subprocess, "Popen", popen)
    monkeypatch.setattr(process.os, "killpg", lambda *a: None)
    first = process.invoke("point", {}, tmp_path, 0.01)
    assert first["status"] == "cleanup_unconfirmed"
    assert not first["termination_confirmed"]
    assert process.invoke("point", {}, tmp_path, 0.01) == first
    assert len(calls) == 1
    with pytest.raises(ValueError, match="identity"):
        process.invoke("point", {"changed": True}, tmp_path, 0.01)


def test_real_point_deadline_is_bounded(tmp_path):
    result = process.invoke("point", {}, tmp_path, 0.001)
    assert result["status"] == "timeout"
    assert result["termination_confirmed"]
    assert result["elapsed_seconds"] < 11


def test_interrupted_operation_never_refreshes_budget(tmp_path, monkeypatch):
    from autoformalism.benchmarks.audited_release import seal

    seal(
        tmp_path / "started.json",
        {"mode": "point", "payload_sha256": public.content_sha256({}), "seconds": 1},
    )
    monkeypatch.setattr(
        process.subprocess, "Popen", lambda *a, **k: pytest.fail("relaunched")
    )
    result = process.invoke("point", {}, tmp_path, 1)
    assert result["status"] == "interrupted" and not result["budget_restarted"]


def test_plan_and_worker_boundaries(saved, tmp_path):
    _, inputs, _, _ = saved
    root = tmp_path / "campaign"
    policy = campaign.MeshPolicy(targets=(12, None), minimum_intervals=1)
    assert campaign.prepare(root, inputs, policy)["tasks"] == 3
    plan, data = campaign.verify(root)
    base = campaign.bases(data)["linear_s0"]
    assert set(base) == {"training", "request", "coordinates", "source"}
    modified = deepcopy(data)
    modified["case"]["reference_parameters"] = {"a": -999}
    modified["case"]["validation"] = {"unavailable": True}
    assert campaign.bases(modified) == campaign.bases(data)
    for level in plan["policy"]["targets"]:
        grid, audit = worker.grids(base, level, policy.minimum_intervals)
        assert (
            audit["all_observations_retained"]
            and audit["input_interpolation_preserved"]
        )
        p = worker.native_payload(base, base["source"]["parameters"], grid, {})
        assert not {"validation", "reference_parameters", "source"} & set(p)
    modified["case"]["training"]["name"] = "val"
    with pytest.raises(ValueError):
        campaign.bases(modified)
    modified = deepcopy(data)
    modified["assisted"][0]["payload"]["assisted_start"]["parameters"]["a"] += 1
    with pytest.raises(ValueError, match="payload"):
        campaign.bases(modified)
    with pytest.raises(ValueError, match="artifact differs"):
        campaign.prepare(
            root, inputs, policy.model_copy(update={"minimum_intervals": 2})
        )
    assert campaign.report(root)["status"] == "incomplete"


def test_ladder_rejects_ambiguous_order():
    for targets in ((None,), (12, 12, None), (12, 6, None), (None, 12), (0, None)):
        with pytest.raises(ValueError):
            campaign.MeshPolicy(targets=targets)


def test_continuation_transfers_endpoint_but_keeps_better_incumbent(
    saved, tmp_path, monkeypatch
):
    _, inputs, _, _ = saved
    base = campaign.bases(read_seal(inputs))["linear_s0"]
    initial = base["source"]["parameters"]
    calls = []
    monkeypatch.setattr(
        fitting, "point", lambda *a: {"status": "complete", "training_nmse": 1e-14}
    )

    def level(base, folder, target, policy, previous):
        calls.append(previous)
        return {
            "status": "complete",
            "advance_qualified": True,
            "endpoint_parameters": {"a": 99},
            "endpoint_screen": {"status": "complete", "training_nmse": 0.1},
            "released": {},
        }

    monkeypatch.setattr(fitting, "solve_level", level)
    monkeypatch.setattr(fitting, "_read_detail", lambda *a: {"parameters": {"a": 99}})
    result = fitting.fit(
        base,
        {"levels": [0, 1]},
        campaign.MeshPolicy(targets=(12, None)).model_dump(mode="json"),
        tmp_path,
    )
    assert calls == [None, {"parameters": {"a": 99}}]
    assert result["selected"]["parameters"] == initial
    assert result["selected"]["origin"] == "supplied_M7"
    assert result["stop_reason"] == "planned_levels_complete"


def test_no_refinement_after_unfinished_solve(saved, tmp_path, monkeypatch):
    _, inputs, _, _ = saved
    base = campaign.bases(read_seal(inputs))["linear_s0"]
    monkeypatch.setattr(
        fitting, "point", lambda *a: {"status": "complete", "training_nmse": 1e-14}
    )
    calls = []

    def level(*args):
        calls.append(args)
        return {"status": "fixed_solve_unqualified", "advance_qualified": False}

    monkeypatch.setattr(fitting, "solve_level", level)
    result = fitting.fit(
        base,
        {"levels": [0, 1]},
        campaign.MeshPolicy(targets=(12, None)).model_dump(mode="json"),
        tmp_path,
    )
    assert len(calls) == 1 and len(result["levels"]) == 1
    assert result["stop_reason"] == "unqualified_level_no_refinement"


def test_evaluation_after_freeze_and_terminal_resume(saved, tmp_path, monkeypatch):
    _, inputs, _, truth = saved
    root = tmp_path / "campaign"
    campaign.prepare(root, inputs, campaign.MeshPolicy(targets=(12, None)))
    plan, _ = campaign.verify(root)
    folder = root / "results" / plan["tasks"][0]["task_id"]
    monkeypatch.setattr(
        fitting,
        "fit",
        lambda *a: {
            "source_fit_seconds": 9,
            "levels": [],
            "selected": {"parameters": truth, "origin": "supplied_M7"},
            "stop_reason": "planned_levels_complete",
        },
    )
    calls = []

    def evaluation(*args):
        assert (folder / "backend.json").exists()
        calls.append(args)
        return {"status": "complete"}

    monkeypatch.setattr(campaign, "_evaluate", evaluation)
    first = campaign.run_task(root, 0)
    assert len(calls) == 1 and first["selected_evaluation"]["status"] == "complete"
    assert campaign.run_task(root, 0) == first and len(calls) == 1
    assert campaign.report(root)["recorded"] == 1


def test_no_evaluation_after_unconfirmed_child(saved, tmp_path, monkeypatch):
    _, inputs, _, truth = saved
    root = tmp_path / "campaign"
    campaign.prepare(root, inputs, campaign.MeshPolicy(targets=(12, None)))

    def fit(base, task, policy, folder):
        seal(folder / "bad/process.json", {"termination_confirmed": False})
        return {
            "source_fit_seconds": 9,
            "levels": [],
            "selected": {"parameters": truth},
            "stop_reason": "cleanup_or_interruption_stop",
        }

    monkeypatch.setattr(fitting, "fit", fit)
    monkeypatch.setattr(campaign, "_evaluate", lambda *a: pytest.fail("child launched"))
    result = campaign.run_task(root, 0)
    assert result["evaluation_blocked_by_cleanup"] and not result["cost_complete"]


def test_scheduler_cpu_arrays_and_idempotent_receipts(saved, tmp_path, monkeypatch):
    import sys

    from scripts import submit_phase_c_mesh_refinement as submit

    _, inputs, _, _ = saved
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(submit.subprocess, "check_output", lambda *a, **k: "revision\n")
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        assert "--partition=cpu" in argv and "--mem=16G" in argv
        assert not any("--gpus" in a for a in argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submit.subprocess, "run", sbatch)
    config = tmp_path / "config.json"
    config.write_text('{"targets":[12,null],"minimum_intervals":1}')
    root = tmp_path / "submit"
    result = submit.submit(root, config, account="test", concurrency=2, inputs=inputs)
    assert "--array=0-2%2" in calls[1] and "--dependency=afterok:101" in calls[1]
    assert "--dependency=afterany:101:102" in calls[2]
    assert (
        submit.submit(root, config, account="test", concurrency=2, inputs=inputs)
        == result
    )
    assert len(calls) == 3


def test_training_point_rejects_different_vector(saved, tmp_path, monkeypatch):
    _, inputs, _, _ = saved
    base = campaign.bases(read_seal(inputs))["linear_s0"]

    def invoke(mode, payload, folder, seconds):
        public._write(
            folder / "result.json",
            {
                "status": "complete",
                "payload_sha256": public.content_sha256(payload),
                "parameters": {"wrong": 1},
                "training_nmse": 0,
            },
        )
        return {"status": "complete", "termination_confirmed": True}

    monkeypatch.setattr(process, "invoke", invoke)
    with pytest.raises(ValueError, match="vector"):
        fitting.point(base, base["source"]["parameters"], tmp_path, 1)


def test_launch_failure_is_terminal(tmp_path, monkeypatch):
    def fail(*a, **k):
        raise OSError("worker unavailable")

    monkeypatch.setattr(process.subprocess, "Popen", fail)
    result = process.invoke("native", {}, tmp_path, 1)
    assert result["status"] == "launch_failed" and result["termination_confirmed"]
    monkeypatch.setattr(
        process.subprocess, "Popen", lambda *a, **k: pytest.fail("retry")
    )
    assert process.invoke("native", {}, tmp_path, 1) == result


def test_native_outputs_are_bound_to_operation(tmp_path, monkeypatch):
    def popen(*args, **kwargs):
        public._write(tmp_path / "nodes.json", {"nodes": [1]})
        return SimpleNamespace(pid=123, returncode=0, wait=lambda **k: None)

    monkeypatch.setattr(process.subprocess, "Popen", popen)
    first = process.invoke("nodes", {}, tmp_path, 1)
    assert first["status"] == "complete"
    public._write(tmp_path / "nodes.json", {"nodes": [2]})
    with pytest.raises(ValueError, match="output differs"):
        process.invoke("nodes", {}, tmp_path, 1)
