"""Generic-start roster, training-only mesh/checks, budgets and scheduler safety."""

import subprocess
from copy import deepcopy
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import generic_recovery as campaign
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_fit as fitting
from autoformalism.fitting import recovery_mesh as mesh
from autoformalism.fitting import recovery_worker as worker
from tests.test_screening_replay import saved as source_fixture


@pytest.fixture
def saved(tmp_path, monkeypatch):
    _, source, _, _ = source_fixture.__wrapped__(tmp_path, monkeypatch)
    inputs = tmp_path / "generic.json"
    campaign.export_inputs(source, inputs)
    return inputs


def small_policy():
    return campaign.RecoveryPolicy(
        seconds=40,
        certificate_seconds=10,
        native_seconds=5,
        point_seconds=2,
        minimum_intervals=1,
        targets=(12, None),
    )


def test_export_keeps_generic_start_and_excludes_assistance(saved):
    data = read_seal(saved)
    assert set(data["commons"]["linear_s0"]) == {
        "request",
        "coordinates",
        "nodes",
        "case",
        "seed",
        "start",
    }
    base = campaign.bases(data)
    changed = deepcopy(data)
    changed["case"]["reference_parameters"] = {"a": 999}
    changed["case"]["validation"] = {"unavailable": True}
    assert campaign.bases(changed) == base
    changed["commons"]["linear_s0"]["start"]["a"] += 1
    with pytest.raises(ValueError, match="start mismatch"):
        campaign.bases(changed)


def test_roster_cannot_silently_drop_failed_start(saved):
    data = read_seal(saved)
    data["case_name"] = "alien_hard"
    data["commons"] = {"alien_hard_s0": data["commons"]["linear_s0"]}
    with pytest.raises(ValueError, match="all original"):
        campaign.bases(data)


def test_start_identity_cannot_be_relabelled(saved):
    data = read_seal(saved)
    data["commons"]["linear_s0"]["seed"] = 2
    with pytest.raises(ValueError, match="seed identity"):
        campaign.bases(data)


def test_training_only_mesh_and_nested_inputs(saved):
    base = campaign.bases(read_seal(saved))["linear_s0"]
    previous = None
    for target in (12, 400, None):
        grids, audit = mesh.grids(base, target, 1, 8)
        assert (
            audit["all_observations_retained"]
            and audit["input_interpolation_preserved"]
        )
        if previous:
            assert all(set(previous[k]) <= set(v) for k, v in grids.items())
        previous = grids
        assert all(
            np.isfinite(v).all()
            for v in mesh.initial_guesses(base, grids)["nodes"].values()
        )
    base["training"]["name"] = "val"
    with pytest.raises(ValueError):
        mesh.grids(base, 12, 1, 8)


def test_observation_anchors_bracket_abrupt_change_and_are_bounded():
    t = np.arange(100.0)
    y = np.where(t < 40, 0.0, 1.0)
    anchors = mesh.observation_anchors(t, {"y": y}, 2)
    assert {39, 40} <= set(anchors) and len(anchors) <= 6
    assert mesh.observation_anchors(t, {"y": np.ones(100)}, 2) == []
    assert mesh.observation_anchors(t, {"y": y}, 0) == []


@pytest.mark.parametrize("failure", ["none", "one_bad", "disagree", "partial"])
def test_certificate_needs_all_trajectories_and_two_solvers(
    saved, tmp_path, monkeypatch, failure
):
    base = campaign.bases(read_seal(saved))["linear_s0"]
    calls = []

    def simulate(model, trajectory, parameters, initials, config, **kwargs):
        calls.append(config.integration_method)
        pred = {k: np.array(v).copy() for k, v in trajectory.targets.items()}
        if failure == "one_bad" or (
            failure == "disagree" and config.integration_method == "DOP853"
        ):
            pred = {k: v + 1 for k, v in pred.items()}
        return SimpleNamespace(
            success=failure != "partial", predictions=pred, message="failed"
        )

    monkeypatch.setattr(worker, "simulate_trajectory", simulate)
    public._write(tmp_path / "launch.json", {"monotonic": monotonic()})
    p = {k: base[k] for k in ("training", "request")} | {
        "parameters": base["start"],
        "seconds": 20,
        "training_nmse": 1e-6,
        "trajectory_nmse": 1e-5,
        "solver_agreement": 1e-5,
    }
    result = worker.certify(p, tmp_path)
    assert result["passed"] is (failure == "none")
    assert "Radau" in calls


def test_completed_certificate_skips_native_and_refinement(
    saved, tmp_path, monkeypatch
):
    base = campaign.bases(read_seal(saved))["linear_s0"]
    calls = []

    def invoke(mode, payload, folder, seconds):
        calls.append(mode)
        assert mode in {"point", "recovery_check"}
        folder.mkdir(parents=True, exist_ok=True)
        public._write(
            folder / "result.json",
            {
                "payload_sha256": public.content_sha256(payload),
                "parameters": payload["parameters"],
                "status": "complete",
                "training_nmse": 0.0,
                "maximum_trajectory_nmse": 0.0,
                "passed": True,
            },
        )
        return {"status": "complete", "termination_confirmed": True}

    monkeypatch.setattr(fitting.process, "invoke", invoke)
    r = fitting.fit(
        base, "mesh_rollout", small_policy().model_dump(mode="json"), tmp_path
    )
    assert r["stop_reason"] == "training_prediction_certified" and r["levels"] == []
    assert calls == ["point", "recovery_check"]


def test_interruption_never_refreshes_coordinator_budget(saved, tmp_path, monkeypatch):
    base = campaign.bases(read_seal(saved))["linear_s0"]
    policy = small_policy().model_dump(mode="json")
    identity = public.content_sha256(
        {"base": base, "arm": "mesh_rollout", "policy": policy}
    )
    seal(tmp_path / "fit-started.json", {"identity": identity})
    monkeypatch.setattr(fitting.process, "invoke", lambda *a: pytest.fail("relaunched"))
    result = fitting.fit(base, "mesh_rollout", policy, tmp_path)
    assert (
        result["stop_reason"] == "interrupted_no_fit_restart"
        and not result["budget_restarted"]
    )


def test_unfinished_native_retains_best_pair_and_uses_rollout(
    saved, tmp_path, monkeypatch
):
    base = campaign.bases(read_seal(saved))["linear_s0"]
    calls = []
    original = base["start"]
    worse = {k: v * 1.01 for k, v in original.items()}

    def invoke(mode, payload, folder, seconds):
        calls.append(mode)
        folder.mkdir(parents=True, exist_ok=True)
        value = {"payload_sha256": public.content_sha256(payload)}
        if mode == "native":
            public._write(
                folder / "final_checkpoint_diagnostics.json",
                {"parameters": worse, "maximum_scaled_defect": 1e-15},
            )
            return {"status": "timeout", "termination_confirmed": True}
        if mode == "point":
            value.update(
                status="complete",
                parameters=payload["parameters"],
                training_nmse=1.0 if payload["parameters"] == original else 2.0,
                maximum_trajectory_nmse=2.0,
            )
        elif mode == "recovery_rollout":
            assert payload["points"][0]["parameters"] == original
            value.update(parameters=original, training_nmse=1e-8)
        else:
            assert mode == "recovery_check"
            value.update(parameters=original, passed=True)
        public._write(folder / "result.json", value)
        return {"status": "complete", "termination_confirmed": True}

    monkeypatch.setattr(fitting.process, "invoke", invoke)
    result = fitting.fit(
        base, "mesh_rollout", small_policy().model_dump(mode="json"), tmp_path
    )
    assert calls.count("native") == 1
    assert not result["levels"][0]["qualified_for_transfer"]
    assert result["selected"]["origin"] == "rollout_refinement"
    assert result["stop_reason"] == "training_prediction_certified"


def test_evaluation_waits_for_frozen_backend_and_resumes(saved, tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    campaign.prepare(root, saved, small_policy())
    _, data = campaign.verify(root)
    base = campaign.bases(data)["linear_s0"]
    monkeypatch.setattr(
        fitting,
        "fit",
        lambda *a: {
            "selected": {"parameters": base["start"]},
            "levels": [],
            "stop_reason": "budget_or_optimizer_stop",
        },
    )

    def evaluate(folder, *args):
        assert (folder.parent / "backend.json").exists()
        return {"status": "complete", "accuracy_passed": False}

    monkeypatch.setattr(campaign.replay, "_evaluation", evaluate)
    result = campaign.run_task(root, 0)
    assert campaign.run_task(root, 0) == result
    summary = campaign.report(root)
    assert summary["recorded"] == 1 and summary["expected"] == 3
    assert summary["groups"]["rollout_only"]["accuracy_passed"] == 0


def test_unconfirmed_worker_blocks_postfit_evaluation(saved, tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    campaign.prepare(root, saved, small_policy())

    def fit(base, arm, policy, folder):
        seal(folder / "running/started.json", {"worker": "unfinished"})
        return {
            "selected": {"parameters": base["start"]},
            "levels": [],
            "stop_reason": "interrupted_no_fit_restart",
        }

    monkeypatch.setattr(fitting, "fit", fit)
    monkeypatch.setattr(
        campaign.replay, "_evaluation", lambda *a: pytest.fail("evaluation")
    )
    result = campaign.run_task(root, 0)
    assert result["evaluation_blocked_by_cleanup"] and not result["cost_complete"]


def test_scheduler_fixed_cpu_budget_and_receipts(saved, tmp_path, monkeypatch):
    import sys

    from scripts import submit_phase_c_generic_recovery as submit

    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(submit.subprocess, "check_output", lambda *a, **k: "revision\n")
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submit.subprocess, "run", sbatch)
    config = tmp_path / "policy.json"
    config.write_text(small_policy().model_dump_json())
    root = tmp_path / "submit"
    first = submit.submit(root, config, account="test", concurrency=2, inputs=saved)
    assert "--array=0-2%2" in calls[1] and "--time=00:35:00" in calls[1]
    assert (
        submit.submit(root, config, account="test", concurrency=2, inputs=saved)
        == first
    )
    assert len(calls) == 3


def test_inspector_shell_and_wrong_campaign_guard():
    from pathlib import Path

    script = Path("scripts/hpc/inspect_phase_c_generic_recovery_delta.sh")
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "AF_CAMPAIGN" not in text and ".value.protocol" in text
