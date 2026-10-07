"""M18 qualification, paired budgets, post-selection evaluation and scheduler resume."""

import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import larger_coupled_campaign as campaign
from autoformalism.fitting import polishing_fit
from autoformalism.fitting import public_fitting as public
from tests.test_polishing_fit import backend


@pytest.fixture(scope="module")
def larger_inputs(tmp_path_factory):
    path = tmp_path_factory.mktemp("larger-campaign") / "inputs.json"
    campaign.controls.export_inputs(path)
    return path


def policy():
    return campaign.PolishingPolicy.model_validate_json(
        Path("configs/phase_c_larger_coupled_v1.json").read_text()
    )


def fake_qualification(root):
    plan, _ = campaign.verify(root)
    seal(
        root / "qualification/summary.json",
        {"plan_sha256": public.content_sha256(plan), "passed": True},
    )


def test_roster_and_gate_required(larger_inputs, tmp_path):
    assert campaign.prepare(tmp_path, larger_inputs, policy())["tasks"] == 24
    result = campaign.report(tmp_path)
    assert result["expected"] == 24 and result["recorded"] == 0
    assert result["qualification"] is None
    assert all(g["expected"] == 12 for g in result["groups"].values())
    with pytest.raises(FileNotFoundError):
        campaign.run_task(tmp_path, 0)
    plan, _ = campaign.verify(tmp_path)
    seal(
        tmp_path / "qualification/summary.json",
        {"plan_sha256": public.content_sha256(plan), "passed": False},
    )
    with pytest.raises(ValueError, match="failed qualification"):
        campaign.run_task(tmp_path, 0)


def test_host_qualification_and_resume(larger_inputs, tmp_path):
    campaign.prepare(tmp_path, larger_inputs, policy())
    first = campaign.qualify(tmp_path)
    assert first["passed"] and set(first["cases"]) == set(campaign.controls.CASES)
    assert all(r["identifiability"]["passed"] for r in first["cases"].values())
    stamps = {
        p: p.stat().st_mtime_ns
        for p in (tmp_path / "qualification").iterdir()
        if p.is_file()
    }
    assert campaign.qualify(tmp_path) == first
    assert stamps == {p: p.stat().st_mtime_ns for p in stamps}


@pytest.mark.parametrize("arm", campaign.ARMS)
def test_same_budget_and_timed_algorithms(larger_inputs, tmp_path, monkeypatch, arm):
    base = campaign.bases(read_seal(larger_inputs))["linear6_fast_slow_s2"]
    now, calls = [0.0], []

    def fit(b, method, settings, folder, *, rollout_mode, diagnostic_timing):
        assert diagnostic_timing and method == "rollout_only"
        assert rollout_mode == ("recovery_rollout" if arm == "rollout_only" else arm)
        calls.append(deepcopy(settings))
        now[0] += 40
        if len(calls) == 1:
            assert (
                settings["seconds"] == 600 and settings["maximum_rollout_calls"] == 600
            )
            return backend(b["start"], error=8e-9, calls=10)
        assert settings["seconds"] == 560 and settings["maximum_rollout_calls"] == 590
        assert settings["training_nmse"] == 1e-11
        return backend(b["start"], error=1e-12, calls=2)

    monkeypatch.setattr(polishing_fit, "monotonic", lambda: now[0])
    monkeypatch.setattr(campaign.coupled_polishing.recovery_fit, "fit", fit)
    value = campaign.fit(base, arm, policy().model_dump(mode="json"), tmp_path)
    assert value["strict_prediction_certified"] and value["actual_residual_calls"] == 12


def test_post_selection_evaluation_and_idempotence(
    larger_inputs, tmp_path, monkeypatch
):
    campaign.prepare(tmp_path, larger_inputs, policy())
    fake_qualification(tmp_path)
    calls = []

    def fit(base, arm, settings, folder):
        first = base["start"]
        final = {name: value + 0.01 for name, value in first.items()}
        return backend(final) | {
            "first_prediction": {"selected": {"parameters": first}}
        }

    def evaluate(folder, data, entry, parameters, seconds):
        assert (folder.parent / "backend.json").exists()
        if folder.name == "first-evaluation":
            assert (folder.parent / "result.json").exists()
        assert data["case_name"] == entry["case"] and seconds == 180
        calls.append(folder.name)
        return {"status": "complete", "accuracy_passed": True}

    monkeypatch.setattr(campaign, "fit", fit)
    monkeypatch.setattr(common.replay, "_evaluation", evaluate)
    for i in range(24):
        row = campaign.run_task(tmp_path, i)
        assert campaign.run_task(tmp_path, i) == row
    assert len(calls) == 48
    result = campaign.report(tmp_path)
    assert result["status"] == "complete" and result["recorded"] == 24
    folder = tmp_path / "results" / result["rows"][0]["task_id"]
    (folder / "comparison.json").unlink()
    assert campaign.report(tmp_path)["status"] == "incomplete"


def test_submission_and_shells(larger_inputs, tmp_path, monkeypatch):
    from scripts import submit_phase_c_generic_recovery as scheduler
    from scripts import submit_phase_c_larger_coupled as submit

    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        scheduler.subprocess, "check_output", lambda *a, **k: "revision\n"
    )
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    args = {"account": "test", "concurrency": 6, "inputs": larger_inputs}
    config = Path("configs/phase_c_larger_coupled_v1.json")
    result = submit.submit(tmp_path, config, **args)
    assert "--array=0-23%6" in calls[1] and "--partition=cpu" in calls[1]
    assert "--cpus-per-task=1" in calls[1]
    assert submit.submit(tmp_path, config, **args) == result and len(calls) == 3


def test_shell_syntax():
    for kind in ("run", "submit", "inspect"):
        subprocess.run(
            ["bash", "-n", f"scripts/hpc/{kind}_phase_c_larger_coupled_delta.sh"],
            check=True,
        )
