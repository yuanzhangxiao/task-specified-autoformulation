"""M16 roster, causal fitting boundary, deterministic resume and CPU dispatch."""

import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import coupled_campaign as campaign
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import recovery_fit


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    path = tmp_path_factory.mktemp("coupled_campaign") / "inputs.json"
    campaign.export_inputs(path)
    return path


def policy():
    return campaign.CoupledPolicy(seconds=40, certificate_seconds=10)


def test_all_cases_and_starts_training_allowlist(source, tmp_path):
    data = read_seal(source)
    before = campaign.bases(data)
    for case in campaign.CASES:
        assert data["cases"][case]["identifiability"]["passed"]
        data["cases"][case]["reference_parameters"] = {"poison": 999}
        data["cases"][case]["validation"] = {"poison": True}
    assert campaign.bases(data) == before
    assert all(
        set(b) == {"request", "training", "coordinates", "nodes", "start"}
        for b in before.values()
    )
    campaign.prepare(tmp_path, source, policy())
    report = campaign.report(tmp_path)
    assert report["expected"] == 12 and report["recorded"] == 0
    assert report["groups"]["rollout_only"]["expected"] == 6
    for seed in range(3):
        assert (
            before[f"coupled_linear_s{seed}"]["start"]
            == before[f"coupled_fast_slow_s{seed}"]["start"]
        )
    data["commons"].pop("coupled_linear_s2")
    with pytest.raises(ValueError, match="six"):
        campaign.bases(data)


def test_start_and_split_identity(source):
    data = read_seal(source)
    altered = deepcopy(data)
    altered["commons"]["coupled_linear_s0"]["start"]["a"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        campaign.bases(altered)
    altered = deepcopy(data)
    altered["cases"]["coupled_linear"]["training"]["name"] = "val"
    with pytest.raises(ValueError):
        campaign.bases(altered)
    assert campaign.export_inputs(source)["starts"] == 6


def test_dispatch_and_post_freeze_evaluation_resume(source, tmp_path, monkeypatch):
    campaign.prepare(tmp_path, source, policy())
    plan, _ = campaign.verify(tmp_path)
    calls = []

    def fit(base, arm, config, folder, *, rollout_mode):
        assert set(base) == {"request", "training", "coordinates", "nodes", "start"}
        calls.append((arm, rollout_mode))
        return {
            "selected": {"parameters": base["start"]},
            "levels": [],
            "stop_reason": "budget_or_optimizer_stop",
        }

    def evaluate(folder, data, common_data, parameters, seconds):
        assert (folder.parent / "backend.json").is_file()
        assert data["case"]["reference_parameters"]["a"] == (
            18.0 if common_data["case"] == "coupled_fast_slow" else 0.8
        )
        return {"status": "complete", "accuracy_passed": False}

    monkeypatch.setattr(recovery_fit, "fit", fit)
    monkeypatch.setattr(common.replay, "_evaluation", evaluate)
    rows = [campaign.run_task(tmp_path, i) for i in range(len(plan["tasks"]))]
    assert calls == [
        ("rollout_only", mode)
        for _ in range(6)
        for mode in ("recovery_rollout", "coupled_profiled_rollout")
    ]
    assert campaign.run_task(tmp_path, 0) == rows[0] and len(calls) == 12
    assert campaign.report(tmp_path)["status"] == "complete"


def test_cleanup_blocks_evaluation(source, tmp_path, monkeypatch):
    campaign.prepare(tmp_path, source, policy())

    def fitter(base, *args):
        folder = args[-1]
        seal(folder / "operation/started.json", {})
        seal(folder / "operation/process.json", {"termination_confirmed": False})
        return {
            "selected": {"parameters": base["start"]},
            "levels": [],
            "stop_reason": "cleanup_unconfirmed",
        }

    monkeypatch.setattr(campaign, "fit", fitter)
    monkeypatch.setattr(
        common.replay, "_evaluation", lambda *a: pytest.fail("unsafe evaluator")
    )
    row = campaign.run_task(tmp_path, 0)
    assert row["evaluation_blocked_by_cleanup"] and row["evaluation"] is None


def test_cpu_submission_resume(source, tmp_path, monkeypatch):
    from scripts import submit_phase_c_fitting_coupled as submit
    from scripts import submit_phase_c_generic_recovery as scheduler

    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        scheduler.subprocess, "check_output", lambda *a, **k: "revision\n"
    )
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    config = tmp_path / "config.json"
    config.write_text(policy().model_dump_json())
    args = {"account": "test", "concurrency": 6, "inputs": source}
    first = submit.submit(tmp_path / "submit", config, **args)
    assert "--array=0-11%6" in calls[1]
    assert "--cpus-per-task=1" in calls[1] and "--partition=cpu" in calls[1]
    assert calls[1][-2].endswith("run_phase_c_fitting_coupled_delta.sh")
    assert submit.submit(tmp_path / "submit", config, **args) == first
    assert len(calls) == 3


def test_shell_syntax_and_inspection_identity():
    for kind in ("run", "submit", "inspect"):
        subprocess.run(
            ["bash", "-n", f"scripts/hpc/{kind}_phase_c_fitting_coupled_delta.sh"],
            check=True,
        )
    assert (
        '.value.protocol == "phase-c-coupled-profile-1"'
        in Path("scripts/hpc/inspect_phase_c_fitting_coupled_delta.sh").read_text()
    )
