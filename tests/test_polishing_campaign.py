"""Post-selection before/after evaluation, roster and journaled CPU launch."""

import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import polishing_campaign as campaign
from autoformalism.fitting import public_fitting as public
from tests.test_generic_recovery import saved as source_fixture
from tests.test_polishing_fit import backend


@pytest.fixture
def saved(tmp_path, monkeypatch):
    return source_fixture.__wrapped__(tmp_path, monkeypatch)


def policy():
    return campaign.PolishingPolicy(seconds=40, certificate_seconds=10)


def test_targets_must_be_stricter():
    for key, value in (
        ("polish_training_nmse", 1e-6),
        ("polish_trajectory_nmse", 1e-5),
    ):
        with pytest.raises(ValueError, match="stricter"):
            campaign.PolishingPolicy(**{key: value})


def test_paired_roster_reference_and_validation_excluded(saved, tmp_path):
    data = read_seal(saved)
    data["case_name"] = "alien_hard"
    source = data["commons"]["linear_s0"]
    data["commons"] = {
        f"alien_hard_s{i}": deepcopy(source) | {"case": "alien_hard", "seed": i}
        for i in range(3)
    }
    inputs = tmp_path / "six.json"
    seal(inputs, data)
    campaign.prepare(tmp_path / "run", inputs, policy())
    plan, _ = campaign.verify(tmp_path / "run")
    assert [t["arm"] for t in plan["tasks"]] == list(campaign.ARMS) * 3
    before = common.bases(data)
    data["case"]["reference_parameters"] = {"poison": 999}
    data["case"]["validation"] = {"poison": True}
    assert before == common.bases(data)
    assert campaign.report(tmp_path / "run")["expected"] == 6


@pytest.mark.parametrize("same,cleanup", [(True, False), (False, False), (False, True)])
def test_before_after_evaluation_only_after_final_seal_resume(
    saved, tmp_path, monkeypatch, same, cleanup
):
    root = tmp_path / "run"
    campaign.prepare(root, saved, policy())
    calls = []

    def fit(base, arm, settings, folder):
        if cleanup:
            seal(folder / "worker/started.json", {"mode": "profiled_rollout"})
            seal(folder / "worker/process.json", {"termination_confirmed": False})
        initial = base["start"]
        final = initial if same else {k: v + 0.1 for k, v in initial.items()}
        return backend(final) | {
            "first_prediction": {"selected": {"parameters": initial}},
            "stop_reason": "cleanup_unconfirmed"
            if cleanup
            else "training_prediction_certified",
        }

    def evaluate(folder, data, source, parameters, seconds):
        assert (folder.parent / "backend.json").exists()
        if folder.name == "first-evaluation":
            assert (folder.parent / "result.json").exists()
        calls.append(folder.name)
        return {
            "status": "complete",
            "accuracy_passed": True,
            "coefficients_recovered": True,
        }

    monkeypatch.setattr(campaign.polishing_fit, "fit", fit)
    monkeypatch.setattr(common.replay, "_evaluation", evaluate)
    first = campaign.run_task(root, 0)
    assert campaign.run_task(root, 0) == first
    assert calls == (
        []
        if cleanup
        else ["evaluation"]
        if same
        else ["evaluation", "first-evaluation"]
    )
    report = campaign.report(root)
    assert report["recorded"] == 1 and report["expected"] == 2
    if not cleanup:
        assert (
            report["polishing_groups"]["rollout_only"]["first_coefficients_recovered"]
            == 1
        )
    folder = root / "results" / first["task_id"]
    altered = read_seal(folder / "comparison.json")
    altered["identity"] = {"poison": True}
    public._write(
        folder / "comparison.json",
        {"sha256": public.content_sha256(altered), "value": altered},
    )
    with pytest.raises(ValueError, match="identity"):
        campaign.report(root)


def test_pending_comparison_is_not_reported_complete(saved, tmp_path, monkeypatch):
    root = tmp_path / "run"
    campaign.prepare(root, saved, policy())
    monkeypatch.setattr(common.replay, "_evaluation", lambda *a: {"status": "complete"})
    for i in range(2):
        common.run_task(
            root,
            i,
            protocol=campaign.PROTOCOL,
            fitter=lambda b, *a: backend(b["start"]),
        )
    summary = campaign.report(root)
    assert summary["recorded"] == 2 and summary["status"] == "incomplete"


def test_scheduler_cpu_receipts_and_six_concurrency(saved, tmp_path, monkeypatch):
    from scripts import submit_phase_c_fitting_polishing as submit
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
    kwargs = {"account": "test", "concurrency": 6, "inputs": saved}
    first = submit.submit(tmp_path / "submit", config, **kwargs)
    assert "--array=0-1%6" in calls[1] and "--partition=cpu" in calls[1]
    assert "--cpus-per-task=1" in calls[1] and "--time=00:35:00" in calls[1]
    assert calls[1][-2].endswith("run_phase_c_fitting_polishing_delta.sh")
    assert submit.submit(tmp_path / "submit", config, **kwargs) == first
    assert len(calls) == 3


def test_shell_syntax_and_protocol_guard():
    for kind in ("run", "submit", "inspect"):
        path = Path(f"scripts/hpc/{kind}_phase_c_fitting_polishing_delta.sh")
        subprocess.run(["bash", "-n", str(path)], check=True)
    text = Path("scripts/hpc/inspect_phase_c_fitting_polishing_delta.sh").read_text()
    assert '.value.protocol == "phase-c-fitting-polishing-1"' in text
