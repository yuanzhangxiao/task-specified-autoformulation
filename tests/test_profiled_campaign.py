"""M14 training boundary, paired roster, accounting, interruption and submission."""

import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import profiled_campaign as campaign
from autoformalism.fitting import profiled_worker as worker
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_fit
from tests.test_generic_recovery import saved as source_fixture


@pytest.fixture
def saved(tmp_path, monkeypatch):
    return source_fixture.__wrapped__(tmp_path, monkeypatch)


def policy():
    return campaign.ProfiledPolicy(seconds=40, certificate_seconds=10)


def test_all_three_starts_two_arms_and_training_allowlist(saved, tmp_path):
    data = read_seal(saved)
    source = data["commons"]["linear_s0"]
    data["case_name"] = "alien_hard"
    data["commons"] = {
        f"alien_hard_s{i}": deepcopy(source) | {"case": "alien_hard", "seed": i}
        for i in range(3)
    }
    inputs = tmp_path / "six-inputs.json"
    seal(inputs, data)
    campaign.prepare(tmp_path / "campaign", inputs, policy())
    plan, _ = campaign.verify(tmp_path / "campaign")
    assert len(plan["tasks"]) == 6
    assert [t["arm"] for t in plan["tasks"]] == list(campaign.ARMS) * 3
    assert campaign.report(tmp_path / "campaign")["expected"] == 6
    before = common.bases(data)
    data["case"]["reference_parameters"] = {"poison": 999}
    data["case"]["validation"] = {"poison": True}
    assert before == common.bases(data)
    assert all(
        set(b) == {"request", "coordinates", "nodes", "start", "training"}
        for b in before.values()
    )
    del data["commons"]["alien_hard_s2"]
    with pytest.raises(ValueError, match="all original"):
        common.bases(data)


@pytest.mark.parametrize("terminated", [True, False])
def test_timeout_checkpoint_only_after_confirmed_exit_and_no_restart(
    saved, tmp_path, monkeypatch, terminated
):
    base = common.bases(read_seal(saved))["linear_s0"]
    calls = []

    def invoke(mode, payload, folder, seconds):
        calls.append((mode, seconds))
        folder.mkdir(parents=True, exist_ok=True)
        assert set(payload).isdisjoint({"reference_parameters", "validation", "nodes"})
        assert payload["training"]["name"] == "train"
        if mode == "profiled_rollout":
            assert len(payload["points"]) == 1
            (folder / "refinement").mkdir(parents=True, exist_ok=True)
            public._write(
                folder / "refinement/best.json",
                {"parameters": base["start"], "training_nmse": 1e-8},
            )
            return {"status": "timeout", "termination_confirmed": terminated}
        assert mode == "recovery_check" and terminated
        public._write(
            folder / "result.json",
            {
                "payload_sha256": public.content_sha256(payload),
                "parameters": payload["parameters"],
                "passed": True,
            },
        )
        return {"status": "complete", "termination_confirmed": True}

    monkeypatch.setattr(recovery_fit.process, "invoke", invoke)
    folder = tmp_path / "fit"
    first = campaign.fit(base, "profiled_rollout", policy().model_dump(), folder)
    assert bool(first["selected"]) == terminated
    assert first["stop_reason"] == (
        "training_prediction_certified" if terminated else "cleanup_unconfirmed"
    )
    count = len(calls)
    second = campaign.fit(base, "profiled_rollout", policy().model_dump(), folder)
    assert second["selected"] == first["selected"] and len(calls) == count
    assert all(seconds <= 40 for _, seconds in calls)
    with pytest.raises(ValueError, match="identity differs"):
        campaign.fit(base, "rollout_only", policy().model_dump(), folder)


def test_control_dispatch_and_terminal_evaluation_resume(saved, tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    campaign.prepare(root, saved, policy())
    calls = []

    def fit(base, arm, config, folder, *, rollout_mode):
        calls.append((arm, rollout_mode))
        return {
            "selected": {"parameters": base["start"]},
            "levels": [],
            "stop_reason": "budget_or_optimizer_stop",
        }

    def evaluate(folder, *args):
        assert (folder.parent / "backend.json").is_file()
        return {"status": "complete", "accuracy_passed": False}

    monkeypatch.setattr(recovery_fit, "fit", fit)
    monkeypatch.setattr(common.replay, "_evaluation", evaluate)
    rows = [campaign.run_task(root, i) for i in range(2)]
    assert calls == [
        ("rollout_only", "recovery_rollout"),
        ("rollout_only", "profiled_rollout"),
    ]
    assert rows == [campaign.run_task(root, i) for i in range(2)]
    assert len(calls) == 2
    assert campaign.report(root)["recorded"] == 2


def test_worker_rejects_extra_evaluator_fields_before_fitting(tmp_path):
    with pytest.raises(ValueError, match="training-only"):
        worker.rollout({"validation": {}}, tmp_path)


def test_scheduler_cpu_receipts_no_duplicate_submission(saved, tmp_path, monkeypatch):
    from scripts import submit_phase_c_generic_recovery as scheduler
    from scripts import submit_phase_c_profiled_output as submit

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
    kwargs = {"account": "test", "concurrency": 2, "inputs": saved}
    first = submit.submit(tmp_path / "submit", config, **kwargs)
    assert "--array=0-1%2" in calls[1]
    assert "--partition=cpu" in calls[1] and "--cpus-per-task=1" in calls[1]
    assert "--time=00:35:00" in calls[1]
    assert calls[1][-2].endswith("run_phase_c_profiled_output_delta.sh")
    assert submit.submit(tmp_path / "submit", config, **kwargs) == first
    assert len(calls) == 3


def test_launcher_shells_and_wrong_campaign_guard():
    for kind in ("run", "submit", "inspect"):
        path = Path(f"scripts/hpc/{kind}_phase_c_profiled_output_delta.sh")
        subprocess.run(["bash", "-n", str(path)], check=True)
    text = Path("scripts/hpc/inspect_phase_c_profiled_output_delta.sh").read_text()
    assert '.value.protocol == "phase-c-profiled-output-1"' in text
