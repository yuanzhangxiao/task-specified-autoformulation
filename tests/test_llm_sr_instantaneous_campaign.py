"""Exact current-value roster, training isolation and nonrenewable replay costs."""

import copy
import hashlib

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import llm_sr_numerical_control as control
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_instantaneous_campaign as campaign
from tests.test_llm_sr_control_campaign import policy


def inputs(monkeypatch):
    """Use tiny allowlisted test rewrites without executing the native source."""
    tasks, transcriptions = [], []
    source = "never execute this source"
    time = np.linspace(0, 2, 11)
    for task, benchmark, repetition in campaign.ROSTER:
        item = control.Transcription.model_validate(
            {
                "task": task,
                "benchmark_id": benchmark,
                "repetition": repetition,
                "programs": {
                    "x": {
                        "function_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "parameters": [0.7],
                    }
                },
                "parameters": {"a": 0.7},
                "targets": {"x": "-a*x"},
                "rules": ["direct"],
            }
        )
        transcriptions.append(item.model_dump(mode="json"))
        tasks.append(
            {
                "task": task,
                "benchmark_id": benchmark,
                "repetition": repetition,
                "programs": {
                    "x": {
                        "function": source,
                        "parameters": [0.7],
                        "variables": ["x", "area_down"],
                    }
                },
                "context": {"targets": ["x"], "fixed_covariates": ["area_down"]},
                "training": {
                    "name": "train",
                    "fingerprint": "synthetic",
                    "rows": [
                        {
                            "trajectory_id": "train0",
                            "time": time.tolist(),
                            "targets": {"x": np.exp(-0.7 * time).tolist()},
                            "fixed_covariates": {"area_down": 1200},
                        }
                    ],
                },
            }
        )
    monkeypatch.setattr(
        campaign,
        "TRANSCRIPTIONS",
        {
            t["task"]: public.content_sha256(control.Transcription.model_validate(t))
            for t in transcriptions
        },
    )
    monkeypatch.setattr(
        campaign,
        "TRAINING",
        {
            t["benchmark_id"]: public.content_sha256(
                campaign.PublicSplit.model_validate(t["training"])
            )
            for t in tasks
        },
    )
    monkeypatch.setattr(
        campaign, "CONTEXTS", {t["benchmark_id"]: t["context"] for t in tasks}
    )
    return {
        "protocol": "llm-sr-instantaneous-controls-input-1",
        "source_plan_sha256": campaign.SOURCE_PLAN,
        "transcription_set": {
            "protocol": "phase-c-llm-sr-ode-transcriptions-1",
            "status": "proposed_pending_review",
            "source_run": "synthetic",
            "source_plan_sha256": campaign.SOURCE_PLAN,
            "rules": {"direct": "current-value rewrite"},
            "transcriptions": transcriptions,
        },
        "tasks": tasks,
    }


def test_four_controls_replay_and_resume_without_native_execution(
    tmp_path, monkeypatch
):
    data = inputs(monkeypatch)
    prepared = campaign.prepare(tmp_path, data, policy())
    assert [r["identity"]["task"] for r in prepared["tasks"]] == [8, 9, 14, 15]
    for index in range(4):
        row = campaign.run_task(tmp_path, index)
        assert row["status"] == "replay_verified"
        assert row["stages"]["primary"]["normalized_mse"] < 1e-12
        assert not row["native_program_executed"]
        assert not row["native_equivalence_certified"]
        assert not row["validation_or_test_access"]
        assert row["optimizer_calls"] == 0
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    monkeypatch.setattr(
        control, "replay_training", lambda *a, **kw: pytest.fail("replay renewed")
    )
    for index in range(4):
        campaign.run_task(tmp_path, index)
    assert before == {p: p.read_bytes() for p in before}
    assert campaign.report(tmp_path)["status"] == "complete"


@pytest.mark.parametrize(
    "mutation",
    [
        "split",
        "roster",
        "source",
        "coefficients",
        "expression",
        "history",
        "area_guard",
        "metrics",
    ],
)
def test_invalid_inputs_fail_before_work(tmp_path, monkeypatch, mutation):
    data = copy.deepcopy(inputs(monkeypatch))
    task = data["tasks"][2]
    if mutation == "split":
        task["training"]["name"] = "val"
    elif mutation == "roster":
        data["tasks"].reverse()
    elif mutation == "source":
        task["programs"]["x"]["function"] += " "
    elif mutation == "coefficients":
        task["programs"]["x"]["parameters"][0] = 99
    elif mutation == "expression":
        data["transcription_set"]["transcriptions"][2]["targets"]["x"] = "a*x"
    elif mutation == "history":
        data["transcription_set"]["transcriptions"][2]["rules"].append("filter")
    elif mutation == "area_guard":
        task["training"]["rows"][0]["fixed_covariates"]["area_down"] = 0.5
    else:
        data["native_validation_nmse"] = 0.1
    with pytest.raises(ValueError):
        campaign.prepare(tmp_path, data, policy())
    assert not (tmp_path / "plan.json").exists()


def test_interrupted_operation_is_not_restarted(tmp_path, monkeypatch):
    campaign.prepare(tmp_path, inputs(monkeypatch), policy())
    plan, data, _ = campaign.verify(tmp_path)
    identity = campaign.common._identity(plan, data.tasks[0])
    seal(
        tmp_path / "tasks/0/operations/primary/started.json",
        {
            "campaign": identity,
            "stage": "primary",
            "allowance_seconds": 5.0,
        },
    )
    original = control.replay_training
    calls = []

    def replay(item, context, training, settings, **kwargs):
        calls.append(settings.method)
        return original(item, context, training, settings, **kwargs)

    monkeypatch.setattr(control, "replay_training", replay)
    row = campaign.run_task(tmp_path, 0)
    assert calls == ["DOP853"]
    assert row["status"] == "replay_unverified"
    assert row["stages"]["primary"]["budget_charge_seconds"] == 5
    assert not row["stages"]["primary"]["budget_restarted"]


def test_operation_tampering_is_rejected(tmp_path, monkeypatch):
    campaign.prepare(tmp_path, inputs(monkeypatch), policy())
    campaign.run_task(tmp_path, 0)
    path = tmp_path / "tasks/0/operations/primary/progress.json"
    value = public._read(path)
    value["completed_trajectories"] += 1
    public._write(path, value)
    with pytest.raises(ValueError, match="checkpoint differs"):
        campaign.report(tmp_path)
