"""Two frozen repetitions, training-only controls, and nonrenewable stage budgets."""

import copy

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import llm_sr_numerical_control as control
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_control_campaign as campaign
from tests.test_llm_sr_numerical_control import example


def inputs():
    item, programs, _, training = example()
    raw = item.model_dump(mode="json")
    raw["benchmark_id"] = campaign.BENCHMARK
    raw["targets"] = {"Gp": "-a*Gp+z", "I": "-0.3*I", "U": "-0.4*U"}
    raw["hidden_states"]["z"]["initial"] = "0.5*Gp"
    raw["programs"] = dict.fromkeys(raw["targets"], raw["programs"]["x"])
    source = programs["x"].model_dump(mode="json")
    source["variables"] = ["Gp", "I", "U"]
    training = training.model_dump(mode="json")
    row = training["rows"][0]
    time = np.asarray(row["time"])
    row["targets"] = {
        "Gp": row["targets"]["x"],
        "I": (2 * np.exp(-0.3 * time)).tolist(),
        "U": (3 * np.exp(-0.4 * time)).tolist(),
    }
    return {
        "protocol": "llm-sr-numerical-controls-input-1",
        "source_plan_sha256": "a" * 64,
        "transcription_set": {
            "protocol": "phase-c-llm-sr-ode-transcriptions-1",
            "status": "proposed_pending_review",
            "source_run": "synthetic",
            "source_plan_sha256": "a" * 64,
            "rules": {"filter": "synthetic supplied filter"},
            "transcriptions": [raw],
        },
        "training": training,
        "context": {"targets": ["Gp", "I", "U"]},
        "tasks": [
            {
                "task": 4,
                "benchmark_id": campaign.BENCHMARK,
                "repetition": 0,
                "programs": dict.fromkeys(("Gp", "I", "U"), source),
            },
            {
                "task": 5,
                "benchmark_id": campaign.BENCHMARK,
                "repetition": 1,
                "programs": {},
                "unavailable_reason": "No causal transcription supplied.",
            },
        ],
    }


def policy():
    return campaign.Policy(
        primary=control.ReplaySettings(seconds=5),
        verification=control.ReplaySettings(
            method="DOP853",
            relative_tolerance=1e-9,
            absolute_tolerance=1e-11,
            seconds=5,
        ),
    )


def test_complete_fixed_roster_replay_and_read_only_resume(tmp_path, monkeypatch):
    data = inputs()
    prepared = campaign.prepare(tmp_path, data, policy())
    assert [r["identity"]["task"] for r in prepared["tasks"]] == [4, 5]
    assert not (tmp_path / "tasks").exists()
    first = campaign.run_task(tmp_path, 0)
    assert first["status"] == "replay_verified"
    assert first["stages"]["primary"]["normalized_mse"] < 1e-10
    assert first["transcription_review_status"] == "proposed_pending_review"
    assert first["solver_agreement"]["verified"]
    assert not first["refit_ready"] and not first["native_program_executed"]
    missing = campaign.run_task(tmp_path, 1)
    assert missing["status"] == "unavailable_transcription"
    assert missing["stage_cost"]["operations"] == 0
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    monkeypatch.setattr(
        control, "replay_training", lambda *a, **kw: pytest.fail("budget renewed")
    )
    assert campaign.run_task(tmp_path, 0) == first
    assert campaign.prepare(tmp_path, data, policy()) == campaign.report(tmp_path)
    assert before == {p: p.read_bytes() for p in before}
    report = campaign.report(tmp_path)
    assert report["status"] == "complete" and not report["model_selection_performed"]
    assert len(report["tasks"]) == 2


def test_interrupted_stage_is_charged_and_never_retried(tmp_path, monkeypatch):
    campaign.prepare(tmp_path, inputs(), policy())
    plan, data, _ = campaign.verify(tmp_path)
    identity = campaign._identity(plan, data.tasks[0])
    folder = tmp_path / "tasks/0/operations/primary"
    seal(
        folder / "started.json",
        {"campaign": identity, "stage": "primary", "allowance_seconds": 5.0},
    )
    original = control.replay_training
    seen = []

    def replay(item, context, training, settings, **kwargs):
        seen.append(settings.method)
        return original(item, context, training, settings, **kwargs)

    monkeypatch.setattr(control, "replay_training", replay)
    result = campaign.run_task(tmp_path, 0)
    assert seen == ["DOP853"]
    assert result["status"] == "replay_unverified"
    stage = result["stages"]["primary"]
    assert stage["operation_status"] == "interrupted"
    assert stage["budget_charge_seconds"] == 5 and not stage["budget_restarted"]
    assert not result["stage_cost"]["timing_accounting_complete"]
    campaign.run_task(tmp_path, 0)
    assert seen == ["DOP853"]


@pytest.mark.parametrize(
    "mutation", ["split", "roster", "source", "coefficients", "metric"]
)
def test_inputs_fail_closed_before_prepare(tmp_path, mutation):
    data = copy.deepcopy(inputs())
    if mutation == "split":
        data["training"]["name"] = "val"
    elif mutation == "roster":
        data["tasks"] = data["tasks"][:1]
    elif mutation == "source":
        data["tasks"][0]["programs"]["Gp"]["function"] += " "
    elif mutation == "coefficients":
        data["tasks"][0]["programs"]["Gp"]["parameters"][0] = 99
    else:
        data["native_validation_nmse"] = 0.01
    with pytest.raises((ValueError, ValidationError)):
        campaign.prepare(tmp_path, data, policy())
    assert not (tmp_path / "plan.json").exists()


def test_changed_settings_or_checkpoint_cannot_rewrite_a_replay(tmp_path):
    campaign.prepare(tmp_path, inputs(), policy())
    changed = policy().model_copy(update={"solver_agreement_tolerance": 1e-6})
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(tmp_path, inputs(), changed)
    campaign.run_task(tmp_path, 0)
    folder = tmp_path / "tasks/0/operations/primary"
    checkpoint = public._read(folder / "progress.json")
    checkpoint["completed_trajectories"] += 1
    public._write(folder / "progress.json", checkpoint)
    with pytest.raises(ValueError, match="checkpoint differs"):
        campaign.report(tmp_path)


def test_bypassed_settings_validation_cannot_raise_runtime_allowance(tmp_path):
    unchecked = policy().model_copy(
        update={"primary": control.ReplaySettings().model_copy(update={"seconds": 500})}
    )
    with pytest.raises(ValueError):
        campaign.prepare(tmp_path, inputs(), unchecked)
    assert not (tmp_path / "plan.json").exists()


def test_solver_disagreement_is_retained_not_promoted(tmp_path, monkeypatch):
    campaign.prepare(tmp_path, inputs(), policy())
    original = control.replay_training

    def replay(item, context, training, settings, **kwargs):
        result = original(item, context, training, settings, **kwargs)
        if settings.method == "DOP853":
            result["trajectories"][0]["predictions"]["I"][-1] += 1
        return result

    monkeypatch.setattr(control, "replay_training", replay)
    result = campaign.run_task(tmp_path, 0)
    assert result["solver_agreement"]["available"]
    assert not result["solver_agreement"]["verified"]
    assert result["status"] == "replay_unverified"
    assert not result["model_selected"]
    assert read_seal(tmp_path / "tasks/0/operations/verification/result.json")["value"]
