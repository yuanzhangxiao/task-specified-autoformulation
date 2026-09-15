"""Saved-artifact transfer, accurate failure attribution and immutable fit resume."""

import json
from pathlib import Path

import pytest

from autoformalism.fitting import public_fitting
from autoformalism.rebuttal import prefit_fit_handoff as handoff
from autoformalism.rebuttal import prefit_requirements as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.search.repair_provenance import STAGE_AWARE
from scripts.smoke_prefit_requirements import repair_client, synthetic_plan


def fixture_run(base, *, feedback_policy="legacy", truncated=False):
    plan = synthetic_plan(
        base,
        with_validation=True,
        repair_policy="topology-owned-sign-1",
        feedback_policy=feedback_policy,
    )
    source = base / "campaign"
    task = next(
        t
        for t in plan["tasks"]
        if t["arm"] == "requirement_feedback" and t["cohort"] == "repair"
    )
    calls = []
    client = repair_client(source, plan, task, calls)
    original = client.transport

    def transport(*args):
        record = original(*args)
        if truncated and len(calls) == 1:
            record["choices"][0]["finish_reason"] = "length"
        return record

    client.transport = transport
    result = campaign.run_episode(source, plan, task, client)
    selection = handoff.HandoffSelection(
        source_plan_sha256=plan["artifact_sha256"],
        construction_plan_sha256=plan["source_plan_sha256"],
        task_id=task["task_id"],
        profile="general-rollout-v1",
    )
    return selection, result, calls


@pytest.fixture
def selected(tmp_path):
    selection, result, calls = fixture_run(tmp_path, truncated=True)
    return tmp_path, selection, result, calls


def prepare(base, selection):
    return handoff.prepare_handoff(
        base / "campaign",
        base / "source",
        base / "source/public",
        selection,
        base / "output",
    )


def test_export_reconstructs_accepted_model_and_corrects_historical_reporting(selected):
    base, selection, final, _ = selected
    before = {
        p: p.read_bytes()
        for directory in ("source", "campaign")
        for p in (base / directory).rglob("*")
        if p.is_file()
    }
    report = prepare(base, selection)
    exported = sealed_read(base / "output/handoff.json")
    frozen = json.loads((base / "output/fit/freeze.json").read_text())
    assert frozen["lowered_candidate"] == final["final"]["candidate"]
    assert exported["request"]["parameter_guesses"] == {}
    assert not exported["request"]["context"]["fitted_initialization"]
    assert report["repair_provenance"]["rhs_changed"]
    assert report["repair_provenance"]["initialization_plan_preserved"]
    assert not report["repair_provenance"]["selected_slot_preserved_without_call"]
    failure = report["attempt_evidence"][0]["current_diagnosis"]
    assert failure["code"] == "PROVIDER_RESPONSE_TRUNCATED"
    assert not failure["mathematical_reply_evaluated"]
    assert "interaction IDs" not in failure["next_action"]
    assert (
        report["attempt_evidence"][0]["historical_feedback"]["code"]
        == "REPAIR_CONTRACT_VIOLATION"
    )
    assert prepare(base, selection) == report
    assert before == {p: p.read_bytes() for p in before}


def test_live_stage_aware_feedback_and_provenance(tmp_path):
    _, state, calls = fixture_run(tmp_path, feedback_policy=STAGE_AWARE, truncated=True)
    retry = json.loads(calls[1]["messages"][1]["content"].split("\n", 1)[1])[
        "retry_feedback"
    ]
    assert retry["stage"] == "provider_response"
    assert retry["code"] == "PROVIDER_RESPONSE_TRUNCATED"
    selected = state["final"]["selected_function"]
    assert (
        "rhs_changed" not in selected
        and "preserved_without_atomic_call" not in selected
    )
    assert "construction_provenance" in selected
    assert state["final"]["requirement_repair_provenance"]["rhs_changed"]
    selection = handoff.HandoffSelection(
        source_plan_sha256=sealed_read(tmp_path / "campaign/plan.json")[
            "artifact_sha256"
        ],
        construction_plan_sha256=sealed_read(tmp_path / "source/plan.json")[
            "artifact_sha256"
        ],
        task_id=next(
            t["task_id"]
            for t in sealed_read(tmp_path / "campaign/plan.json")["tasks"]
            if t["cohort"] == "repair" and t["arm"] == "requirement_feedback"
        ),
        profile="general-rollout-v1",
    )
    assert prepare(tmp_path, selection)["status"] == "prepared"


@pytest.mark.parametrize("kind", ["candidate", "acceptance", "reply", "unfinished"])
def test_resealed_invalid_repair_cannot_enter_fitting(selected, kind):
    base, selection, _, _ = selected
    path = base / "campaign/results" / selection.task_id / "state.json"
    state = sealed_read(path)
    state.pop("artifact_sha256")
    if kind == "candidate":
        state["final"]["candidate"]["state_equations"][0]["rhs"] = "0"
    elif kind == "acceptance":
        state["attempts"][0]["accepted"] = True
    elif kind == "reply":
        state["attempts"][-1]["response"]["expression"] = "0"
    else:
        state["status"] = "running"
    path.write_text(
        json.dumps({**state, "artifact_sha256": campaign.content_hash(state)})
    )
    with pytest.raises(ValueError):
        prepare(base, selection)
    assert not (base / "output/fit/started.json").exists()


def test_wrong_plan_selection_and_overlapping_destination_fail(selected):
    base, selection, _, _ = selected
    with pytest.raises(ValueError, match="selection"):
        prepare(base, selection.model_copy(update={"source_plan_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="separate"):
        handoff.prepare_handoff(
            base / "campaign",
            base / "source",
            base / "source/public",
            selection,
            base / "source/new",
        )


def test_changed_training_is_rejected_before_loader_or_fitter(selected, monkeypatch):
    base, selection, _, _ = selected
    path = base / "source/public/phase_b_v1" / selection.cell / "train.csv"
    path.write_text(path.read_text() + "\n")
    monkeypatch.setattr(
        handoff.construction,
        "load_development",
        lambda *args: pytest.fail("must reject before loading"),
    )
    with pytest.raises(ValueError, match="public asset"):
        prepare(base, selection)


def test_no_test_or_private_access_during_export(selected, monkeypatch):
    base, selection, _, _ = selected
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.name != "test.csv" and "private_reference" not in path.parts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert prepare(base, selection)["status"] == "prepared"


def test_terminal_fit_feedback_and_resume_are_candidate_bound(selected, monkeypatch):
    base, selection, _, _ = selected
    prepare(base, selection)
    calls = []

    def backend(request, model, training, validation, guesses, settings, directory):
        calls.append(request)
        metric = {
            "normalized_mse": 0.8,
            "per_target_normalized_mse": {"v01": 0.8},
            "failed_trajectories": [],
        }
        return {
            "global_parameters": guesses,
            "training_metrics": metric,
            "validation_metrics": metric,
            "diagnostics": [],
            "best_start_index": None,
        }

    monkeypatch.setattr(public_fitting, "_run_backend", backend)
    report = handoff.run_handoff(base / "output")
    assert report["status"] == "complete"
    assert report["numerical_feedback"]["route"] == "model_review_with_fit_evidence"
    assert not report["numerical_feedback"]["structural_infeasibility_proven"]
    assert report["numerical_feedback"]["scientific_verdict"] is None
    assert len(calls) == 1
    assert handoff.run_handoff(base / "output") == report
    assert len(calls) == 1
    saved = json.loads((base / "output/fit/result.json").read_text())
    saved["result"]["source"]["task_id"] = "another"
    saved["sha256"] = public_fitting.content_sha256(saved["result"])
    (base / "output/fit/result.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="lineage"):
        handoff.summarize_handoff(base / "output")


def test_interruption_does_not_start_another_optimizer(selected, monkeypatch):
    base, selection, _, _ = selected
    prepared = prepare(base, selection)
    (base / "output/fit/started.json").write_text(
        json.dumps({"identity": prepared["capability"]["identity"]})
    )
    monkeypatch.setattr(
        public_fitting, "_run_backend", lambda *args: pytest.fail("fresh budget")
    )
    report = handoff.run_handoff(base / "output")
    assert report["status"] == "interrupted"
    assert report["numerical_feedback"]["route"] == "execution_recovery_required"
