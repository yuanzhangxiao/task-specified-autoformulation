"""Real fixed-point export, lineage rejection, and no renewed interrupted budgets."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from autoformalism.fitting import fit_residual_feedback as feedback
from autoformalism.fitting import public_fitting as public

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC = importlib.util.spec_from_file_location(
        "residual_feedback_smoke", ROOT / "scripts/smoke_residual_feedback.py"
    )
    SMOKE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(SMOKE)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    return SMOKE.fixture_history(tmp_path_factory.mktemp("feedback-history"))


@pytest.fixture
def prepared(history, tmp_path):
    parent, continuation, selection = history
    output = tmp_path / "feedback"
    feedback.prepare_residual_feedback(parent, continuation, output, selection)
    return output


def forbid(*args, **kwargs):
    pytest.fail("unexpected new numerical work")


def test_real_training_replay_carries_initial_map_and_immutable_resume(
    prepared, history, monkeypatch
):
    parent, continuation, _ = history
    before = {
        str(p): p.read_bytes()
        for root in (parent, continuation)
        for p in root.rglob("*.json")
    }
    real = feedback.simulate_trajectory
    calls = []

    def simulate(model, trajectory, parameters, initials, settings, **kwargs):
        assert trajectory.trajectory_id.startswith("train")
        assert initials == {}
        assert "init_z_scale" in parameters
        calls.append(trajectory.trajectory_id)
        return real(model, trajectory, parameters, initials, settings, **kwargs)

    monkeypatch.setattr(feedback, "simulate_trajectory", simulate)
    result = feedback.run_residual_feedback(prepared)
    assert result["status"] == "ready"
    assert result["current_score_agrees"] and result["previous_comparison_available"]
    assert len(calls) == 4
    packet = result["packet"]
    assert packet["normalized_mse"] < packet["previous_normalized_mse"]
    assert packet["training_content_sha256"] == public.content_sha256(
        result["seed"]["parent"]["freeze"]["training"]
    )
    assert result["optimization_calls"] == 0
    monkeypatch.setattr(feedback, "simulate_trajectory", forbid)
    assert feedback.run_residual_feedback(prepared) == result
    assert feedback.load_residual_feedback(prepared) == result
    assert before == {
        str(p): p.read_bytes()
        for root in (parent, continuation)
        for p in root.rglob("*.json")
    }


def test_consumed_export_never_restarts_numerics(prepared, monkeypatch):
    frozen = public._read(prepared / "freeze.json")
    public._write(prepared / "started.json", {"identity": frozen["identity"]})
    monkeypatch.setattr(feedback, "simulate_trajectory", forbid)
    result = feedback.run_residual_feedback(prepared)
    assert result["status"] == "interrupted" and result["packet"] is None
    assert all(
        v == "Fixed replay interrupted; no automatic fresh budget"
        for v in result["errors"].values()
    )


def test_recover_current_packet_when_optional_previous_interrupted(
    prepared, monkeypatch
):
    original = feedback.run_residual_feedback(prepared)
    (prepared / "result.json").unlink()
    (prepared / "points/previous/result.json").unlink()
    monkeypatch.setattr(feedback, "simulate_trajectory", forbid)
    result = feedback.run_residual_feedback(prepared)
    assert result["status"] == "ready" and not result["previous_comparison_available"]
    assert result["packet"]["previous_normalized_mse"] is None
    assert result["packet"]["normalized_mse"] == original["packet"]["normalized_mse"]


def test_recover_only_publication_without_new_clock(prepared, monkeypatch):
    original = feedback.run_residual_feedback(prepared)
    (prepared / "result.json").unlink()
    monkeypatch.setattr(feedback, "simulate_trajectory", forbid)
    assert feedback.run_residual_feedback(prepared) == original


def test_unreproduced_score_does_not_reach_proposer(prepared, monkeypatch):
    real = feedback.simulate_trajectory

    def wrong(*args, **kwargs):
        result = real(*args, **kwargs)
        for value in result.predictions.values():
            value += 10
        return result

    monkeypatch.setattr(feedback, "simulate_trajectory", wrong)
    result = feedback.run_residual_feedback(prepared)
    assert result["status"] == "replay_failed"
    assert result["packet"] is None and not result["current_score_agrees"]


def test_previous_failure_does_not_discard_current(prepared, monkeypatch):
    real = feedback.simulate_trajectory

    def previous_fails(model, trajectory, parameters, *args, **kwargs):
        if parameters["init_z_scale"] == 0.3:
            raise RuntimeError("synthetic previous-only failure")
        return real(model, trajectory, parameters, *args, **kwargs)

    monkeypatch.setattr(feedback, "simulate_trajectory", previous_fails)
    result = feedback.run_residual_feedback(prepared)
    assert result["status"] == "ready"
    assert not result["previous_comparison_available"]
    assert "previous" in result["errors"]


def test_changed_source_rejected_without_replay(prepared, monkeypatch):
    monkeypatch.setattr(public, "_source_identity", lambda: "0" * 64)
    monkeypatch.setattr(feedback, "simulate_trajectory", forbid)
    with pytest.raises(ValueError, match="source or runtime"):
        feedback.run_residual_feedback(prepared)


def test_changed_seed_rejected(prepared, monkeypatch):
    monkeypatch.setattr(feedback, "_seed", lambda *args: {})
    with pytest.raises(ValueError, match="historical seed"):
        feedback.load_residual_feedback(prepared)


def test_point_and_packet_tampering_rejected(prepared):
    feedback.run_residual_feedback(prepared)
    path = prepared / "points/current/result.json"
    original = path.read_bytes()
    value = public._read(path)
    value["result"]["normalized_mse"] += 1
    public._write(path, value)
    with pytest.raises(ValueError):
        feedback.load_residual_feedback(prepared)
    path.write_bytes(original)
    path = prepared / "result.json"
    value = public._read(path)
    value["result"]["packet"]["normalized_mse"] += 1
    public._write(path, feedback.lineage._seal(value["result"]))
    with pytest.raises(ValueError, match="digest"):
        feedback.load_residual_feedback(prepared)


def test_historical_outputs_cannot_be_reused(history):
    parent, continuation, selection = history
    for output in (
        parent.parent / "new",
        continuation.parent / "new",
        parent.parent.parent,
    ):
        with pytest.raises(ValueError, match="separate"):
            feedback.prepare_residual_feedback(parent, continuation, output, selection)
