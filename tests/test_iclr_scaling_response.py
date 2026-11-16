"""Check frozen response diagnostics, coverage, timing scope and resume guards."""

import numpy as np
import pytest

from scripts import iclr_scaling_response as analysis


def test_gain_and_sign_are_separate_from_response_amplitude():
    truth = np.array([0.0, 3.0, -1.0, 0.0])
    zero = np.zeros(4)
    row = analysis.response_metrics(truth, zero, 4 * truth, zero)
    assert row["sign_agreement"] == 1
    assert row["best_scale"] == 0.25
    assert row["shape_rse"] == 0
    # A correct dominant lobe gives 75% agreement, despite a wrong smaller lobe.
    row = analysis.response_metrics(truth, zero, abs(truth), zero)
    assert row["sign_agreement"] == 0.75
    assert row["shape_rse"] == pytest.approx(0.36)


def test_zero_opposite_and_negligible_responses():
    truth = np.array([0.0, 1.0, 2.0])
    zero = np.zeros(3)
    for prediction in (zero, -truth):
        row = analysis.response_metrics(truth, zero, prediction, zero)
        assert row["sign_agreement"] == 0
        assert row["shape_rse"] == 1
    assert analysis.response_metrics(zero, zero, truth, zero) == {
        "status": "negligible_reference"
    }
    with pytest.raises(ValueError, match="finite response vectors"):
        analysis.response_metrics(truth, zero, [0, float("nan"), 0], zero)
    with pytest.raises(ValueError, match="finite response vectors"):
        analysis.response_metrics(truth, zero, [1, 2], zero)


def test_run_direction_accuracy_averages_continuous_scores_without_thresholding():
    half = analysis.response_metrics([0, 1, -1], [0, 0, 0], [0, 1, 1], [0, 0, 0])
    three_quarters = analysis.response_metrics(
        [0, 3, -1], [0, 0, 0], [0, 3, 1], [0, 0, 0]
    )
    assert half["sign_agreement"] == 0.5
    assert three_quarters["sign_agreement"] == 0.75
    diagnostics = [{"status": "negligible_reference"}, half, three_quarters]
    scores = analysis.aggregate_responses(diagnostics)
    assert scores["direction_accuracy_pct"] == 62.5
    assert scores["shape_rse"] == pytest.approx(0.68)
    with pytest.raises(ValueError, match="two active contrasts"):
        analysis.aggregate_responses(diagnostics[:2])


def test_checkpoint_resume_checks_plan_and_reference(tmp_path, monkeypatch):
    calls = []

    def predict(*args):
        calls.append(1)
        return {"y": np.array([0.0, 1.0])}, {"y": np.zeros(2)}

    monkeypatch.setattr(analysis.stress, "predict", predict)
    plan, endpoint, pair = {"artifact_sha256": "p"}, {"source_id": "s"}, {"case": "c"}
    for _ in range(2):
        p, q = analysis.cached_predict(tmp_path, plan, endpoint, {}, pair)
        assert p["y"].tolist() == [0, 1] and q["y"].tolist() == [0, 0]
    assert len(calls) == 1
    with pytest.raises(ValueError, match="stale rollout"):
        analysis.cached_predict(tmp_path, plan, endpoint, {}, pair | {"changed": True})
    with pytest.raises(ValueError, match="stale rollout"):
        analysis.cached_predict(tmp_path, {"artifact_sha256": "q"}, endpoint, {}, pair)


def test_parity_catches_case_changes_even_when_means_match():
    old = {
        "evaluation_status": "complete",
        "test_nmse": 2,
        "response_rse": 3,
        "trajectories": [
            {"case": "p", "channels": {"y": {"nmse": 2, "response_rse": 3}}}
        ],
    }
    analysis.assert_parity(old, old)
    new = old | {
        "trajectories": [
            {"case": "p", "channels": {"y": {"nmse": 1, "response_rse": 3}}}
        ]
    }
    with pytest.raises(ValueError, match="parity failed"):
        analysis.assert_parity(old, new)
    with pytest.raises(ValueError, match="incomplete"):
        analysis.assert_parity(old, old | {"evaluation_status": "unresolved"})


def test_summary_orders_extrema_and_keeps_failed_run_costs():
    base = {
        "dimension": 4,
        "variant": "dimension",
        "status": "complete",
        "test_nmse": 1,
        "response_rse": 1,
        "direction_accuracy_pct": 62.5,
        "shape_rse": 0.5,
        "fit_minutes": 1,
        "total_minutes": 2,
        "proposer_tokens_k": 3,
    }
    rows = [
        base,
        base | {"test_nmse": 5, "direction_accuracy_pct": 87.5},
        base | {"status": "model_unavailable", "total_minutes": 50},
    ]
    (group,) = analysis.summarize(rows)
    assert (group["retained"], group["planned"]) == (2, 3)
    assert group["statistics"]["test_nmse"] == {
        "median": 3,
        "worst": 5,
        "best": 1,
        "n": 2,
    }
    assert group["statistics"]["direction_accuracy_pct"] == {
        "median": 75,
        "worst": 62.5,
        "best": 87.5,
        "n": 2,
    }
    assert group["all_attempt_costs"]["total_minutes"]["worst"] == 50


def test_cost_completeness_and_parts():
    cost = {
        "fit_seconds": 60,
        "visit_seconds": 120,
        "pruning_seconds": 60,
        "search_and_pruning_wall_seconds": 180,
        "observed_proposal_tokens": 4000,
    }
    assert analysis.cost_values(cost) == {
        "fit_minutes": 1,
        "total_minutes": 3,
        "proposer_tokens_k": 4,
    }
    with pytest.raises(ValueError, match="incomplete"):
        analysis.cost_values(cost | {"unknown_visit_costs": 1})
    with pytest.raises(ValueError, match="inconsistent"):
        analysis.cost_values(cost | {"search_and_pruning_wall_seconds": 999})
