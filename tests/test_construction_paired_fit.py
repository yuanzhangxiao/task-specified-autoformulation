"""Matched fits, failure visibility and deterministic resume."""

import json
from copy import deepcopy

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_paired_fit as campaign
from tests.test_construction_paired_inputs import synthetic_inputs


def policy():
    return campaign.Policy(
        population="synthetic-smoke",
        search_seconds=12,
        diagnostic_seconds=12,
        verification_seconds=12,
        inference_seconds=12,
        forecast_seconds=12,
        maximum_calls=60,
        inference_calls=40,
    )


@pytest.fixture
def completed(tmp_path):
    data = synthetic_inputs()
    plan = campaign.prepare(tmp_path, data, policy())
    for index in range(2):
        campaign.run_task(tmp_path, index)
    return tmp_path, data, plan


def test_numerical_pair_and_frozen_prefix_forecast(completed, monkeypatch):
    root, _, plan = completed
    report = campaign.report(root)
    assert report["recorded"] == report["numerically_verified"] == 2
    assert report["validation_available"] == 2
    assert len(report["pairs"]) == 1
    for row in report["rows"]:
        assert row["training_nmse"] < 1e-8
        assert row["validation_score"]["normalized_mse"] < 1e-7
        assert all(s["search_recorded"] for s in row["starts"])
        assert row["cost"]["accounting_complete"] is True
    assert report["pairs"][0]["validation"]["child_minus_parent"] == 0
    assert report["test_data_opened"] is False
    monkeypatch.setattr(
        campaign.numerical,
        "make_oracle",
        lambda *a, **kw: pytest.fail("resume/report must not redo numerics"),
    )
    for task in plan["tasks"]:
        result = campaign.run_task(root, task["index"])
        assert result["numerically_verified"]
    assert campaign.inspect(root)["recorded"] == 2


def test_initial_endpoint_retained_when_qualification_blocks_search(
    tmp_path, monkeypatch
):
    campaign.prepare(tmp_path, synthetic_inputs(), policy())
    monkeypatch.setattr(
        campaign.numerical,
        "qualification",
        lambda *a, **kw: {"status": "complete", "passed": False},
    )
    result = campaign.run_task(tmp_path, 0)
    assert result["numerically_verified"]
    assert result["optimization_status"] == "optimization_unresolved"
    assert all(s["stop_reason"] == "not_started" for s in result["starts"])
    assert not any(s["search_recorded"] for s in result["starts"])


def test_failed_verification_cannot_produce_reportable_model(tmp_path, monkeypatch):
    campaign.prepare(tmp_path, synthetic_inputs(), policy())
    monkeypatch.setattr(
        campaign.numerical,
        "verify_endpoint_ladder",
        lambda *a, **kw: {"status": "complete", "original_equations_verified": False},
    )
    result = campaign.run_task(tmp_path, 0)
    assert result["numerically_verified"] is False
    assert result["training_nmse"] is None
    assert result["validation_score"] is None
    report = campaign.report(tmp_path)
    assert report["expected"] == 2 and report["recorded"] == 1
    assert report["aggregates"]["validation"]["unavailable_pairs"] == 1


def test_receipt_tampering_and_selection_tampering_rejected(completed):
    root, _, _ = completed
    folder = root / "results/child-00"
    path = folder / "training_selection.json"
    packet = json.loads(path.read_text())
    packet["value"]["origin"] = "forged"
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        campaign.run_task(root, 0)
    with pytest.raises(ValueError):
        campaign.report(root)


def test_unknown_operations_and_runtime_drift_rejected(tmp_path, monkeypatch):
    campaign.prepare(tmp_path, synthetic_inputs(), policy())
    (tmp_path / "results/child-00/operations/extra_restart").mkdir(parents=True)
    with pytest.raises(ValueError, match="unknown paired fit operation"):
        campaign.report(tmp_path)
    monkeypatch.setattr(public, "_source_identity", lambda: "different")
    with pytest.raises(ValueError, match="source/runtime"):
        campaign.verify(tmp_path)
    campaign.verify(tmp_path, runtime=False)


def test_interrupted_reservation_never_restarts(tmp_path):
    data = synthetic_inputs()
    plan = campaign.prepare(tmp_path, data, policy())
    task = plan["tasks"][0]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = tmp_path / "results" / task["task_id"]
    allowance = campaign._allowances(policy(), data)["declared_search"]

    def interrupted(save):
        save({"marker": "persisted"})
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        campaign.journal._operation(
            folder, identity, "declared_search", allowance, interrupted
        )
    resumed = campaign.journal._operation(
        folder,
        identity,
        "declared_search",
        allowance,
        lambda save: pytest.fail("interrupted allowance must not restart"),
    )
    assert resumed["budget_restarted"] is False
    assert resumed["observed_progress"] == {"marker": "persisted"}


def test_two_start_policy_name_stable_and_bounded():
    class Oracle:
        names = ("a", "trajectory:one:z")
        start = dict.fromkeys(names, 0.0)
        units = np.asarray([2.0, 4.0])
        lower = np.asarray([-0.1, -10.0])
        upper = np.asarray([0.1, 10.0])

        def vector(self, p):
            return np.asarray([p[n] for n in self.names])

        def parameters(self, x):
            return dict(zip(self.names, x, strict=True))

    oracle = Oracle()
    first = campaign.starts(oracle)
    assert first == campaign.starts(oracle)
    assert first[0] == oracle.start and first[1] != oracle.start
    assert -0.1 <= first[1]["a"] <= 0.1
    oracle.names = tuple(reversed(oracle.names))
    oracle.units = oracle.units[::-1]
    oracle.lower, oracle.upper = oracle.lower[::-1], oracle.upper[::-1]
    assert campaign.starts(oracle)[1] == first[1]


def test_prepare_immutable_and_no_validation_selection(tmp_path):
    data = synthetic_inputs()
    plan = campaign.prepare(tmp_path, data, policy())
    assert read_seal(tmp_path / "plan.json") == plan
    assert campaign.prepare(tmp_path, data, policy()) == plan
    changed = deepcopy(data)
    changed["validation"]["rows"][0]["targets"]["y"][-1] += 100
    with pytest.raises(ValueError):
        campaign.prepare(tmp_path, changed, policy())
    assert plan["selection"] == "verified_training_error_only_before_validation"
    assert plan["model_selection"] is False


def test_validation_aggregate_requires_all_trajectories():
    score = {"sample_count": 2, "per_target_normalized_mse": {"y": 3.0}}
    assert campaign._aggregate([score], 2) is None
    assert campaign._aggregate([score, score], 2)["normalized_mse"] == 3.0
    assert campaign._spread([1, 2, 1000]) == {"count": 3, "median": 2, "mad": 1}


def test_nonfinite_validation_score_is_unavailable(completed, monkeypatch):
    root, data, plan = completed
    task = plan["tasks"][0]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    operations = campaign._operations(
        root / "results" / task["task_id"],
        identity,
        campaign._allowances(policy(), data),
    )
    monkeypatch.setattr(
        campaign.kernel,
        "score_forecast",
        lambda *a: {
            "sample_count": 1,
            "normalized_mse": float("inf"),
            "per_target_normalized_mse": {"y": float("inf")},
        },
    )
    result = campaign._assemble(identity, operations, data)
    assert result["validation_score"] is None
    assert result["validation_available_rows"] == 0
    assert all(r["scoring_error"] for r in result["validation_rows"])


def test_large_finite_scores_have_finite_descriptive_statistics():
    values = campaign._spread([1e308, 1e308])
    assert values == {"count": 2, "median": 1e308, "mad": 0.0}
    score = {"sample_count": 2, "per_target_normalized_mse": {"y": 1e308, "z": 1e308}}
    assert campaign._aggregate([score, score], 2)["normalized_mse"] == 1e308
