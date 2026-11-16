"""Check historical provenance, held-out contracts, coverage and metric pooling."""

import copy

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.staged_topology import content_hash
from scripts import iclr_historical_observation as study


def test_freeze_reads_archived_dataset_and_requires_matched_training_targets(
    tmp_path, monkeypatch
):
    rows = [
        {"cell": cell, "status": "ready", "context": {"targets": ["Gp"]}}
        for cell in study.CELLS
    ]
    campaign = {"public_assets": {c: {"train.csv": "csv"} for c in study.CELLS}}
    monkeypatch.setattr(study, "load_archive", lambda _: ({}, campaign))
    monkeypatch.setattr(study, "sealed_read", lambda _: {"artifact_sha256": "inv"})
    monkeypatch.setattr(study, "retained_rows", lambda *_: rows)
    monkeypatch.setattr(study.metrics, "digest", lambda _: "source")
    changed = []

    def training(path, *_):
        values = [1.0, 2.0, 3.0]
        if changed and "t1_canonical_named_hard" in str(path):
            values[-1] = 4.0
        tr = Trajectory(
            trajectory_id="train",
            time=np.array([0.0, 1.0, 2.0]),
            targets={"Gp": np.array(values)},
            auxiliaries={},
            external_inputs={},
            fixed_covariates={},
            derivatives={},
        )
        return DatasetSplit(SplitName.TRAIN, (tr,), "csv")

    monkeypatch.setattr(study, "read_training", training)
    plan = study.freeze(tmp_path, tmp_path, tmp_path, tmp_path)
    assert all(
        s["Gp"] == pytest.approx(np.std([1, 2, 3]))
        for s in plan["training_sd"].values()
    )
    changed.append(True)
    with pytest.raises(ValueError, match="common targets differ"):
        study.freeze(tmp_path, tmp_path, tmp_path, tmp_path)


def test_schedules_keep_legacy_encoding_and_matched_controls():
    for task in ("T1", "T2"):
        tests = study.schedules(task, "original_heldout")
        contrasts = [v for v in tests.values() if v["role"] == "contrast"]
        assert len(contrasts) == 6
        shifted = next(
            v for v in contrasts if "initial_shift" in v["protocol"]["specification"]
        )
        assert (
            tests[shifted["control"]]["protocol"]["specification"]["initial_shift"]
            == shifted["protocol"]["specification"]["initial_shift"]
        )
        confirmation = study.schedules(task, "confirmation_schedules")
        assert confirmation["fed"]["protocol"]["specification"]["meals"] == [[41, 53]]
        assert confirmation["split"]["protocol"]["specification"]["meals"] == [
            [41, 26.5],
            [109, 26.5],
        ]
        assert confirmation["split"]["control"] == "fed"
        for item in list(tests.values()) + list(confirmation.values()):
            p = item["protocol"]
            assert p["input_contract"] == "legacy-events-1"
            assert "meal_event_g" in p["input_names"]
            assert "meal_rate_g_per_min" not in p["input_names"]
            assert p["split"] == "test"
    with pytest.raises(ValueError, match="unknown panel"):
        study.schedules("T1", "new")


def archive_fixture():
    rows, occurrences, models = [], [], {}
    for i, cell in enumerate(study.CELLS):
        for seed in (0, 1):
            task = f"cell{i}_seed{seed}_full_c1v1s1"
            ready = (i, seed) != (2, 0)
            rows.append(
                {
                    "task_id": task,
                    "round": study.CUTOFF,
                    "cell": cell,
                    "seed": seed,
                    "arm": "full",
                    "result_sha256": task,
                    "retained_validation_nmse": 0.5 if ready else None,
                }
            )
            if not ready:
                continue
            model = {
                "cell": cell,
                "parameters": {"k": 1},
                "lowered_candidate_sha256": "lowered",
                "parameter_status": "saved_fit_vector",
                "candidate": {},
                "lowered_context": {},
                "initialization_plan": {},
            }
            key = content_hash(model)
            models[key] = model
            occurrences.append(
                {
                    "task": {"task_id": task},
                    "role": "selected",
                    "round": study.CUTOFF,
                    "source_sha256": task,
                    "model_id": key,
                    "fit": {
                        "parameters": {"k": 1},
                        "lowered_candidate_sha256": "lowered",
                        "status": "complete",
                        "validation_initials_fitted": False,
                        "training_only_parameter_estimation": True,
                        "validation": {"normalized_mse": 0.5},
                        "identity": "fit",
                    },
                }
            )
    return {"rows": rows}, {
        "plan_sha256": study.PLAN,
        "test_data_opened": False,
        "models": models,
        "occurrences": occurrences,
    }


def test_exact_historical_selection_keeps_missing_seed_and_rejects_wrong_fit():
    search, inv = archive_fixture()
    rows = study.retained_rows(search, inv)
    assert len(rows) == 8
    assert sum(r["status"] == "ready" for r in rows) == 7
    assert sum(r["status"] == "no_retained_model" for r in rows) == 1
    broken = copy.deepcopy(inv)
    broken["occurrences"][0]["source_sha256"] = "later round"
    with pytest.raises(ValueError, match="selected receipt"):
        study.retained_rows(search, broken)
    broken = copy.deepcopy(inv)
    broken["occurrences"][0]["fit"]["parameters"]["k"] = 2
    with pytest.raises(ValueError, match="fitting receipt"):
        study.retained_rows(search, broken)
    with pytest.raises(ValueError, match="denominator"):
        study.retained_rows({"rows": search["rows"][:-1]}, inv)


def test_future_targets_and_hidden_reference_are_not_exposed():
    row = {
        "context": {
            "targets": ["Gp", "I"],
            "auxiliaries": ["Gt"],
            "external_inputs": ["meal_event_g"],
        }
    }
    record = {
        "time": [0, 1, 2],
        "fields": {
            "Gp": [1, 999, 888],
            "I": [2, 33, 44],
            "Gt": [4, 5, 6],
            "meal_event_g": [0, 53, 0],
            "private_X": [91, 92, 93],
        },
    }
    tr = study.public_trajectory(row, record)
    assert tr.targets["Gp"].tolist() == [1, 1, 1]
    assert tr.targets["I"].tolist() == [2, 2, 2]
    assert tr.auxiliaries["Gt"].tolist() == [4, 5, 6]
    assert not tr.derivatives and not tr.fixed_covariates
    assert set(tr.external_inputs) == {"meal_event_g"}


def test_continuous_direction_shape_zero_effect_and_solver_failure():
    ref = {"time": [0, 1, 2], "fields": {"Gp": [0, 3, -1]}}
    control = {"time": [0, 1, 2], "fields": {"Gp": [0, 0, 0]}}
    a = {
        "status": "complete",
        "predicted": {"Gp": [0, 3, 1]},
        "alternate": {"Gp": [0, 3, 1]},
    }
    b = {
        "status": "complete",
        "predicted": {"Gp": [0, 0, 0]},
        "alternate": {"Gp": [0, 0, 0]},
    }
    result = study.score_pair(a, b, ref, control, "Gp", 2, "confirmation_schedules")
    assert result["test_nmse"] == pytest.approx(1 / 3)
    assert result["response_rse"] == 0.4
    assert result["directional_accuracy"] == 0.75
    assert result["shape_error"] == pytest.approx(0.36)
    empty = study.score_pair(b, b, control, control, "Gp", 2, "original_heldout")
    assert empty["test_nmse"] == 0 and empty["response_rse"] is None
    assert empty["effect_status"] == "negligible_reference"
    a["alternate"]["Gp"] = [0, 30, 1]
    assert (
        study.score_pair(a, b, ref, control, "Gp", 2, "original_heldout")["status"]
        == "effect_solver_disagreement"
    )


def test_pooling_uses_requested_centers_and_does_not_hide_failed_rollouts():
    rows = [
        {
            "status": "complete",
            "test_nmse": x,
            "response_rse": x,
            "directional_accuracy": a,
            "shape_error": a,
        }
        for x, a in [(1, 0.1), (2, 0.2), (100, 0.9)]
    ]
    result = study.pool(rows)
    assert result["test_nmse"] == result["response_rse"] == 2
    assert result["directional_accuracy"] == pytest.approx(0.4)
    assert result["shape_error"] == pytest.approx(0.4)
    assert study.pool([*rows, {"status": "rollout_unavailable"}]) == {
        "status": "evaluation_incomplete"
    }


def test_prediction_resume_binds_model_reference_and_plan(tmp_path, monkeypatch):
    calls = []

    def predict(*args):
        calls.append(1)
        return {"predicted": {"Gp": [0, 1]}, "alternate": {"Gp": [0, 1]}}

    monkeypatch.setattr(study, "predict", predict)
    plan, row = {"artifact_sha256": "plan"}, {"source_id": "s", "model_id": "m"}
    ref = {"artifact_sha256": "ref", "reference": {}}
    for _ in range(2):
        assert study.cached_prediction(tmp_path, plan, row, ref)["status"] == "complete"
    assert len(calls) == 1
    with pytest.raises(ValueError, match="stale model"):
        study.cached_prediction(tmp_path, plan, row | {"model_id": "changed"}, ref)
    with pytest.raises(ValueError, match="stale model"):
        study.cached_prediction(tmp_path, {"artifact_sha256": "other"}, row, ref)


def test_failed_prediction_is_checkpointed_without_successful_rerun(
    tmp_path, monkeypatch
):
    def fail(*args):
        raise ValueError("integration failed")

    monkeypatch.setattr(study, "predict", fail)
    args = (
        tmp_path,
        {"artifact_sha256": "plan"},
        {"source_id": "s", "model_id": "m"},
        {"artifact_sha256": "ref", "reference": {}},
    )
    assert study.cached_prediction(*args)["status"] == "rollout_unavailable"
    monkeypatch.setattr(study, "predict", lambda *a: pytest.fail("reran failure"))
    assert study.cached_prediction(*args)["reason"] == "integration failed"
