"""Match graph origins on the same train-only constructor, checks and fit budget."""

import numpy as np
import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import topology_forecast_fit as trial
from tests.test_ordered_driver_certificate import model, observations


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    graph, formula = model()
    frozen = {
        "case_id": "synthetic",
        "evidence": observations().model_dump(mode="json"),
        "public_brief": {"public_variables": []},
        "histories": ["x", "w"],
    }
    source = tmp_path / "topology.json"
    topo = sealed_write(source, {"frozen_cases": {"synthetic": frozen}})
    rows = frozen["evidence"]["trajectories"]
    train = sealed_write(tmp_path / "train.json", {"train": {"d": rows}})
    monkeypatch.setattr(trial.intake, "INSPECTION", source)
    monkeypatch.setattr(trial.intake, "ORIGINAL_INTAKE", topo["artifact_sha256"])
    monkeypatch.setattr(trial, "TRAIN", train["artifact_sha256"])
    monkeypatch.setattr(trial, "INPUTS", tmp_path)
    monkeypatch.setattr(trial.pointwise, "DATASET", "d")
    monkeypatch.setattr(trial.pointwise, "TARGET", "y")
    trial._training.cache_clear()
    yield {"case": "synthetic", "dataset": "d", "graph": graph}, formula, rows
    trial._training.cache_clear()
    trial._registered.cache_clear()


def test_common_constructor_budget_and_postfit_scores(cohort, tmp_path, monkeypatch):
    row, _, rows = cohort
    row = {**row, "graph": model("b*x")[0]}
    formulas = trial.construct({**row, "arm": "algorithm"})
    assert len(formulas) == 2
    assert formulas == trial.construct({**row, "arm": "20b_repair"})
    result = trial.fit(row, formulas[0], 0)
    assert result["status"] == "fitted"
    assert result["evaluations"] <= 1000
    assert result["budget"] == {**trial.FIT_BUDGET, "seed": 100}
    assert result["starts_attempted"] == 2
    assert np.isfinite(result["training_nmse"])
    # Construction, diagnostics and training finish before this file exists.
    decision = trial.prefit(row, formulas[0])
    assert decision["status"] in {"eligible", "pointwise_conflict"}
    validation = sealed_write(
        tmp_path / "validation.json",
        {
            "rows": {"d": [{**rows[0], "trajectory_id": "val_0"}]},
            "split": "val",
            "training_intake_sha256": trial.TRAIN,
        },
    )
    monkeypatch.setattr(trial, "VALIDATION", validation["artifact_sha256"])
    scored = trial.validate(row, formulas[0], result)
    assert scored["status"] == "scored"
    assert scored["curves"][0]["trajectory_id"] == "val_0"
    assert np.isfinite(scored["pooled_nmse"])
    assert scored["per_event"]["val_0"]["pooled_nmse"] == scored["pooled_nmse"]


def test_order_certificate_records_conflict_before_fitting(cohort):
    row, formula, _ = cohort
    review = trial.prefit(row, formula)
    assert review["status"] == "pointwise_conflict"
    assert review["ordered_checks"]["y"]["exact_lower_bound"] == "2"
    assert review["balance_check"]["status"] == "not_applicable"
    assert (
        trial.prefit(row, {**formula, "equations": {"y": "unknown"}})["status"]
        == "invalid_formula"
    )


def test_dataset_binding_and_rank(cohort, tmp_path, monkeypatch):
    row, formula, rows = cohort
    with pytest.raises(ValueError, match="rank"):
        trial.fit(row, formula, -1)
    train = sealed_write(tmp_path / "other.json", {"train": {"d": rows[::-1]}})
    (tmp_path / "other.json").replace(tmp_path / "train.json")
    monkeypatch.setattr(trial, "TRAIN", train["artifact_sha256"])
    with pytest.raises(ValueError, match="disagree"):
        trial.construct(row)
    with pytest.raises(ValueError, match="Unexpected registered artifact"):
        trial._registered(tmp_path / "train.json", "wrong")
