from pathlib import Path

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import run_t2_initial_memory_holdouts as study


def _attempt(name: str, nmse: float, eligible=True):
    return {
        "start": name,
        "replay": {"eligible": eligible, "summary": {"pooled_nmse": nmse}},
    }


def test_only_eligible_training_endpoints_compete():
    fits = [
        _attempt("short", 0.05, False),
        _attempt("long", 0.1),
        _attempt("nested", 0.1),
    ]
    assert study._best(fits)["start"] == "long"
    with pytest.raises(ValueError, match="passed"):
        study._best([fits[0], _attempt("long", float("nan"))])


def test_resume_binds_training_fold(tmp_path: Path):
    path = tmp_path / "fit.json"
    assert study._checkpoint(path, {"held": "train_011"}, lambda: 2) == 2
    assert study._checkpoint(path, {"held": "train_011"}, lambda: pytest.fail()) == 2
    with pytest.raises(ValueError, match="identity"):
        study._checkpoint(path, {"held": "train_012"}, lambda: 5)


def test_all_forecasts_are_frozen_before_any_scoring(tmp_path, monkeypatch):
    held_ids = study.folds.HOLDOUTS
    selections = {
        h: {"artifact_sha256": h, "models": {a: {"model": a} for a in study.ARMS}}
        for h in held_ids
    }

    def training(_):
        return [{"trajectory_id": h} for h in held_ids]

    predictions = []

    def predict(model, pub):
        predictions.append((pub, model))
        return model

    def score(model, prediction, held):
        assert len(predictions) == 4
        assert len(sealed_read(tmp_path / "forecasts.json")["result"]) == 2
        return {"id": held["trajectory_id"], "model": model}

    monkeypatch.setattr(study.intake, "load_training", training)
    monkeypatch.setattr(
        study.forecast, "public_event", lambda row: row["trajectory_id"]
    )
    monkeypatch.setattr(study.forecast, "predict", predict)
    monkeypatch.setattr(study.forecast, "score", score)
    monkeypatch.setattr(study.forecast, "plot", lambda scores, path: None)
    with pytest.raises(ValueError, match="four"):
        study._future(tmp_path, {held_ids[0]: selections[held_ids[0]]}, study.TRAIN)
    assert predictions == []
    result = study._future(tmp_path, selections, study.TRAIN)
    assert len(result) == 2
    assert set(predictions) == {(h, a) for h in held_ids for a in study.ARMS}
