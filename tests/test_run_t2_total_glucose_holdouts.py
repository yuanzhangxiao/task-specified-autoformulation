from copy import deepcopy

import pytest

from scripts import run_t2_total_glucose_holdouts as study


def _selections():
    return {
        h: {
            "held": h,
            "models": {a: {"model": f"{h}-{a}"} for a in study.ARMS},
            "artifact_sha256": h,
        }
        for h in study.PARENTS
    }


def test_frozen_models_enforces_fold_and_comparator():
    selected = _selections()
    models = study._frozen_models(selected)
    assert all(set(d) == set(study.ARMS) for d in models.values())
    selected["train_012"]["held"] = "train_011"
    with pytest.raises(ValueError, match="both models"):
        study._frozen_models(selected)
    selected = _selections()
    del selected["train_011"]["models"]["total"]
    with pytest.raises(ValueError, match="both models"):
        study._frozen_models(selected)


def test_control_identity_and_all_four_forecasts_before_scoring(tmp_path, monkeypatch):
    from scripts import t2_total_glucose_replay as replay

    selections = _selections()
    truth = [{"trajectory_id": h} for h in study.PARENTS]
    monkeypatch.setattr(study.previous.intake, "load_training", lambda _: truth)
    monkeypatch.setattr(
        study.previous.forecast, "public_event", lambda x: x["trajectory_id"]
    )
    calls = []

    def pred(m, h):
        calls.append(m)
        assert m.startswith(h)
        return m

    for module in (replay, study.previous.forecast):
        monkeypatch.setattr(module, "predict", pred)
    controls = {
        "identity": study.PARENTS,
        "result": {h: {"fading": h + "-fading"} for h in study.PARENTS},
    }
    monkeypatch.setattr(study, "sealed_read", lambda p: deepcopy(controls))

    def score(m, p, row):
        assert len(calls) == 4 and (tmp_path / "forecasts.json").is_file()
        return {"model": m, "event": row["trajectory_id"]}

    monkeypatch.setattr(study.previous.forecast, "score", score)
    assert len(study.held_scores(tmp_path, selections)["scores"]) == 2
    controls["result"]["train_011"]["fading"] = "unregistered"
    with pytest.raises(ValueError, match="Control forecast"):
        study.held_scores(tmp_path, selections)
