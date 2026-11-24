"""Select and score only physically eligible, training-verified formulas."""

import math

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.staged_topology import content_hash
from scripts import topology_forecast_eligible as study

GOOD = {
    "status": "fitted",
    "parameters": {"a": 1},
    "training_nmse": 0.1,
    "maximum_scaled_grid_difference": 0.001,
    "scales": {"x": 1, "y": 2},
    "negative_samples": {"x": 0, "y": 0},
}


@pytest.mark.parametrize(
    "invalid",
    [
        {"status": "resolution_unverified"},
        {"training_nmse": math.nan},
        {"maximum_scaled_grid_difference": 0.011},
        {"parameters": None},
        {"negative_samples": {"x": 0}},
        {"negative_samples": {"x": 1, "y": 0}},
    ],
)
def test_every_training_condition_is_required(invalid):
    assert study.qualifies({"status": "eligible"}, GOOD)
    assert not study.qualifies({"status": "eligible"}, {**GOOD, **invalid})
    assert not study.qualifies({"status": "pointwise_conflict"}, GOOD)


def test_eligible_freeze_scans_both_slots_and_resumes_development(
    tmp_path, monkeypatch
):
    specifications = [
        ("z", "algorithm", [("eligible", 0.1), ("eligible", 0.1)]),
        ("a", "algorithm", [("pointwise_conflict", 0.001), ("eligible", 0.1)]),
        ("r", "20b_repair", [("pointwise_conflict", 0.0), ("pointwise_conflict", 0.9)]),
    ]
    rows, slots, losses = [], [], {}
    for i, (gid, arm, variants) in enumerate(specifications):
        rows.append({"inventory": "inv", "graph_id": gid, "arm": arm})
        for rank, (status, loss) in enumerate(variants):
            key = f"{i}-{rank}"
            slots.append(
                {
                    "graph_index": i,
                    "fit_key": key,
                    "rank": rank,
                    "prefit": {"status": status},
                    "formula": {"index": key},
                }
            )
            losses[key] = loss
    plan = sealed_write(
        tmp_path / "plan.json", {"cohort": {"rows": rows}, "slots": slots}
    )
    study.register(tmp_path)
    with pytest.raises(FileNotFoundError):
        study.development(tmp_path)
    for slot in slots:
        key = slot["fit_key"]
        sealed_write(
            tmp_path / "fits" / f"{key}.json",
            {
                "fit_key": key,
                "plan": plan["artifact_sha256"],
                "slot_sha256": content_hash(slot),
                "formula": slot["formula"],
                "result": {**GOOD, "training_nmse": losses[key]},
            },
        )
    calls = []

    def validate(graph, formula, fit):
        calls.append(formula["index"])
        return {"status": "scored", "pooled_nmse": 0.2}

    monkeypatch.setattr(study.experiment.adapter, "validate", validate)
    receipt = study.freeze(tmp_path)
    assert calls == []
    group = {g["arm"]: g for g in receipt["groups"]}
    assert group["20b_repair"]["winner"] is None
    algo = group["algorithm"]
    assert (
        algo["advanced_graphs"],
        algo["formula_slots"],
        algo["eligible_graphs"],
        algo["eligible_slots"],
        algo["winner"]["fit_key"],
    ) == (2, 4, 2, 3, "1-1")
    scored = study.development(tmp_path)
    assert calls == ["1-1"]
    assert scored["freeze"] == receipt["artifact_sha256"]
    assert study.development(tmp_path) == scored and calls == ["1-1"]
