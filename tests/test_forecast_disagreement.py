"""Whole-event forecast disagreement, disjoint choice, qualification and bookkeeping."""

from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import forecast_disagreement as rule
from scripts import report_hidden_form_forecasts as study


def test_disagreement_uses_only_post_initial_predictions_and_fit_scale():
    preds = {
        "affine": {"q": [100, 0, 0], "r": [-50, 1, 2]},
        "square": {"q": [0, 4, 4], "r": [5, 1, 2]},
    }
    assert rule.prediction_gap(preds, {"q": 2, "r": 1}) == pytest.approx(2)
    with pytest.raises(ValueError, match="scale"):
        rule.prediction_gap(preds, {"q": 0, "r": 1})
    assert rule.largest_gaps({"b": 3, "a": 3, "d": 0, "c": 3}) == ["a", "b"]


def test_all_subsets_and_forced_or_unavailable_families():
    losses = {e: {"affine": 0.1, "square": 0.2} for e in "abcd"}
    losses["d"]["affine"] = 1.0
    gaps = {"a": 5, "b": 4, "c": 3, "d": 2}
    result = rule.choose(losses, gaps)
    assert result["votes"]["high_disagreement"] == {"affine": 1}
    assert result["votes"]["all_events"] == {"square": 1}
    assert result["votes"]["random_same_size"] == {"affine": 3, "square": 3}
    losses["b"]["affine"] = None
    gaps.pop("b")
    result = rule.choose(losses, gaps)
    assert result["eligible"] == ["square"] and result["votes"][
        "high_disagreement"
    ] == {"square": 1}
    losses["a"]["square"] = None
    gaps.pop("a")
    assert rule.choose(losses, gaps)["votes"]["high_disagreement"] == {}
    with pytest.raises(ValueError, match="gap"):
        rule.choose(losses, {})


def test_scoring_a_random_subset_keeps_failure_probability():
    held = {"x": {"affine": 0.3, "square": 0.6}, "y": {"affine": 0.9, "square": 0.3}}
    votes = {"affine": 3, "square": 1}
    x = rule.score_votes(votes, held)
    assert x["mean_event_nmse"] == pytest.approx(0.5625)
    held["x"]["square"] = None
    assert rule.score_votes(votes, held) == {
        "full_forecast_probability": 0.75,
        "event_nmse": {},
        "mean_event_nmse": None,
    }
    assert rule.score_votes({}, held)["full_forecast_probability"] == 0


def fixture(root, change_held=False):
    events = [f"train_{i:03}" for i in range(16)]
    rots = [
        {"selection_ids": events[8:12], "held_ids": events[12:16]},
        {"selection_ids": events[12:16], "held_ids": events[8:12]},
    ]
    p = sealed_write(
        root / "plan.json",
        {
            "panels": [
                {
                    "panel_id": "A",
                    "graph_name": "g",
                    "source": "x",
                    "target": "q",
                    "fit_ids": events[:8],
                    "choice_rotations": rots,
                }
            ],
            "starts": ["start"],
            "per_start_budget": {"max_evaluations": 1, "wall_seconds": 1.0},
        },
    )
    for family in rule.FAMILIES:
        fit = sealed_write(
            root / "fits/A" / family / "best.json",
            {"panel_id": "A", "family": family, "plan": p["artifact_sha256"]},
        )
        for j, e in enumerate(events[8:]):
            truth = 10 if j >= 4 and change_held else 1
            forecast = j + 2 if family == "affine" else 0
            sealed_write(
                root / "curves/A" / family / f"{e}.json",
                {
                    "event_id": e,
                    "panel_id": "A",
                    "plan": p["artifact_sha256"],
                    "fit": fit["artifact_sha256"],
                    "family": family,
                    "observed": {"q": [1, truth]},
                    "predicted": {"q": [1, forecast]},
                    "scales": {"q": 1},
                    "time": [0, 1],
                    "status": "scored",
                    "nmse": float((truth - forecast) ** 2),
                },
            )
    return p


def test_choices_are_frozen_without_held_outcomes_and_scaling_is_replayed(tmp_path):
    paths = [tmp_path / "a", tmp_path / "b"]
    for j, root in enumerate(paths):
        fixture(root, bool(j))
        with pytest.raises(FileNotFoundError):
            study.score(root)
        manifest = study.select(root)
        assert study.select(root) == manifest
    decisions = [sealed_read(p / "choices/A/rotation0.json") for p in paths]
    assert decisions[0] == decisions[1]  # Only the other rotation's outcomes changed.
    reports = [study.score(p) for p in paths]
    assert reports[0]["rows"][0]["scores"] != reports[1]["rows"][0]["scores"]
    assert reports[0]["summary"]["common"] == 2
    root = paths[0]
    plan = deepcopy(sealed_read(root / "plan.json"))
    plan["panels"][0]["choice_rotations"][0]["held_ids"] = plan["panels"][0]["fit_ids"][
        :4
    ]
    plan.pop("artifact_sha256")
    sealed_write(tmp_path / "overlap/plan.json", plan)
    with pytest.raises(ValueError, match="disjoint"):
        study.select(tmp_path / "overlap")


def test_optional_validation_report_requires_the_same_prior_decisions(tmp_path):
    assert study._validation_section(tmp_path, "frozen") == ""
    sealed_write(
        tmp_path / "official-validation/summary.json",
        {
            "choice_manifest": "frozen",
            "events": ["val_000"],
            "failures": [],
            "summary": study.compare([]),
        },
    )
    with pytest.raises(ValueError, match="different training choices"):
        study._validation_section(tmp_path, "other")
    text = study._validation_section(tmp_path, "frozen")
    assert "Official validation events" in text and "0/0" in text
