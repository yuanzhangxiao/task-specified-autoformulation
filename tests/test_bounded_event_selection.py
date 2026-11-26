"""Exact selection and stopping certificates across event evaluation orders."""

from itertools import product

import pytest

from autoformalism.search.bounded_event_selection import (
    FittedCandidate,
    select_on_all_events,
)


def test_partial_loss_cannot_prune_ties_or_prefer_an_invalid_endpoint():
    candidates = [
        FittedCandidate(n, k, i)
        for i, (n, k) in enumerate(
            [("first", 3), ("bad", 1), ("tie", 2), ("failed", 1)]
        )
    ]
    values = {"first": [1, 0], "bad": [2, 0], "tie": [1, 0], "failed": [0, None]}
    events = ["a", "b"]

    def loss(name, event):
        return values[name][events.index(event)]

    got = select_on_all_events(candidates, events, loss)
    assert got["winner"] == "tie" and got["selection_nmse"] == 0.5
    assert got["event_evaluations"] == 7
    assert [e["status"] for e in got["trace"]] == [
        "complete",
        "above_incumbent",
        "complete",
        "failed_qualification",
    ]


def test_matches_exhaustive_qualified_winner_including_all_abstain():
    models = [FittedCandidate("a", 3, 0.0), FittedCandidate("b", 1, 1.0)]
    for values in product([None, 0.0, 0.5], repeat=4):

        def loss(name, event, values=values):
            return values[2 * (name == "b") + (event == "later")]

        a = select_on_all_events(models, ["early", "later"], loss)
        b = select_on_all_events(models, ["early", "later"], loss, early_stop=False)
        assert (a["winner"], a["selection_nmse"]) == (b["winner"], b["selection_nmse"])
        assert a["event_evaluations"] <= b["event_evaluations"]


def test_declines_negative_or_missing_event_design():
    models = [FittedCandidate("a", 1, 1.0)]
    with pytest.raises(ValueError, match="nonnegative event loss"):
        select_on_all_events(models, ["a"], lambda *_: -1)
    with pytest.raises(ValueError, match="nonempty"):
        select_on_all_events(models, [], lambda *_: 0)


def test_cross_graph_decision_requires_the_same_training_events():
    from scripts.export_t2_disposal_models import shared_events

    fit = [f"train_{n:03}" for n in range(8)]
    choice = [f"train_{n:03}" for n in range(8, 12)]
    panels = [
        {
            "graph_name": graph,
            "fit_ids": fit,
            "choice_rotations": [{"selection_ids": choice}],
        }
        for graph in ("g1", "g2")
    ]
    assert shared_events(panels, 0) == (fit, choice)
    panels[1]["choice_rotations"] = [{"selection_ids": [*choice[:3], "val_003"]}]
    with pytest.raises(ValueError, match="differ"):
        shared_events(panels, 0)
    panels[0]["choice_rotations"] = panels[1]["choice_rotations"]
    with pytest.raises(ValueError, match="distinct training"):
        shared_events(panels, 0)
