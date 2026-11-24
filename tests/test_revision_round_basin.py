"""Strict basin edit grammar, equivalent balances, and fully training-only ranking."""

from itertools import product

import pytest

from scripts import revision_round_basin as basin

GRAPH = {
    "variable_types": {"h_down": "differential", "h_up": "differential"},
    "sources": {
        "h_down": [
            "area_down",
            "crest_down",
            "crest_up",
            "h_down",
            "h_up",
            "inflow_down",
        ],
        "h_up": ["area_up", "crest_up", "h_up", "inflow_up"],
    },
    "initializers": {
        "h_down": {"mode": "observed_initial", "channel": "h_down"},
        "h_up": {"mode": "observed_initial", "channel": "initial_up"},
    },
}


def test_all_25_candidate_powers_compile_and_conserve_transfer():
    seen = set()
    assert basin.content_hash(GRAPH) == basin.family.REVISED
    for u, d in product(basin.POWERS, repeat=2):
        form = basin.validate_formula(GRAPH, basin._formula({"up": u, "down": d}))
        assert set(form["parameters"].values()) == {10.0}
        status = basin.check_transfer_balance(
            {"provenance": {"case": basin.family.BASIN}}, GRAPH, form
        )
        assert status["status"] == "pass"
        seen.add(basin.content_hash(form))
    assert len(seen) == 25
    assert basin.normalize({"down": 2, "up": 1}) == {"up": 1.0, "down": 2.0}


@pytest.mark.parametrize(
    "edit",
    [
        {"up": True, "down": 1},
        {"up": 1, "down": "1.5"},
        {"up": 1, "down": float("inf")},
        {"up": -1, "down": 1},
        {"up": 1, "down": 1, "k": 2},
        {"up": 1},
    ],
)
def test_invalid_edits_never_fit_or_open_data(edit, monkeypatch):
    monkeypatch.setattr(basin, "_training", lambda: pytest.fail("opened data"))
    result = basin.score(edit)
    assert result["eligible"] is False
    assert result["reason"]


def test_folds_use_common_budget_and_separate_events(monkeypatch):
    rows = [{"trajectory_id": f"train_{i:03}"} for i in range(6)]
    monkeypatch.setattr(basin, "_training", lambda: (GRAPH, rows))
    calls = []
    freeze = {}

    def fit(graph, form, train, **kw):
        assert kw == basin.family.BUDGET
        calls.append(tuple(r["trajectory_id"] for r in train))
        return {
            "evaluations": 3,
            "status": "fitted",
            "training_nmse": 0.1,
            "negative_samples": {"h_down": 0},
            "parameters": {"k": 2},
            "graph_sha256": basin.content_hash(graph),
            "formula_sha256": basin.content_hash(form),
            "scales": {"h_down": 1},
            "minimum_prediction": {"h_down": 0.2},
            "maximum_storage_gain_above_inflow_m3": 0.0,
        }

    def held(graph, form, fit, event):
        assert event["trajectory_id"] not in calls[-1]
        return {
            "status": "scored",
            "pooled_nmse": 0.5,
            "negative_samples": {"h_down": 0},
        }

    monkeypatch.setattr(basin, "fit_formula", fit)
    monkeypatch.setattr(basin.folds, "_held_score", held)
    monkeypatch.setattr(
        basin, "sealed_read", lambda _: pytest.fail("opened validation")
    )
    saved = basin.score(basin.PARENT)
    assert saved["eligible"] and saved["train_nmse"] == 0.1
    assert saved["selection_score"] == 0.5 and saved["evaluations"] == 21
    assert len(calls) == 7 and len(calls[0]) == 6 and len(set(calls[1:])) == 6
    with pytest.raises(ValueError, match="Frozen fit"):
        basin.validate({"up": 1, "down": 1.5}, saved)
    with pytest.raises(ValueError, match="Frozen fit"):
        basin.validate(basin.PARENT, {"eligible": False})

    def validation(path):
        assert path.name == "validation.json"
        freeze["opened"] = True
        return {"artifact_sha256": "wrong", "training_intake_sha256": basin.trial.TRAIN}

    monkeypatch.setattr(basin, "sealed_read", validation)
    with pytest.raises(ValueError, match="Unexpected validation"):
        basin.validate(basin.PARENT, saved)
    assert freeze["opened"]


def test_interval_ranking_matches_returned_evidence_and_schema(monkeypatch):
    monkeypatch.setattr(
        basin, "_training", lambda: (GRAPH, [{"trajectory_id": "train_A"}])
    )
    monkeypatch.setattr(basin, "_rows", lambda rows, split: rows)

    def profile(best):
        return {
            "status": "estimated",
            "zero_law_rmse_m3_per_min": 1.0,
            "profile": [
                {"power": p, "flow_rmse_m3_per_min": abs(p - best)}
                for p in basin.POWERS
            ],
        }

    monkeypatch.setattr(
        basin.balance,
        "estimate",
        lambda rows, stride: {"upstream": profile(1.75), "downstream": profile(1.25)},
    )
    result = basin.packet()
    assert result["parent"] not in result["algorithm_edits"]
    assert len(result["algorithm_edits"]) == 3
    assert result["algorithm_edits"][0] == {"up": 1.75, "down": 1.25}
    assert result["edit_schema"]["properties"]["up"]["enum"] == list(basin.POWERS)
    assert result["evidence"]["training_events"] == ["train_A"]
