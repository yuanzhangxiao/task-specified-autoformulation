"""Training event exclusion, selection order and checkpoint integrity."""

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import basin_family_stability as stability


@pytest.mark.parametrize("policy, n_starts", [("original", 1), ("geometric", 4)])
def test_all_selections_precede_held_scores_and_resume(
    tmp_path, monkeypatch, policy, n_starts
):
    root = tmp_path / "study"
    ids = [f"train_{i:03}" for i in range(6)]
    rows = [{"trajectory_id": n} for n in ids]
    models = [
        {"up": float(i + 1), "down": 1.0, "formula": {"i": i, "parameters": {"k": 0.1}}}
        for i in range(2)
    ]
    fixture = {
        "artifact_sha256": "family",
        "selection": "training loss",
        "graph": {},
        "models": models,
        "budget": {"seed": 100},
    }
    monkeypatch.setattr(stability.family, "prepare", lambda *args: (fixture, rows))
    events = []

    def fit(graph, formula, training, *, seed):
        seen = {row["trajectory_id"] for row in training}
        assert len(seen) == 5 and seed == 100
        held = (set(ids) - seen).pop()
        events.append(("fit", held, formula["i"]))
        start = formula["parameters"]["k"]
        loss = 0.25 + formula["i"]
        if start == 1:
            loss = 0.0  # Invalid rate must never win.
        if start == 10 and formula["i"] == 0:
            loss = 0.15
        return {
            "initial": start,
            "evaluations": 30,
            "training_nmse": loss,
            "status": "fitted",
            "parameters": {"k": -1 if start == 1 else 1},
            "negative_samples": {"h_down": 0},
            "training": sorted(seen),
        }

    def score(graph, formula, fit, row):
        assert row["trajectory_id"] not in fit["training"]
        assert formula["parameters"]["k"] == fit["initial"]
        assert (root / "rankings.json").is_file()
        assert len(sealed_read(root / "rankings.json")["rankings"]) == 6
        events.append(("score", row["trajectory_id"], formula["i"]))
        return {"status": "scored", "pooled_nmse": 1.25 - formula["i"]}

    monkeypatch.setattr(stability.family, "fit_formula", fit)
    monkeypatch.setattr(stability, "_held_score", score)
    summary = stability.run(tmp_path / "source", root, policy)
    assert [f["held"] for f in summary["folds"]] == ids
    assert all(f["winner"] == 0 for f in summary["folds"])
    assert all(len(f["scores"]) == 2 for f in summary["folds"])
    assert summary["fit_receipts"] == 12 * n_starts
    assert summary["evaluations"] == 30 * 12 * n_starts
    assert all(
        f["scores"][0]["start"] == (10.0 if policy == "geometric" else None)
        and f["scores"][1]["start"] is None
        for f in summary["folds"]
    )
    assert [e[0] for e in events] == ["fit"] * (12 * n_starts) + ["score"] * 12
    assert stability.run(tmp_path / "source", root, policy) == summary
    assert len(events) == 12 * (n_starts + 1)
    rows[0]["trajectory_id"] = "val_000"
    with pytest.raises(ValueError, match="six unique training"):
        stability.run(tmp_path / "source", root)


def test_fold_scorer_rejects_mismatched_model_or_held_split():
    from autoformalism.staged_topology import content_hash

    fit = {"graph_sha256": content_hash({}), "formula_sha256": "changed"}
    with pytest.raises(ValueError, match="differs"):
        stability._held_score({}, {}, fit, {})
    row = {
        "trajectory_id": "val_000",
        "time": [0.0, 1.0],
        "targets": {"h_down": [1.0, 1.0]},
        "auxiliaries": {},
        "inputs": {},
        "covariates": {},
    }
    fit["formula_sha256"] = content_hash({})
    with pytest.raises(ValueError, match="train trajectory"):
        stability._held_score({}, {}, fit, row)
