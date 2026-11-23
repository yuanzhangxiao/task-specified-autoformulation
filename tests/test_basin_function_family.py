"""Check physical equations, train-only selection and immutable fit receipts."""

from itertools import product

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_basin_balance import check_transfer_balance
from autoformalism.research.trajectory_function_fitting import validate_formula
from autoformalism.staged_topology import content_hash
from scripts import basin_function_family as family

GRAPH = {
    "variable_types": {"h_down": "differential", "h_up": "differential"},
    "initializers": {
        "h_up": {"mode": "observed_initial", "channel": "initial_up"},
        "h_down": {"mode": "observed_initial", "channel": "h_down"},
    },
    "sources": {
        "h_up": ["area_up", "crest_up", "h_up", "inflow_up"],
        "h_down": [
            "area_down",
            "crest_down",
            "crest_up",
            "h_down",
            "h_up",
            "inflow_down",
        ],
    },
}


def test_all_forms_use_the_frozen_graph_and_cancel_internal_transfer():
    assert content_hash(GRAPH) == family.REVISED
    case = {"provenance": {"case": family.BASIN}}
    identities = set()
    for up, down in product(family.POWERS, repeat=2):
        formula = validate_formula(GRAPH, family.formula(up, down))
        assert check_transfer_balance(case, GRAPH, formula)["status"] == "pass"
        identities.add(content_hash(formula))
    assert len(identities) == 9
    with pytest.raises(ValueError, match="Power"):
        family.formula(3, 1)


def test_modified_public_cohort_and_overlapping_roots_are_rejected(tmp_path):
    sealed_write(tmp_path / "intake.json", {"source": family.trial.TRAIN})
    sealed_write(tmp_path / "plan.json", {"intake_sha256": family.INTAKE})
    with pytest.raises(ValueError, match="separate"):
        family.prepare(tmp_path, tmp_path / "within")
    with pytest.raises(ValueError, match="Unexpected public training"):
        family.prepare(tmp_path, tmp_path.parent)


def test_training_freeze_controls_validation_and_resumes_without_new_fits(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    cohort = [
        {"up": u, "down": 1.5, "formula": family.formula(u, 1.5)} for u in family.POWERS
    ]
    monkeypatch.setattr(family, "_code", lambda: {"script": "frozen"})
    plan = sealed_write(
        root / "plan.json",
        {
            "models": cohort,
            "graph": GRAPH,
            "code": family._code(),
            "budget": family.BUDGET,
        },
    )
    monkeypatch.setattr(family, "prepare", lambda c, r: (plan, ["training"]))
    val = tmp_path / "validation.json"
    with pytest.raises(FileNotFoundError, match=r"ranking\.json"):
        family.validate(root, val)
    count = []

    def fit(graph, f, rows, **kw):
        assert rows == ["training"] and kw == family.BUDGET
        i = len(count)
        count.append(1)
        return {
            "status": "fitted",
            "training_nmse": [0.1, 0.2, 0.01][i],
            "parameters": {"a": -1 if i == 2 else 1},
            "negative_samples": {"h_down": 0},
        }

    monkeypatch.setattr(family, "fit_formula", fit)
    ranking = family.train(tmp_path / "source", root)
    assert ranking["training_order"] == [0, 1]
    assert not ranking["validation_opened"]
    assert family.train(tmp_path / "source", root) == ranking and len(count) == 3
    with pytest.raises(ValueError, match="registered validation"):
        family.validate(root, tmp_path / "test.json")
    payload = sealed_write(
        val,
        {
            "training_intake_sha256": family.trial.TRAIN,
            "rows": {family.BASIN: ["validation"]},
        },
    )
    with pytest.raises(ValueError, match="Unexpected validation"):
        family.validate(root, val)
    monkeypatch.setattr(family.trial, "VALIDATION", payload["artifact_sha256"])
    scores = []

    def score(graph, f, result, val_rows):
        assert val_rows == ["validation"]
        scores.append(1)
        # The validation ordering deliberately reverses the training winner.
        return {"status": "scored", "pooled_nmse": 1 - result["training_nmse"]}

    monkeypatch.setattr(family, "score_formula", score)
    summary = family.validate(root, val)
    assert summary["training_order"] == [0, 1]
    assert summary["rows"][2]["validation_status"] == "no_admissible_training_fit"
    assert family.validate(root, val) == summary and len(scores) == 2
    assert sealed_read(root / "ranking.json") == ranking


def test_interrupted_fit_consumes_reserved_budget(tmp_path, monkeypatch):
    form = family.formula(1, 1)
    plan = {"models": [{"formula": form}], "artifact_sha256": "abc"}
    identity = {"slot": 0, "formula": content_hash(form), "plan": "abc"}
    sealed_write(tmp_path / "fits/00/reservation.json", identity)
    monkeypatch.setattr(
        family, "fit_formula", lambda *a, **kw: pytest.fail("spent twice")
    )
    report = family._fit(tmp_path, plan, 0, [])
    assert report["result"]["status"] == "interrupted"
    assert family._fit(tmp_path, plan, 0, []) == report
