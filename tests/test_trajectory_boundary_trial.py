"""Exercise training-only boundary repair, the equation and the matched allowance."""

import casadi as ca
import pytest

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.casadi_initializer import _translate
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import trajectory_boundary_trial as trial


def test_boundary_preserves_positive_rates_and_tapers_only_outward_flux():
    f = {
        "equations": {"I": "gain*u", "x": "I"},
        "parameters": {"gain": 1},
        "initializers": {},
    }
    repaired = trial.boundary(f, "I", 2.0)
    assert f["equations"]["I"] == "gain*u"
    assert (
        repaired["equations"]["x"] == "I" and repaired["parameters"] == f["parameters"]
    )
    node = RestrictedParser().parse(repaired["equations"]["I"], location="I")
    i, u = ca.SX.sym("i"), ca.SX.sym("u")
    rhs = ca.Function(
        "rhs", [i, u], [_translate(ca, node, {"gain": 1.0, "I": i, "u": u})]
    )
    for state, rate, expected in (
        (0, -3, 0),
        (0, 3, 3),
        (2, -3, -3),
        (1, -3, -1.5),
        (50, -3, -3),
    ):
        assert float(rhs(state, rate)) == pytest.approx(expected)
    with pytest.raises(ValueError, match="positive"):
        trial.boundary(f, "I", 0.0)


def test_scale_uses_only_strictly_positive_training_measurements():
    def point(name, *values):
        return {"trajectory_id": name, "targets": {"I": list(values)}}

    assert trial.training_scale(
        [point("train_1", 1, 0, 2), point("train_2", 3, -1)], "I"
    ) == pytest.approx(1.1)
    for row in (
        point("val_1", 1),
        point("train_1", 0, 1),
        point("train_1", 1, float("nan")),
    ):
        with pytest.raises(ValueError):
            trial.training_scale([row], "I")


def test_selections_compare_algorithm_candidates_with_verified_training_fits(
    tmp_path, monkeypatch
):
    import scripts.trajectory_functions as campaign

    models = [("algorithm", 0.3), ("algorithm", 0.2), ("hybrid", 0.01)]
    g = {"example": True}
    orig = sealed_write(tmp_path / "plan.json", {"global": "source"})
    intake = {
        "graphs": [
            {"graph": g, "structure_sha256": content_hash(g), "dataset": trial.T2}
        ]
    }
    monkeypatch.setattr(
        campaign, "slots", lambda data: [(0, a, j) for j, (a, _) in enumerate(models)]
    )
    receipts = []
    candidates = []
    for i, (arm, nmse) in enumerate(models):
        formula = {"k": i}
        c = sealed_write(
            tmp_path / "candidates" / f"{i:03}.json",
            {
                "index": i,
                "arm": arm,
                "graph": content_hash(g),
                "formula": formula,
                "plan": orig["artifact_sha256"],
            },
        )
        f = sealed_write(
            tmp_path / "fits" / f"{i:03}" / "result.json",
            {
                "index": i,
                "candidate": c["artifact_sha256"],
                "generation": "g",
                "result": {
                    "status": "fitted",
                    "training_nmse": nmse,
                    "formula_sha256": content_hash(formula),
                    "graph_sha256": content_hash(g),
                },
            },
        )
        receipts.append(f["artifact_sha256"])
        candidates.append(c["artifact_sha256"])
    generation = {"artifact_sha256": "g", "candidates": candidates}
    assert (
        trial._best_fits(
            tmp_path, orig, intake, generation, {"fit_receipts": receipts}
        )[0][1]
        == 1
    )
    receipts[1] = "changed"
    with pytest.raises(ValueError, match="identity"):
        trial._best_fits(tmp_path, orig, intake, generation, {"fit_receipts": receipts})


def test_matched_budget_and_verified_incumbent_resume(tmp_path, monkeypatch):
    root = tmp_path / "run"
    input_rows = ["training"]
    model = {"graph": {}, "formula": {"parameters": {"k": 2}}, "source_slot": 2}
    plan = {"artifact_sha256": "p", "models": [model], "budget": trial.BUDGET}
    calls = []

    def fake_fit(graph, formula, rows, **settings):
        calls.append(settings)
        assert rows == input_rows and formula["parameters"] == {"k": 2}
        return {
            "status": "fitted",
            "training_nmse": float(len(calls)),
            "evaluations": 1,
        }

    monkeypatch.setattr(trial, "fit_formula", fake_fit)
    result = trial._fit(root, plan, 0, input_rows)
    assert result["selected_option"] == 0 and result["evaluation_count_complete"]
    assert [s["max_evaluations"] for s in calls] == [1, 999]
    assert [s["starts"] for s in calls] == [1, 2] and [s["seed"] for s in calls] == [
        100,
        100,
    ]
    assert trial._fit(root, plan, 0, input_rows) == result and len(calls) == 2
    other = tmp_path / "interrupted"
    folder = other / "fits" / "00"
    sealed_write(folder / "reservation.json", {"plan": "p", "index": 0})
    warm = sealed_read(root / "fits" / "00" / "warm.json")
    sealed_write(
        folder / "warm.json", {k: v for k, v in warm.items() if k != "artifact_sha256"}
    )
    after = trial._fit(other, plan, 0, input_rows)
    assert after["status"] == "interrupted" and not after["evaluation_count_complete"]
    assert after["selected_option"] == 0 and len(calls) == 2


def test_validation_requires_a_frozen_ranking_before_accessing_validation(tmp_path):
    root = tmp_path / "experiment"
    sealed_write(root / "plan.json", {"info": 1})
    with pytest.raises(FileNotFoundError, match=r"ranking\.json"):
        trial.validate(root, tmp_path / "test.json")
