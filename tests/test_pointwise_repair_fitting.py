from pathlib import Path

import numpy as np
import pytest

from scripts import pointwise_repair_fitting as exp


def model(equation, parameters, sources, *, incumbent=None):
    val = {
        "graph": {
            "variable_types": {"I": "differential"},
            "initializers": {"I": {"mode": "observed_initial", "channel": "I"}},
            "sources": {"I": list(sources)},
        },
        "formula": {
            "equations": {"I": equation},
            "initializers": {},
            "parameters": parameters,
        },
    }
    if incumbent is not None:
        val["incumbent"] = incumbent
    return val


def parent():
    return model("a+b*x", {"a": 0.1, "b": 0.2}, ["x"], incumbent={"a": 0.3, "b": 0.4})


def test_transfer_preserves_matched_parameter_shifts_under_new_or_renamed_terms():
    p = parent()
    child = model("a+b*x+gain*(2-I)", {"a": 32, "b": 8, "gain": 0.2}, ["x", "I"])
    rule, init = exp.starting_points(p, child, 14)
    _, compare = exp.starting_points(p, p, 14)
    assert rule["nested"] == "zero_added_terms"
    assert init[0] == {"a": 0.3, "b": 0.4, "gain": 0.0}
    assert init[1]["a"] == compare[1]["a"] and init[1]["b"] == compare[1]["b"]
    assert init[1]["gain"] != 0
    renamed = model("c+d*x", {"c": -11, "d": 8}, ["x"])
    mapping, other = exp.starting_points(p, renamed, 14)
    assert mapping["nested"] == "same_family" and other[0] == {"c": 0.3, "d": 0.4}
    assert other[1]["d"] == compare[1]["b"]
    changed = model("c+b*I", {"c": -11, "b": 8}, ["I"])
    mapping, other = exp.starting_points(p, changed, 14)
    assert mapping["nested"] == "unverified" and other[0] == {"c": 0.3, "b": 8}


def test_domain_ambiguity_and_changed_initials_disallow_zero_embedding():
    p = parent()
    form = model("a+b*x+extra*(1/extra)", {"a": 1, "b": 1, "extra": 1}, ["x"])
    assert exp.parameter_relation(p, form)["nested"] == "unverified"
    form = model("a+b*x+extra*sqrt(I)", {"a": 1, "b": 1, "extra": 1}, ["x", "I"])
    assert exp.parameter_relation(p, form)["nested"] == "unverified"
    form = model("a+b*x+extra*I", {"a": 1, "b": 1, "extra": 1}, ["x", "I"])
    form["graph"]["initializers"]["I"] = {"mode": "value"}
    assert exp.parameter_relation(p, form)["nested"] == "unverified"


def test_metrics_include_insulin_margin_negativity_and_measured_equivalence():
    pred = {"train_0": {"I": [4.0, -2.0, 0.2]}}
    c = [
        {
            "trajectory_id": "train_0",
            "observed": {"I": [5.0, 4.0, 3.0]},
            "predicted": pred["train_0"],
        }
    ]
    row = exp._point_error(c)
    assert row["max_abs_I_error"] == 6 and not row["within_1_unit"]
    assert row["negative_I"] == 1 and row["near_zero_I_while_observed_above_one"] == 1
    p = {
        "status": "scored",
        "scales": {"I": 2},
        "predictions": pred,
        "pooled_nmse": 1,
        "per_target_nmse": {"I": 1},
        "maximum_scaled_grid_difference": 0.01,
    }
    assert exp.compare(p, p)["same_training_predictions"]
    child = p | {"pooled_nmse": 0.99, "per_target_nmse": {"I": 0.99}}
    comparison = exp.compare(p, child)
    assert comparison["both_improved"]
    assert not comparison["insulin_ordering_separated_across_grids"]
    with pytest.raises(ValueError, match="normalization"):
        exp.compare(p, p | {"scales": {"I": 1.1}})
    with pytest.raises(ValueError, match="grid lengths"):
        exp.numerical_distance(pred, {"train_0": {"I": [4]}}, {"I": 2})


def synthetic_sources():
    p = parent()
    rows = []
    for i in range(3):
        t = np.linspace(0, 2, 9)
        x = float(i)
        y = 2 + 2 * x * (1 - np.exp(-t / 2))
        rows.append(
            {
                "trajectory_id": f"train_{i:03}",
                "time": t.tolist(),
                "inputs": {"x": [x] * len(t)},
                "auxiliaries": {},
                "covariates": {},
                "targets": {"I": y.tolist()},
            }
        )
    src = {
        k: {
            **p,
            "archive_candidate": f"c{k}",
            "archive_fit": f"f{k}",
            "archived_train_nmse": 1.0,
            "slot": k,
        }
        for k in exp.SLOTS
    }
    return rows, src


def mock_panel(rows, src):
    records = []
    for k in exp.SLOTS:
        for seed in (1000, 1001, 1002):
            for arm in ("control", "witness"):
                i = len(records)
                rec = {
                    "index": i,
                    "parent_slot": k,
                    "seed": seed,
                    "condition": arm,
                    "artifact_sha256": f"saved-{i}",
                    "status": "pointwise_conflict",
                }
                if i == 1:
                    m = model(
                        "a+b*x+gain*(2-I)",
                        {"a": 0.1, "b": 0.2, "gain": 0.2},
                        ["x", "I"],
                    )
                    rec.update(
                        status="eligible", graph_dict=m["graph"], formula=m["formula"]
                    )
                records.append(rec)
    parents = {
        str(k): {
            "graph": p["graph"],
            "formula": p["formula"],
            "archive_candidate_sha256": p["archive_candidate"],
        }
        for k, p in src.items()
    }
    return (
        {"artifact_sha256": "mock-complete-panel"},
        {"parents": parents, "train": rows},
        records,
    )


def test_freeze_complete_panel_matched_fit_and_deterministic_replay(
    monkeypatch, tmp_path
):
    rows, src = synthetic_sources()
    batch = mock_panel(rows, src)
    monkeypatch.setattr(exp, "parents_from_archive", lambda _: (rows, src))
    monkeypatch.setattr(exp.archived, "_code", lambda: {})
    monkeypatch.setattr(exp, "panel_data", lambda _: batch)
    monkeypatch.setattr(
        exp,
        "BUDGET",
        {"starts": 1, "max_evaluations": 60, "wall_seconds": 10.0, "seed": 100},
    )
    root = tmp_path / "root"
    plan, raw = exp.prepare(Path("archive"), Path("panel"), root)
    assert len(plan["models"]) == 4 and exp.report(root)["recorded"] == 0
    for i in range(4):
        r = exp.run_one(root, plan, raw, i)
        assert r["status"] == "scored" and r["evaluations"] <= 120
    report = exp.report(root)
    assert report["conditions"]["control"]["requested"] == 9
    assert report["conditions"]["witness"]["verified_pairs"] == 1
    child = next(r for r in report["decisions"] if r["model"] is not None)
    assert child["parent_embedded_after_fitting"]["matches_refitted_parent"]
    assert not report["validation_opened"] and not report["test_data_opened"]
    monkeypatch.setattr(
        exp.archived, "_fit", lambda *_: pytest.fail("Repeated an optimization")
    )
    assert exp.prepare(Path("archive"), Path("panel"), root)[0] == plan
    exp.run_one(root, plan, raw, 3)
    assert exp.report(root) == report
    monkeypatch.setattr(
        exp, "panel_data", lambda _: (batch[0], batch[1], batch[2][:-1])
    )
    with pytest.raises(ValueError, match="18 terminal"):
        exp.prepare(Path("archive"), Path("panel"), root)
