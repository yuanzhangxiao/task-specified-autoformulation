"""Check seed-specific starts, immutable selection and paired gain transfers."""

from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import basin_process_start_control as study
from tests.test_basin_balance_inference import event
from tests.test_basin_function_family import GRAPH


def design(key, up="k_overflow", down="k_outflow"):
    return {
        "key": key,
        "submitted_starts": {up: 0.1, down: 0.05},
        "authored_factors": {
            "upstream_factor": f"{up} * sqrt(head)",
            "downstream_factor": f"{down} * sqrt(head)",
        },
    }


def form(up, down):
    f = study.family.formula(1.5, 1.5)
    f["equations"] = {
        k: v.replace("k_outflow", down).replace("k_overflow", up)
        for k, v in f["equations"].items()
    }
    f["parameters"] = {up: 1.0, down: 1.0}
    return f


def test_cross_transfer_matches_role_even_with_different_parameter_order():
    d1 = design("full-100")
    d2 = design("control-100", "a", "z")
    plan = {
        "graph": GRAPH,
        "designs": [d1, d2],
        "models": [
            {"formula": form("k_overflow", "k_outflow")},
            {"formula": form("a", "z")},
        ],
    }
    ranks = {"selected": {"full-100": 0, "control-100": 1}}
    fits = [
        {"result": {"parameters": {"k_overflow": 4.0, "k_outflow": 6.0}}},
        {"result": {"parameters": {"a": 2.0, "z": 10.0}}},
    ]
    rows = [event(0, 0.4, 0.6), event(1, 0.5, 0.8)]
    cross = study.cross_rollouts(plan, ranks, fits, rows, "train")
    assert cross["status"] == "agreement" and cross["max_abs_m"] == 0.0
    assert [(r["upstream_gain"], r["downstream_gain"]) for r in cross["vectors"]] == [
        (4.0, 6.0),
        (2.0, 10.0),
    ]
    for r in cross["vectors"]:
        assert len(r["events"]) == 2
        assert set(r["events"][0]["max_abs_m"]) == {"h_up", "h_down"}
    # Preserve the typed graph but deliberately alter one scientific scale.
    altered = deepcopy(plan)
    f = altered["models"][1]["formula"]
    f["equations"]["h_down"] = (
        f["equations"]["h_down"].replace("z *", "2*z *")
        if "z *" in f["equations"]["h_down"]
        else f["equations"]["h_down"].replace("z*", "2*z*")
    )
    assert (
        study.cross_rollouts(altered, ranks, fits, rows, "train")["status"]
        == "different"
    )


def test_shared_or_unused_gains_cannot_be_cross_transferred():
    with pytest.raises(ValueError, match="distinct"):
        study._factor_gains(design("full-100", "k", "k"))
    d = design("control-100")
    d["authored_factors"]["downstream_factor"] = "0.3 * sqrt(head)"
    with pytest.raises(ValueError, match="one fitted"):
        study._factor_gains(d)


def test_plan_has_three_common_starts_and_one_per_literal(tmp_path, monkeypatch):
    def fake(root, profile):
        p = {
            "system": "one",
            "schema": {},
            "user": {"outlet_evidence": {"rows": [1] if profile == "full" else []}},
        }
        cells = []
        for seed in (100, 101, 102):
            d = design(f"{profile}-{seed}")
            f = form("k_overflow", "k_outflow")
            if seed == 102:
                f["parameters"] = {}
            cells.append(
                {
                    "seed": seed,
                    "authored_factors": d["authored_factors"],
                    "formula": f,
                    "revised_graph": GRAPH,
                    "artifact_sha256": str(seed),
                }
            )
        return {"train": {study.proposals.BASIN: []}}, p, cells

    monkeypatch.setattr(study, "read_source", fake)
    plan, _ = study.prepare(tmp_path / "a", tmp_path / "b", tmp_path / "r")
    assert len(plan["designs"]) == 6 and len(plan["models"]) == 14
    assert (
        plan["fixed_budget"]["max_evaluations"] == plan["fixed_budget"]["starts"] == 1
    )
    assert plan["budget"]["max_evaluations"] == 1000
    for d in plan["designs"]:
        models = [m for m in plan["models"] if m["key"] == d["key"]]
        assert [m["start"] for m in models] == (
            [1, 10, 100] if d["submitted_starts"] else [None]
        )
        for m in models:
            assert all(x == m["start"] for x in m["formula"]["parameters"].values())


def test_freeze_precedes_validation(tmp_path):
    root = tmp_path / "r"
    sealed_write(root / "plan.json", {"registered": True})
    with pytest.raises(FileNotFoundError, match=r"ranking\.json"):
        study.validate(root, tmp_path / "test.json")
