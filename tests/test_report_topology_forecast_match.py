"""Verify matched forecast reporting, including failed and unphysical candidates."""

from pathlib import Path

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.staged_topology import content_hash
from scripts import topology_forecast_eligible as eligibility
from scripts.matched_topology_forecast import freeze
from scripts.report_topology_forecast_match import ARMS, cross_pairs, render


def _score(nmse, low):
    return {
        "status": "scored",
        "pooled_nmse": nmse,
        "per_target_nmse": {"I": nmse},
        "negative_samples": {"I": int(low < 0)},
        "minimum_prediction": {"I": low},
        "curves": [
            {
                "trajectory_id": "val_007",
                "time": [0, 1],
                "observed": {"I": [0, 1]},
                "predicted": {"I": [low, 2]},
            }
        ],
    }


def _fixture(root: Path) -> tuple[dict, dict]:
    graph = {
        "variable_types": {"I": "differential"},
        "initializers": {"I": {"mode": "observed_initial", "channel": "I"}},
    }
    formula = {"equations": {"I": "q*I"}, "initializers": {}, "parameters": {"q": 1}}
    rows, slots = [], []
    for index, arm in enumerate(ARMS):
        rows.append(
            {
                "graph": graph,
                "graph_id": f"g{index}",
                "inventory": "t2_fixture",
                "arm": arm,
            }
        )
        for rank in (0, 1):
            check = "pointwise_conflict" if rank == 0 else "eligible"
            slots.append(
                {
                    "graph_index": index,
                    "rank": rank,
                    "fit_key": f"f{index}-{rank}",
                    "formula": formula,
                    "prefit": {"status": check},
                }
            )
    counts = {
        "selected_per_arm": 1,
        "selected_in_both_arms": [],
        "arms": {
            key: {
                "admitted_occurrences": 1,
                "distinct_admitted": 1,
                "matching_inventory": 1,
            }
            for key in ARMS
        },
    }
    plan = sealed_write(
        root / "plan.json",
        {
            "fit_budget": {"starts": 2},
            "slots": slots,
            "cohort": {
                "rows": rows,
                "inventory_counts": {"t2_fixture": counts},
                "cap_per_inventory_arm": 1,
            },
        },
    )
    for slot in slots:
        index, rank = slot["graph_index"], slot["rank"]
        result = {
            "status": "fit_failed",
            "training_nmse": None,
            "evaluations": 12,
            "failed_rollouts": 2,
            "scales": {"I": 1},
            "maximum_scaled_grid_difference": 0.0,
        }
        if index == 0:
            result.update(
                status="fitted",
                training_nmse=0.1 + rank,
                parameters={"q": 2},
                negative_samples={"I": int(rank == 0)},
            )
        sealed_write(
            root / "fits" / f"{slot['fit_key']}.json",
            {
                "plan": plan["artifact_sha256"],
                "slot_sha256": content_hash(slot),
                "fit_key": slot["fit_key"],
                "formula": formula,
                "result": result,
            },
        )
    frozen = freeze(root, plan)
    eligibility.register(root)
    qualified = eligibility.freeze(root)
    qualified_rows = [
        {**g["winner"], "score": _score(0.4, 0.0)}
        for g in qualified["groups"]
        if g["winner"]
    ]
    sealed_write(
        root / "eligibility-development.json",
        {"freeze": qualified["artifact_sha256"], "rows": qualified_rows},
    )
    dev = {
        "freeze": frozen["artifact_sha256"],
        "rows": [
            {**row, "score": _score(0.3, -2) if row["rank"] == 0 else None}
            for row in frozen["selections"]
        ],
        "nonnegative_rows": [
            {
                **frozen["nonnegative_winners"]["t2_fixture"]["algorithm"],
                "score": _score(0.4, 0.0),
            }
        ],
    }
    return frozen, dev


def test_report_preserves_all_slots_and_training_selected_winners(tmp_path):
    _, dev = _fixture(tmp_path)
    sealed_write(tmp_path / "development.json", dev)
    html = render(tmp_path).read_text()
    for text in (
        "pointwise_conflict",
        "Nonnegative training forecasts",
        "0.4",
        "no_verified_training_fit",
        "d(I)/dt = q*I",
        "I(0) = observed I(0)",
        "q=2",
        "I -2 pmol/L",
        "val_007",
    ):
        assert text in html
    assert (tmp_path / "portfolio-forecast-0.png").stat().st_size > 0
    assert cross_pairs(dev["rows"], [dev["rows"][1]]) == (1, 1, 0)


def test_report_rejects_a_different_frozen_formula(tmp_path):
    _, dev = _fixture(tmp_path)
    dev["rows"][0]["fit_key"] = "f0-1"
    sealed_write(tmp_path / "development.json", dev)
    with pytest.raises(ValueError, match="differs from training freeze"):
        render(tmp_path)
