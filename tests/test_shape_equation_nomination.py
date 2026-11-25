"""Check observable shape selection, graph closure and public evidence boundaries."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.research.shape_equation_nomination import build_shape_shortlist
from autoformalism.research.trajectory_function_fitting import validate_formula


def sample(seed, kind, names, *, step=False):
    a, y = names
    rng = np.random.default_rng(seed)
    t = np.r_[0, np.cumsum(rng.uniform(0.4, 1.5, 40))]
    x = np.r_[0, rng.uniform(-2, 2, 40)]
    rate = 0.4 + 2 * x - x**2
    observed = rate
    if kind == "differential":
        interval = rate[:-1] if step else (rate[1:] + rate[:-1]) / 2
        observed = np.r_[0, np.cumsum(interval * np.diff(t))]
    return {
        "trajectory_id": f"train_{seed:03}",
        "time": t.tolist(),
        "targets": {y: observed.tolist()},
        "inputs": {a: x.tolist()},
        "auxiliaries": {},
        "covariates": {},
    }


def setup(kind, names, *, step=False):
    a, y = names
    graph = {"sources": {y: [a]}, "variable_types": {y: kind}, "initializers": {}}
    if kind == "differential":
        graph["initializers"][y] = {"mode": "observed_initial", "channel": y}
    rows = [sample(i, kind, names, step=step) for i in range(4)]
    return graph, rows


@pytest.mark.parametrize(
    ("kind", "step"),
    [
        ("algebraic", False),
        ("differential", False),
        ("differential", True),
    ],
)
def test_unknown_quantity_names_recover_curvature_from_values_or_integral_windows(
    kind, step
):
    winners = []
    for names in [("q", "s"), ("shape1", "nonnegative_glucose")]:
        graph, rows = setup(kind, names, step=step)
        a, y = names
        menu = build_shape_shortlist(
            graph, rows, interpolation={a: "previous_complete_knots"} if step else {}
        )
        assert len(menu["candidates"]) == 2
        assert menu["candidate_families"][0][y] == f"square {a}"
        diag = menu["evidence"]["equations"][y]
        assert (
            diag["status"] == "all_arguments_observed"
            and diag["metric"] == "whole_event_cv_rmse"
        )
        winners.append(diag["shortlist"][0]["rmse"])
        assert winners[-1] < 1e-5
        for model in menu["candidates"]:
            assert model == validate_formula(graph, model)
        assert menu == build_shape_shortlist(
            graph,
            rows[::-1],
            interpolation={a: "previous_complete_knots"} if step else {},
        )
    assert winners[0] == winners[1]


def test_incomplete_observability_keeps_latent_sources_and_initials():
    graph, rows = setup("algebraic", ("a", "y"))
    graph["sources"].update(y=["a", "hidden"], hidden=["hidden", "a"])
    graph["variable_types"]["hidden"] = "differential"
    graph["initializers"]["hidden"] = {
        "mode": "map",
        "public_initials": ["a"],
        "state_initials": [],
    }
    result = build_shape_shortlist(graph, rows)
    details = result["evidence"]["equations"]
    assert details["y"]["status"] == "observed_projection"
    assert details["hidden"]["status"] == "rollout_required"
    assert "shortlist" not in details["hidden"]
    for model in result["candidates"]:
        assert (
            "hidden" in model["equations"]["y"]
            and "a" in model["initializers"]["hidden"]
        )
        assert validate_formula(graph, model) == model


def test_rejects_heldout_rows_duplicate_ids_and_undeclared_sources():
    graph, rows = setup("algebraic", ("a", "y"))
    assert (
        build_shape_shortlist(graph, rows[:1])["evidence"]["equations"]["y"]["metric"]
        == "training_rmse_only"
    )
    with pytest.raises(ValueError, match="unique train"):
        build_shape_shortlist(graph, [*rows, rows[0]])
    val = deepcopy(rows)
    val[-1]["trajectory_id"] = "val_000"
    with pytest.raises(ValueError, match="train"):
        build_shape_shortlist(graph, val)
    graph["sources"]["y"] = ["invented"]
    with pytest.raises(ValueError, match="undeclared"):
        build_shape_shortlist(graph, rows)
