"""Tests for matched initial arguments and conservative shape evidence."""

from copy import deepcopy

import pytest

from autoformalism.research.hidden_start_contrasts import matched_starts


def _graph(z=None, *, algebraic=False):
    return {
        "variable_types": {
            "y": "algebraic" if algebraic else "differential",
            "z": "differential",
        },
        "sources": {"y": ["u", "z"], "z": ["u"]},
        "initializers": {
            **(
                {} if algebraic else {"y": {"mode": "observed_initial", "channel": "y"}}
            ),
            "z": z or {"mode": "value", "scope": "shared_training"},
        },
    }


def _rows(levels=(0, 1, 2)):
    return [
        {
            "trajectory_id": f"train_{i}",
            "time": [0, 1, 2],
            "targets": {"y": [level**2, level**2 + 1, level**2 + 1]},
            "inputs": {"u": [level, level + 1, level + 1]},
            "auxiliaries": {},
            "covariates": {"c": 5},
        }
        for i, level in enumerate(levels)
    ]


def _first(graph, rows, **kwargs):
    return matched_starts(graph, rows, **kwargs)["equations"][0]


def test_matched_unknowns_and_certified_rate_bounds_exclude_affine_only():
    rows = _rows()
    graph = _graph({"mode": "map", "public_initials": ["c"], "state_initials": []})
    rates = {
        "y": {
            r["trajectory_id"]: (x * x - 0.01, x * x + 0.01) for x, r in enumerate(rows)
        }
    }
    naive = _first(graph, rows)["sources"][0]
    assert naive["maximum_levels"] == 3 and naive["isolated_pairs"] == 3
    assert naive["status"] == "three_or_more_levels_without_affine_exclusion"
    certified = _first(graph, rows, right_rate_bounds=rates)["sources"][0]
    assert certified["status"] == "affine_excluded_on_slice"
    assert certified["witness"]["levels"] == [0, 1, 2]
    assert certified["witness"]["trajectory_ids"] == [
        ["train_0"],
        ["train_1"],
        ["train_2"],
    ]
    uncertain = _first(graph, rows, right_rate_bounds=rates, exact_conditioning=False)
    assert uncertain["sources"][0]["witness"] is None


@pytest.mark.parametrize(
    "initializer",
    [
        {"mode": "value", "scope": "trajectory"},
        {"mode": "map", "public_initials": ["u"], "state_initials": []},
        {"mode": "map", "public_initials": [], "state_initials": ["y"]},
    ],
)
def test_separate_or_unmatched_latent_starts_cannot_isolate_source(initializer):
    eq = _first(_graph(initializer), _rows())
    assert eq["equal_hidden_initial_pairs"] == 0
    assert eq["sources"][0]["isolated_pairs"] == 0


def test_every_other_argument_must_match():
    graph = _graph()
    graph["sources"]["y"].append("c")
    rows = _rows()
    for i, row in enumerate(rows):
        row["covariates"]["c"] = i
    eq = _first(graph, rows)
    assert eq["equal_hidden_initial_pairs"] == 3
    assert all(s["isolated_pairs"] == 0 for s in eq["sources"])


def test_exact_algebraic_bounds_and_repeated_level_inconsistency():
    graph = _graph(algebraic=True)
    rows = _rows()
    assert _first(graph, rows, algebraic_error={"y": 0})["sources"][0]["status"] == (
        "affine_excluded_on_slice"
    )
    assert (
        _first(graph, rows, algebraic_error={"y": 10})["sources"][0]["witness"] is None
    )
    rows = _rows((0, 0, 2))
    rows[1]["targets"]["y"][0] = 4
    out = _first(graph, rows, algebraic_error={"y": 0})["sources"][0]
    assert out["status"] == "inconsistent_repeated_level"
    assert out["witness"]["source_level"] == 0


def test_motifs_respect_public_input_interpolation_and_graph():
    graph = _graph()
    rows = _rows()
    for method, count in [("linear_complete_knots", 0), ("previous_complete_knots", 3)]:
        eq = _first(graph, rows, interpolation={"u": method})
        assert eq["jump_candidates"] == [
            {"source": "u", "isolated_interior_jumps": count}
        ]
        assert not eq["filter_candidates"]
    graph["sources"] = {"y": ["y", "z"], "z": ["z", "u"]}
    eq = _first(graph, rows, interpolation={"u": "linear_complete_knots"})
    assert eq["filter_candidates"] == [
        {"hidden": "z", "input": "u", "constant_input_levels": [1, 2, 3]}
    ]
    graph["sources"]["y"].append("u")
    assert not _first(graph, rows)["filter_candidates"]


def test_hidden_algebraic_equalities_follow_their_own_dependencies():
    graph = _graph()
    graph["variable_types"]["z"] = "algebraic"
    del graph["initializers"]["z"]
    graph["sources"]["z"] = ["c"]
    assert _first(graph, _rows())["equal_hidden_initial_pairs"] == 3
    graph["sources"]["z"] = ["u"]
    assert _first(graph, _rows())["sources"][0]["maximum_levels"] == 1


def test_invalid_rate_bounds_split_and_cyclic_initialization():
    with pytest.raises(ValueError, match="interval"):
        _first(_graph(), _rows(), right_rate_bounds={"y": {"train_0": (3, 1)}})
    cases = deepcopy(_rows())
    cases[0]["trajectory_id"] = "test_0"
    with pytest.raises(ValueError, match="train"):
        _first(_graph(), cases)
    graph = _graph({"mode": "map", "public_initials": [], "state_initials": ["z"]})
    with pytest.raises(ValueError, match="Cyclic"):
        _first(graph, _rows())


def test_pooled_sensitivity_uses_half_of_absolute_secant_departure():
    from scripts.audit_pooled_hidden_shape_t2 import _rate_sensitivity

    rows = _rows()
    for i, r in enumerate(rows):
        r["targets"]["y"] = [0, i**2, i**2]
    result = _first(_graph(), rows)
    result["graph"] = "example"
    diagnostic = _rate_sensitivity([result])
    assert diagnostic["required_strict_uniform_rate_error"] == 0.5
    assert diagnostic["affine_middle_from_endpoint_secants"] == 2
    assert diagnostic["given_rate_error"] is None
    rates = {
        r["trajectory_id"]: (i * i - 0.49, i * i + 0.49) for i, r in enumerate(rows)
    }
    checked = _first(_graph(), rows, right_rate_bounds={"y": rates})
    assert checked["sources"][0]["status"] == "affine_excluded_on_slice"
    wide = {r["trajectory_id"]: (i * i - 0.5, i * i + 0.5) for i, r in enumerate(rows)}
    boundary = _first(_graph(), rows, right_rate_bounds={"y": wide})
    assert boundary["sources"][0]["witness"] is None
