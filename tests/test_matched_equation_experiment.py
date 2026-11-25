"""Check the graph matching, omission split and opaque-name audit."""

from copy import deepcopy

import pytest

from scripts import audit_equation_constructor_names as audit
from scripts import matched_equation_experiment as study


def test_fixed_roster_and_complementary_fold_assignment():
    rows = [
        {"dataset": "d", "inventory": "one", "arm": a, "id": i}
        for i, a in enumerate(("old", "new", "old", "new"))
    ]
    other = {**rows[0], "dataset": "elsewhere"}
    result = study.select_graphs({"rows": [other, *rows]}, "d")
    assert sorted(r["id"] for r in result) == [0, 1]
    events = [{"trajectory_id": f"train_{i:02d}"} for i in range(5)]
    fit, held = study.split_events(events[::-1], 0)
    assert (held, fit) == study.split_events(events, 1)
    assert fit == events[1::2] and held == events[::2]
    with pytest.raises(ValueError, match="distinct training events"):
        study.split_events([*events[:-1], {"trajectory_id": "val_00"}], 0)


def test_renamed_equation_keeps_parameter_identity_and_start_value():
    base = {
        "equations": {"y": "k*x+b", "x": "k*y"},
        "initializers": {},
        "parameters": {"k": 1.5, "b": -2},
    }
    renamed = {
        "equations": {"v00": "p*v01+q", "v01": "p*v00"},
        "initializers": {},
        "parameters": {"p": 1.5, "q": -2},
    }
    mapping = {"y": "v00", "x": "v01"}
    signature = audit.formula_signature(base, {})
    assert audit.formula_signature(renamed, mapping) == signature
    split = deepcopy(renamed)
    split["equations"]["v01"] = "r*v00"
    split["parameters"]["r"] = 1.5
    assert audit.formula_signature(split, mapping) != signature
    renamed["parameters"]["q"] = 2
    assert audit.formula_signature(renamed, mapping) != signature
