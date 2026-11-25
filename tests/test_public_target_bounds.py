"""Check that public target domains are inherited only with identified evidence."""

from copy import deepcopy

import pytest

from scripts.public_target_bounds import matched_bound_contract, observed_target_bounds


def brief(descriptions):
    return {
        "public_variables": [
            {"name": name, "data_role": "target"} for name in descriptions
        ]
        + [{"name": "supplied", "data_role": "auxiliary"}],
        "scientific_context": "\n".join(
            ["Target channels:"]
            + [f"- {name}(t): {text}" for name, text in descriptions.items()]
            + ["Supplied auxiliaries:", "- supplied(t): primary nonnegative target"]
        ),
    }


def test_explicit_label_survives_obfuscation_and_missing_semantic_evidence():
    targets = observed_target_bounds(
        brief(
            {
                "v01": "primary nonnegative stored-quantity target",
                "I": "channel with unknown semantics",
                "x": "insulin concentration [unit: pmol L^-1]",
                "y": "total glucose utilization/disposal rate, including "
                "insulin-independent and insulin-dependent contributions",
                "change": "signed deviation from a nonnegative reference concentration",
                "z": "nonnegative target only during recovery",
            }
        )
    )
    assert targets["v01"]["basis"] == "explicit_public"
    assert targets["v01"]["lower"] == 0
    assert targets["x"]["basis"] == targets["y"]["basis"] == "reviewed_quantity"
    assert targets["change"]["lower"] is targets["I"]["lower"] is None
    assert targets["z"]["lower"] is None
    assert "supplied" not in targets


def test_matched_inventories_share_the_same_public_constraint():
    item = {"public_brief": brief({"v": "primary nonnegative target"})}
    cohort = {
        "cases": {"a": item, "b": deepcopy(item)},
        "rows": [{"case": "a"}, {"case": "b"}],
    }
    assert matched_bound_contract(cohort)["targets"] == ["v"]
    cohort["cases"]["b"]["public_brief"] = brief({"v": "signed deviation"})
    with pytest.raises(ValueError, match="disagree"):
        matched_bound_contract(cohort)
