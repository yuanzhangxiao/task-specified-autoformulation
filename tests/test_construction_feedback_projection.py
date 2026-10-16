"""Opt-in feedback keeps distinct facts and historical payloads unchanged."""

from copy import deepcopy

import pytest

from autoformalism.search import construction_feedback_projection as projection
from autoformalism.search import construction_readable as readable
from tests.test_construction_ledger import apply, equation, process, variable
from tests.test_construction_readable import packet


def test_factoring_preserves_each_distinct_issue_status_and_numerical_evidence():
    common = {
        "action": "Correct the record only; preserve the intended physical effect. "
        * 12,
        "components": ["state", "flow"],
        "scope": "Declarations and explicit predicates only. " * 12,
    }
    witness = {
        "target": "x",
        "property": "closed_subsystem",
        "witnesses": [{"trajectory": "train_0", "value": 2, "uncertainty": 0.1}],
        "assumptions": ["identical shared initial values"],
    }
    original = {
        "model_view": {"balances": [{"topology": "d(x)/dt = + p"}]},
        "feedback": {
            "issues": [
                {"code": "realization", "status": "inconsistent", **common},
                {"code": "declaration", "status": "missing", **common},
                {"code": "other", "action": "Distinct required correction."},
            ],
            "scientific_review": {"questions": [witness]},
        },
        "requirements_checklist": [
            {"id": "mechanism:x", "status": "unavailable", **common}
        ],
        "public_brief": {"observations": [1, 2, 3]},
        "training_highlights": {"witness": witness},
    }
    saved = deepcopy(original)
    compact = projection.factor(original)
    assert compact["feedback_projection"]["policy"] == projection.POLICY
    assert compact == projection.factor(original)
    assert projection.expand(compact) == original == saved
    assert len(compact["feedback"]["issues"]) == 3
    assert [r["status"] for r in compact["feedback"]["issues"][:2]] == [
        "inconsistent",
        "missing",
    ]
    assert (
        compact["feedback"]["scientific_review"]
        == saved["feedback"]["scientific_review"]
    )
    for key in ("model_view", "training_highlights", "public_brief"):
        assert compact[key] == saved[key]


def test_matching_driver_lists_are_not_merged_or_reordered():
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[process()],
        equations=[equation("x", "u", "x"), equation("y", "u", "x")],
    )
    original = packet(d)
    projected = projection.project(original, d)
    assert projected["model_view"] == original["model_view"]
    assert d.processes[0].uses[0].sign == "negative"
    assert readable.model_view(d) == projected["model_view"]


def test_missing_process_use_does_not_claim_the_rhs_already_contains_it():
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[
            process(kind="influence", uses=[{"target": "x", "sign": "positive"}])
        ],
        equations=[equation("y", "p", "u"), equation("x", "u")],
    )
    original = packet(d)
    before = deepcopy(original)
    old = next(
        r
        for r in original["feedback"]["issues"]
        if r["code"] == "process_consumer_decision"
    )
    assert "already includes" in old["decision"]
    compact = projection.expand(projection.project(original, d))
    row = next(
        r
        for r in compact["feedback"]["issues"]
        if r["code"] == "process_consumer_decision"
    )
    assert row["existing_contribution"] == row["declared_uses_for_target"] == []
    assert "No signed use of p targets y" in row["decision"]
    assert "add the intended signed use" in row["decision"]
    assert "distinct composite law" in row["decision"]
    assert "cannot split" in row["decision"]
    assert original == before == packet(d)  # Historical prompt policy is unchanged.


def test_existing_process_use_keeps_actual_sign_and_conversion():
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[process()],
        equations=[equation("y", "p", "u"), equation("x", "u")],
    )
    issue = {"code": "process_consumer_decision", "process": "p", "target": "y"}
    row = readable.qualified_consumer_feedback(issue, d)
    assert "already includes p" in row["decision"]
    assert row["existing_contribution"] == [
        {"process": "p", "target": "y", "sign": "positive", "conversion": "1/area"}
    ]
    assert issue == {"code": "process_consumer_decision", "process": "p", "target": "y"}


def test_unknown_process_feedback_fails_instead_of_guessing():
    with pytest.raises(ValueError, match="retained process"):
        readable.qualified_consumer_feedback(
            {"code": "process_consumer_decision", "process": "missing", "target": "y"},
            apply(variables=[variable("y")]),
        )


@pytest.mark.parametrize("reserved", ["feedback_projection", "feedback_details"])
def test_reserved_projection_keys_fail_closed(reserved):
    with pytest.raises(ValueError, match="names collide"):
        projection.factor({reserved: {}})


def test_small_packets_keep_original_shape_and_no_truncation():
    value = {"feedback": {"issues": [{"code": "missing", "action": "repair"}]}}
    assert projection.factor(value) == value
    assert projection.expand(value) == value


def test_factor_only_exact_values_not_similar_prose_or_conflicting_status():
    value = {
        "feedback": {
            "issues": [
                {"code": "x", "status": "passed", "action": "a" * 300},
                {"code": "x", "status": "failed", "action": "b" * 300},
            ]
        }
    }
    assert projection.factor(value) == value
