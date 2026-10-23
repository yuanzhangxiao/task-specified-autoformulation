"""Feedback direction and repetition refer to equations, never changed prose."""

from copy import deepcopy

import pytest

from autoformalism.research import construction_directional_feedback as feedback
from autoformalism.research.construction_function_consistency import assess
from tests.test_construction_function_consistency import evidence, problem


@pytest.mark.parametrize("upper", [True, False])
def test_directional_bound_is_not_described_as_observed_requirement(upper):
    source = problem(rhs="-k*x" if upper else "u+k*m")
    data = evidence(source, y=(0, 2, 4) if upper else (2, 2, 2))
    original = assess(source, data)
    before = deepcopy(original)
    check = next(
        c for c in feedback.function_view(original)["conflicts"] if c["target"] == "y"
    )
    witness = check["witness"]
    assert "required_change" not in witness
    assert witness["model_implied_change_bound"]["comparison"] == (
        "at_most" if upper else "at_least"
    )
    assert ("upper bound is too low" if upper else "lower bound is too high") in (
        witness["directional_explanation"]
    )
    assert original == before


def patch():
    return {
        "hypothesis": "why",
        "evidence_ids": ["public_task"],
        "expected_observable_effect": "effect",
        "edits": [
            {"component": "X", "kind": "dynamic", "sources": [], "expression": "-k*X+u"}
        ],
        "new_parameters": [{"name": "k", "role": "coefficient"}],
    }


def test_repeat_signature_ignores_prose_labels_order_and_advisory_sources():
    original = patch()
    changed = deepcopy(original)
    changed.update(
        hypothesis="another explanation", evidence_ids=["training_consistency"]
    )
    changed["edits"][0].update(expression="d(X)/dt = (-k * X) + u", sources=["u"])
    assert feedback.patch_signature(original) == feedback.patch_signature(changed)
    changed["edits"][0]["expression"] = "-k*X-u"
    assert feedback.patch_signature(original) != feedback.patch_signature(changed)
    changed = deepcopy(original)
    changed["new_parameters"][0]["role"] = "nonnegative_coefficient"
    assert feedback.patch_signature(original) != feedback.patch_signature(changed)


@pytest.mark.parametrize(
    "bad", [None, {"edits": "bad"}, {"edits": [{}]}, {"edits": [{}] * 33}]
)
def test_bad_signature_never_controls_acceptance(bad):
    assert feedback.patch_signature(bad) is None


def test_nested_section_citation_alias_is_metadata_only():
    original = patch()
    original["evidence_ids"] = [
        "function_consistency",
        "training_consistency",
        "unknown",
    ]
    before = deepcopy(original)
    normalized, changes = feedback.normalize_citations(original)
    assert normalized["evidence_ids"] == ["training_consistency", "unknown"]
    assert normalized["edits"] == original["edits"]
    assert changes == [{"from": "function_consistency", "to": "training_consistency"}]
    assert original == before


@pytest.mark.parametrize("value", [[], None, "text", 3])
def test_non_object_citation_payload_is_schema_error(value):
    with pytest.raises(ValueError, match="JSON object"):
        feedback.normalize_citations(value)
