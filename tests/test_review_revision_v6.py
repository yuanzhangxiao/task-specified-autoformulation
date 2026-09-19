"""Whole-model patches retain grammar, role, closure and initializer checks."""

import copy

import pytest
from pydantic import ValidationError

from autoformalism.expressions import ModelValidationError
from autoformalism.search import review_revision_v5 as old
from autoformalism.search import review_revision_v6 as edits
from tests.test_review_deadline_v2 import example


def large_patch(kind="algebraic", count=12):
    return {
        "hypothesis": "Revise a connected chain of scientific definitions together.",
        "equations": [
            {
                "component": f"q{i}",
                "kind": kind,
                "expression": "m" if i == 0 else f"q{i - 1}",
            }
            for i in range(count)
        ]
        + [{"component": "f", "expression": f"tanh(q{count - 1})"}],
        "initializers": (
            [{"state": f"q{i}", "causal_map": None} for i in range(count)]
            if kind == "dynamic"
            else []
        ),
    }


@pytest.mark.parametrize("kind", ["algebraic", "dynamic"])
def test_large_patch_compiles_without_changing_legacy_or_source(kind):
    bundle, packet, values, *_ = example()
    raw = large_patch(kind)
    before = copy.deepcopy((bundle, raw))
    with pytest.raises(ValidationError):
        old.apply_edits(bundle, packet, raw)
    result = edits.apply_edits(bundle, packet, raw)
    assert result["outcome"] == "committed"
    audit = result["provenance"]
    assert audit["protocol"] == "scientific-content-revision-6"
    assert audit["size_audit"]["after"]["counts"]["generated_variables"] == 15
    assert (bundle, raw) == before
    assert edits.payload(bundle, packet, values)["patch_count_limits"] is None


def test_schema_and_prompt_do_not_advertise_removed_quotas():
    schema = edits.ScientificRevision.model_json_schema()["properties"]
    for key in ("equations", "initializers", "remove_variables", "new_parameters"):
        assert "maxItems" not in schema[key]
    assert "six" not in edits.SYSTEM_PROMPT
    assert "two new" not in edits.SYSTEM_PROMPT
    assert "serialization_capacity" in edits.SYSTEM_PROMPT
    assert edits.serialization_capacity()["states"] == 64


@pytest.mark.parametrize(
    "expression", ["__import__('os')", "unavailable*m", "v01+m", "q0"]
)
def test_large_patch_does_not_weaken_grammar_symbols_leakage_or_cycles(expression):
    bundle, packet, *_ = example()
    raw = large_patch()
    raw["equations"][0]["expression"] = expression
    before = copy.deepcopy(bundle)
    with pytest.raises((ValueError, ModelValidationError)):
        edits.apply_edits(bundle, packet, raw)
    assert bundle == before


def test_large_new_latent_inventory_still_requires_all_boundaries():
    bundle, packet, *_ = example()
    raw = large_patch("dynamic")
    raw["initializers"].pop()
    with pytest.raises(ValueError, match="explicit initializer"):
        edits.apply_edits(bundle, packet, raw)


def test_more_than_two_variables_can_be_removed_atomically():
    bundle, packet, *_ = example()
    result = edits.apply_edits(bundle, packet, large_patch())
    new = result["bundle"]
    from autoformalism.staged_topology import content_hash

    packet = {**packet, "candidate_sha256": content_hash(new["candidate"])}
    result = edits.apply_edits(
        new,
        packet,
        {
            "hypothesis": "Remove the entire unnecessary chain.",
            "remove_variables": [f"q{i}" for i in range(12)],
            "equations": [{"component": "f", "expression": "tanh(m)"}],
        },
    )
    assert (
        result["provenance"]["size_audit"]["after"]["counts"]["generated_variables"]
        == 3
    )


def test_unused_cleanup_and_internal_role_error_are_retained():
    bundle, packet, *_ = example()
    raw = large_patch()
    raw["new_parameters"] = [{"name": "unused"}]
    result = edits.apply_edits(bundle, packet, raw)
    assert len(result["provenance"]["unused_new_declarations_removed"]) == 1
    raw["new_parameters"] = [{"name": "shape_guess"}]
    raw["equations"][0]["expression"] = "tanh(shape_guess*m)"
    with pytest.raises(ValueError) as caught:
        edits.apply_edits(bundle, packet, raw)
    assert caught.value.code == "AMBIGUOUS_INTERNAL_PARAMETER_ROLE"
