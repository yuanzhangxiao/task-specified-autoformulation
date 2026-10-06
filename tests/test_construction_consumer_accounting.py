"""Use preservation, covariate-only overlap and actionable whole-draft feedback."""

import pytest

from autoformalism.search import construction_handoff as handoff
from autoformalism.search import construction_ledger as ledger
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    optional_memory_brief,
    patch,
    process,
    variable,
)
from tests.test_topology_clarification import overlapping_draft


def local_law(*, target="y"):
    return apply(
        variables=[variable("y")],
        equations=[equation("y", "u")],
        processes=[
            process(
                depends_on=["y"],
                kind="influence",
                uses=[{"target": target, "sign": "negative", "conversion": "1/area"}],
            )
        ],
    )


@pytest.mark.parametrize("target", ["p", "q"])
def test_no_process_definition_can_silently_consume_a_use(target):
    d = local_law(target=target)
    if target == "q":
        d = apply(
            d,
            processes=[
                process(
                    name="q",
                    depends_on=["u"],
                    kind="influence",
                    uses=[{"target": "y", "sign": "positive"}],
                )
            ],
        )
    a = ledger.assess(brief(), context(), {}, d)
    assert not a["eligible"]
    issue = next(e for e in a["errors"] if e["code"] == "process_definition_consumer")
    assert issue["process"] == "p" and issue["use"]["target"] == target
    assert all(r["scope"] == "unverified" for r in a["process_usage"])
    assert all(
        u["assembled_count"] is None for r in a["process_usage"] for u in r["uses"]
    )
    with pytest.raises(ValueError, match="runtime-generated definitions"):
        ledger.assembled_equations(brief(), d)
    assert d.processes[0].uses[0].target == target  # No invented reassignment.


def test_local_consumer_retains_sign_conversion_and_is_assembled_once():
    d = local_law()
    a = ledger.assess(brief(), context(), {}, d)
    assert a["eligible"], a["errors"]
    usage = a["process_usage"][0]
    assert usage["assembly_verified"] and usage["scope"] == "local"
    assert usage["assembled_consumers"] == ["y"]
    assert usage["uses"] == [
        {
            "target": "y",
            "sign": "negative",
            "conversion": "1/area",
            "assembled_count": 1,
        }
    ]
    assert a["shared_process_bindings"][0]["signed_declaration"]["uses"] == [
        {k: v for k, v in usage["uses"][0].items() if k != "assembled_count"}
    ]


@pytest.mark.parametrize("fault", ["missing", "twice", "wrong_sign"])
def test_independent_count_does_not_certify_bad_assembly(fault):
    d = local_law()
    equations = list(ledger.assembled_equations(brief(), d))
    e = equations[-1]
    terms = (
        e.terms[:-1]
        if fault == "missing"
        else (
            (*e.terms, e.terms[-1])
            if fault == "twice"
            else (
                *e.terms[:-1],
                e.terms[-1].model_copy(
                    update={
                        "outer_weight_sign": e.terms[-1].outer_weight_sign.__class__(
                            "positive"
                        )
                    }
                ),
            )
        )
    )
    equations[-1] = e.model_copy(update={"terms": terms})
    usage = ledger.process_usage(d, tuple(equations))[0]
    assert not usage["assembly_verified"] and usage["scope"] == "unverified"


def test_forward_consumer_is_pending_until_its_equation_is_declared():
    d = local_law(target="future")
    assert "future" in ledger.pending(brief(), d)["missing_declarations"]
    assert ledger.snapshot(brief(), d)["process_usage"][0]["scope"] == "unverified"
    with pytest.raises(ValueError, match="every declared consumer"):
        ledger.assembled_equations(brief(), d)
    d = apply(
        d, variables=[variable("future")], equations=[{"name": "future", "terms": []}]
    )
    assert ledger.assess(brief(), context(), {}, d)["eligible"]


def test_covariate_only_difference_requires_a_bound_scientific_choice():
    d = overlapping_draft()
    d = apply(d, equations=[equation("y", "u", "area")])
    overlaps = ledger.contribution_overlaps(d, brief())
    assert len(overlaps) == 1
    issue = overlaps[0]
    assert issue["fixed_covariate_comparison"]["matching_non_covariate_drivers"] == [
        "u"
    ]
    assert not ledger.assess(brief(), context(), {}, d, clarify_overlaps=True)[
        "eligible"
    ]
    confirmed = apply(
        d,
        overlap_confirmations=[
            {
                "overlap_id": issue["overlap_id"],
                "scientific_distinction": "A distinct response.",
            }
        ],
    )
    assert ledger.assess(brief(), context(), {}, confirmed, clarify_overlaps=True)[
        "eligible"
    ]
    changed = apply(
        confirmed,
        processes=[
            process(
                depends_on=["u"],
                kind="influence",
                uses=[{"target": "y", "sign": "positive", "conversion": "1/area"}],
            )
        ],
    )
    assert (
        ledger.contribution_overlaps(changed, brief())[0]["status"]
        == "clarification_required"
    )
    # An observation or input is never stripped as though it were a covariate.
    for source in ["a", "y"]:
        other = apply(d, equations=[equation("y", "u", source)])
        assert ledger.contribution_overlaps(other, brief()) == []
    # Covariate-only laws with different sources have no shared dynamic driver.
    other = apply(
        d,
        processes=[
            process(
                depends_on=["area"],
                kind="influence",
                uses=[{"target": "y", "sign": "positive"}],
            )
        ],
    )
    assert ledger.contribution_overlaps(other, brief()) == []


def test_algebraic_cycle_feedback_names_cycle_but_allows_dynamic_feedback():
    d = apply(local_law(), variables=[variable("y", "algebraic")])
    a = ledger.assess(brief(), context(), {}, d)
    issue = next(e for e in a["errors"] if e["code"] == "algebraic_dependency_cycle")
    assert issue["cycle"] == ["p", "y", "p"]
    assert {e["name"] for e in issue["equations"]} == {"p", "y"}
    assert ledger.assess(brief(), context(), {}, local_law())["eligible"]


def test_binding_feedback_names_actual_field_and_optional_removal():
    b = optional_memory_brief()
    d = apply(
        public=b,
        variables=[variable("y"), variable("storage")],
        equations=[equation("y", "u"), equation("storage", "u")],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["storage"]}],
    )
    a = ledger.assess(b, context(), {}, d)
    issue = next(e for e in a["errors"] if e["code"] == "memory_target_path")
    assert issue["target_ancestors"] == ["u"]
    feedback = issue["binding_feedback"]
    assert feedback["field"] == "mechanism_bindings"
    assert feedback["entry"]["memory_states"] == ["storage"]
    assert not feedback["mandatory"] and "remove_bindings" in str(
        feedback["repair_options"]
    )
    mandatory = ledger.assess(brief(True), context(), {}, d)
    issue = next(e for e in mandatory["errors"] if e["code"] == "memory_target_path")
    assert "remove_bindings" not in str(issue["binding_feedback"]["repair_options"])


@pytest.mark.parametrize("fault", [None, "source", "sign", "role", "conflict"])
def test_redundant_process_definition_normalized_only_if_exact(fault):
    d = local_law()
    p = process(
        depends_on=["u"], kind="influence", uses=[{"target": "y", "sign": "negative"}]
    )
    e = equation("p", "u")
    e["terms"][0]["scientific_role"] = p["scientific_meaning"]
    if fault == "source":
        e["terms"][0]["sources"] = ["y"]
    elif fault == "sign":
        e["terms"][0]["outer_weight_sign"] = "negative"
    elif fault == "role":
        e["terms"][0]["scientific_role"] = "Another meaning"
    raw = patch(
        processes=[p],
        equations=[e],
        stage_complete=True,
        remove_equations=["p"] if fault == "conflict" else [],
    ).model_dump(mode="json")
    normalized, log = ledger.normalize_reply(brief(), raw, draft=d)
    if fault:
        assert normalized == raw and log == []
        with pytest.raises(ValueError):
            ledger.apply_patch(brief(), d, ledger.DraftPatch.model_validate(normalized))
    else:
        assert log[0]["code"] == "repeated_process_definition"
        updated = ledger.apply_patch(
            brief(), d, ledger.DraftPatch.model_validate(normalized)
        )
        assert updated.processes[0].depends_on == ("u",)
        assert len(ledger.assembled_equations(brief(), updated)) == 2
    assert len(raw["equations"]) == 1  # Raw provider evidence stays immutable.


def test_edit_effects_show_new_variable_declarations():
    before = ledger.Draft()
    after = apply(variables=[variable("y")])
    assert handoff.edit_effects(before, after)["variables"][0]["key"] == "y"
