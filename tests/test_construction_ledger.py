"""Forward references, atomic edits and final structural closure."""

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_ledger as ledger


def brief(memory=False):
    return PublicScientificBrief.model_validate(
        {
            "scientific_context": "Generate y from u; a is a supplied channel.",
            "public_variables": [
                {"name": "y", "data_role": "target"},
                {"name": "u", "data_role": "external_input"},
                {"name": "a", "data_role": "auxiliary"},
                {"name": "area", "data_role": "covariate"},
            ],
            "requirements": [
                {
                    "id": "memory",
                    "public_requirement": "Delayed input effect",
                    "drivers": ["u"],
                    "targets": ["y"],
                    "requires_dynamic_memory": True,
                }
            ]
            if memory
            else [],
        }
    )


def context():
    return ValidationContext(
        targets=("y",),
        auxiliaries=("a",),
        external_inputs=("u",),
        fixed_covariates=("area",),
    )


def variable(name, definition="differential"):
    return {
        "name": name,
        "definition": definition,
        "scientific_role": "physical quantity",
    }


def equation(name, *sources):
    return {
        "name": name,
        "terms": [
            {
                "sources": list(sources),
                "outer_weight_sign": "unrestricted",
                "scientific_role": "joint effect",
            }
        ],
    }


def patch(**kwargs):
    return ledger.DraftPatch.model_validate({"stage_complete": False, **kwargs})


def apply(draft=None, *, public=None, **kwargs):
    return ledger.apply_patch(
        public or brief(), draft or ledger.Draft(), patch(**kwargs)
    )


def assess(draft, memory=False):
    return ledger.assess(brief(memory), context(), {}, draft)


def process(**updates):
    return {
        "name": "p",
        "depends_on": ["x"],
        "kind": "transfer",
        "scientific_meaning": "shared release",
        "uses": [
            {"target": "x", "sign": "negative", "conversion": "1"},
            {"target": "y", "sign": "positive", "conversion": "1/area"},
        ],
        **updates,
    }


def test_available_source_needs_no_activation_and_unused_is_derived():
    d = apply(variables=[variable("y")], equations=[equation("y", "a")])
    assert assess(d)["eligible"]
    assert {
        v.name for v in ledger.inventory(brief(), d) if v.definition == "supplied"
    } == {"u", "a", "area"}
    assert len(d.variables) == 1


def test_new_rhs_and_lhs_before_declaration_are_pending_not_rejected():
    d = apply(variables=[variable("y")], equations=[equation("y", "x")])
    assert ledger.pending(brief(), d)["missing_declarations"] == ["x"]
    d = apply(d, equations=[equation("x", "u", "x")])
    assert not assess(d)["eligible"]
    d = apply(d, variables=[variable("x")])
    assert assess(d)["eligible"]


def test_indirect_memory_path_waits_for_all_equations():
    b = brief(True)
    d = apply(
        public=b,
        variables=[variable("y"), variable("x"), variable("z", "algebraic")],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["x"]}],
        equations=[equation("y", "x"), equation("x", "z", "x")],
    )
    assert not assess(d, True)["eligible"]
    d = apply(d, public=b, equations=[equation("z", "u")])
    assert assess(d, True)["eligible"]


def test_differential_cycles_are_valid_algebraic_cycles_are_not():
    d = apply(
        variables=[variable("y"), variable("x")],
        equations=[equation("y", "x"), equation("x", "y")],
    )
    assert assess(d)["eligible"]
    d = apply(d, variables=[variable("y", "algebraic"), variable("x", "algebraic")])
    assert "algebraic cycle" in str(assess(d)["errors"])


def test_process_update_propagates_once_and_removal_exposes_pending_rhs():
    d = apply(
        variables=[variable("y"), variable("x")],
        processes=[process()],
        equations=[{"name": "y", "terms": []}, equation("x", "u")],
    )
    a = assess(d)
    assert a["eligible"], a
    assert [e["name"] for e in a["equations"]].count("p") == 1
    d = apply(d, processes=[process(depends_on=["x", "a"])])
    assert assess(d)["equations"][0]["terms"][0]["sources"] == ["x", "a"]
    assert len(assess(d)["equations"][1]["terms"]) == 1
    d = apply(d, remove_processes=["p"])
    assert not assess(d)["eligible"]  # Empty y now requires an actual contribution.


def test_one_sided_transfer_is_draft_until_global_repair():
    p = process(uses=[{"target": "y", "sign": "positive", "conversion": None}])
    d = apply(
        variables=[variable("y"), variable("x")],
        processes=[p],
        equations=[{"name": "y", "terms": []}, equation("x", "u")],
    )
    assert any(e["code"] == "process_uses" for e in assess(d)["errors"])
    d = apply(d, processes=[{**p, "kind": "influence"}])
    assert assess(d)["eligible"]


@pytest.mark.parametrize(
    "changes",
    [
        {"variables": [variable("u")]},
        {"remove_variables": ["missing"]},
        {"variables": [variable("y")], "remove_variables": ["y"]},
        {"variables": [variable("y"), variable("y")]},
        {
            "processes": [
                process(
                    uses=[
                        {
                            "target": "y",
                            "sign": "positive",
                            "conversion": "__import__('os')",
                        }
                    ]
                )
            ]
        },
    ],
)
def test_invalid_batches_do_not_partially_mutate_draft(changes):
    d = apply(variables=[variable("y")], equations=[equation("y", "u")])
    before = d.model_dump()
    with pytest.raises(ValueError):
        apply(d, **changes)
    assert d.model_dump() == before


def test_public_target_never_becomes_supplied_and_role_prose_not_a_gate():
    d = apply(
        variables=[{**variable("y"), "scientific_role": "imperfect explanation"}],
        equations=[equation("y", "u")],
    )
    assert assess(d)["eligible"]
    d = apply(d, remove_variables=["y"])
    assert not assess(d)["eligible"]
    assert "y" in ledger.pending(brief(), d)["missing_declarations"]


def test_target_type_and_supplied_consumer_checked_at_completion():
    d = apply(
        variables=[variable("y")],
        equations=[equation("y", "u")],
        processes=[
            process(
                depends_on=["u"],
                kind="influence",
                uses=[{"target": "a", "sign": "positive", "conversion": None}],
            )
        ],
    )
    result = ledger.assess(brief(), context(), {"y": "algebraic"}, d)
    assert {e["code"] for e in result["errors"]} >= {
        "public_target_type",
        "supplied_process_consumers",
    }


def optional_memory_brief(*, drivers=("u",), targets=("y",)):
    """A real named requirement may permit memory without mandating a mediator."""
    b = brief(True)
    return b.model_copy(
        update={
            "requirements": tuple(
                r.model_copy(
                    update={
                        "requires_dynamic_memory": False,
                        "drivers": drivers,
                        "targets": targets,
                    }
                )
                for r in b.requirements
            )
        }
    )


@pytest.mark.parametrize("state", ["x", "y"])
def test_optional_memory_allows_latent_or_target_accumulator(state):
    b = optional_memory_brief()
    d = apply(
        public=b,
        variables=[variable("y"), variable("x")],
        equations=[equation("y", "x"), equation("x", "u")],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": [state]}],
    )
    result = ledger.assess(b, context(), {}, d)
    assert result["eligible"], result["errors"]
    assert result["memory_binding_checks"][0]["status"] == "passed"
    assert not result["memory_binding_checks"][0]["mandatory"]


def test_optional_binding_with_underspecified_endpoints_is_unresolved():
    b = optional_memory_brief(drivers=())
    d = apply(
        public=b,
        variables=[variable("y")],
        equations=[equation("y", "u")],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["y"]}],
    )
    result = ledger.assess(b, context(), {}, d)
    assert result["eligible"]
    assert result["memory_binding_checks"][0]["status"] == "unresolved_public_endpoints"


@pytest.mark.parametrize("bad", ["unknown_id", "algebraic", "disconnected"])
def test_optional_memory_does_not_waive_id_type_or_path_errors(bad):
    b = optional_memory_brief()
    d = apply(
        public=b,
        variables=[
            variable("y"),
            variable("x", "algebraic" if bad == "algebraic" else "differential"),
        ],
        equations=[
            equation("y", "u" if bad == "disconnected" else "x"),
            equation("x", "u"),
        ],
        mechanism_bindings=[
            {
                "requirement_id": "oops" if bad == "unknown_id" else "memory",
                "memory_states": ["x"],
            }
        ],
    )
    result = ledger.assess(b, context(), {}, d)
    expected = {
        "unknown_id": "unknown_memory_requirement",
        "algebraic": "memory_type",
        "disconnected": "memory_target_path",
    }[bad]
    assert expected in {e["code"] for e in result["errors"]}


def test_mandatory_delayed_memory_still_requires_distinct_mediator():
    b = brief(True)
    d = apply(
        public=b,
        variables=[variable("y")],
        equations=[equation("y", "u")],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["y"]}],
    )
    assert "memory_type" in {
        e["code"] for e in ledger.assess(b, context(), {}, d)["errors"]
    }


def test_conversion_feedback_names_use_and_does_not_guess_fitted_coefficient():
    d = apply(variables=[variable("y"), variable("x")])
    before = d.model_dump()
    with pytest.raises(
        ValueError, match=r"p -> y: invalid fixed conversion 'k'.*Use null"
    ):
        apply(
            d,
            processes=[
                process(
                    uses=[{"target": "y", "sign": "positive", "conversion": "k"}],
                    kind="influence",
                )
            ],
        )
    assert d.model_dump() == before


def test_constant_term_supported_but_empty_unassembled_rhs_explained():
    d = apply(
        variables=[variable("y", "algebraic"), variable("x")],
        equations=[equation("y"), equation("x", "u")],
    )
    assert assess(d)["eligible"]
    d = apply(d, equations=[{"name": "y", "terms": []}])
    assert "empty assembled RHS" in str(assess(d)["errors"])
