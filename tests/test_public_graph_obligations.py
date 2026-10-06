"""Public feedback/isolation checks allow equivalent coordinates, not fake proofs."""

import pytest

from autoformalism.research import construction_obligations as profiles
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_ledger as ledger
from autoformalism.search.public_graph_obligations import (
    GraphObligation,
    PublicGraphContract,
)
from autoformalism.staged_topology import content_hash
from tests.test_construction_ledger import apply, brief, context, equation, variable


def contract(kind="dynamic_feedback"):
    return PublicGraphContract(
        source_brief_sha256=content_hash(brief().model_dump(mode="json")),
        obligations=[
            GraphObligation(
                id="public_rule",
                kind=kind,
                target="y",
                source="u" if kind == "forbidden_path" else None,
                public_quote="Generate y from u",
                interpretation="Explicit test policy",
            )
        ],
        deferred_scientific_checks=["Function signs and behavior remain unassessed."],
    )


def assessment(draft, kind="dynamic_feedback"):
    return ledger.assess(brief(), context(), {}, draft, graph_contract=contract(kind))


@pytest.mark.parametrize("mode", ["direct", "coupled", "energy_readout", "shared"])
def test_feedback_accepts_equivalent_coordinates(mode):
    if mode == "direct":
        d = apply(variables=[variable("y")], equations=[equation("y", "u", "y")])
    elif mode == "coupled":
        d = apply(
            variables=[variable("y"), variable("x")],
            equations=[equation("y", "x"), equation("x", "y", "u")],
        )
    elif mode == "energy_readout":
        d = apply(
            variables=[variable("y", "algebraic"), variable("energy")],
            equations=[equation("y", "energy"), equation("energy", "u", "energy")],
        )
    else:
        d = apply(
            variables=[variable("y"), variable("x")],
            processes=[
                {
                    "name": "transfer",
                    "depends_on": ["y"],
                    "kind": "transfer",
                    "scientific_meaning": "shared flow",
                    "uses": [
                        {"target": "y", "sign": "negative", "conversion": None},
                        {"target": "x", "sign": "positive", "conversion": None},
                    ],
                }
            ],
            equations=[{"name": "y", "terms": []}, {"name": "x", "terms": []}],
        )
    result = assessment(d)
    assert result["eligible"], result["errors"]
    check = result["reviewed_public_graph_checks"][0]
    assert check["passed"] is True
    assert len(check["witness"]["cycle"]) >= 2
    assert result["scientific_adequacy"] == "not_assessed"


@pytest.mark.parametrize("disconnected_cycle", [False, True])
def test_accumulator_fails_only_when_explicitly_required(disconnected_cycle):
    d = apply(variables=[variable("y")], equations=[equation("y", "u")])
    if disconnected_cycle:
        d = apply(d, variables=[variable("x")], equations=[equation("x", "x")])
    assert ledger.assess(brief(), context(), {}, d)["eligible"]
    result = assessment(d)
    assert not result["eligible"]
    assert result["reviewed_public_graph_checks"][0]["status"] == "fail"


@pytest.mark.parametrize("broken", ["missing", "algebraic_loop"])
def test_uncompiled_graph_is_unavailable(broken):
    d = apply(
        variables=[variable("y", "algebraic")],
        equations=[equation("y", "missing" if broken == "missing" else "y")],
    )
    result = assessment(d)
    assert not result["eligible"]
    assert result["reviewed_public_graph_checks"][0]["passed"] is None
    assert not any(e["code"] == "reviewed_public_graph" for e in result["errors"])


def test_forbidden_path_checks_indirect_influence_and_keeps_disconnected_states():
    d = apply(
        variables=[variable("y"), variable("x")],
        equations=[equation("x", "u"), equation("y", "x")],
    )
    result = assessment(d, "forbidden_path")
    assert not result["eligible"]
    assert result["reviewed_public_graph_checks"][0]["witness"] == ["u", "x", "y"]
    d = apply(d, equations=[equation("y", "y")])
    assert assessment(d, "forbidden_path")["eligible"]


def test_contract_rejects_unsupported_endpoints_quotes_and_duplicate_ids():
    c = contract()
    with pytest.raises(ValueError, match="quote differs"):
        c.validate_public(
            brief().model_copy(update={"scientific_context": "Other task"})
        )
    with pytest.raises(ValueError, match="endpoint"):
        c.model_copy(
            update={
                "obligations": (c.obligations[0].model_copy(update={"target": "z"}),)
            }
        ).validate_public(brief())
    with pytest.raises(ValueError, match="duplicate"):
        PublicGraphContract.model_validate(
            {**c.model_dump(), "obligations": c.obligations * 2}
        )
    with pytest.raises(ValueError, match="requires a source"):
        GraphObligation.model_validate(
            {**c.obligations[0].model_dump(), "kind": "forbidden_path"}
        )


def test_reviewed_profiles_are_explicit_and_do_not_add_universal_decay():
    cstr = PublicScientificBrief(
        scientific_context=(
            "a reactor-temperature balance that distinguishes feed transport, "
            "reaction heat generation, and heat exchange with the jacket"
        ),
        public_variables=[{"name": "T", "data_role": "target"}],
        requirements=[],
    )
    c = profiles.reviewed_contract(profiles.CSTR, cstr)
    assert len(c.obligations) == 1
    assert c.obligations[0].kind == "target_feedback"
    for name in (*profiles.DALLA, profiles.ALIEN):
        assert profiles.reviewed_contract(name, brief()).obligations == ()
    with pytest.raises(ValueError, match="quote differs"):
        profiles.reviewed_contract(
            profiles.CSTR, cstr.model_copy(update={"scientific_context": "Unrelated"})
        )
    with pytest.raises(ValueError, match="no reviewed"):
        profiles.reviewed_contract("unknown", brief())
