"""Assembly ownership never infers scientific content from names or prose."""

from copy import deepcopy

import pytest

from autoformalism.expressions import ModelValidationError
from autoformalism.schemas.staged_functions import InteractionFunctionReply
from autoformalism.search import process_assembly_contract as assembly
from scripts import smoke_function_dependencies as smoke


def selected(*, sign="unrestricted", shared=True, conversion="1/area"):
    slot = {"lhs": "q", "outer_weight_sign": sign}
    bindings = (
        [
            {
                "proposal": {"name": "q"},
                "signed_declaration": {
                    "uses": [
                        {"target": "x", "sign": "negative", "conversion": conversion},
                        {"target": "y", "sign": "positive", "conversion": "1"},
                    ]
                },
            }
        ]
        if shared
        else []
    )
    return assembly.decorate(slot, bindings)


@pytest.mark.parametrize(
    "shared,sign", [(True, "unrestricted"), (False, "negative"), (False, "positive")]
)
def test_owned_outer_signs_normalized_once(shared, sign):
    slot = selected(shared=shared, sign=sign)
    raw = InteractionFunctionReply(expression="-(h-crest)*area", parameters=[])
    new, event = assembly.prepare(raw, slot)
    assert new.expression == "(h - crest) * area"
    assert event["sign_normalizations"][0]["removed_negative_factors"] == 1
    assert raw.expression == "-(h-crest)*area"
    assert assembly.prepare(new, slot)[1]["sign_normalizations"] == []


@pytest.mark.parametrize("expression", ["h-crest", "exp(-h)", "(-h)**2", "-h+crest"])
def test_internal_signs_survive(expression):
    raw = InteractionFunctionReply(expression=expression, parameters=[])
    assert assembly.prepare(raw, selected())[0] == raw


def test_unrestricted_ordinary_law_and_automatic_identities_survive():
    raw = InteractionFunctionReply(expression="-h", parameters=[])
    assert assembly.prepare(raw, selected(shared=False))[0] == raw
    identity = InteractionFunctionReply(expression="q", parameters=[])
    assert (
        assembly.prepare(identity, selected(shared=False, sign="negative"))[0]
        == identity
    )


@pytest.mark.parametrize("expression", ["h/area", "h*(1/area)", "(h-crest)/area"])
def test_conversion_overlap_requires_proposer_decision(expression):
    raw = InteractionFunctionReply(expression=expression, parameters=[])
    with pytest.raises(ValueError, match="CONSUMER_CONVERSION_OVERLAP"):
        assembly.prepare(raw, selected())
    explicit = assembly.AssemblyFunctionReply(
        **raw.model_dump(), conversion_factor_is_intrinsic=True
    )
    prepared, event = assembly.prepare(explicit, selected())
    assert prepared == raw
    assert event["intrinsic_factor_confirmed"]
    assert event["consumer_conversion_overlap"][0]["target"] == "x"


@pytest.mark.parametrize(
    "expression", ["max(0,h-area)", "h*area", "h/(area+h)", "(h/area)+h"]
)
def test_covariate_reuse_is_not_automatically_a_conversion(expression):
    assert not assembly.conversion_overlap(expression, selected())


def test_ratio_conversion_and_unresolved_conversion():
    slot = selected(conversion="area_up/area_down")
    assert assembly.conversion_overlap("area_up*h/area_down", slot)
    assert not assembly.conversion_overlap("h/area_down", slot)
    assert not assembly.conversion_overlap("h/area", selected(conversion=None))


def test_downstream_connection_is_preserved_even_if_driver_path_survives():
    brief, _, source = smoke.fixture()
    # q still drives x, and x drives y directly; make an extra named diagnostic
    # process whose only target path is via the ordinary y term.
    source["equations"].insert(
        0, {"name": "outlet", "definition": "algebraic", "terms": [{"sources": ["y"]}]}
    )
    source["equations"][-1]["terms"][0]["sources"] = ["outlet"]
    after = deepcopy(source)
    after["equations"][-1]["terms"][0]["sources"] = ["y"]
    with pytest.raises(ValueError, match=r"PROCESS_TARGET_PATH_LOST.*outlet"):
        assembly.preserve_paths(brief, source, after)
    assembly.preserve_paths(brief, after, after)  # Already disconnected is advisory.


def test_failed_local_path_edit_leaves_source_untouched():
    from autoformalism.schemas.staged_topology import EquationDefinition, EquationTerm
    from autoformalism.search import function_dependencies as dep

    brief, context, source = smoke.fixture()
    source["inventory"].append(
        {"name": "outlet", "definition": "algebraic", "scientific_role": "outlet"}
    )
    source["equations"].append(
        EquationDefinition(
            name="outlet", definition="algebraic",
            terms=(EquationTerm(
                sources=("y",), outer_weight_sign="positive", scientific_role="outlet"
            ),),
        ).model_dump(mode="json")
    )
    source["equations"][2]["terms"][0]["sources"] = ["outlet"]
    before = deepcopy(source)
    reply = dep.DependencyFunctionReply(
        expression="y", parameters=[], revise_dependencies=True
    )
    # The old checks still pass: memory and shared immediate consumers survive.
    assert dep.prepare(brief, context, source, "term_2_0", reply)[1]
    with pytest.raises(ValueError, match="PROCESS_TARGET_PATH_LOST"):
        dep.prepare(
            brief, context, source, "term_2_0", reply, preserve_process_paths=True
        )
    assert source == before


def test_metadata_cannot_authorize_code_or_coerced_confirmation():
    with pytest.raises(ValueError):
        assembly.AssemblyFunctionReply(
            expression="h", parameters=[], conversion_factor_is_intrinsic="yes"
        )
    with pytest.raises(ModelValidationError):
        assembly.prepare(
            InteractionFunctionReply(expression="__import__('os')", parameters=[]),
            selected(),
        )


def test_complete_live_reconstruction_and_resume(tmp_path):
    from scripts import smoke_process_assembly_contract as live

    assert live.run(tmp_path)["status"] == "passed"


@pytest.mark.parametrize("damage", ["normalized", "missing", "source", "policy", "raw"])
def test_reconstruction_rejects_tampered_assembly(tmp_path, damage):
    from autoformalism.rebuttal.prefit_construction_audit import reconstruct
    from scripts import smoke_process_assembly_contract as live

    brief, context, source, result = live.construct(tmp_path, [])
    assert result["complete_model"]
    result = deepcopy(result)
    if damage == "normalized":
        result["assembly_decisions"][0]["normalized_reply"]["expression"] = "x"
    elif damage == "missing":
        result["assembly_decisions"].pop()
    elif damage == "source":
        result["provider_visible_accepted_functions"][0]["expression"] = "x"
    elif damage == "policy":
        result["assembly_policy"] = "unknown"
    else:
        result["assembly_decisions"][0]["raw_reply"][
            "conversion_factor_is_intrinsic"
        ] = True
    with pytest.raises(ValueError):
        reconstruct(
            {
                "brief": brief.model_dump(mode="json"),
                "context": context.model_dump(mode="json"),
            },
            {"topology": source, "functions": result},
        )


def test_new_policy_requires_new_root_and_hybrid_mode(tmp_path):
    from scripts import smoke_process_assembly_contract as live

    live.construct(tmp_path, [])
    with pytest.raises(ValueError, match="contract differs"):
        live.construct(tmp_path, [], assembly_policy="legacy")
