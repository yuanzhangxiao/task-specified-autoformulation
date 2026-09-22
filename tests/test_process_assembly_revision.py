"""Atomic assembly revisions preserve hard contracts and historical artifacts."""

from copy import deepcopy

import pytest

from autoformalism.search import process_assembly_revision as revision
from autoformalism.search import signed_processes as signed
from scripts import smoke_function_dependencies as toy


def fixture():
    brief, context, source = toy.fixture()
    # The old proposal included area solely because a consumer uses 1/area.
    source["equations"][0]["terms"][0]["sources"] = ["x", "crest", "area"]
    binding = source["shared_process_contract"]["bindings"][0]
    binding["signed_declaration"]["depends_on"] = ["x", "crest", "area"]
    binding.update(
        signed.binding_for(
            signed.SignedProcess.model_validate(binding["signed_declaration"])
        )
    )
    topology, _ = revision.bind_known(brief, context, source, {})
    source["topology"] = topology.model_dump(mode="json")
    return brief, context, source


def reply(expression="max(0,x-crest)", **kwargs):
    return revision.RevisionReply(expression=expression, parameters=[], **kwargs)


def test_conversion_owned_covariate_can_be_omitted_without_fake_factor():
    brief, context, source = fixture()
    before = deepcopy(source)
    result = revision.prepare(brief, context, source, "term_0_0", reply())
    assert result["dependency_changes"][0]["conversion_only_omissions"] == ["area"]
    effective = result["effective_source"]
    assert effective["equations"][0]["terms"][0]["sources"] == ["x", "crest"]
    binding = effective["shared_process_contract"]["bindings"][0]
    assert binding["signed_declaration"]["uses"][0]["conversion"] == "1/area"
    assert source == before
    assert revision.replay(brief, context, source, {}, result) == result


def test_shape_covariate_and_dynamic_dependencies_need_explicit_decision():
    brief, context, source = fixture()
    with pytest.raises(ValueError, match=r"DEPENDENCY_DECISION_REQUIRED.*crest"):
        revision.prepare(brief, context, source, "term_0_0", reply("x"))
    with pytest.raises(ValueError, match="PUBLIC_PATHWAY_REPAIR_REQUIRED"):
        revision.prepare(
            brief, context, source, "term_0_0", reply("u", revise_dependencies=True)
        )


def test_law_and_conversion_change_together_without_altering_signs():
    brief, context, source = fixture()
    before = deepcopy(source)
    # Move the area scaling into the common law, explicitly adjust both consumers.
    result = revision.prepare(
        brief,
        context,
        source,
        "term_0_0",
        reply(
            "max(0,x-crest)/area",
            consumer_conversions=[
                {"target": "x", "conversion": "1"},
                {"target": "y", "conversion": "area"},
            ],
        ),
    )
    assert not result["remaining_conversion_overlap"]
    assert len(result["conversion_changes"]) == 2
    assert [
        u["sign"]
        for u in result["effective_source"]["shared_process_contract"]["bindings"][0][
            "signed_declaration"
        ]["uses"]
    ] == ["negative", "positive"]
    assert source == before


@pytest.mark.parametrize(
    "conversion", ["-1", "0", "u", "area+1", "fitted_gain", "__import__('os')"]
)
def test_unsafe_or_nonfixed_conversions_do_not_mutate_source(conversion):
    brief, context, source = fixture()
    before = deepcopy(source)
    with pytest.raises(ValueError):
        revision.prepare(
            brief,
            context,
            source,
            "term_0_0",
            reply(
                consumer_conversions=[{"target": "x", "conversion": conversion}],
            ),
        )
    assert source == before


def test_actual_overlap_needs_target_specific_decision_and_stays_unverified():
    brief, context, source = fixture()
    with pytest.raises(ValueError, match="CONVERSION_REPAIR_REQUIRED"):
        revision.prepare(
            brief,
            context,
            source,
            "term_0_0",
            reply("x/area", revise_dependencies=True),
        )
    result = revision.prepare(
        brief,
        context,
        source,
        "term_0_0",
        reply(
            "x/area",
            revise_dependencies=True,
            retain_intrinsic_for=["x"],
        ),
    )
    assert result["conversion_status"] == "retained_intrinsic_unverified"
    assert not result["scientific_validity_certified"]
    with pytest.raises(ValueError, match="actual overlapping"):
        revision.prepare(
            brief, context, source, "term_0_0", reply(retain_intrinsic_for=["x"])
        )
    with pytest.raises(ValueError, match="Extra inputs"):
        reply(conversion_factor_is_intrinsic=True)


def test_bounded_topology_repair_validates_final_graph_not_intermediate_edits():
    brief, context, source = fixture()
    r = reply(
        "u",
        revise_dependencies=True,
        companion_laws=[
            {
                "interaction_id": "term_2_0",
                "expression": "y+x",
                "parameters": [],
                "revise_dependencies": True,
            },
        ],
    )
    with pytest.raises(ValueError, match="bounded topology"):
        revision.prepare(brief, context, source, "term_0_0", r)
    result = revision.prepare(
        brief, context, source, "term_0_0", r, allow_topology_repair=True
    )
    assert all(
        c["passed"] for c in result["effective_source"]["public_structure_checks"]
    )
    assert result["functions"]["term_2_0"]["expression"] == "y+x"
    assert revision.replay(brief, context, source, {}, result) == result


@pytest.mark.parametrize("slot", ["term_1_1", "term_0_0", "term_99_0"])
def test_topology_scope_cannot_overwrite_shared_identity_or_repeat_focus(slot):
    brief, context, source = fixture()
    with pytest.raises(ValueError):
        revision.prepare(
            brief,
            context,
            source,
            "term_0_0",
            reply(
                companion_laws=[
                    {
                        "interaction_id": slot,
                        "expression": "u",
                        "parameters": [],
                        "revise_dependencies": True,
                    },
                ]
            ),
            allow_topology_repair=True,
        )


def test_retained_functions_are_revalidated_and_transaction_is_all_or_nothing():
    brief, context, source = fixture()
    known = {"term_1_0": {"expression": "private_value", "parameters": []}}
    before = deepcopy(source)
    with pytest.raises(ValueError):
        revision.prepare(
            brief, context, source, "term_0_0", reply(), known_functions=known
        )
    assert source == before


@pytest.mark.parametrize(
    "field", ["effective_source", "functions", "conversion_status", "reply", "scope"]
)
def test_transaction_replay_rejects_tampering(field):
    brief, context, source = fixture()
    result = revision.prepare(brief, context, source, "term_0_0", reply())
    damaged = deepcopy(result)
    damaged[field] = {} if isinstance(result[field], dict) else "tampered"
    with pytest.raises(ValueError):
        revision.replay(brief, context, source, {}, damaged)
