"""General realization declarations preserve memory and public graph contracts."""

from copy import deepcopy

import pytest

from autoformalism.search import construction_ledger as ledger
from autoformalism.search import mechanism_realizations as realizations
from autoformalism.search import scientific_verification
from autoformalism.staged_topology import content_hash
from tests.test_construction_ledger import (
    brief,
    context,
    equation,
    optional_memory_brief,
    process,
    variable,
)
from tests.test_public_graph_obligations import contract


def record(*components, memory_states=(), requirement_id="memory"):
    return {
        "requirement_id": requirement_id,
        "components": list(components),
        "memory_states": list(memory_states),
    }


def apply(draft=None, *, public=None, **edits):
    return realizations.apply_patch(
        public or optional_memory_brief(),
        draft or realizations.Draft(),
        realizations.DraftPatch.model_validate({"stage_complete": False, **edits}),
    )


def assess(draft, *, public=None, graph=None):
    return realizations.assess(
        public or optional_memory_brief(),
        context(),
        {},
        draft,
        graph_contract=graph,
    )


def rows(draft, *, public=None):
    return realizations.variable_checklist(public or optional_memory_brief(), draft, {})


def test_algebraic_mechanism_needs_no_memory_and_type_is_inferred():
    draft = apply(
        variables=[variable("y"), variable("effect", "algebraic")],
        mechanism_realizations=[record("effect")],
    )
    early = rows(draft)
    assert early["status"] == "ready"
    row = early["mechanism_realization_checks"][0]
    assert row["inferred_type"] == "algebraic"
    assert row["structural_status"] == "deferred"
    assert row["behavior_status"] == "unverified"
    assert draft.mechanism_bindings == ()
    draft = apply(draft, equations=[equation("y", "effect"), equation("effect", "u")])
    checked = assess(draft)
    assert checked["eligible"], checked["errors"]
    assert checked["mechanism_realization_checks"][0]["structural_status"] == "passed"
    assert checked["scientific_adequacy"] == "not_assessed"


def test_every_requirement_needs_record_not_every_record_needs_a_state():
    draft = apply(variables=[variable("y")])
    assert rows(draft)["blocking_items"] == ["mechanism_realization:memory"]
    assert realizations.pending(optional_memory_brief(), draft)[
        "required_mechanism_realizations"
    ] == ["memory"]
    draft = apply(draft, mechanism_realizations=[record("y")])
    assert rows(draft)["status"] == "ready"
    assert len(draft.variables) == 1


def test_supplied_sources_are_references_not_a_required_generated_declaration():
    draft = apply(variables=[variable("y")], mechanism_realizations=[record("u")])
    assert rows(draft)["status"] == "ready"
    assert (
        rows(draft)["mechanism_realization_checks"][0]["inferred_type"] == "unresolved"
    )
    assert (
        realizations.pending(optional_memory_brief(), draft)["missing_declarations"]
        == []
    )
    draft = apply(draft, equations=[equation("y", "u")])
    assert assess(draft)["eligible"]


@pytest.mark.parametrize("name", ["unknown", "missing_quantity"])
def test_unknown_id_or_reference_is_pending_then_blocks_closure(name):
    draft = apply(
        variables=[variable("y")],
        mechanism_realizations=[
            record("y", requirement_id="unknown") if name == "unknown" else record(name)
        ],
    )
    assert rows(draft)["status"] == "incomplete"
    assert not assess(draft)["eligible"]
    assert any(
        error["code"] == "mechanism_realization_declaration"
        for error in assess(draft)["errors"]
    )
    if name == "missing_quantity":
        assert (
            name
            in realizations.pending(optional_memory_brief(), draft)[
                "missing_declarations"
            ]
        )
        draft = apply(draft, variables=[variable(name, "algebraic")])
        assert rows(draft)["status"] == "ready"


@pytest.mark.parametrize(
    "bad_record",
    [
        record(),
        record("x", "x"),
        record("x", memory_states=("x", "x")),
        record("y", memory_states=("x",)),
        {**record("y"), "realized": True},
        {**record("y"), "type": "differential"},
    ],
)
def test_strict_record_schema_rejects_bad_or_self_certifying_annotations(bad_record):
    with pytest.raises(ValueError):
        realizations.MechanismRealization.model_validate(bad_record)


def test_optional_memory_can_use_target_but_mandatory_mediator_contract_is_preserved():
    draft = apply(
        variables=[variable("y")],
        equations=[equation("y", "u")],
        mechanism_realizations=[record("y", memory_states=("y",))],
    )
    assert assess(draft)["eligible"]
    assert rows(draft, public=brief(True))["status"] == "incomplete"
    assert "memory_type" in {
        error["code"] for error in assess(draft, public=brief(True))["errors"]
    }
    draft = apply(draft, mechanism_realizations=[record("y")])
    assert rows(draft, public=brief(True))["status"] == "incomplete"
    assert "required_memory_annotations" in {
        error["code"] for error in assess(draft, public=brief(True))["errors"]
    }


def test_selected_memory_is_differential_and_exact_path_checks_still_apply():
    draft = apply(
        variables=[variable("y"), variable("x"), variable("z", "algebraic")],
        equations=[equation("y", "u"), equation("x", "u"), equation("z", "u")],
        mechanism_realizations=[record("x", memory_states=("x",))],
    )
    result = assess(draft)
    assert not result["eligible"]
    assert "memory_target_path" in {error["code"] for error in result["errors"]}
    feedback = next(
        error["realization_feedback"]
        for error in result["errors"]
        if error["code"] == "memory_target_path"
    )
    assert feedback["field"] == "mechanism_realizations"
    assert feedback["entry"]["components"] == ["x"]
    assert "memory_states=[]" in " ".join(feedback["repair_options"])
    draft = apply(draft, mechanism_realizations=[record("z", memory_states=("z",))])
    assert rows(draft)["status"] == "incomplete"
    assert rows(draft)["mechanism_realization_checks"][0][
        "nondifferential_memory_states"
    ] == ["z"]


def test_component_alone_does_not_prove_path_and_disconnected_component_is_reported():
    draft = apply(
        variables=[variable("y"), variable("x")],
        equations=[equation("y", "a"), equation("x", "u")],
        mechanism_realizations=[record("u", "y")],
    )
    result = assess(draft)
    assert result["mechanism_realization_checks"][0]["structural_status"] == "failed"
    draft = apply(
        draft,
        equations=[equation("y", "u")],
        mechanism_realizations=[record("y", "x")],
    )
    assert assess(draft)["mechanism_realization_checks"][0][
        "disconnected_components"
    ] == ["x"]


def test_supporting_covariate_branch_is_allowed_and_mixed_type_is_derived():
    draft = apply(
        variables=[variable("y"), variable("gain", "algebraic")],
        equations=[equation("y", "u", "gain"), equation("gain", "area")],
        mechanism_realizations=[record("y", "gain", "area")],
    )
    result = assess(draft)
    assert result["eligible"], result["errors"]
    row = result["mechanism_realization_checks"][0]
    assert row["inferred_type"] == "mixed"
    assert row["disconnected_components"] == []


def test_named_process_can_be_referenced_and_type_is_algebraic():
    draft = apply(
        variables=[variable("y")],
        processes=[
            process(
                depends_on=["u"],
                kind="influence",
                uses=[{"target": "y", "sign": "positive", "conversion": None}],
            )
        ],
        equations=[{"name": "y", "terms": []}],
        mechanism_realizations=[record("p")],
    )
    result = assess(draft)
    assert result["eligible"], result["errors"]
    assert result["mechanism_realization_checks"][0]["inferred_type"] == "algebraic"


def test_missing_graph_is_unavailable_and_missing_public_endpoints_are_unresolved():
    draft = apply(variables=[variable("y")], mechanism_realizations=[record("y")])
    result = assess(draft)
    assert (
        result["mechanism_realization_checks"][0]["structural_status"]
        == "unavailable_graph"
    )
    assert not any(e["code"] == "mechanism_realization_path" for e in result["errors"])
    draft = apply(draft, equations=[equation("y", "u")])
    public = optional_memory_brief(drivers=())
    result = assess(draft, public=public)
    assert result["eligible"]
    assert (
        result["mechanism_realization_checks"][0]["structural_status"]
        == "unresolved_public_endpoints"
    )
    assert result["mechanism_realization_checks"][0]["behavior_status"] == "unverified"


def test_feedback_self_path_cannot_be_faked_by_listing_the_target():
    public = optional_memory_brief(drivers=("y",))
    draft = apply(
        public=public,
        variables=[variable("y")],
        equations=[equation("y", "u")],
        mechanism_realizations=[record("y")],
    )
    result = assess(draft, public=public)
    assert not result["eligible"]
    assert (
        result["mechanism_realization_checks"][0]["path_checks"][0]["passed"] is False
    )
    draft = apply(draft, public=public, equations=[equation("y", "u", "y")])
    assert assess(draft, public=public)["eligible"]


def test_public_graph_predicates_are_preserved_alongside_realizations():
    draft = apply(
        variables=[variable("y")],
        equations=[equation("y", "u")],
        mechanism_realizations=[record("y")],
    )
    graph = contract("target_feedback")
    result = assess(draft, graph=graph)
    legacy = ledger.assess(
        optional_memory_brief(),
        context(),
        {},
        realizations.project(draft),
        graph_contract=graph,
    )
    assert (
        result["reviewed_public_graph_checks"] == legacy["reviewed_public_graph_checks"]
    )
    assert not result["eligible"]
    assert result["reviewed_public_graph_checks"][0]["passed"] is False


@pytest.mark.parametrize(
    "edits",
    [
        {"remove_realizations": ["unknown"]},
        {"remove_realizations": ["memory", "memory"]},
        {"remove_realizations": ["memory"], "mechanism_realizations": [record("y")]},
        {"mechanism_realizations": [record("y"), record("y")]},
        {"mechanism_bindings": [{"requirement_id": "memory", "memory_states": ["y"]}]},
        {"remove_bindings": ["memory"]},
    ],
)
def test_invalid_transaction_never_partly_mutates_existing_draft(edits):
    draft = apply(variables=[variable("y")], mechanism_realizations=[record("y")])
    before = draft.model_dump(mode="json")
    with pytest.raises(ValueError):
        apply(draft, variables=[variable("new_state")], **edits)
    assert draft.model_dump(mode="json") == before


def test_withdrawing_optional_memory_keeps_the_general_record():
    draft = apply(
        variables=[variable("y")],
        mechanism_realizations=[record("y", memory_states=("y",))],
    )
    retained = apply(draft, mechanism_realizations=[], stage_complete=True)
    assert retained == draft
    changed = apply(draft, mechanism_realizations=[record("y")])
    assert changed.mechanism_bindings == ()
    assert changed.mechanism_realizations[0].components == ("y",)
    effects = realizations.edit_effects(draft, changed)
    assert "mechanism_bindings" not in effects
    assert effects["mechanism_realizations"][0]["after"]["memory_states"] == []
    assert not apply(changed, remove_realizations=["memory"]).mechanism_realizations


def test_schema_version_loading_and_preview_projection_do_not_change_legacy_hashes():
    legacy = ledger.Draft(
        variables=(ledger.GeneratedVariable(**variable("y")),),
    )
    before = content_hash(legacy.model_dump(mode="json"))
    assert isinstance(realizations.load_draft(legacy.model_dump()), ledger.Draft)
    preview = realizations.from_legacy(legacy)
    assert preview.mechanism_realizations == ()
    assert rows(preview)["status"] == "incomplete"
    assert isinstance(realizations.load_draft(preview.model_dump()), realizations.Draft)
    assert "mechanism_bindings" not in preview.model_dump()
    assert "mechanism_realizations" not in legacy.model_dump()
    assert content_hash(legacy.model_dump(mode="json")) == before
    with pytest.raises(ValueError, match="mixed"):
        realizations.load_draft({**legacy.model_dump(), **preview.model_dump()})


def test_normalization_preserves_realizations_and_records_quoted_null_only():
    raw = {
        "stage_complete": True,
        "mechanism_realizations": [record("p")],
        "processes": [
            process(uses=[{"target": "y", "sign": "positive", "conversion": "null"}])
        ],
    }
    original = deepcopy(raw)
    normalized, changes = realizations.normalize_reply(optional_memory_brief(), raw)
    assert raw == original
    assert normalized["mechanism_realizations"] == raw["mechanism_realizations"]
    assert normalized["processes"][0]["uses"][0]["conversion"] is None
    assert changes[0]["code"] == "quoted_null_conversion"


def test_stage_schema_and_final_handoff_keep_general_map_and_exact_topology_hash():
    for stage in ("variables", "shared_laws", "equations", "repair"):
        fields = realizations.response_model(stage).model_fields
        assert "mechanism_realizations" in fields
        assert "remove_realizations" in fields
        assert "mechanism_bindings" not in fields
        if stage == "variables":
            assert "processes" not in fields and "equations" not in fields
    draft = apply(
        variables=[variable("y")],
        equations=[equation("y", "u")],
        mechanism_realizations=[record("y")],
    )
    result = assess(draft)
    handoff = realizations.final_handoff(draft, result)
    assert handoff["topology_hash"] == content_hash(result["topology"])
    assert handoff["mechanism_realizations"] == [record("y")]
    assert handoff["behavior_status"] == "unverified"
    assert handoff["structurally_eligible"] is True


def test_several_mechanisms_can_reuse_components_without_extra_states():
    public = optional_memory_brief()
    public = public.model_copy(
        update={
            "requirements": (
                *public.requirements,
                public.requirements[0].model_copy(update={"id": "second"}),
            )
        }
    )
    draft = apply(
        public=public,
        variables=[variable("y")],
        equations=[equation("y", "u")],
        mechanism_realizations=[record("y")],
    )
    assert rows(draft, public=public)["blocking_items"] == [
        "mechanism_realization:second"
    ]
    draft = apply(
        draft,
        public=public,
        mechanism_realizations=[record("y", requirement_id="second")],
    )
    assert assess(draft, public=public)["eligible"]
    assert len(draft.variables) == 1


def test_consumer_and_repeated_definition_normalization_preserve_full_records():
    public = optional_memory_brief()
    draft = apply(
        variables=[variable("y"), variable("x")],
        processes=[process()],
        mechanism_realizations=[record("p", "x", memory_states=("x",))],
    )
    raw = {
        "stage_complete": True,
        "mechanism_realizations": [record("p", "x", memory_states=("x",))],
        "equations": [
            {
                "name": "p",
                "terms": [
                    {
                        "sources": ["x"],
                        "outer_weight_sign": "unrestricted",
                        "scientific_role": "shared release",
                    }
                ],
            }
        ],
    }
    normalized, changes = realizations.normalize_reply(public, raw, draft=draft)
    assert normalized["equations"] == []
    assert changes[0]["code"] == "repeated_process_definition"
    edit = realizations.DraftPatch.model_validate(
        {
            "stage_complete": True,
            "equations": [
                {
                    "name": "y",
                    "terms": [
                        {
                            "sources": ["p"],
                            "outer_weight_sign": "positive",
                            "scientific_role": "shared release",
                        }
                    ],
                }
            ],
        }
    )
    updated = realizations.apply_patch(public, draft, edit)
    normalized_draft, changes = realizations.normalize_consumers(updated, edit, draft)
    assert changes[0]["code"] == "repeated_process_use"
    assert normalized_draft.mechanism_realizations == draft.mechanism_realizations
    assert normalized_draft.equations[0].terms == ()


def test_legacy_preview_projects_only_existing_memory_and_does_not_fill_other_records():
    legacy = ledger.Draft.model_validate(
        {
            "variables": [variable("y"), variable("x")],
            "mechanism_bindings": [
                {"requirement_id": "memory", "memory_states": ["x"]}
            ],
        }
    )
    preview = realizations.from_legacy(legacy)
    assert preview.mechanism_realizations[0].model_dump(mode="json") == record(
        "x", memory_states=("x",)
    )
    assert realizations.project(preview) == legacy


def test_new_removal_control_is_never_accepted_as_a_generated_quantity():
    draft = apply(variables=[variable("y")], mechanism_realizations=[record("y")])
    before = draft.model_dump()
    with pytest.raises(ValueError, match="reply-control fields"):
        apply(draft, variables=[variable("remove_realizations"), variable("x")])
    assert draft.model_dump() == before


def test_realization_rows_reuse_existing_path_scope_without_new_public_gates():
    draft = apply(
        variables=[variable("y")],
        equations=[equation("y", "a")],
        mechanism_realizations=[record("y")],
    )
    with scientific_verification.scope(False):
        result = assess(draft)
        legacy = ledger.assess(
            optional_memory_brief(), context(), {}, realizations.project(draft)
        )
    assert result["eligible"] == legacy["eligible"]
    assert result["public_structure_checks"] == []
    assert result["mechanism_realization_checks"][0]["structural_status"] == "deferred"
    assert (
        result["mechanism_realization_checks"][0]["public_path_check_status"]
        == "not_requested"
    )


def test_missing_driver_endpoints_do_not_hide_failed_known_selected_memory_path():
    public = optional_memory_brief(drivers=())
    draft = apply(
        public=public,
        variables=[variable("y"), variable("x")],
        equations=[equation("x", "u"), equation("y", "u")],
        mechanism_realizations=[record("x", memory_states=("x",))],
    )
    result = assess(draft, public=public)
    assert result["mechanism_realization_checks"][0]["structural_status"] == "failed"
    assert "memory_target_path" in {error["code"] for error in result["errors"]}


def test_fixed_conversion_covariate_is_relevant_without_changing_public_paths():
    draft = apply(
        variables=[variable("y")],
        processes=[
            process(
                depends_on=["u"],
                kind="influence",
                uses=[{"target": "y", "sign": "positive", "conversion": "1/area"}],
            )
        ],
        equations=[{"name": "y", "terms": []}],
        mechanism_realizations=[record("p", "area")],
    )
    result = assess(draft)
    legacy = ledger.assess(
        optional_memory_brief(), context(), {}, realizations.project(draft)
    )
    assert result["eligible"]
    assert result["public_structure_checks"] == legacy["public_structure_checks"]
    assert result["topology"] == legacy["topology"]
    row = result["mechanism_realization_checks"][0]
    assert row["disconnected_components"] == []
    assert row["path_checks"][0]["components_on_path"] == ["p"]
    public = optional_memory_brief(drivers=("area",))
    assert not assess(draft, public=public)["eligible"]
