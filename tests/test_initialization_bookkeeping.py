"""V6 bookkeeping removes inherited debris without changing scientific choices."""

from copy import deepcopy

import pytest

from autoformalism.research import initialization_bookkeeping as subject
from autoformalism.search import construction_initialized_repair as initialized
from autoformalism.search.topology_initialization import InitializerDependencyHandoff
from tests.test_construction_ledger import brief, equation, variable
from tests.test_initialized_topology_repair import prepared  # noqa: F401
from tests.test_saved_topology_repair import setup as saved_setup  # noqa: F401


def handoff(*states, mode="shared_parameter", arguments=()):
    return InitializerDependencyHandoff.model_validate(
        {
            "records": [
                {"state": state, "rule": {"mode": mode, "arguments": arguments}}
                for state in states
            ]
        }
    )


@pytest.fixture
def draft():
    return initialized.Draft.model_validate(
        {
            "variables": [variable("y"), variable("z"), variable("w")],
            "equations": [
                equation("y", "z"),
                equation("z", "u"),
                equation("w", "u"),
            ],
        }
    )


def assemble(draft, **kwargs):
    patch = initialized.DraftPatch.model_validate({"stage_complete": True, **kwargs})
    candidate = initialized.apply_patch(brief(), draft, patch)
    candidate, _ = initialized.normalize_consumers(candidate, patch, draft)
    return patch, candidate


@pytest.mark.parametrize("operation", ["remove", "retype"])
def test_cleanup_only_realized_explicit_state_changes(draft, operation):
    edits = (
        {"remove_variables": ["z"], "remove_equations": ["z"]}
        if operation == "remove"
        else {"variables": [variable("z", "algebraic")]}
    )
    patch, candidate = assemble(draft, **edits)
    retained = handoff("z", "w")
    originals = deepcopy((draft, candidate, retained, patch))
    result = subject.cleanup_inherited_handoff(draft, candidate, retained, patch)
    assert result == handoff("w")
    assert (draft, candidate, retained, patch) == originals


@pytest.mark.parametrize("replacement", [handoff(), handoff("z", "w")])
def test_explicit_handoff_is_untouched_even_when_it_is_stale(draft, replacement):
    patch, candidate = assemble(
        draft,
        variables=[variable("z", "algebraic")],
        initialization=replacement.model_dump(mode="json"),
    )
    result = subject.cleanup_inherited_handoff(
        draft, candidate, handoff("z", "w"), patch
    )
    assert result is patch.initialization
    assert result == replacement


def test_unrelated_stale_records_are_not_silently_repaired(draft):
    patch, candidate = assemble(draft, variables=[variable("z", "algebraic")])
    result = subject.cleanup_inherited_handoff(
        draft, candidate, handoff("z", "w", "already_missing"), patch
    )
    assert result == handoff("w", "already_missing")


def test_unchanged_final_type_prevents_cleanup(draft):
    patch, _ = assemble(draft, variables=[variable("z", "algebraic")])
    retained = handoff("z")
    assert subject.cleanup_inherited_handoff(draft, draft, retained, patch) == retained


def test_unrequested_type_change_and_differential_restatement_prevent_cleanup(draft):
    _, candidate = assemble(draft, variables=[variable("z", "algebraic")])
    retained = handoff("z")
    for changes in ({}, {"variables": [variable("z")]}):
        patch, _ = assemble(draft, **changes)
        assert subject.cleanup_inherited_handoff(
            draft, candidate, retained, patch
        ) == retained


def test_explicitly_touching_preexisting_algebraic_state_does_not_clean_it(draft):
    _, algebraic = assemble(draft, variables=[variable("z", "algebraic")])
    patch, candidate = assemble(algebraic, variables=[variable("z", "algebraic")])
    retained = handoff("z")
    assert subject.cleanup_inherited_handoff(
        algebraic, candidate, retained, patch
    ) == retained


def test_cleaned_inherited_handoff_validates_but_explicit_stale_choice_fails(
    prepared,  # noqa: F811 -- imported pytest fixture
):
    _, _, problem, draft, contract = prepared
    retained = handoff("z")
    patch, candidate = assemble(draft, variables=[variable("z", "algebraic")])
    cleaned = subject.cleanup_inherited_handoff(draft, candidate, retained, patch)
    assert initialized.handoff_status(
        candidate, problem.evidence, contract, cleaned
    )["complete"]
    explicit = patch.model_copy(update={"initialization": retained})
    untouched = subject.cleanup_inherited_handoff(draft, candidate, retained, explicit)
    with pytest.raises(ValueError, match="not a differential state"):
        initialized.handoff_status(candidate, problem.evidence, contract, untouched)
    assert retained == handoff("z")


def signature(
    draft, candidate, proposed, *, retained=None, error="invalid initializer"
):
    return subject.rejected_initializer_attempt_signature(
        draft,
        {"errors": [], "eligible": True},
        {"advisory_questions": [{"question_id": "unresolved"}]},
        candidate,
        retained=(retained or handoff("z", "w")).model_dump(mode="json"),
        proposed=proposed,
        error=error,
    )


def test_redundant_topology_restatements_do_not_reset_rejected_attempt(draft):
    bad = handoff("z", mode="observed_initial").model_dump(mode="json")
    _, first = assemble(draft, variables=[variable("z", "algebraic")])
    repeated_equation = equation("y", "z")
    repeated_equation["terms"][0]["scientific_role"] += "."
    _, second = assemble(
        draft,
        variables=[{**variable("z", "algebraic"), "scientific_role": "Auxiliary."}],
        equations=[repeated_equation],
    )
    assert first != second
    assert signature(draft, first, bad) == signature(draft, second, bad)


@pytest.mark.parametrize("change", ["dependency", "sign", "multiplicity", "kind"])
def test_different_attempted_topology_effects_remain_distinct(draft, change):
    proposed = handoff("z", mode="observed_initial").model_dump(mode="json")
    _, first = assemble(draft)
    edited = equation("y", "z")
    if change == "dependency":
        edited["terms"][0]["sources"] = ["w"]
    elif change == "sign":
        edited["terms"][0]["outer_weight_sign"] = "negative"
    elif change == "multiplicity":
        edited["terms"].append(deepcopy(edited["terms"][0]))
    changes = (
        {"variables": [variable("z", "algebraic")]}
        if change == "kind"
        else {"equations": [edited]}
    )
    _, second = assemble(draft, **changes)
    assert signature(draft, first, proposed) != signature(draft, second, proposed)


def test_rejected_initializer_choices_and_malformed_alternatives_remain_distinct(draft):
    _, candidate = assemble(draft)
    alternatives = [
        None,
        {},
        "malformed handoff",
        handoff("z", mode="observed_initial").model_dump(mode="json"),
        handoff("z", mode="common_map", arguments=("missing",)).model_dump(mode="json"),
    ]
    assert len({signature(draft, candidate, value) for value in alternatives}) == len(
        alternatives
    )


def test_initializer_record_order_collapses_but_argument_order_does_not(draft):
    _, candidate = assemble(draft)
    first = handoff("z", "w", mode="common_map", arguments=("u", "y"))
    reordered = handoff("w", "z", mode="common_map", arguments=("u", "y"))
    changed_arguments = handoff("z", "w", mode="common_map", arguments=("y", "u"))
    original = deepcopy((draft, candidate, first))
    assert signature(draft, candidate, first.model_dump(mode="json")) == signature(
        draft, candidate, reordered.model_dump(mode="json")
    )
    assert signature(draft, candidate, first.model_dump(mode="json")) != signature(
        draft, candidate, changed_arguments.model_dump(mode="json")
    )
    assert (draft, candidate, first) == original


def test_proposed_record_multiplicity_is_retained_without_mutation(draft):
    proposed = handoff("z", mode="observed_initial").model_dump(mode="json")
    duplicate = deepcopy(proposed)
    duplicate["records"].append(deepcopy(duplicate["records"][0]))
    originals = deepcopy((proposed, duplicate))
    assert signature(draft, draft, proposed) != signature(draft, draft, duplicate)
    assert (proposed, duplicate) == originals


def test_error_wording_and_retained_record_order_are_not_new_attempts(draft):
    proposed = handoff("z", mode="observed_initial").model_dump(mode="json")
    assert signature(draft, draft, proposed, retained=handoff("z", "w")) == signature(
        draft,
        draft,
        proposed,
        retained=handoff("w", "z"),
        error="same invalid initializer, different diagnostic wording",
    )


@pytest.mark.parametrize("error", ["", "no draft change; make an explicit edit"])
def test_non_rejection_cannot_use_rejected_initializer_fingerprint(draft, error):
    with pytest.raises(ValueError, match="rejected initializer validation error"):
        signature(draft, draft, None, error=error)
