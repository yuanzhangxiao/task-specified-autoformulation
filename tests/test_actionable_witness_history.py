"""Every intermediate matched pair survives repairs under the opt-in policy."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.research.actionable_repair_status import (
    ALL_SEEN_WITNESSES,
    build_state,
    retain_witness_history,
)
from autoformalism.research.initialization_handoff_audit import qualified_feedback
from tests.test_actionable_repair_status import report
from tests.test_topology_compatibility import contract, draft, evidence, trajectory
from tests.test_topology_initialization import choice, information


def question(identifier, index):
    """Use explicit sample coordinates and independently retained question IDs."""
    value = report(question_id=identifier)["advisory_questions"][0]
    value["witnesses"] = [
        {
            "trajectories": ["left", "right"],
            "sample_indices": [index, index],
            "times": [index, index],
            "observations": [1, 2],
            "error_intervals_disjoint": True,
        }
    ]
    return value


def with_question(value):
    result = report(question_id=value["question_id"])
    result["advisory_questions"] = [value]
    return result


def test_new_sample_coordinates_survive_same_target_property_and_question_id():
    first, second = question("same-question", 1), question("same-question", 2)
    initial = build_state(
        with_question(first), None, True, witness_policy=ALL_SEEN_WITNESSES
    )
    changed = build_state(
        with_question(second), initial, True, witness_policy=ALL_SEEN_WITNESSES
    )
    assert changed["tracked_conflicts"] == [
        {**first, "witnesses": first["witnesses"] + second["witnesses"]}
    ]
    assert changed["witness_history_policy"] == ALL_SEEN_WITNESSES
    assert build_state(
        with_question(second), changed, True, witness_policy=ALL_SEEN_WITNESSES
    ) == changed


def test_new_question_identity_survives_and_inputs_are_not_mutated():
    first, second = question("first", 1), question("later", 2)
    originals = deepcopy((first, second))
    retained = retain_witness_history([first], [second, second])
    assert retained == [first, second]
    assert (first, second) == originals
    retained[0]["witnesses"][0]["times"][0] = 99
    assert (first, second) == originals


def test_default_keeps_the_historical_first_question_policy():
    first, second = question("first", 1), question("later", 2)
    initial = build_state(with_question(first), None, True)
    changed = build_state(with_question(second), initial, True)
    assert changed["tracked_conflicts"] == [first]
    assert "witness_history_policy" not in changed


@pytest.mark.parametrize("bad", [[], None, "missing witnesses"])
def test_malformed_or_empty_history_is_not_silently_replaced(bad):
    invalid, valid = question("same", 1), question("same", 2)
    invalid["witnesses"] = bad
    assert retain_witness_history([invalid], [valid]) == [invalid, valid]


def test_unknown_witness_policy_cannot_change_a_checkpoint_silently():
    with pytest.raises(ValueError, match="unsupported actionable witness policy"):
        build_state(report(), None, True, witness_policy="unrecognized")


def test_later_scalar_restriction_is_not_erased_by_first_record_scope_change():
    first = report(property_="scalar_autonomous", scope=("a",))
    second = report(
        property_="scalar_autonomous", scope=("a", "u"), question_id="later"
    )
    state = build_state(first, None, True, witness_policy=ALL_SEEN_WITNESSES)
    state = build_state(second, state, True, witness_policy=ALL_SEEN_WITNESSES)
    current = report(
        property_="scalar_autonomous",
        scope=("a", "u"),
        conflict=False,
        status="deferred",
        comparisons=0,
        unavailable=["no qualifying observed comparison"],
    )
    final = build_state(current, state, True, witness_policy=ALL_SEEN_WITNESSES)
    assert not final["completion_allowed"]
    assert final["baseline_outcomes"][0]["status"] == "deferred_after_revision"


def three_preparations():
    """A later forcing change distinguishes whole horizons but not early samples."""
    context = ValidationContext(
        targets=("a",), external_inputs=("u", "v"), auxiliaries=("b", "c")
    )
    settings = (("A", 0, 0, 0), ("B", 1, 1, 1), ("C", 3, 1, 0))
    rows = []
    for index, (name, output, b, c) in enumerate(settings):
        row = trajectory(
            name, (0, output, output), aux={"b": (b, b, b), "c": (c, c, c)}
        )
        rows.append(
            replace(
                row,
                external_inputs={
                    **row.external_inputs,
                    "v": np.asarray((0, 0, index), float),
                },
            )
        )
    packet = evidence(
        *rows,
        context=context,
        config=contract(
            interpolation={"u": "linear_complete_knots", "v": "linear_complete_knots"},
            absolute_error_bounds={"a": 0, "u": 0, "v": 0, "b": 0, "c": 0},
        ),
    )
    model = draft(sources=("z", "u"), extras=[("z", "differential", ())])
    return model, packet, information(permitted_symbols=("a", "b", "c", "u", "v", "t"))


def test_intermediate_pair_remains_after_old_pair_is_distinguished():
    """The saved run exposed this pattern: a new pair reused an existing test key."""
    model, packet, contract_ = three_preparations()
    first = qualified_feedback(model, packet, contract_, choice(arguments=("b",)))
    initial = build_state(first, None, True, witness_policy=ALL_SEEN_WITNESSES)
    assert first["advisory_questions"][0]["witnesses"][0]["trajectories"] == ["B", "C"]

    # A changed initializer introduces a different strongest pair for the same
    # target/property. The old policy retains only the first question.
    second = qualified_feedback(
        model,
        packet,
        contract_,
        choice("shared_parameter", ()),
        previous_questions=initial["tracked_conflicts"],
    )
    assert second["advisory_questions"][0]["witnesses"][0]["trajectories"] == ["A", "C"]
    state = build_state(second, initial, True, witness_policy=ALL_SEEN_WITNESSES)
    legacy = build_state(second, build_state(first, None, True), True)
    assert len(state["tracked_conflicts"]) == 2
    assert len(legacy["tracked_conflicts"]) == 1

    # The new initial map distinguishes B/C, but A/C still match until t=1.
    # v differs only later and cannot account for that earlier mismatch.
    final_model = draft(sources=("z", "u", "v"), extras=[("z", "differential", ())])
    final_report = qualified_feedback(
        final_model,
        packet,
        contract_,
        choice(arguments=("c",)),
        previous_questions=state["tracked_conflicts"],
    )
    final = build_state(
        final_report, state, True, witness_policy=ALL_SEEN_WITNESSES
    )
    assert final["repair_status"] == "repair_required"
    checks = final_report["saved_witness_recheck"]["checks"]
    assert any(
        row["trajectories"] == ["A", "C"]
        and row["conditioning_end_time"] == 1
        and row["status"] == "preserved"
        for row in checks
    )
    legacy_report = qualified_feedback(
        final_model,
        packet,
        contract_,
        choice(arguments=("c",)),
        previous_questions=legacy["tracked_conflicts"],
    )
    assert build_state(legacy_report, legacy, True)["completion_allowed"]


def test_missing_one_intermediate_prefix_receipt_cannot_clear_the_history():
    from tests.test_initializer_prefix_resolution import initialized_pair

    before, after = initialized_pair()
    initial = build_state(before, None, True, witness_policy=ALL_SEEN_WITNESSES)
    changed = deepcopy(before)
    changed["advisory_questions"][0]["question_id"] = "intermediate"
    state = build_state(changed, initial, True, witness_policy=ALL_SEEN_WITNESSES)
    final = build_state(after, state, True, witness_policy=ALL_SEEN_WITNESSES)
    assert not final["completion_allowed"]
    assert final["repair_status"] == "review_deferred"
    receipt = deepcopy(after["saved_witness_recheck"]["checks"][0])
    receipt["previous_question_id"] = "intermediate"
    after["saved_witness_recheck"]["checks"].append(receipt)
    assert build_state(
        after, state, True, witness_policy=ALL_SEEN_WITNESSES
    )["completion_allowed"]
