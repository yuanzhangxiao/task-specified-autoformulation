"""Initializer progress follows committed meaning, not equivalent reply syntax."""

from copy import deepcopy

from autoformalism.research import repair_stagnation as subject


def handoff(*states, mode="shared_parameter", arguments=()):
    return {
        "protocol": "topology-initializer-dependencies-1",
        "records": [
            {"state": state, "rule": {"mode": mode, "arguments": list(arguments)}}
            for state in states
        ],
    }


def signature(retained, proposed, error=None):
    return subject.initializer_attempt_signature(
        "unchanged-scientific-model", retained, proposed, error=error
    )


def test_accepted_null_restatement_and_record_reordering_are_one_attempt():
    retained = handoff("z", "w")
    original = deepcopy(retained)
    assert signature(retained, None) == signature(retained, retained)
    assert signature(retained, retained) == signature(handoff("w", "z"), retained)
    assert signature(retained, None, "no draft change; make an explicit edit") == (
        signature(retained, retained)
    )
    assert retained == original


def test_accepted_mathematical_initializer_change_changes_identity():
    old = handoff("z")
    new = handoff("z", mode="common_map", arguments=("y",))
    assert signature(old, None) != signature(new, new)
    assert signature(old, None) != signature(handoff("w"), None)


def test_distinct_rejected_alternatives_are_retained():
    retained = handoff("z")
    first = handoff("z", mode="observed_initial")
    second = handoff("z", mode="common_map", arguments=("missing",))
    error = "invalid initializer"
    assert signature(retained, first, error) != signature(retained, second, error)
    assert signature(retained, first, error) != signature(retained, None, error)
    assert signature(retained, retained, error) == signature(retained, None, error)


def test_rejected_alternative_record_order_is_not_a_new_attempt():
    retained = handoff("z", "w")
    first = handoff("z", "w", mode="observed_initial")
    second = handoff("w", "z", mode="observed_initial")
    assert signature(retained, first, "invalid") == signature(
        retained, second, "invalid"
    )


def test_malformed_alternatives_and_multiplicity_are_not_erased():
    retained = handoff("z")
    assert signature(retained, {}, "invalid") != signature(retained, None, "invalid")
    assert signature(retained, "invalid text", "invalid") != signature(
        retained, {}, "invalid"
    )
    assert signature(retained, handoff("z", "z"), "invalid") != signature(
        retained, retained, "invalid"
    )


def test_argument_order_is_preserved_and_input_is_not_mutated():
    first = handoff("z", mode="common_map", arguments=("y", "u"))
    second = handoff("z", mode="common_map", arguments=("u", "y"))
    original = deepcopy(first)
    assert subject.initializer_semantics(first) != subject.initializer_semantics(second)
    assert first == original
