"""Consistent runtime duplicates are a display-only opt-in, not a rule repair."""

from copy import deepcopy

import pytest

from autoformalism.research import initialization_equality_feedback as equality
from autoformalism.research import initialization_evidence_feedback as numeric
from tests.test_initialization_equality_feedback import scenario
from tests.test_initialization_evidence_feedback import fixture


def duplicate_runtime(packet: dict, state: str) -> None:
    """Repeat an exact fixed/retained rule in the other namespace."""
    initialization = packet["initialization"]
    fixed = initialization["fixed_runtime_rules"]
    records = initialization["retained_handoff"]["records"]
    if state in fixed:
        records.append({"state": state, "rule": deepcopy(fixed[state])})
    else:
        fixed[state] = deepcopy(
            next(row["rule"] for row in records if row["state"] == state)
        )


@pytest.mark.parametrize("initial", [True, False])
@pytest.mark.parametrize("state", ["x", "z"])
def test_identical_runtime_restatement_restores_the_same_equality_card(initial, state):
    packet, draft = scenario(initial=initial)
    expected_card = equality.project(packet, draft)["model_data_conflicts"]
    duplicate_runtime(packet, state)
    original = deepcopy(packet)
    draft_before = draft.model_dump(mode="json")

    # V4/V5 callers retain their exact historical omission behavior.
    assert equality.project(packet, draft) == original
    assert equality.project(packet, draft, consistent_runtime_rules=False) == original

    result = equality.project(packet, draft, consistent_runtime_rules=True)
    assert result.pop("model_data_conflicts") == expected_card
    assert result == packet == original
    assert draft.model_dump(mode="json") == draft_before


@pytest.mark.parametrize("initial", [True, False])
@pytest.mark.parametrize("state", ["x", "z"])
def test_identical_runtime_restatement_restores_numeric_values_only_when_opted_in(
    initial, state
):
    packet, draft, evidence = fixture(initial=initial)
    expected = numeric.project(packet, draft, evidence)["model_data_conflicts"]
    duplicate_runtime(packet, state)
    original = deepcopy(packet)

    # The numeric adapter independently needs the opt-in after equality projection.
    assert numeric.project(packet, draft, evidence) == original
    assert (
        numeric.project(packet, draft, evidence, consistent_runtime_rules=False)
        == original
    )
    result = numeric.project(packet, draft, evidence, consistent_runtime_rules=True)
    assert result["model_data_conflicts"] == expected
    result["model_data_conflicts"][0].pop("paired_recorded_values")
    assert result == packet == original


@pytest.mark.parametrize(
    "conflicting_rule",
    [
        {"mode": "shared_parameter", "arguments": []},
        {"mode": "observed_initial", "arguments": ["u"]},
        {"mode": "common_map", "arguments": ["u"]},
    ],
)
def test_conflicting_runtime_duplicates_fail_closed_in_both_projections(
    conflicting_rule,
):
    packet, draft, evidence = fixture(initial=False)
    duplicate_runtime(packet, "x")
    packet["initialization"]["retained_handoff"]["records"][-1]["rule"] = (
        conflicting_rule
    )
    original = deepcopy(packet)
    with pytest.raises(ValueError, match="duplicate current initialization rule"):
        equality._rules(packet["initialization"], consistent_runtime_rules=True)
    assert (
        numeric.project(packet, draft, evidence, consistent_runtime_rules=True)
        == original
    )

    packet.pop("model_data_conflicts")
    assert equality.project(packet, draft, consistent_runtime_rules=True) == packet


def test_proposer_only_duplicate_remains_unsupported_even_when_identical():
    packet, draft = scenario()
    records = packet["initialization"]["retained_handoff"]["records"]
    records.append(deepcopy(records[0]))
    assert equality.project(packet, draft, consistent_runtime_rules=True) == packet


def test_two_identical_retained_records_do_not_merge_with_a_fixed_runtime_rule():
    packet, draft, evidence = fixture(initial=False)
    duplicate_runtime(packet, "x")
    records = packet["initialization"]["retained_handoff"]["records"]
    records.append(deepcopy(records[-1]))
    with pytest.raises(ValueError, match="duplicate current initialization rule"):
        equality._rules(packet["initialization"], consistent_runtime_rules=True)
    assert (
        numeric.project(packet, draft, evidence, consistent_runtime_rules=True)
        == packet
    )
    packet.pop("model_data_conflicts")
    assert equality.project(packet, draft, consistent_runtime_rules=True) == packet


@pytest.mark.parametrize("initial", [True, False])
def test_opt_in_without_duplicates_leaves_existing_outputs_exactly_unchanged(initial):
    packet, draft, evidence = fixture(initial=initial)
    assert numeric.project(packet, draft, evidence, consistent_runtime_rules=True) == (
        numeric.project(packet, draft, evidence)
    )
    packet.pop("model_data_conflicts")
    assert equality.project(packet, draft, consistent_runtime_rules=True) == (
        equality.project(packet, draft)
    )
