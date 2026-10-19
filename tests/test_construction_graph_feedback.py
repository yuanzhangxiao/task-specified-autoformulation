"""Current conditional contradictions gate initializer repair; deferral is explicit."""

import copy

import pytest

from autoformalism.search.construction_graph_feedback import acceptance_feedback
from autoformalism.search.construction_initialization_graph import (
    GraphDependencyHandoff,
    build_contract,
)
from autoformalism.staged_topology import content_hash
from tests.test_construction_graph_evidence import graph_fixture, saved_question
from tests.test_topology_compatibility import draft, evidence, trajectory


def test_initial_conflict_blocks_and_deduplicates_three_reporting_surfaces():
    model, packet, runtime, handoff = graph_fixture()
    result = acceptance_feedback(model, packet, runtime, handoff, [saved_question()])
    assert result["blocked"] and not result["eligible"]
    assert result["status"] == "model_data_conflict"
    assert result["conflict_count"] == 1
    assert result["witness_count"] == 1
    card = result["conflicts"][0]
    assert set(card["sources"]) == {
        "declared_report",
        "initial_time_report",
        "retained_witness_report",
    }
    assert [p["observed_target"] for p in card["observations"]] == [1, 2]
    assert "Later ODE evolution cannot change" in card["conflict"]
    assert not result["scientific_repair_established"]


def test_repair_card_uses_current_graph_and_complete_permitted_initial_values():
    model, packet, runtime, handoff = graph_fixture()
    prior = saved_question()
    prior["model_relation"] = "stale model"
    prior["paired_recorded_values"] = {"u": [-999, -999]}
    before = copy.deepcopy(prior)
    result = acceptance_feedback(model, packet, runtime, handoff, [prior])
    assert prior == before
    card = result["conflicts"][0]
    assert card["model_relation"] == ["a = f(z2)"]
    assert any("z2(t0) = g_z2(z1(t0)" in s for s in card["initializer_relations"])
    assert any(
        s == "z0(t0) = one shared training constant"
        for s in card["initializer_relations"]
    )
    table = card["paired_permitted_initial_measurements"]
    assert table["trajectories"] == ["left", "right"]
    rows = {row["channel"]: row for row in table["rows"]}
    assert rows["a"]["values"] == [1, 2] and not rows["a"]["equal"]
    assert rows["u"]["values"] == [0, 0] and rows["u"]["equal"]
    assert rows["t"]["values"] == [0, 0]
    assert "stale model" not in str(card) and "-999" not in str(card)


@pytest.mark.parametrize("window", (True, False))
def test_trajectory_root_can_be_eligible_but_never_turns_deferral_into_repair(window):
    model, packet, runtime, handoff = graph_fixture(scope="trajectory", window=window)
    result = acceptance_feedback(model, packet, runtime, handoff, [saved_question()])
    assert not result["blocked"] and result["eligible"]
    assert result["status"] == "eligible_with_deferred_evidence"
    assert result["conflict_count"] == result["witness_count"] == 0
    assert result["deferred_count"] > 0
    assert any(
        row.get("prefix_inferred_roots") == ["z0"] for row in result["deferred_checks"]
    )
    assert not result["scientific_repair_established"]
    assert not result["assessment"]["scientific_pass"]


def test_unknown_conditioning_and_invalid_old_coordinates_remain_unresolved():
    model, packet, runtime, handoff = graph_fixture()
    raw = packet.model_dump(mode="json", exclude={"sha256"})
    raw["contract"]["absolute_error_bounds"]["u"] = None
    packet = type(packet).model_validate({**raw, "sha256": content_hash(raw)})
    result = acceptance_feedback(model, packet, runtime, handoff, [saved_question(100)])
    assert not result["blocked"]
    assert result["status"] == "eligible_with_deferred_evidence"
    assert any(
        row["status"] == "invalid_saved_witness" for row in result["deferred_checks"]
    )
    assert result["assessment"]["retained_witness_report"]["original_questions"]


def test_matching_forcing_is_rechecked_only_through_saved_conflicting_observation():
    model, _, runtime, handoff = graph_fixture()
    packet = evidence(
        trajectory("left", (0, 1, 2)),
        trajectory("right", (0, 2, 3), (0, 0, 1)),
    )
    result = acceptance_feedback(model, packet, runtime, handoff, [saved_question(1)])
    card = result["conflicts"][0]
    assert card["forcing_prefix"]["end"] == 1
    assert card["forcing_prefix"]["equal"]
    assert [p["observed_target"] for p in card["observations"]] == [1, 2]
    assert "unique solutions" in " ".join(card["logical_assumptions"])


@pytest.mark.parametrize(
    "kind,property_",
    (("differential", "scalar_autonomous"), ("algebraic", "algebraic_map")),
)
def test_independent_topology_contradictions_are_not_hidden_by_prefix_root(
    kind, property_
):
    model = draft(kind, ("u",), extras=[("z", "differential", ())])
    packet = evidence(trajectory("single", (0, 1, 0)))
    runtime = build_contract(
        model,
        packet.context,
        permitted_public_initials=("a", "u", "t"),
        observed_initial_channels={"a": "a"} if kind == "differential" else {},
        trajectory_initialization="observation_prefix",
        provenance="Synthetic test.",
    )
    handoff = GraphDependencyHandoff.model_validate(
        {
            "records": [
                {
                    "state": "z",
                    "rule": {"mode": "value", "scope": "trajectory"},
                    "justification": "Independent preparation.",
                }
            ]
        }
    )
    result = acceptance_feedback(model, packet, runtime, handoff)
    assert result["blocked"]
    assert any(row["property"] == property_ for row in result["conflicts"])
    if kind == "algebraic":
        card = next(row for row in result["conflicts"] if row["property"] == property_)
        assert card["matched_algebraic_arguments"] == [
            {"channel": "u", "values": [0, 0]}
        ]


def test_a_changed_permitted_argument_clears_only_that_conditional_equality():
    model, packet, runtime, handoff = graph_fixture(different_input=True)
    result = acceptance_feedback(model, packet, runtime, handoff, [saved_question()])
    assert not result["blocked"] and result["eligible"]
    checks = result["assessment"]["retained_witness_report"]["checks"]
    assert checks[0]["status"] == "conditioning_distinction"
    assert not result["scientific_repair_established"]


def test_stale_or_heldout_source_is_rejected_before_feedback():
    model, packet, runtime, handoff = graph_fixture()
    stale = packet.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="digest"):
        acceptance_feedback(model, stale, runtime, handoff)
    heldout = packet.model_copy(update={"split": "test"})
    with pytest.raises(ValueError):
        acceptance_feedback(model, heldout, runtime, handoff)


def test_old_scalar_card_is_not_reused_as_a_current_contradiction():
    model, packet, runtime, handoff = graph_fixture(different_input=True)
    question = {**saved_question(), "property": "scalar_autonomous"}
    result = acceptance_feedback(model, packet, runtime, handoff, [question])
    assert not result["blocked"]
    assert not result["conflicts"]
    assert any(
        row["status"] == "retained_unresolved" for row in result["deferred_checks"]
    )
