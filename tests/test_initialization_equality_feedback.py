"""Equality presentation uses current dependencies and qualified saved evidence."""

import json
from copy import deepcopy

import pytest

from autoformalism.research import initialization_equality_feedback as feedback
from autoformalism.search.construction_initialization_focus import (
    factor_compatibility_details,
)
from autoformalism.search.construction_initialized_repair import Draft
from autoformalism.search.topology_initialization import INITIAL_TIME_POLICY


def scenario(*, initial=True):
    """Provide a generic readout or two-state ODE with one qualified pair."""
    variables = [
        {"name": name, "definition": kind, "scientific_role": "quantity"}
        for name, kind in (
            ("x", "differential"),
            ("z", "differential"),
            ("y", "algebraic"),
        )
    ]
    equations = [
        {
            "name": name,
            "terms": [
                {
                    "sources": sources,
                    "outer_weight_sign": "positive",
                    "scientific_role": "contribution",
                }
            ],
        }
        for name, sources in (
            ("x", ["drive"] if initial else ["z", "drive"]),
            ("z", ["x", "z"]),
            ("y", ["u", "z"]),
        )
    ]
    draft = Draft.model_validate({"variables": variables, "equations": equations})
    rule = (
        {"mode": "shared_parameter", "arguments": []}
        if initial
        else {"mode": "common_map", "arguments": ["x"]}
    )
    initial_information = {
        "z": {
            "rule": rule["mode"],
            "recorded_arguments": ({} if initial else {"x": 3.0}),
        }
    }
    if not initial:
        initial_information["x"] = {
            "rule": "observed_initial",
            "recorded_arguments": {"x": 3.0},
        }
    conditioning = {
        "matched_initial_information": {
            name: deepcopy(initial_information) for name in ("train_a", "train_b")
        },
        (
            "matched_instantaneous_channels"
            if initial
            else "matched_complete_forcing_channels"
        ): ["u"] if initial else ["drive"],
    }
    if initial:
        conditioning["initial_time_policy"] = INITIAL_TIME_POLICY
    question = {
        "question_id": "q1",
        "target": "y" if initial else "x",
        "property": "closed_subsystem",
        "repair_status": "actionable_conflict",
        "conditioning": conditioning,
        "assumptions": ["Shared deterministic laws have unique solutions."],
        "initialization_qualification": {
            "family_available": True,
            "contract": {
                "complete": True,
                "scope": "recorded_t0",
                "shared_deterministic": True,
            },
        },
        "witnesses": [
            {
                "trajectories": ["train_a", "train_b"],
                "times": [0.0, 0.0] if initial else [8.0, 8.0],
                "sample_indices": [0, 0] if initial else [8, 8],
                "observations": [1.2, 1.9],
                "error_intervals_disjoint": True,
            }
        ],
    }
    packet = {
        "stage": "repair",
        "stage_instructions": "Repair justified conflicts.",
        "feedback": {"repair_work": {"actionable_conflicts": ["q1"]}},
        "initialization": {
            "fixed_runtime_rules": {"x": {"mode": "observed_initial", "arguments": []}},
            "retained_handoff": {"records": [{"state": "z", "rule": rule}]},
        },
        "compatibility_questions": [question],
        "public_brief": {
            "public_variables": [
                {"name": "x", "data_role": "target"},
                {"name": "y", "data_role": "target"},
                {"name": "u", "data_role": "auxiliary"},
                {"name": "drive", "data_role": "external_input"},
                {"name": "t", "data_role": "time"},
            ]
        },
    }
    return packet, draft


def test_initial_readout_stops_at_memory_and_preserves_complete_evidence():
    packet, draft = scenario()
    original, original_draft = deepcopy(packet), draft.model_dump(mode="json")
    result = feedback.project(packet, draft)
    assert list(result)[:2] == ["stage", "model_data_conflicts"]
    row = result.pop("model_data_conflicts")[0]
    assert row["comparison"] == "initial_readout"
    assert row["model_relation"] == "y(t0) = H_y(u(t0), z(t0))"
    assert row["initial_equalities"] == ["z(t0) = one shared constant"]
    assert "ODE RHS arguments alone" in row["equality_argument"]
    assert "drive" not in json.dumps(row)
    assert row["assumptions"] == packet["compatibility_questions"][0]["assumptions"]
    assert row["evidence_reference"] == "compatibility_questions[0].witnesses[0]"
    assert result == original == packet
    assert draft.model_dump(mode="json") == original_draft


def test_later_feedback_includes_coupled_states_and_actual_common_map():
    packet, draft = scenario(initial=False)
    row = feedback.project(packet, draft)["model_data_conflicts"][0]
    assert row["comparison"] == "later_subsystem"
    assert row["model_relation"] == "d(x, z)/dt = F(drive, x, z)"
    assert (
        "z(t0) = h_z(x(t0)), with equal recorded arguments"
        in (row["initial_equalities"])
    )
    assert "matched supplied histories/covariates drive" in row["equality_argument"]
    assert "same subsystem solution and x(8)" in row["equality_argument"]
    assert "not selected formulas" in row["qualification"]


def test_shared_process_is_expanded_for_initial_dependency_equality():
    packet, draft = scenario()
    raw = draft.model_dump(mode="json")
    raw["equations"][-1]["terms"] = []
    raw["processes"] = [
        {
            "name": "readout",
            "depends_on": ["u", "z"],
            "kind": "influence",
            "uses": [{"target": "y", "sign": "positive", "conversion": None}],
            "scientific_meaning": "Generated readout contribution",
        }
    ]
    row = feedback.project(packet, Draft.model_validate(raw))["model_data_conflicts"][0]
    assert row["model_relation"] == "y(t0) = H_y(u(t0), z(t0))"


@pytest.mark.parametrize(
    "change",
    [
        "inactive",
        "uncertain",
        "no_assumptions",
        "unavailable",
        "unknown_rule",
        "stale_arguments",
        "unequal_initial_arguments",
        "missing_forcing",
        "wrong_time",
        "bad_number",
        "noninitial_index",
        "incomplete_contract",
    ],
)
def test_incomplete_or_unqualified_equality_is_not_summarized(change):
    packet, draft = scenario(initial=change == "noninitial_index")
    question = packet["compatibility_questions"][0]
    witness = question["witnesses"][0]
    if change == "inactive":
        packet["feedback"]["repair_work"]["actionable_conflicts"] = []
    elif change == "uncertain":
        witness["error_intervals_disjoint"] = False
    elif change == "no_assumptions":
        question["assumptions"] = []
    elif change == "unavailable":
        question["unavailable"] = ["unique solutions not established"]
    elif change in {"unknown_rule", "stale_arguments"}:
        packet["initialization"]["retained_handoff"]["records"][0]["rule"] = {
            "mode": "unknown" if change == "unknown_rule" else "common_map",
            "arguments": [] if change == "unknown_rule" else ["u"],
        }
    elif change == "unequal_initial_arguments":
        question["conditioning"]["matched_initial_information"]["train_b"]["z"][
            "recorded_arguments"
        ]["x"] = 9.0
    elif change == "missing_forcing":
        question["conditioning"]["matched_complete_forcing_channels"] = []
    elif change == "wrong_time":
        witness["times"][1] = 9.0
    elif change == "bad_number":
        witness["observations"][0] = "1.2"
    elif change == "noninitial_index":
        witness["sample_indices"] = [1, 1]
    else:
        question["initialization_qualification"]["contract"]["complete"] = False
    assert feedback.project(packet, draft) == packet


def test_new_rhs_source_requires_new_coverage_before_later_summary():
    packet, draft = scenario(initial=False)
    raw = draft.model_dump(mode="json")
    raw["equations"][0]["terms"][0]["sources"].append("u")
    assert feedback.project(packet, Draft.model_validate(raw)) == packet


def test_missing_topology_does_not_receive_a_conditional_proof():
    packet, draft = scenario()
    raw = draft.model_dump(mode="json")
    raw["equations"] = raw["equations"][:-1]
    assert feedback.project(packet, Draft.model_validate(raw)) == packet


def test_fixed_conversion_dependencies_must_be_covered():
    packet, draft = scenario()
    packet["public_brief"]["public_variables"].append(
        {"name": "scale", "data_role": "covariate"}
    )
    raw = draft.model_dump(mode="json")
    raw["processes"] = [
        {
            "name": "P",
            "depends_on": ["z"],
            "kind": "influence",
            "uses": [{"target": "y", "sign": "positive", "conversion": "scale"}],
            "scientific_meaning": "Readout contribution with fixed conversion",
        }
    ]
    assert feedback.project(packet, Draft.model_validate(raw)) == packet


def test_factored_evidence_is_kept_and_only_two_arguments_are_shown():
    packet, draft = scenario()
    original = packet["compatibility_questions"][0]
    original["verified"] = ["Verified conditioning. " * 100]
    packet["compatibility_questions"] = [
        {**deepcopy(original), "question_id": f"q{index}"} for index in range(4)
    ]
    packet["feedback"]["repair_work"]["actionable_conflicts"] = [
        f"q{index}" for index in range(4)
    ]
    compact = factor_compatibility_details(packet)
    assert "compatibility_details" in compact
    result = feedback.project(compact, draft)
    assert len(result.pop("model_data_conflicts")) == 2
    assert result == compact


def test_other_stages_and_no_conflicts_remain_unchanged():
    packet, draft = scenario()
    packet["stage"] = "initialization"
    assert feedback.project(packet, draft) == packet
    packet["stage"] = "repair"
    packet["compatibility_questions"] = []
    assert feedback.project(packet, draft) == packet


def test_repeated_projection_fails_instead_of_reusing_a_stale_summary():
    packet, draft = scenario()
    with pytest.raises(ValueError, match="already applied"):
        feedback.project(feedback.project(packet, draft), draft)
