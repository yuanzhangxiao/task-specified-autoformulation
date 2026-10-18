"""V5 adds bounded recorded values and concrete rejected-initializer guidance."""

from copy import deepcopy

import pytest

from autoformalism.research import initialization_equality_feedback as equality
from autoformalism.research import initialization_evidence_feedback as feedback
from autoformalism.search.construction_initialization_focus import (
    factor_compatibility_details,
)
from autoformalism.search.construction_initialized_repair import Draft
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from autoformalism.staged_topology import content_hash
from tests.test_initialization_equality_feedback import scenario


def sealed(raw):
    """Seal a fully materialized synthetic train fixture."""
    evidence = TrainingCompatibilityEvidence.model_validate(raw)
    return evidence.model_copy(
        update={
            "sha256": content_hash(evidence.model_dump(mode="json", exclude={"sha256"}))
        }
    )


def fixture(*, initial=True):
    packet, draft = scenario(initial=initial)
    trajectories = []
    for name, observed in (("train_a", 1.2), ("train_b", 1.9)):
        trajectories.append(
            {
                "trajectory_id": name,
                "time": list(range(9)),
                "targets": {
                    "x": [3.0] * 8 + [observed],
                    "y": [observed] * 9,
                },
                "auxiliaries": {"u": [7.0] * 9},
                "inputs": {"drive": list(range(9))},
                "covariates": {},
            }
        )
    evidence = sealed(
        {
            "context": {
                "targets": ["x", "y"],
                "auxiliaries": ["u"],
                "external_inputs": ["drive"],
            },
            "contract": {
                "provenance": "Synthetic complete public train evidence.",
                "absolute_error_bounds": dict.fromkeys(("x", "y", "u", "drive"), 0),
                "interpolation": {
                    "u": "linear_complete_knots",
                    "drive": "linear_complete_knots",
                },
                "initial_rules": {"x": {"mode": "observed_initial"}},
            },
            "trajectories": trajectories,
            "sha256": "0" * 64,
        }
    )
    packet["initialization"].update(
        {
            "original_evidence_sha256": evidence.sha256,
            "runtime_initial_information": {
                "scope": "recorded_t0",
                "permitted_symbols": ["x", "y", "u", "drive", "t"],
            },
            "editable_states": ["z"],
        }
    )
    return equality.project(packet, draft), draft, evidence


def values(result):
    return result["model_data_conflicts"][0]["paired_recorded_values"]


def test_initial_outputs_and_supplied_arguments_are_distinct_and_exact():
    packet, draft, evidence = fixture()
    original = deepcopy(packet)
    before = draft.model_dump(mode="json")
    result = feedback.project(packet, draft, evidence)
    numbers = values(result)
    assert numbers["evidence_sha256"] == evidence.sha256
    assert numbers["observed_output"] == [
        {"trajectory": "train_a", "channel": "y", "time": 0, "value": 1.2},
        {"trajectory": "train_b", "channel": "y", "time": 0, "value": 1.9},
    ]
    row = numbers["conditioning_values"][0]
    assert row["role"] == "instantaneous_supplied_argument"
    assert row["readings"] == [
        {"trajectory": "train_a", "channel": "u", "time": 0, "value": 7},
        {"trajectory": "train_b", "channel": "u", "time": 0, "value": 7},
    ]
    assert "supplied_histories" not in numbers
    result["model_data_conflicts"][0].pop("paired_recorded_values")
    assert result == packet == original
    assert draft.model_dump(mode="json") == before


def test_later_common_map_arguments_and_nonconstant_complete_forcing_are_labeled():
    packet, draft, evidence = fixture(initial=False)
    numbers = values(feedback.project(packet, draft, evidence))
    initials = numbers["conditioning_values"]
    assert [row["state"] for row in initials] == ["x", "z"]
    assert initials[1]["retained_rule"] == {"mode": "common_map", "arguments": ["x"]}
    assert [row["value"] for row in initials[1]["readings"]] == [3, 3]
    assert [row["time"] for row in initials[1]["readings"]] == [0, 0]
    assert [row["time"] for row in numbers["observed_output"]] == [8, 8]
    forcing = numbers["supplied_histories"][0]
    assert forcing["channel"] == "drive"
    assert forcing["interval"] == [0, 8]
    assert forcing["complete_histories_match"] is True
    assert forcing["trajectories"][0]["knots"] == [
        {"time": i, "value": i} for i in (0, 1, 2, 8)
    ]
    assert forcing["trajectories"][0]["omitted_knot_count"] == 5
    assert "constant" not in str(forcing)
    assert "not the resulting state value" in numbers["interpretation"]


@pytest.mark.parametrize(
    "change",
    [
        "unknown_id",
        "missing_witness",
        "missing_time",
        "missing_value",
        "time",
        "index",
        "output",
        "reference",
        "source_hash",
        "seal",
        "split",
        "missing_channel",
        "initial_argument",
        "forcing",
        "interpolation",
    ],
)
def test_unsupported_numeric_evidence_is_omitted_without_changing_status(change):
    packet, draft, evidence = fixture(initial=False)
    witness = packet["compatibility_questions"][0]["witnesses"][0]
    raw = evidence.model_dump(mode="json")
    if change == "unknown_id":
        witness["trajectories"][0] = "unknown"
    elif change == "missing_witness":
        packet["compatibility_questions"][0]["witnesses"] = []
    elif change == "missing_time":
        witness.pop("times")
    elif change == "missing_value":
        witness.pop("observations")
    elif change == "time":
        witness["times"] = [7, 7]
    elif change == "index":
        witness["sample_indices"] = [99, 99]
    elif change == "output":
        witness["observations"][0] = -50
    elif change == "reference":
        packet["model_data_conflicts"][0]["evidence_reference"] = "anything_else"
    elif change == "source_hash":
        packet["initialization"]["original_evidence_sha256"] = "0" * 64
    elif change == "seal":
        evidence = evidence.model_copy(update={"sha256": "0" * 64})
    elif change == "split":
        raw["split"] = "test"
        evidence = raw
    elif change == "missing_channel":
        raw["trajectories"][0]["inputs"] = {}
        evidence = sealed(raw)
    elif change == "initial_argument":
        raw["trajectories"][0]["targets"]["x"][0] = 9
        evidence = sealed(raw)
    elif change == "forcing":
        # Hidden from the four displayed knots, but checked against the full history.
        raw["trajectories"][1]["inputs"]["drive"][5] = 99
        evidence = sealed(raw)
    else:
        raw["contract"]["interpolation"]["drive"] = "unknown"
        evidence = sealed(raw)
    if change in {"missing_channel", "initial_argument", "forcing", "interpolation"}:
        packet["initialization"]["original_evidence_sha256"] = evidence.sha256
    assert feedback.project(packet, draft, evidence) == packet


def test_dict_evidence_and_factored_questions_work_without_expanding_packet():
    packet, draft, evidence = fixture()
    packet["compatibility_questions"][0]["verified"] = ["Long saved evidence. " * 100]
    packet["compatibility_questions"].append(
        deepcopy(packet["compatibility_questions"][0])
    )
    packet = factor_compatibility_details(packet)
    result = feedback.project(packet, draft, evidence.model_dump(mode="json"))
    assert "compatibility_details" in result
    assert values(result)["conditioning_values"]
    result["model_data_conflicts"][0].pop("paired_recorded_values")
    assert result == packet


def reject(packet, states):
    packet["previous_edit"] = {"status": "rejected", "error": "opaque runtime error"}
    packet["feedback"]["repeated_repair_feedback"] = {
        "observations": {
            "previous_reply": {
                "initialization": {
                    "records": [
                        {
                            "state": state,
                            "rule": {"mode": "observed_initial", "arguments": []},
                        }
                        for state in states
                    ]
                }
            }
        }
    }


def test_rejected_observed_initial_guidance_names_each_state_and_retained_rule():
    packet, draft, evidence = fixture()
    raw = draft.model_dump(mode="json")
    raw["variables"].append(
        {"name": "w", "definition": "differential", "scientific_role": "memory"}
    )
    draft = Draft.model_validate(raw)
    packet["initialization"]["editable_states"].append("w")
    packet["initialization"]["retained_handoff"]["records"].append(
        {"state": "w", "rule": {"mode": "common_map", "arguments": ["u"]}}
    )
    reject(packet, ["z", "w", "x", "undeclared", "z"])
    result = feedback.project(packet, draft, evidence)
    rows = result["feedback"]["unsupported_observed_initial"]
    assert [row["state"] for row in rows] == ["z", "w"]
    assert rows[0]["retained_rule"] == {"mode": "shared_parameter", "arguments": []}
    assert rows[1]["retained_rule"] == {"mode": "common_map", "arguments": ["u"]}
    assert rows[0]["available_initial_public_quantities"] == [
        "drive",
        "t",
        "u",
        "x",
        "y",
    ]
    assert set(rows[0]["legal_editable_choices"]) == {"shared_parameter", "common_map"}
    assert result["previous_edit"] == packet["previous_edit"]
    assert (
        result["feedback"]["repeated_repair_feedback"]
        == packet["feedback"]["repeated_repair_feedback"]
    )


@pytest.mark.parametrize("mode", ["observed_initial", "common_map", "shared_parameter"])
def test_fixed_runtime_binding_is_not_flagged_as_missing(mode):
    packet, draft, evidence = fixture()
    packet["initialization"]["fixed_runtime_rules"]["z"] = {
        "mode": mode,
        "arguments": ["u"] if mode == "common_map" else [],
    }
    reject(packet, ["z"])
    assert (
        "unsupported_observed_initial"
        not in feedback.project(packet, draft, evidence)["feedback"]
    )


@pytest.mark.parametrize("change", ["accepted", "prose_only", "no_reply", "no_runtime"])
def test_guidance_requires_structured_rejected_edit_and_runtime_information(change):
    packet, draft, evidence = fixture()
    reject(packet, ["z"])
    if change == "accepted":
        packet["previous_edit"]["status"] = "accepted"
    elif change == "prose_only":
        packet["feedback"]["repeated_repair_feedback"]["observations"][
            "previous_reply"
        ] = "use observed_initial for z"
    elif change == "no_reply":
        packet["feedback"].pop("repeated_repair_feedback")
    else:
        packet["initialization"].pop("runtime_initial_information")
    assert (
        "unsupported_observed_initial"
        not in feedback.project(packet, draft, evidence)["feedback"]
    )


def test_no_conflict_or_missing_evidence_stays_unchanged():
    packet, draft, evidence = fixture()
    packet.pop("model_data_conflicts")
    assert feedback.project(packet, draft, evidence) == packet
    packet, draft, _ = fixture()
    assert feedback.project(packet, draft, {}) == packet
