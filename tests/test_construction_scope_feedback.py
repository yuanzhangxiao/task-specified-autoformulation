"""Scope-aware repair preserves contradictions while locating admissible edits."""

import copy
import json

import pytest

from autoformalism.search.construction_graph_repair import (
    construct_graph_dependencies,
    dependency_construction_identity,
    dependency_request,
)
from autoformalism.search.construction_initialization_graph import (
    GraphDependencyHandoff,
    build_contract,
)
from autoformalism.staged_topology import content_hash
from tests.test_construction_graph_evidence import graph_fixture, saved_question
from tests.test_construction_graph_runner import setup

POLICY = "scope-aware-2"


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(
        "autoformalism.rebuttal.staged_topology_campaign.runtime_source_hash",
        lambda: "scope_feedback_test_source",
    )


def inputs():
    """A readout DAG plus unrelated, observed and prepared differential states."""
    model, packet, _, trajectory = graph_fixture(scope="trajectory", window=False)
    raw = model.model_dump(mode="json")
    for name in ("unrelated", "observed", "prepared"):
        raw["variables"].append(
            {
                "name": name,
                "definition": "differential",
                "scientific_role": "Synthetic separate coordinate",
            }
        )
        raw["equations"].append({"name": name, "terms": []})
    model = type(model).model_validate(raw)
    contract = build_contract(
        model,
        packet.context,
        permitted_public_initials=("a", "u", "t"),
        observed_initial_channels={"observed": "a"},
        known_initials={
            "prepared": {"value": 0, "justification": "Initially empty vessel"}
        },
        trajectory_initialization="observation_prefix",
        provenance="Synthetic scope boundaries; no benchmark state identities.",
    )
    raw = trajectory.model_dump(mode="json")
    raw["records"] += [
        {
            "state": "unrelated",
            "rule": {"mode": "value", "scope": "shared_training"},
            "justification": "Unrelated initial coordinate",
        },
        {
            "state": "prepared",
            "rule": {"mode": "known"},
            "justification": "Fixed runtime preparation",
        },
    ]
    trajectory = GraphDependencyHandoff.model_validate(raw)
    raw["records"][0]["rule"]["scope"] = "shared_training"
    shared = GraphDependencyHandoff.model_validate(raw)
    return model, packet, contract, shared, trajectory


def reply(handoff):
    return {
        "decision": "revise",
        "records": handoff.model_dump(mode="json")["records"],
        "explanation": "A synthetic initializer choice.",
    }


def run(tmp_path, client, *, feedback_policy=POLICY):
    model, packet, contract, shared, _ = inputs()
    return construct_graph_dependencies(
        model,
        packet.context,
        "",
        contract,
        client,
        tmp_path / "result",
        evidence=packet,
        initial_handoff=shared,
        previous_questions=(saved_question(),),
        feedback_policy=feedback_policy,
    )


def append_record(payload, name, rule=None):
    payload["records"].append(
        {
            "state": name,
            "rule": rule or {"mode": "value", "scope": "shared_training"},
            "justification": "An intentionally unsupported initializer target.",
        }
    )
    return payload


def test_invalid_algebraic_removal_cannot_accept_original_conflicting_graph(tmp_path):
    shared = inputs()[3]
    invalid = append_record(reply(shared), "a")
    client, calls = setup(tmp_path, [invalid, reply(shared)])
    result = run(tmp_path, client)
    assert result["status"] == "repair_exhausted"
    assert not result["accepted"] and result["handoff"] is None
    assert result["structural_valid"] and result["conflict_count"] == 1
    assert [event["structural_valid"] for event in result["attempts"]] == [False, True]
    feedback = calls[1]["feedback"]
    assert feedback["current_checks"]["blocked"]
    guidance = feedback["scope_guidance"]
    assert "cannot be accepted" in guidance["acceptance_obligation"]
    invalid_target = next(
        row for row in guidance["invalid_record_targets"] if row["state"] == "a"
    )
    assert "algebraic" in str(invalid_target)
    assert calls[1]["current_dependency_records"] == shared.model_dump(mode="json")
    assert result["attempts"][1]["repeated_dependency_choices"] >= 1
    assert "contradiction remains" in result["attempts"][1]["repair"]


def test_unknown_parameter_record_does_not_become_state_or_obscure_editable_scope(
    tmp_path,
):
    invalid = append_record(reply(inputs()[3]), "gain_coefficient")
    client, calls = setup(tmp_path, [invalid, reply(inputs()[4])])
    result = run(tmp_path, client)
    scope = calls[1]["feedback"]["scope_guidance"]
    assert set(scope["editable_initializers"]) == {"z0", "z1", "z2", "unrelated"}
    row = next(
        row
        for row in scope["invalid_record_targets"]
        if row["state"] == "gain_coefficient"
    )
    assert "undeclared" in str(row) or "not declared" in str(row)
    assert result["accepted"] and result["compatibility_status"] == "deferred"
    assert all(
        record["state"] != "gain_coefficient" for record in result["handoff"]["records"]
    )


def test_conflict_scope_traverses_initializer_maps_without_prescribing_unrelated_roots(
    tmp_path,
):
    client, calls = setup(tmp_path, [reply(inputs()[4])])
    run(tmp_path, client)
    scope = calls[0]["feedback"]["scope_guidance"]
    targets = [row for row in scope["conflict_repair_scope"] if row["target"] == "a"]
    assert targets and all(row["property"] == "closed_subsystem" for row in targets)
    assert all(
        set(row["editable_initializers"]) == {"z0", "z1", "z2"} for row in targets
    )
    assert all("unrelated" not in str(row["current_rules"]) for row in targets)
    assert "observed" not in scope["editable_initializers"]
    assert "prepared" not in scope["editable_initializers"]


@pytest.mark.parametrize("invalid_target", ["observed", "prepared"])
def test_runtime_fixed_initials_remain_immutable_in_specific_feedback(
    tmp_path, invalid_target
):
    invalid = reply(inputs()[3])
    if invalid_target == "prepared":
        next(row for row in invalid["records"] if row["state"] == "prepared")[
            "rule"
        ] = {"mode": "value", "scope": "trajectory"}
    else:
        append_record(invalid, invalid_target)
    client, calls = setup(tmp_path, [invalid, reply(inputs()[4])])
    result = run(tmp_path, client)
    assert not result["attempts"][0]["structural_valid"]
    scope = calls[1]["feedback"]["scope_guidance"]
    assert invalid_target not in scope["editable_initializers"]
    assert any(
        row["state"] == invalid_target for row in scope["invalid_record_targets"]
    )
    accepted = {row["state"]: row["rule"] for row in result["handoff"]["records"]}
    assert "observed" not in accepted
    assert accepted["prepared"] == {"mode": "known"}


def test_recorded_algebraic_public_initial_is_allowed_but_state_initial_is_not(
    tmp_path,
):
    shared = inputs()[3]
    invalid = reply(shared)
    invalid["records"][0]["rule"] = {"mode": "map", "state_initials": ["a"]}
    valid = reply(shared)
    valid["records"][0]["rule"] = {"mode": "map", "public_initials": ["a"]}
    client, calls = setup(tmp_path, [invalid, valid])
    result = run(tmp_path, client)
    assert not result["attempts"][0]["structural_valid"]
    assert calls[1]["feedback"]["current_checks"]["blocked"]
    assert result["accepted"] and result["conflict_count"] == 0
    chosen = next(row for row in result["handoff"]["records"] if row["state"] == "z0")
    assert chosen["rule"]["public_initials"] == ["a"]
    assert chosen["rule"]["state_initials"] == []
    assert not result["scientific_correctness_established"]


def test_permitted_trajectory_root_retains_evidence_and_exact_resume(tmp_path):
    client, calls = setup(tmp_path, [reply(inputs()[4])])
    result = run(tmp_path, client)
    assert result["accepted"] and result["compatibility_status"] == "deferred"
    assert result["deferred_count"] > 0 and result["conflict_count"] == 0
    assert not result["scientific_correctness_established"]
    restored, new_calls = setup(tmp_path, [reply(inputs()[3])])
    assert run(tmp_path, restored) == result
    assert not new_calls and result["physical_requests"] == len(calls) == 1


def test_scope_policy_is_explicit_and_separated_from_legacy_identity(tmp_path):
    model, packet, contract, shared, trajectory = inputs()
    client, _ = setup(tmp_path, [reply(trajectory)])
    kwargs = {
        "evidence": packet,
        "initial_handoff": shared,
        "previous_questions": (saved_question(),),
    }
    legacy_identity = dependency_construction_identity(
        model, packet.context, "", contract, client, **kwargs
    )
    explicit_identity = dependency_construction_identity(
        model,
        packet.context,
        "",
        contract,
        client,
        feedback_policy="legacy-1",
        **kwargs,
    )
    new_identity = dependency_construction_identity(
        model, packet.context, "", contract, client, feedback_policy=POLICY, **kwargs
    )
    assert legacy_identity == explicit_identity != new_identity
    old = dependency_request(model, packet.context, "", contract, **kwargs)
    explicit = dependency_request(
        model, packet.context, "", contract, feedback_policy="legacy-1", **kwargs
    )
    new = dependency_request(
        model, packet.context, "", contract, feedback_policy=POLICY, **kwargs
    )
    assert old == explicit
    assert "scope_guidance" not in json.loads(old.split("\n", 1)[1])["feedback"]
    assert "scope_guidance" in json.loads(new.split("\n", 1)[1])["feedback"]
    run(tmp_path, client)
    replay, new_calls = setup(tmp_path, [reply(trajectory)])
    with pytest.raises(ValueError, match="checkpoint differs"):
        run(tmp_path, replay, feedback_policy="legacy-1")
    assert not new_calls


def test_resealed_feedback_tampering_does_not_survive_cached_reconstruction(tmp_path):
    client, _ = setup(tmp_path, [reply(inputs()[4])])
    result = run(tmp_path, client)
    state_path = tmp_path / "result/state.json"
    state = json.loads(state_path.read_text())
    state["result"]["accepted"] = False
    state.pop("state_sha256")
    state["state_sha256"] = content_hash(state)
    state_path.write_text(json.dumps(state))
    restored, calls = setup(tmp_path, [reply(inputs()[4])])
    with pytest.raises(ValueError, match="result differs"):
        run(tmp_path, restored)
    assert not calls and result["accepted"]


def test_no_feedback_path_mutates_saved_initial_graph(tmp_path):
    model, packet, contract, shared, _ = inputs()
    before = copy.deepcopy(shared.model_dump(mode="json"))
    dependency_request(
        model,
        packet.context,
        "",
        contract,
        evidence=packet,
        initial_handoff=shared,
        feedback_policy=POLICY,
    )
    assert shared.model_dump(mode="json") == before


def test_scope_system_preserves_legacy_but_explains_recorded_algebraic_readings():
    from autoformalism.search.construction_graph_repair import SYSTEM, system_prompt

    assert system_prompt() == system_prompt("legacy-1") == SYSTEM
    revised = system_prompt(POLICY)
    assert "retaining valid records" not in revised
    assert "retaining unaffected" in revised
    assert "a reading\nof an algebraic output" in revised
    assert "cannot be accepted while the conflict remains" in revised
    with pytest.raises(ValueError, match="unknown initializer feedback policy"):
        system_prompt("unknown")


def test_unknown_policy_rejected_before_any_request(tmp_path):
    client, calls = setup(tmp_path, [reply(inputs()[4])])
    with pytest.raises(ValueError, match="unknown initializer feedback policy"):
        run(tmp_path, client, feedback_policy="scope-aware-typo")
    assert not calls
