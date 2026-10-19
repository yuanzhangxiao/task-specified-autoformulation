"""Deterministic model/data conflicts block acceptance, including after resume."""

import json

import pytest

from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.search.construction_graph_repair import (
    construct_graph_dependencies,
    dependency_construction_identity,
    dependency_request,
)
from autoformalism.staged_topology import content_hash
from tests.test_construction_graph_evidence import graph_fixture, saved_question
from tests.test_construction_graph_runner import setup
from tests.test_staged_function_runner import response


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(
        "autoformalism.rebuttal.staged_topology_campaign.runtime_source_hash",
        lambda: "fixed_repair_test_source",
    )


def inputs():
    model, evidence, contract, trajectory = graph_fixture(
        scope="trajectory", window=False
    )
    raw = trajectory.model_dump(mode="json")
    raw["records"][0]["rule"]["scope"] = "shared_training"
    shared = type(trajectory).model_validate(raw)
    return model, evidence, contract, shared, trajectory


def reply(handoff):
    return {
        "decision": "revise",
        "records": handoff.model_dump(mode="json")["records"],
        "explanation": "Explicit preparation hypothesis.",
    }


def run(tmp_path, client, **kwargs):
    model, evidence, contract, shared, _ = inputs()
    return construct_graph_dependencies(
        model,
        evidence.context,
        "",
        contract,
        client,
        tmp_path / "result",
        evidence=evidence,
        initial_handoff=shared,
        previous_questions=(saved_question(),),
        **kwargs,
    )


def test_conflict_rejected_then_trajectory_root_accepted_with_explicit_deferral(
    tmp_path,
):
    *_, shared, trajectory = inputs()
    client, calls = setup(tmp_path, [reply(shared), reply(trajectory)])
    result = run(tmp_path, client)
    assert result["status"] == "complete"
    assert result["accepted"] and result["structural_valid"]
    assert result["compatibility_status"] == "deferred"
    assert result["deferred_count"] > 0 and result["conflict_count"] == 0
    assert not result["scientific_correctness_established"]
    assert [e["accepted"] for e in result["attempts"]] == [False, True]
    assert all(e["structural_valid"] for e in result["attempts"])
    assert calls[0]["feedback"]["current_checks"]["blocked"]
    assert calls[1]["feedback"]["current_checks"]["blocked"]
    assert "repeats" in calls[1]["feedback"]["repair"]
    assert result["physical_requests"] == 2
    resumed, fresh_calls = setup(tmp_path, [reply(shared)])
    assert run(tmp_path, resumed) == result
    assert not fresh_calls and len(resumed.records) == 2


def test_exhausted_contradictions_never_export_accepted_graph(tmp_path):
    shared = inputs()[3]
    client, calls = setup(tmp_path, [reply(shared)])
    client.settings = StagedModelSettings(attempts_per_step=3)
    result = run(tmp_path, client)
    assert len(calls) == result["physical_requests"] == 3
    assert result["status"] == "repair_exhausted"
    assert result["structural_valid"] and not result["accepted"]
    assert result["compatibility_status"] == "conflict"
    assert result["conflict_count"] > 0
    assert result["handoff"] is None
    assert result["latest_candidate"] == shared.model_dump(mode="json")
    assert result["attempts"][-1]["repeated_dependency_choices"] == 3
    resumed, fresh_calls = setup(tmp_path, [reply(inputs()[4])])
    resumed.settings = client.settings
    assert run(tmp_path, resumed) == result
    assert not fresh_calls


def test_out_of_scope_decision_stops_unresolved_not_accepted(tmp_path):
    client, calls = setup(
        tmp_path,
        [
            {
                "decision": "needs_topology_revision",
                "records": [],
                "explanation": "The output needs another physical dependence.",
            }
        ],
    )
    result = run(tmp_path, client)
    assert len(calls) == 1
    assert result["status"] == "needs_topology_revision"
    assert not result["accepted"] and result["handoff"] is None
    assert result["compatibility_status"] == "conflict"
    assert result["conflict_count"] > 0


def test_invalid_choice_does_not_erase_verified_conflict(tmp_path):
    shared = inputs()[3]
    invalid = reply(shared)
    invalid["records"].pop()
    client, calls = setup(tmp_path, [invalid, reply(shared)])
    result = run(tmp_path, client)
    assert not result["attempts"][0]["structural_valid"]
    assert calls[1]["feedback"]["current_checks"]["blocked"]
    assert result["status"] == "repair_exhausted"
    assert not result["accepted"]


def test_deferred_allocation_replays_verified_feedback_without_refund(tmp_path):
    *_, shared, trajectory = inputs()
    client, calls = setup(tmp_path, [reply(shared)])
    client.can_start = lambda: not calls
    with pytest.raises(DeferredCall):
        run(tmp_path, client)
    resumed, new_calls = setup(tmp_path, [reply(trajectory)])
    result = run(tmp_path, resumed)
    assert result["physical_requests"] == 2
    assert result["budget_charge"] == 200
    assert len(new_calls) == 1
    assert new_calls[0]["feedback"]["current_checks"]["blocked"]


@pytest.mark.parametrize("failure", ["truncated", "provider_failure", "interrupt"])
def test_unavailable_response_is_charged_and_not_silently_retried(tmp_path, failure):
    raw = response(reply(inputs()[3]))
    if failure == "truncated":
        raw["choices"][0]["finish_reason"] = "length"
    elif failure == "provider_failure":
        raw = TimeoutError("test timeout")
    else:
        raw = KeyboardInterrupt()
    client, _ = setup(tmp_path, [raw])
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            run(tmp_path, client)
        client, calls = setup(tmp_path, [reply(inputs()[4])])
        result = run(tmp_path, client)
        assert not calls
    else:
        result = run(tmp_path, client)
    assert result["status"] == "unavailable"
    assert not result["accepted"]
    assert result["physical_requests"] == 1 and result["budget_charge"] > 0
    resumed, calls = setup(tmp_path, [reply(inputs()[4])])
    assert run(tmp_path, resumed) == result
    assert not calls


@pytest.mark.parametrize(
    "tamper", ["event", "gate", "result", "missing_cache", "replaced_request"]
)
def test_resealed_checkpoint_or_cache_changes_cannot_forge_acceptance(tmp_path, tamper):
    client, _ = setup(tmp_path, [reply(inputs()[4])])
    result = run(tmp_path, client)
    path = tmp_path / "result/state.json"
    state = json.loads(path.read_text())
    if tamper == "event":
        state["attempts"][0]["explanation"] = "forged"
    elif tamper == "gate":
        state["gate"]["conflict_count"] = 999
    elif tamper == "result":
        state["result"]["accepted"] = False
    else:
        call_path = (
            tmp_path / "calls" / (result["attempts"][0]["request_hash"] + ".json")
        )
        if tamper == "missing_cache":
            call_path.unlink()
        else:
            call = json.loads(call_path.read_text())
            call["request"]["body"]["messages"][0]["content"] = "altered"
            call_path.write_text(json.dumps(call))
    state.pop("state_sha256")
    state["state_sha256"] = content_hash(state)
    path.write_text(json.dumps(state))
    resumed, calls = setup(tmp_path, [reply(inputs()[4])])
    with pytest.raises(ValueError, match=r"checkpoint|cached request|result differs"):
        run(tmp_path, resumed)
    assert not calls


def test_preview_equals_first_call_and_seeds_change_identity(tmp_path):
    model, evidence, contract, shared, trajectory = inputs()
    client, calls = setup(tmp_path, [reply(trajectory)])
    kwargs = {
        "evidence": evidence,
        "previous_questions": (saved_question(),),
        "initial_handoff": shared,
    }
    identity = dependency_construction_identity(
        model, evidence.context, "", contract, client, **kwargs
    )
    preview = dependency_request(
        model, evidence.context, "", contract, construction_identity=identity, **kwargs
    )
    result = run(tmp_path, client)
    assert json.loads(preview.split("\n", 1)[1]) == calls[0]
    assert result["identity"] == identity
    client.seed += 1
    assert (
        dependency_construction_identity(
            model, evidence.context, "", contract, client, **kwargs
        )
        != identity
    )


def test_unsealed_evidence_rejected_before_any_provider_call(tmp_path):
    model, evidence, contract, shared, trajectory = inputs()
    bad = evidence.model_copy(update={"sha256": "0" * 64})
    client, calls = setup(tmp_path, [reply(trajectory)])
    with pytest.raises(ValueError, match="evidence digest"):
        construct_graph_dependencies(
            model,
            evidence.context,
            "",
            contract,
            client,
            tmp_path / "result",
            evidence=bad,
            initial_handoff=shared,
        )
    assert not calls


def test_response_cannot_modify_observed_initials_or_claim_completion_flag(tmp_path):
    invalid = reply(inputs()[3])
    invalid["complete"] = True
    client, calls = setup(tmp_path, [invalid])
    result = run(tmp_path, client)
    assert result["status"] == "repair_exhausted"
    assert not result["structural_valid"] and not result["accepted"]
    assert result["handoff"] is None
    assert calls[1]["feedback"]["attempted_records"]["complete"] is True


def test_last_attempt_saved_before_terminal_report_resumes_without_new_call(tmp_path):
    client, _ = setup(tmp_path, [reply(inputs()[3])])
    result = run(tmp_path, client)
    path = tmp_path / "result/state.json"
    state = json.loads(path.read_text())
    state.pop("result")
    state.pop("state_sha256")
    state["status"] = "running"
    state["state_sha256"] = content_hash(state)
    path.write_text(json.dumps(state))
    (tmp_path / "result/result.json").unlink()
    resumed, calls = setup(tmp_path, [reply(inputs()[4])])
    assert run(tmp_path, resumed) == result
    assert not calls


def test_shared_map_with_distinguishing_inputs_is_not_forced_to_trajectory_root(
    tmp_path,
):
    shared = inputs()[3]
    raw = shared.model_dump(mode="json")
    raw["records"][0]["rule"] = {"mode": "map", "public_initials": ["a"]}
    distinguishing = type(shared).model_validate(raw)
    client, calls = setup(tmp_path, [reply(distinguishing)])
    result = run(tmp_path, client)
    assert result["accepted"] and result["conflict_count"] == 0
    assert len(calls) == 1
    assert result["handoff"] == distinguishing.model_dump(mode="json")
    assert not result["scientific_correctness_established"]
    assert "all_conflict_witnesses" not in calls[0]["feedback"]["current_checks"]
    assert "deferred_checks" not in calls[0]["feedback"]["current_checks"]
