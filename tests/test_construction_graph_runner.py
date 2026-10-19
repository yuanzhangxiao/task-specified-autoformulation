"""Bounded graph declaration calls preserve evidence, costs and terminal outcomes."""

import copy
import json

import pytest

from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    StagedTopologyClient,
)
from autoformalism.search.construction_graph_runner import (
    construct_graph_dependencies,
    dependency_construction_identity,
    dependency_request,
)
from autoformalism.staged_topology import content_hash
from tests.test_construction_graph_evidence import graph_fixture, saved_question
from tests.test_construction_initialization_graph import fixture
from tests.test_staged_function_runner import response


@pytest.fixture(autouse=True)
def stable_source_hash(monkeypatch):
    """Concurrent work in other modules must not alter this isolated replay fixture."""
    monkeypatch.setattr(
        "autoformalism.rebuttal.staged_topology_campaign.runtime_source_hash",
        lambda: "fixed_test_runtime",
    )


def setup(tmp_path, replies, *, can_start=lambda: True, namespace="dependencies"):
    """Create a fresh physical client while retaining prior on-disk call records."""
    calls = []

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        value = replies[min(len(calls) - 1, len(replies) - 1)]
        if isinstance(value, BaseException):
            raise value
        return value if "choices" in value else response(value)

    client = StagedTopologyClient(
        settings=StagedModelSettings(attempts_per_step=2),
        base_url="http://unused",
        directory=tmp_path / "calls",
        namespace=namespace,
        seed=0,
        transport=transport,
        can_start=can_start,
    )
    return client, calls


def run(tmp_path, client, *, model=None, context=None, contract=None, **kwargs):
    """Use a fully bound runtime with observed, known and dependent state initials."""
    default_model, default_context, default_contract, _ = fixture()
    return construct_graph_dependencies(
        model or default_model,
        context or default_context,
        "Measured signal is y, in the same units." if model is None else "",
        contract or default_contract,
        client,
        tmp_path / "result",
        **kwargs,
    )


def valid_reply():
    return fixture()[3].model_dump(mode="json")


def invalid_reply():
    payload = valid_reply()
    payload["records"].pop()
    return payload


def test_atomic_repair_retains_conditional_witnesses_and_exact_resume(tmp_path):
    question = {
        "question_id": "old_question",
        "property": "closed_subsystem",
        "witnesses": [{"trajectories": ["first", "second"], "times": [1, 1]}],
        "assumptions": ["shared latent initial boundary"],
        "question": "Which initial preparation assumption applies?",
    }
    before = copy.deepcopy(question)
    client, calls = setup(tmp_path, [invalid_reply(), valid_reply()])
    result = run(tmp_path, client, previous_questions=(question,))
    assert result["status"] == "complete"
    assert [event["accepted"] for event in result["attempts"]] == [False, True]
    assert calls[0]["feedback"] is None
    assert calls[1]["feedback"]["attempted_records"] == invalid_reply()
    assert "complete corrected record list" in calls[1]["feedback"]["repair"]
    for call in calls:
        retained = call["retained_conditional_evidence"]
        assert "earlier initializer assumptions" in retained["interpretation"]
        assert retained["questions"][0]["witnesses"] == question["witnesses"]
        assert retained["questions"][0]["assumptions"] == question["assumptions"]
        assert retained["questions"][0]["question"] == question["question"]
        assert retained["retained_history_count"] == retained["display_count"] == 1
    assert question == before
    assert result["handoff"] == valid_reply()
    assert result["physical_requests"] == 2
    assert result["budget_charge"] == result["observed_total_tokens"] == 200
    assert not result["scientific_correctness_established"]
    assert result["optimizer_calls"] == 0
    restored, new_calls = setup(tmp_path, [valid_reply()])
    assert run(tmp_path, restored, previous_questions=(question,)) == result
    assert len(restored.records) == 2
    assert new_calls == []


def test_exhausted_repairs_are_terminal_without_partial_handoff(tmp_path):
    client, calls = setup(tmp_path, [invalid_reply()])
    result = run(tmp_path, client)
    assert result["status"] == "repair_exhausted"
    assert result["handoff"] is None
    assert len(calls) == 2
    restored, new_calls = setup(tmp_path, [valid_reply()])
    assert run(tmp_path, restored) == result
    assert new_calls == []
    assert sum(item["budget_charge"] for item in restored.records) == 200


@pytest.mark.parametrize("failure", ["truncated", "missing", "provider_failure"])
def test_unavailable_outcomes_remain_terminal_and_charged(tmp_path, failure):
    value = response(valid_reply())
    if failure == "truncated":
        value["choices"][0]["finish_reason"] = "length"
    elif failure == "missing":
        value["choices"][0]["message"]["content"] = ""
    else:
        value = TimeoutError("test provider timeout")
    client, calls = setup(tmp_path, [value, valid_reply()])
    result = run(tmp_path, client)
    assert result["status"] == "unavailable"
    assert result["attempts"][0]["unavailable"] is True
    assert result["handoff"] is None
    assert result["physical_requests"] == len(calls) == 1
    assert result["budget_charge"] > 0
    assert result["unmeasured_requests"] == (failure == "provider_failure")
    restored, new_calls = setup(tmp_path, [valid_reply()])
    assert run(tmp_path, restored) == result
    assert new_calls == []


def test_interrupted_call_replays_as_charged_uncertain_without_new_provider(tmp_path):
    client, calls = setup(tmp_path, [KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, client)
    assert len(calls) == 1
    restored, new_calls = setup(tmp_path, [valid_reply()])
    result = run(tmp_path, restored)
    assert result["status"] == "unavailable"
    assert restored.records[0]["status"] == "uncertain"
    assert result["physical_requests"] == result["unmeasured_requests"] == 1
    assert result["budget_charge"] > 0
    assert new_calls == []
    again, again_calls = setup(tmp_path, [valid_reply()])
    assert run(tmp_path, again) == result
    assert again_calls == []


def test_deferred_repair_retains_consumed_attempt_and_feedback(tmp_path):
    client, calls = setup(tmp_path, [invalid_reply()])
    client.can_start = lambda: not calls
    with pytest.raises(DeferredCall):
        run(tmp_path, client)
    state = json.loads((tmp_path / "result/state.json").read_text())
    assert len(state["attempts"]) == 1
    assert state.get("handoff") is None
    restored, new_calls = setup(tmp_path, [valid_reply()])
    result = run(tmp_path, restored)
    assert result["status"] == "complete"
    assert result["physical_requests"] == 2
    assert new_calls[0]["feedback"]["attempt"] == 0
    assert len(new_calls) == 1


@pytest.mark.parametrize("tamper", ["digest", "resealed_handoff", "result", "cache"])
def test_checkpoint_and_cached_reply_tampering_are_rejected(tmp_path, tamper):
    client, _ = setup(tmp_path, [valid_reply()])
    result = run(tmp_path, client)
    path = tmp_path / "result/state.json"
    state = json.loads(path.read_text())
    if tamper == "digest":
        state["attempts"] = []
    elif tamper == "cache":
        call_path = (
            tmp_path / "calls" / (result["attempts"][0]["request_hash"] + ".json")
        )
        record = json.loads(call_path.read_text())
        record["request"]["namespace"] = "other"
        call_path.write_text(json.dumps(record))
    else:
        if tamper == "resealed_handoff":
            state["handoff"]["records"][0]["justification"] = "Changed after acceptance"
            state["result"]["handoff"] = copy.deepcopy(state["handoff"])
            state["result"].pop("result_sha256")
            state["result"]["result_sha256"] = content_hash(state["result"])
        else:
            state["result"]["scientific_correctness_established"] = True
        state.pop("state_sha256")
        state["state_sha256"] = content_hash(state)
    path.write_text(json.dumps(state))
    restored, calls = setup(tmp_path, [valid_reply()])
    with pytest.raises(ValueError, match=r"checkpoint|provenance|result differs"):
        run(tmp_path, restored)
    assert calls == []


def test_changed_namespace_rejects_resume_before_provider(tmp_path):
    client, _ = setup(tmp_path, [valid_reply()])
    run(tmp_path, client)
    changed, calls = setup(tmp_path, [valid_reply()], namespace="new")
    with pytest.raises(ValueError, match="checkpoint differs"):
        run(tmp_path, changed)
    assert calls == []


def test_fresh_construction_with_new_runtime_has_new_cache_key(tmp_path, monkeypatch):
    client, _ = setup(tmp_path, [valid_reply()])
    previous = run(tmp_path, client)
    next_client, calls = setup(tmp_path, [valid_reply()])
    monkeypatch.setattr(
        "autoformalism.rebuttal.staged_topology_campaign.runtime_source_hash",
        lambda: "new_test_runtime",
    )
    draft, context, contract, _ = fixture()
    result = construct_graph_dependencies(
        draft,
        context,
        "Measured signal is y, in the same units.",
        contract,
        next_client,
        tmp_path / "new_result",
    )
    assert len(calls) == 1
    assert result["identity"] != previous["identity"]
    assert (
        result["attempts"][0]["request_hash"] != previous["attempts"][0]["request_hash"]
    )


@pytest.mark.parametrize("terminal", ["complete", "repair_exhausted"])
def test_local_serving_port_change_replays_terminal_outcome_and_budget(
    tmp_path, terminal
):
    replies = [valid_reply()] if terminal == "complete" else [invalid_reply()]
    client, calls = setup(tmp_path, replies)
    client.base_url = "http://127.0.0.1:8001"
    original = run(tmp_path, client)
    assert original["status"] == terminal
    resumed, new_calls = setup(tmp_path, [valid_reply()])
    resumed.base_url = "http://127.0.0.1:49152"
    assert run(tmp_path, resumed) == original
    assert new_calls == []
    assert len(resumed.records) == len(calls)
    assert (
        sum(record["budget_charge"] for record in resumed.records)
        == original["budget_charge"]
    )


def test_invalid_evidence_is_rejected_before_spending_a_request(tmp_path):
    model, packet, contract, handoff = graph_fixture()
    changed = packet.model_copy(update={"sha256": "0" * 64})
    client, calls = setup(tmp_path, [handoff.model_dump(mode="json")])
    with pytest.raises(ValueError, match="evidence digest"):
        run(
            tmp_path,
            client,
            model=model,
            context=packet.context,
            contract=contract,
            evidence=changed,
        )
    assert calls == []


def test_complete_graph_preserves_scientific_conflict_as_separate_assessment(tmp_path):
    model, packet, contract, handoff = graph_fixture()
    client, calls = setup(tmp_path, [handoff.model_dump(mode="json")])
    result = run(
        tmp_path,
        client,
        model=model,
        context=packet.context,
        contract=contract,
        evidence=packet,
    )
    assert len(calls) == 1
    assert result["status"] == "complete"
    assert result["assessment"]["scientific_pass"] is False
    assert any(
        row["status"] == "conditional_conflict"
        for row in result["assessment"]["declared_report"]["diagnostics"]
    )


def test_latest_prompt_view_preserves_conditioning_and_full_evidence_history(tmp_path):
    model, packet, contract, handoff = graph_fixture()
    old = saved_question()
    old["conditioning"] = {"initial_u": [0, 0], "z0": "shared training root"}
    repeated = copy.deepcopy(old)
    repeated["question_id"] = "historical_rendering_variant"
    unrelated = copy.deepcopy(old)
    unrelated.update(question_id="other_property", property="algebraic_map")
    latest = copy.deepcopy(old)
    latest.update(
        question_id="current_advisory",
        conditioning={"initial_u": [0, 0], "initial_z0": "shared across rows"},
        question="Can the initialization graph explain this matched contrast?",
        model_relation="z2(0) depends transitively on shared z0(0)",
        paired_recorded_values={"u": [0, 0]},
        assumptions=["shared roots", "equal permitted initial input"],
    )
    history = (old, repeated, unrelated, latest)
    original_history = copy.deepcopy(history)
    client, calls = setup(tmp_path, [handoff.model_dump(mode="json")])
    result = run(
        tmp_path,
        client,
        model=model,
        context=packet.context,
        contract=contract,
        evidence=packet,
        previous_questions=history,
    )
    projected = calls[0]["retained_conditional_evidence"]
    assert projected["retained_history_count"] == 4
    assert projected["display_count"] == 2
    displayed = next(
        row for row in projected["questions"] if row["property"] == "closed_subsystem"
    )
    for field in (
        "question_id",
        "conditioning",
        "question",
        "model_relation",
        "paired_recorded_values",
        "assumptions",
        "witnesses",
    ):
        assert displayed[field] == latest[field]
    retained = result["assessment"]["retained_witness_report"]
    assert retained["original_questions"] == list(history)
    assert {row["source_question_index"] for row in retained["checks"]} == set(range(4))
    assert history == original_history
    expected_identity = dependency_construction_identity(
        model,
        packet.context,
        "",
        contract,
        client,
        evidence=packet,
        previous_questions=history,
    )
    assert result["identity"] == calls[0]["construction_identity"] == expected_identity
    altered = copy.deepcopy(history)
    altered[0]["reason"] = (
        "Changed historical detail omitted by latest-view compression"
    )
    assert (
        dependency_construction_identity(
            model,
            packet.context,
            "",
            contract,
            client,
            evidence=packet,
            previous_questions=altered,
        )
        != expected_identity
    )


def test_shared_identity_helper_and_request_renderer_match_first_execution(tmp_path):
    model, context, contract, handoff = fixture()
    client, calls = setup(tmp_path, [handoff.model_dump(mode="json")])
    identity = dependency_construction_identity(
        model,
        context,
        "Measured signal is y, in the same units.",
        contract,
        client,
    )
    preview = dependency_request(
        model,
        context,
        "Measured signal is y, in the same units.",
        contract,
        construction_identity=identity,
    )
    result = run(tmp_path, client)
    assert json.loads(preview.split("\n", 1)[1]) == calls[0]
    assert result["identity"] == identity
