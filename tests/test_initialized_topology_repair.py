"""Topology-owned initialization records, atomic repair and saved live episodes."""

import json
from copy import deepcopy

import pytest

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import initialization_handoff_audit as audit
from autoformalism.research import initialized_topology_repair as live_campaign
from autoformalism.research import saved_topology_repair as campaign
from autoformalism.schemas.staged_topology import EquationDefinition, ScientificVariable
from autoformalism.search import construction_initialized_repair as initialized
from autoformalism.search import shared_process_contract as shared
from autoformalism.search.causal_initialization import validate_dependency_handoff
from autoformalism.search.topology_initialization import InitializerDependencyHandoff
from autoformalism.staged_topology import content_hash, lower_topology
from tests.test_construction_ledger import equation, variable
from tests.test_explicit_variable_bindings import response
from tests.test_saved_topology_repair import execute
from tests.test_saved_topology_repair import setup as saved_setup  # noqa: F401


def choices(*states, mode="shared_parameter"):
    return InitializerDependencyHandoff.model_validate(
        {"records": [{"state": state, "rule": {"mode": mode}} for state in states]}
    )


@pytest.fixture
def prepared(saved_setup):  # noqa: F811 -- imported pytest fixture
    plan, tasks, problem = saved_setup
    draft = initialized.Draft.model_validate(
        {
            "variables": [variable("y"), variable("z")],
            "equations": [equation("y", "z"), equation("z", "u")],
        }
    )
    contract = audit.information_contract(problem.evidence, "causal_runtime")
    return plan, tasks[1], problem, draft, contract


@pytest.fixture
def live(prepared):
    plan, task, problem, draft, contract = prepared
    start = plan["starts"]["saved"]
    baseline = deepcopy(start["diagnostics"])
    start.update(
        draft=draft.model_dump(mode="json"),
        assessment=problem.assess(draft),
        diagnostics=problem.diagnose(draft),
        previous_conflicts=baseline,
        initial_information_contract=contract.model_dump(mode="json"),
    )
    plan.update(
        protocol=campaign.PROTOCOL_INITIALIZATION,
        actionable_policy=initialized.POLICY,
        tasks=[task],
    )
    return plan, task, problem


def packet_and_reply(body, *, complete=True):
    packet = json.loads(body["messages"][1]["content"])
    return packet, {**packet["response_template"], "stage_complete": complete}


def events(root):
    return [
        sealed_read(path)
        for path in sorted((root / "results/actionable/events").glob("*.json"))
    ]


@pytest.mark.parametrize("handoff", [choices(), choices("z", mode="unknown")])
def test_missing_or_unknown_choices_leave_completion_unfinished(prepared, handoff):
    _, _, problem, draft, contract = prepared
    status = initialized.handoff_status(draft, problem.evidence, contract, handoff)
    assert not status["complete"]
    assert status["unresolved_states"] == ["z"]
    state = initialized.qualify_completion({"completion_allowed": True}, status)
    assert not state["completion_allowed"]
    assert not state["initialization_complete"]


def test_initialization_schema_cannot_edit_equations_or_choose_formulas():
    schema = initialized.response_model("initialization")
    base = {
        "stage_complete": True,
        "initialization": choices("z").model_dump(mode="json"),
    }
    assert schema.model_validate(base).stage_complete
    with pytest.raises(ValueError):
        schema.model_validate({**base, "equations": [equation("z", "u")]})
    changed = deepcopy(base)
    changed["initialization"]["records"][0]["rule"]["expression"] = "u"
    with pytest.raises(ValueError):
        schema.model_validate(changed)


@pytest.mark.parametrize("complete", [False, True])
def test_dedicated_reply_normalizes_to_a_record_only_patch(prepared, complete):
    _, _, problem, draft, _ = prepared
    value, notes = initialized.normalize_reply(
        problem.brief,
        {
            "stage_complete": complete,
            "initialization": choices("z").model_dump(mode="json"),
        },
        stage="initialization",
    )
    patch = initialized.DraftPatch.model_validate(value)
    assert patch.initialization == choices("z")
    assert initialized.apply_patch(problem.brief, draft, patch) == draft
    assert notes == []


def test_null_retains_records_and_late_states_stay_unresolved(prepared):
    _, _, problem, draft, contract = prepared
    retained = choices("z")
    patch = initialized.DraftPatch.model_validate(
        {
            "stage_complete": True,
            "variables": [variable("w")],
            "equations": [equation("w", "u")],
        }
    )
    proposed = initialized.apply_patch(problem.brief, draft, patch)
    result = initialized.changed_handoff(
        proposed, problem.evidence, contract, retained, patch
    )
    assert result == retained
    status = initialized.handoff_status(proposed, problem.evidence, contract, result)
    assert status["unresolved_states"] == ["w"]
    assert not status["complete"]


def test_deleted_state_record_requires_explicit_same_transaction_removal(prepared):
    _, _, problem, draft, contract = prepared
    retained = choices("z")
    raw = {
        "stage_complete": True,
        "remove_variables": ["z"],
        "remove_equations": ["z"],
        "equations": [equation("y", "u")],
    }
    patch = initialized.DraftPatch.model_validate(raw)
    proposed = initialized.apply_patch(problem.brief, draft, patch)
    with pytest.raises(ValueError, match="differential"):
        initialized.changed_handoff(
            proposed, problem.evidence, contract, retained, patch
        )
    patch = initialized.DraftPatch.model_validate(
        {**raw, "initialization": choices().model_dump(mode="json")}
    )
    replacement = initialized.changed_handoff(
        proposed, problem.evidence, contract, retained, patch
    )
    assert initialized.handoff_status(
        proposed, problem.evidence, contract, replacement
    )["complete"]
    assert retained == choices("z")


def test_invalid_dependency_rejects_proposed_record_without_mutating_inputs(prepared):
    _, _, problem, draft, contract = prepared
    retained = choices("z")
    before = draft.model_dump(mode="json")
    patch = initialized.DraftPatch.model_validate(
        {
            "stage_complete": True,
            "equations": [equation("y", "z", "t")],
            "initialization": {
                "records": [
                    {
                        "state": "z",
                        "rule": {"mode": "common_map", "arguments": ["future"]},
                    }
                ]
            },
        }
    )
    proposed = initialized.apply_patch(problem.brief, draft, patch)
    assert proposed != draft
    with pytest.raises(ValueError, match="unavailable initializer arguments"):
        initialized.changed_handoff(
            proposed, problem.evidence, contract, retained, patch
        )
    assert draft.model_dump(mode="json") == before
    assert retained == choices("z")


def test_payload_displays_initial_choices_and_disallows_silent_rhs_copy(prepared):
    _, _, problem, draft, contract = prepared
    packet = initialized.payload(
        "initialization",
        problem.brief,
        problem.context,
        problem.targets,
        problem.enriched,
        draft,
        problem.graph,
        evidence=problem.evidence,
        initial_contract=contract,
        handoff=choices(),
        boundary_contract=problem.boundary,
    )
    assert packet["initialization"]["unresolved_states"] == ["z"]
    assert "Do not copy RHS arguments automatically" in packet["stage_instructions"]
    assert set(packet["response_template"]) == {"stage_complete", "initialization"}
    assert packet["feedback"]["repair_work"]["mode"] == "repair_required"
    conditioning = packet["initialization"]["diagnostic_conditioning"]
    assert conditioning["unresolved_states"] == ["z"]
    assert "not a retained or proposed initializer" in conditioning["interpretation"]
    assert "not a request to copy" in conditioning["interpretation"]


def test_exported_interaction_source_has_usable_shared_and_initializer_contracts(
    prepared,
):
    _, _, problem, draft, contract = prepared
    check = problem.assess(draft, qualified_memory=True)
    assert check["eligible"]
    assert (
        initialized.interaction_source(
            problem.brief, draft, check, problem.evidence, contract, choices()
        )
        is None
    )
    source = initialized.interaction_source(
        problem.brief, draft, check, problem.evidence, contract, choices("z")
    )
    inventory = tuple(
        ScientificVariable.model_validate(row) for row in source["inventory"]
    )
    equations = tuple(
        EquationDefinition.model_validate(row) for row in source["equations"]
    )
    assert (
        shared.validate_contract(
            source["shared_process_contract"], equations, inventory
        )
        == []
    )
    topology, aliases = lower_topology(
        problem.brief, inventory, equations, problem.context
    )
    assert topology.model_dump(mode="json") == source["topology"]
    resolved = validate_dependency_handoff(
        problem.context,
        tuple(state.name for state in topology.states),
        {"y"},
        contract,
        choices("z"),
        state_aliases=aliases,
    )
    assert resolved == choices("z")
    assert source["initializer_handoff"] == choices("z").model_dump(mode="json")
    assert source["proposer_initializer_handoff"] == choices("z").model_dump(
        mode="json"
    )
    assert source["initializer_handoff_origins"]["z"]["origin"] == "proposer"
    assert source["scientific_correctness"] == "unverified"


@pytest.mark.parametrize(
    "fixed_rule",
    [
        {"mode": "shared_parameter", "arguments": []},
        {"mode": "common_map", "arguments": ["y"]},
    ],
)
def test_export_materializes_fixed_latent_rules_with_runtime_provenance(
    prepared, fixed_rule
):
    """A complete topology boundary must export consumer-ready fixed choices."""
    _, _, problem, draft, contract = prepared
    raw = problem.evidence.model_dump(mode="json", exclude={"sha256"})
    raw["contract"]["initial_rules"]["z"] = fixed_rule
    evidence = type(problem.evidence).model_validate(
        {**raw, "sha256": content_hash(raw)}
    )
    original_evidence = evidence.model_dump(mode="json")
    retained = choices()
    original_choices = retained.model_dump(mode="json")
    assert initialized.handoff_status(draft, evidence, contract, retained)["complete"]
    source = initialized.interaction_source(
        problem.brief,
        draft,
        problem.assess(draft, qualified_memory=True),
        evidence,
        contract,
        retained,
    )
    assert source is not None
    effective = InitializerDependencyHandoff.model_validate(
        source["initializer_handoff"]
    )
    rules = {
        record.state: record.rule.model_dump(mode="json")
        for record in effective.records
    }
    assert rules["z"] == fixed_rule
    assert source["proposer_initializer_handoff"] == original_choices
    assert source["initializer_handoff_origins"]["z"] == {
        "origin": "runtime",
        "evidence_sha256": evidence.sha256,
    }
    validated = validate_dependency_handoff(
        problem.context, ("y", "z"), {"y"}, contract, effective
    )
    assert validated == effective
    assert retained.model_dump(mode="json") == original_choices
    assert evidence.model_dump(mode="json") == original_evidence
    assert source["training_evidence_sha256"] == evidence.sha256


def test_unsupported_fixed_observation_rule_does_not_export_a_substitute(prepared):
    _, _, problem, draft, contract = prepared
    raw = problem.evidence.model_dump(mode="json", exclude={"sha256"})
    raw["contract"]["initial_rules"]["z"] = {
        "mode": "observed_initial", "arguments": []
    }
    evidence = type(problem.evidence).model_validate(
        {**raw, "sha256": content_hash(raw)}
    )
    retained = choices()
    assert initialized.handoff_status(draft, evidence, contract, retained)["complete"]
    assert initialized.interaction_source(
        problem.brief, draft, problem.assess(draft, qualified_memory=True),
        evidence, contract, retained
    ) is None
    assert retained == choices()


@pytest.mark.parametrize("unresolved", [choices(), choices("z", mode="unknown")])
def test_live_unresolved_choices_get_three_dedicated_calls_then_joint_repair(
    tmp_path, live, unresolved
):
    plan, task, _ = live
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body)
        calls.append(packet)
        reply["initialization"] = unresolved.model_dump(mode="json")
        return response(reply)

    result = execute(tmp_path, plan, task, send)
    recorded = events(tmp_path)
    assert [row["stage"] for row in recorded[:3]] == ["initialization"] * 3
    assert recorded[3]["stage"] == "repair"
    assert len(recorded) == len(calls) <= campaign.MAX_CALLS
    assert result["initialization_status"]["unresolved_states"] == ["z"]
    assert not result["scientific_state"]["completion_allowed"]
    assert result["stop_reason"] != "completed_requested_work"
    assert result["interaction_source"] is None


def test_live_invalid_initialization_and_topology_edit_commit_neither(tmp_path, live):
    plan, task, _ = live
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body, complete=bool(calls))
        calls.append(packet)
        if len(calls) == 1:
            reply["initialization"] = choices("z").model_dump(mode="json")
        elif len(calls) == 2:
            reply["variables"] = [variable("w")]
            reply["equations"] = [equation("y", "z", "w"), equation("w", "u")]
            reply["initialization"] = {
                "records": [
                    *choices("z").model_dump(mode="json")["records"],
                    {
                        "state": "w",
                        "rule": {"mode": "common_map", "arguments": ["future"]},
                    },
                ]
            }
        return response(reply)

    result = execute(tmp_path, plan, task, send)
    accepted, rejected, finished = events(tmp_path)
    assert accepted["stage"] == "initialization"
    assert accepted["accepted"]
    assert rejected["stage"] == "repair"
    assert not rejected["accepted"]
    assert "unavailable initializer arguments" in rejected["error"]
    assert rejected["before"] == rejected["after"] == accepted["after"]
    assert rejected["initialization_before"] == rejected["initialization_after"]
    assert finished["accepted"]
    assert result["initializer_handoff"] == choices("z").model_dump(mode="json")
    assert result["draft"] == plan["starts"]["saved"]["draft"]


def test_live_late_state_requires_new_choice_and_retains_earlier_record(tmp_path, live):
    plan, task, _ = live
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body, complete=len(calls) == 2)
        calls.append(packet)
        if len(calls) == 1:
            reply["initialization"] = choices("z").model_dump(mode="json")
        elif len(calls) == 2:
            assert reply["initialization"] is None
            reply["variables"] = [variable("w")]
            reply["equations"] = [equation("y", "z", "w"), equation("w", "u")]
        else:
            assert packet["initialization"]["unresolved_states"] == ["w"]
            assert packet["initialization"]["retained_handoff"] == choices(
                "z"
            ).model_dump(mode="json")
            reply["initialization"] = choices("z", "w").model_dump(mode="json")
        return response(reply)

    result = execute(tmp_path, plan, task, send)
    recorded = events(tmp_path)
    assert [row["stage"] for row in recorded] == [
        "initialization",
        "repair",
        "initialization",
    ]
    assert all(row["accepted"] for row in recorded)
    assert not recorded[1]["scientific_state"]["initialization_complete"]
    assert not recorded[1]["scientific_state"]["completion_allowed"]
    assert result["initializer_handoff"] == choices("z", "w").model_dump(mode="json")
    assert result["initialization_status"]["complete"]
    assert (
        result["interaction_source"]["initializer_handoff"]
        == result["initializer_handoff"]
    )


def test_live_initialization_progress_and_cost_survive_exact_resume(tmp_path, live):
    plan, task, _ = live
    calls = []

    def send(url, body, timeout):
        packet, reply = packet_and_reply(body, complete=bool(calls))
        calls.append(packet)
        if len(calls) == 1:
            reply["initialization"] = choices("z").model_dump(mode="json")
        else:
            assert packet["initialization"]["retained_handoff"] == choices(
                "z"
            ).model_dump(mode="json")
        return response(reply)

    with pytest.raises(DeferredCall):
        execute(tmp_path, plan, task, send, can_start=lambda: len(calls) < 1)
    original = (tmp_path / "results/actionable/events/000.json").read_bytes()
    first = events(tmp_path)[0]
    assert first["initialization_after"] == choices("z").model_dump(mode="json")
    assert first["consecutive_unchanged"] == 0
    result = execute(tmp_path, plan, task, send)
    assert len(calls) == result["cost"]["physical_requests"] == 2
    assert (tmp_path / "results/actionable/events/000.json").read_bytes() == original
    assert (
        execute(tmp_path, plan, task, lambda *args: pytest.fail("uncached call"))
        == result
    )
    assert all(
        row["initial_information_contract"]
        == plan["starts"]["saved"]["initial_information_contract"]
        for row in events(tmp_path)
    )
    assert (
        result["interaction_source"]["initial_information_contract"]
        == result["initial_information_contract"]
    )
    assert not result["scientific_accuracy_established"]


def test_freezer_uses_sealed_v2_final_draft_and_preserves_history(
    tmp_path, prepared, monkeypatch
):
    plan, task, problem, draft, contract = prepared
    monkeypatch.setattr(campaign, "source_identity", lambda: "synthetic-runtime")
    source, destination = tmp_path / "v2", tmp_path / "initialized"
    plan.pop("artifact_sha256")
    plan.update(
        protocol=campaign.PROTOCOL_V2,
        actionable_policy=campaign.actionable.POLICY_V2,
        test_data_opened=False,
        tasks=[task],
    )
    parent = sealed_write(source / "plan.json", plan)

    def send(url, body, timeout):
        _, reply = packet_and_reply(body)
        reply["variables"] = [variable("z")]
        reply["equations"] = [equation("y", "z"), equation("z", "u")]
        return response(reply)

    result = execute(source, parent, task, send)
    assert result["draft"] == draft.model_dump(mode="json")
    original = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    frozen = live_campaign.freeze(source, destination)
    assert live_campaign.freeze(source, destination) == frozen
    assert original == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert frozen["protocol"] == campaign.PROTOCOL_INITIALIZATION
    assert frozen["actionable_policy"] == initialized.POLICY
    assert frozen["source_plan_sha256"] == parent["artifact_sha256"]
    assert frozen["settings"] == parent["settings"]
    assert frozen["tasks"] == [
        {**task, "task_id": "actionable_initialization", "source_task": "actionable"}
    ]
    start = frozen["starts"]["actionable"]
    assert start["draft"] == result["draft"]
    assert start["source_result_sha256"] == result["artifact_sha256"]
    assert start["initial_information_contract"] == contract.model_dump(mode="json")
    original_questions = parent["starts"]["saved"]["diagnostics"]["advisory_questions"]
    assert {q["question_id"] for q in original_questions} <= {
        q["question_id"] for q in start["previous_conflicts"]["tracked_conflicts"]
    }
    assert not live_campaign.Session(problem, start).status(draft)["complete"]
    assert campaign.verify(destination) == frozen
