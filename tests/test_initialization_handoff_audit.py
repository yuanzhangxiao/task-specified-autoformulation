"""Saved initializer handoffs preserve historical evidence and exact resume."""

import json

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import initialization_handoff_audit as audit
from autoformalism.research import saved_topology_repair as saved
from autoformalism.search import topology_initialization as initial
from autoformalism.staged_topology import content_hash
from tests.test_construction_ledger import equation, variable
from tests.test_explicit_variable_bindings import response
from tests.test_saved_topology_repair import execute
from tests.test_saved_topology_repair import setup as repair_setup  # noqa: F401


@pytest.fixture
def source(tmp_path, repair_setup):  # noqa: F811 -- imported pytest fixture
    """Use genuine sealed plans/results, with an in-memory mock proposer response."""
    plan, tasks, problem = repair_setup
    plan.pop("artifact_sha256")
    plan.update(protocol=saved.PROTOCOL, test_data_opened=False, tasks=[tasks[1]])
    folder = tmp_path / "source"
    plan = sealed_write(folder / "plan.json", plan)

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        patch = {**packet["response_template"], "stage_complete": True}
        patch["variables"] = [variable("z")]
        patch["equations"] = [equation("y", "z"), equation("z", "u")]
        return response(patch)

    result = execute(folder, plan, tasks[1], send)
    return folder, plan, problem, result


def test_saved_audit_keeps_history_sealed_and_resume_exact(tmp_path, source):
    folder, _, _, _ = source
    original = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    root = tmp_path / "audit"
    result = audit.audit(folder, root, mode="causal_runtime")
    assert audit.audit(folder, root, mode="causal_runtime") == result
    assert original == {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    assert result["llm_calls"] == result["optimizer_calls"] == 0
    assert not result["proposer_choices_inferred"]
    assert not result["test_data_opened"]
    assert (root / "PROMPTS.html").exists()
    recorded = sealed_read(root / "results/actionable/assessment.json")
    assert not recorded["structural_eligibility_changed"]
    assert recorded["family_qualified_state"]["tracked_conflicts"]


def test_unknown_contract_keeps_unavailable_initialization(tmp_path, source):
    folder, _, _, _ = source
    audit.audit(folder, tmp_path / "audit")
    record = sealed_read(tmp_path / "audit/results/actionable/assessment.json")
    assert (
        record["declared_feedback"]["counts"]
        == (record["family_qualified_state"]["counts"])
    )
    assert record["family_qualified_state"]["repair_status"] == "review_deferred"


def test_changed_policy_cannot_overwrite_an_audit(tmp_path, source):
    folder, _, _, _ = source
    root = tmp_path / "audit"
    audit.audit(folder, root)
    with pytest.raises(ValueError, match="frozen artifact differs"):
        audit.audit(folder, root, mode="causal_runtime")


def test_cannot_write_into_historical_source(tmp_path, source):
    folder, _, _, _ = source
    with pytest.raises(ValueError, match="outside"):
        audit.audit(folder, folder / "new")


def test_stale_handoff_is_not_applied(tmp_path, source):
    folder, _, problem, result = source
    drafts = tmp_path / "choices"
    receipt = {
        "topology_sha256": "0" * 64,
        "evidence_sha256": problem.evidence.sha256,
        "handoff": initial.InitializerDependencyHandoff().model_dump(mode="json"),
    }
    sealed_write(drafts / "actionable.json", receipt)
    with pytest.raises(ValueError, match="topology differs"):
        audit.audit(folder, tmp_path / "audit", handoffs=drafts)
    assert content_hash(result["draft"]) != receipt["topology_sha256"]


def test_foreign_result_identity_is_rejected(tmp_path, source):
    folder, _, _, _ = source
    path = folder / "results/actionable/result.json"
    value = sealed_read(path)
    value.pop("artifact_sha256")
    value["identity"] = "foreign"
    path.unlink()
    sealed_write(path, value)
    with pytest.raises(ValueError, match="identity differs"):
        audit.audit(folder, tmp_path / "audit")


def test_empty_final_tracking_does_not_erase_original_witnesses():
    question = {
        "question_id": "original",
        "target": "a",
        "property": "closed_subsystem",
    }
    start = {
        "diagnostics": {
            "evidence_sha256": "original-training",
            "advisory_questions": [question],
        }
    }
    result = {
        "scientific_state": {
            "evidence_sha256": "original-training",
            "tracked_conflicts": [],
        }
    }
    baseline = audit._saved_conflicts(start, result)
    assert baseline["tracked_conflicts"] == [question]
    assert result["scientific_state"]["tracked_conflicts"] == []


def test_final_scientific_state_requires_original_evidence_identity():
    start = {"diagnostics": {"evidence_sha256": "original-training"}}
    result = {"scientific_state": {"evidence_sha256": "foreign-training"}}
    with pytest.raises(ValueError, match="training evidence differs"):
        audit._saved_conflicts(start, result)


def test_foreign_event_does_not_replace_validated_final_tracking(tmp_path, source):
    folder, _, problem, _ = source
    expected = audit.audit(folder, tmp_path / "before", mode="causal_runtime")
    sealed_write(
        folder / "results/actionable/events/999.json",
        {
            "identity": "foreign-task",
            "index": 999,
            "scientific_state": {
                "evidence_sha256": problem.evidence.sha256,
                "tracked_conflicts": [],
            },
        },
    )
    actual = audit.audit(folder, tmp_path / "after", mode="causal_runtime")
    assert actual == expected
    before = sealed_read(tmp_path / "before/results/actionable/assessment.json")
    after = sealed_read(tmp_path / "after/results/actionable/assessment.json")
    assert before["family_qualified_state"] == after["family_qualified_state"]


@pytest.mark.parametrize("name", ["../x", ".", "..", "/tmp/x", ""])
def test_task_path_is_not_trusted(name):
    with pytest.raises(ValueError, match="unsafe"):
        audit._name(name)


def test_new_latent_state_cannot_erase_matched_information_conflict():
    from autoformalism.research import construction_training as training
    from autoformalism.research.actionable_repair_status import build_state
    from tests.test_topology_compatibility import draft
    from tests.test_topology_initialization import fixture

    model, evidence, contract = fixture()
    before = training.compatibility_feedback(
        draft(), evidence, identity_policy=training.SEMANTIC_QUESTION_IDENTITY
    )
    declared = audit.qualified_feedback(
        model, evidence, contract, initial.InitializerDependencyHandoff()
    )
    family = audit.qualified_feedback(
        model, evidence, contract, initial.InitializerDependencyHandoff(), family=True
    )
    assert declared["evidence_sha256"] == family["evidence_sha256"] == evidence.sha256
    assert family["initialization_evidence_sha256"] != evidence.sha256
    assert build_state(declared, before, True)["repair_status"] == "review_deferred"
    state = build_state(family, before, True)
    assert state["repair_status"] == "repair_required"
    closed = [
        q for q in state["pending_questions"] if q["property"] == "closed_subsystem"
    ]
    assert len(closed) == 1
    assert closed[0]["initialization_qualification"]["family_available"]
    assert state["scientific_correctness"] == "unverified"


def test_different_permitted_initial_information_is_not_forced_to_match():
    from autoformalism.research import construction_training as training
    from autoformalism.research.actionable_repair_status import build_state
    from tests.test_topology_compatibility import draft
    from tests.test_topology_initialization import fixture

    model, evidence, contract = fixture(extra_initial=True)
    before = training.compatibility_feedback(
        draft(), evidence, identity_policy=training.SEMANTIC_QUESTION_IDENTITY
    )
    family = audit.qualified_feedback(
        model, evidence, contract, initial.InitializerDependencyHandoff(), family=True
    )
    state = build_state(family, before, True)
    assert not [
        q for q in state["pending_questions"] if q["property"] == "closed_subsystem"
    ]
    assert state["scientific_correctness"] == "unverified"


@pytest.mark.parametrize("early_forcing", [False, True])
def test_saved_conflict_checks_forcing_only_through_its_time(early_forcing):
    from autoformalism.research.actionable_repair_status import build_state
    from tests.test_topology_initialization import prefix_fixture

    model, evidence, old = prefix_fixture(early_forcing=early_forcing)
    old["scope"] = ["a", "u"]
    baseline = {"evidence_sha256": evidence.sha256, "advisory_questions": [old]}
    contract = audit.information_contract(evidence, "causal_runtime")
    feedback = audit.qualified_feedback(
        model,
        evidence,
        contract,
        initial.InitializerDependencyHandoff(),
        family=True,
        previous_questions=[old],
    )
    state = build_state(feedback, baseline, True)
    recheck = feedback["saved_witness_recheck"]["checks"][0]
    if early_forcing:
        assert recheck["status"] == "conditioning_distinction"
        assert state["baseline_outcomes"][0]["resolved"]
    else:
        assert recheck["status"] == "preserved"
        assert state["repair_status"] == "repair_required"
        assert not state["baseline_outcomes"][0]["resolved"]
        question = next(
            q for q in state["pending_questions"] if q["property"] == "closed_subsystem"
        )
        assert (
            "through each saved witness time" in question["conditioning"]["comparison"]
        )
        assert question["witnesses"][0]["observations"] == [1, 2]


def test_invalid_saved_pair_does_not_become_a_verified_resolution():
    from autoformalism.research.actionable_repair_status import build_state
    from tests.test_topology_initialization import prefix_fixture

    model, evidence, old = prefix_fixture(early_forcing=True)
    old["scope"] = ["a", "u"]
    old["witnesses"][0]["sample_indices"] = [100, 100]
    feedback = audit.qualified_feedback(
        model,
        evidence,
        audit.information_contract(evidence, "causal_runtime"),
        initial.InitializerDependencyHandoff(),
        family=True,
        previous_questions=[old],
    )
    state = build_state(
        feedback,
        {"evidence_sha256": evidence.sha256, "advisory_questions": [old]},
        True,
    )
    assert not state["baseline_outcomes"][0]["resolved"]
    assert state["repair_status"] == "review_deferred"


def test_supported_saved_prefix_conflict_outweighs_uncertain_current_witness():
    """Later forcing divergence must not hide the larger supported early gap."""
    from dataclasses import replace

    import numpy as np

    from autoformalism.expressions import ValidationContext
    from autoformalism.research.actionable_repair_status import build_state
    from autoformalism.search.topology_compatibility import InitialRule
    from tests.test_topology_compatibility import contract, draft, evidence, trajectory

    rows = [
        trajectory("left", (0, 0, 0, 0), (0, 0, 0, 0)),
        trajectory("right", (0, 3, 0, 0), (0, 0, 0, 0)),
        trajectory("third", (0, 1, 0, 0), (0, 0, 0, 0)),
    ]
    rows = [
        replace(
            row,
            external_inputs={
                **row.external_inputs,
                "v": np.asarray(
                    (0, 0, 0, 1)
                    if row.trajectory_id == "right"
                    else (0, 0, 0, 0),
                    float,
                ),
            },
        )
        for row in rows
    ]
    packet = evidence(
        *rows,
        context=ValidationContext(targets=("a",), external_inputs=("u", "v")),
        config=contract(
            initial_rules={"z": InitialRule(mode="shared_parameter")},
            interpolation={"u": "linear_complete_knots", "v": "linear_complete_knots"},
            absolute_error_bounds={"a": 1, "u": 0, "v": 0},
        ),
    )
    model = draft(
        "algebraic", ("z",), extras=[("z", "differential", ("u", "v"))]
    )
    old = {
        "question_id": "strong-old-pair",
        "property": "closed_subsystem",
        "target": "a",
        "scope": ["a", "z", "u"],
        "witnesses": [
            {
                "trajectories": ["left", "right"],
                "sample_indices": [1, 1],
                "times": [1, 1],
                "observations": [0, 3],
                "error_intervals_disjoint": True,
            }
        ],
    }
    runtime = audit.information_contract(packet, "causal_runtime")
    choices = initial.InitializerDependencyHandoff()
    current = audit.qualified_feedback(model, packet, runtime, choices, family=True)
    before = next(
        q for q in current["advisory_questions"] if q["property"] == "closed_subsystem"
    )
    assert before["witnesses"][0]["observations"] == [0, 1]
    assert before["witnesses"][0]["error_intervals_disjoint"] is False

    feedback = audit.qualified_feedback(
        model, packet, runtime, choices, family=True, previous_questions=[old]
    )
    state = build_state(
        feedback,
        {"evidence_sha256": packet.sha256, "advisory_questions": [old]},
        True,
    )
    assert feedback["saved_witness_recheck"]["checks"][0]["status"] == "preserved"
    assert state["repair_status"] == "repair_required"
    assert not state["baseline_outcomes"][0]["resolved"]
    question = next(
        q for q in state["pending_questions"] if q["property"] == "closed_subsystem"
    )
    assert question["witnesses"][0]["observations"] == [0, 3]
    assert question["witnesses"][0]["error_intervals_disjoint"] is True
