"""Verified, bounded training advice without equation edits or solver calls."""

import copy
import json

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import interaction_feedback as bridge
from autoformalism.fitting import staged_feedback as staged
from autoformalism.schemas.candidate import CandidateModel
from tests.test_fitting_feedback import fitted as fitted_fixture

fitted = fitted_fixture


@pytest.fixture
def diagnostic(fitted, tmp_path):
    folder = tmp_path / "diagnostic"
    staged.run(*fitted, folder)
    return folder


def reseal(path, value):
    path.unlink()
    seal(path, value)


def test_real_snapshot_export_and_resume_without_model_execution(
    diagnostic, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        staged, "evaluate_stage", lambda *a, **kw: pytest.fail("new rollout")
    )
    source_before = {str(p): p.read_bytes() for p in diagnostic.rglob("*.json")}
    out = tmp_path / "export"
    packet = bridge.export_packet(diagnostic, out)
    assert packet.status == "ready_for_advisory_review"
    assert packet.feedback.diagnosis == "solved_restricted_problem_poor_fit"
    assert packet.feedback.witness_bound_to_retained_vector
    assert packet.feedback.specification_verdict == "not_assessed"
    assert packet.edit_contract_status == "requires_constructor_binding"
    assert (
        packet.new_optimization_calls
        == packet.new_rollout_calls
        == packet.live_llm_calls
        == 0
    )
    assert bridge.completed(diagnostic, out) == packet
    assert bridge.export_packet(diagnostic, out) == packet
    assert {str(p): p.read_bytes() for p in diagnostic.rglob("*.json")} == source_before
    user = json.loads(bridge.messages(packet)[1]["content"])
    assert "candidate_id" not in user["model_view"]
    assert "source" not in user["model_view"]
    assert "backend" not in user and "sampled_states" not in user
    assert not user["validation_used"] and not user["test_data_opened"]
    assert set(packet.response.evidence_ids) <= set(packet.evidence_catalog)


def test_partial_delivery_can_resume_without_fitting(diagnostic, tmp_path):
    out = tmp_path / "export"
    packet = bridge.export_packet(diagnostic, out)
    (out / "request.json").unlink()
    assert bridge.export_packet(diagnostic, out) == packet
    assert bridge.completed(diagnostic, out) == packet


@pytest.mark.parametrize("file", ["result.json", "request.json"])
def test_resealed_payload_tampering_is_rejected(diagnostic, tmp_path, file):
    out = tmp_path / "export"
    bridge.export_packet(diagnostic, out)
    raw = read_seal(out / file)
    if file == "result.json":
        raw["feedback"]["diagnosis"] = "accurate_local_evidence"
    else:
        raw["messages"][1]["content"] = "Use evaluator coefficients instead"
    reseal(out / file, raw)
    with pytest.raises(ValueError, match="differs"):
        bridge.completed(diagnostic, out)


@pytest.mark.parametrize("field", ["validation", "test", "reference_parameters"])
def test_evaluator_fields_in_source_rejected(diagnostic, tmp_path, field):
    path = diagnostic / "context.json"
    raw = read_seal(path)
    raw["problem"][field] = {"hidden": 1}
    reseal(path, raw)
    with pytest.raises(ValueError):
        bridge.export_packet(diagnostic, tmp_path / "export")


def test_saved_retained_vector_substitution_is_rejected(diagnostic, tmp_path):
    path = diagnostic / "context.json"
    raw = read_seal(path)
    raw["fit"]["parameters"]["a"] += 1
    reseal(path, raw)
    with pytest.raises(ValueError, match=r"identity|vector"):
        bridge.export_packet(diagnostic, tmp_path / "export")


def test_missing_alternate_does_not_produce_verified_response(
    fitted, tmp_path, monkeypatch
):
    evaluate = staged.evaluate_stage

    def fail(oracle, vector, deadline, checkpoint, method, sensitivities, **kwargs):
        if method == "Radau":
            checkpoint({"calls": 1, "calls_completed": 0, "failure_kind": "timeout"})
            raise TimeoutError("unavailable alternate")
        return evaluate(
            oracle, vector, deadline, checkpoint, method, sensitivities, **kwargs
        )

    monkeypatch.setattr(staged, "evaluate_stage", fail)
    source = tmp_path / "diagnostic"
    staged.run(*fitted, source)
    packet = bridge.export_packet(source, tmp_path / "export")
    assert packet.status == "numerical_evidence_unavailable"
    assert packet.response.status == "unavailable"
    assert not packet.response.examples
    assert set(packet.response.targets) == set(fitted[0].request.context.targets)


def test_prompt_cap_fails_explicitly_without_silent_target_truncation(
    diagnostic, tmp_path
):
    with pytest.raises(ValueError, match="prompt byte cap"):
        bridge.export_packet(
            diagnostic, tmp_path / "export", bridge.Policy(maximum_user_bytes=1024)
        )
    assert not (tmp_path / "export/result.json").exists()


def test_policy_change_cannot_reuse_export(diagnostic, tmp_path):
    out = tmp_path / "export"
    bridge.export_packet(diagnostic, out)
    with pytest.raises(ValueError, match="sealed artifact differs"):
        bridge.export_packet(diagnostic, out, bridge.Policy(examples_per_target=1))


def test_citation_mistakes_recorded_without_equation_or_scientific_acceptance(
    diagnostic, tmp_path
):
    packet = bridge.export_packet(diagnostic, tmp_path / "export")
    before = copy.deepcopy(packet.model_dump(mode="json"))
    reply = bridge.Hypothesis(
        action="function_hypothesis",
        location_id="ode:z",
        measured_mismatch="Displayed residual remains nonzero.",
        hypothesis="A different allowed response law may help.",
        expected_observable_change="A smaller discrepancy in the cited trajectory.",
        evidence_ids=("numerics", "missing", "numerics"),
    )
    receipt = bridge.audit_hypothesis(packet, reply)
    assert receipt["known_references"] == ["numerics"]
    assert receipt["unverified_references"] == ["missing"]
    assert not receipt["model_changed"] and not receipt["scientific_support_verified"]
    assert receipt["status"] == "recorded_with_citation_warnings"
    assert packet.model_dump(mode="json") == before
    with pytest.raises(ValueError, match="undisplayed"):
        bridge.audit_hypothesis(
            packet, reply.model_copy(update={"location_id": "ode:missing"})
        )
    with pytest.raises(ValueError, match="topology_revision_needed"):
        bridge.audit_hypothesis(
            packet, reply.model_copy(update={"location_id": "output:y"})
        )
    request = reply.model_copy(
        update={"action": "topology_revision_needed", "location_id": None}
    )
    assert not bridge.audit_hypothesis(packet, request)["model_changed"]


def test_shared_process_dependencies_include_all_targets_without_inventing_signs():
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "shared",
            "parent_candidate_id": None,
            "states": [{"name": n, "kind": "observed"} for n in ("x", "z")],
            "state_equations": [
                {"state": "x", "rhs": "-q"},
                {"state": "z", "rhs": "q"},
            ],
            "processes": [{"name": "q", "expression": "a*x"}],
            "parameters": [{"name": "a", "role": "coefficient", "scope": "global"}],
            "observation_mappings": [
                {"channel": "y", "expression": "x"},
                {"channel": "w", "expression": "z"},
            ],
        }
    )
    law = next(law for law in bridge._laws(candidate) if law.location_id == "process:q")
    assert set(law.direct_consumers) == {"ode:x", "ode:z"}
    assert set(law.affected_targets) == {"y", "w"}
    assert "sign" not in law.model_dump()
    assert law.scope == "syntactic_dependency_not_causal_blame"


def test_observation_mapping_does_not_change_state_with_matching_name():
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "direct_observation",
            "parent_candidate_id": None,
            "states": [{"name": n, "kind": "observed"} for n in ("x", "z")],
            "state_equations": [
                {"state": "x", "rhs": "-x"},
                {"state": "z", "rhs": "x-z"},
            ],
            "observation_mappings": [
                {"channel": "x", "expression": "x"},
                {"channel": "y", "expression": "z"},
            ],
        }
    )
    laws = {law.location_id: law for law in bridge._laws(candidate)}
    assert laws["output:x"].affected_targets == ("x",)
    assert set(laws["ode:x"].affected_targets) == {"x", "y"}
