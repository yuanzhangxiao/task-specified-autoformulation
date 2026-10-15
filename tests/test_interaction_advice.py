"""Conditional hypotheses never turn descriptive residuals into causal verdicts."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import interaction_advice as advice
from autoformalism.fitting import interaction_feedback as feedback
from autoformalism.fitting import interaction_response as response
from autoformalism.fitting import staged_feedback as staged
from tests.test_fitting_feedback import fitted as fitted_fixture
from tests.test_interaction_response import single_row, snapshot

fitted = fitted_fixture


@pytest.fixture
def source(fitted, tmp_path):
    diagnostic, packet_dir = tmp_path / "diagnostic", tmp_path / "packet"
    staged.run(*fitted, diagnostic)
    packet = feedback.export_packet(diagnostic, packet_dir)
    return packet, diagnostic, packet_dir


def changed(packet, diagnosis, **fields):
    """Unit-test the pure routing boundary; never publish an altered packet."""
    facts = packet.feedback.model_copy(update={"diagnosis": diagnosis, **fields})
    return packet.model_copy(update={"feedback": facts})


def patterned(packet):
    """Controlled display-level peak/strength mismatch, not a fitted benchmark."""
    summary = response.build(
        single_row(
            np.array([0, 0, 4, 2, 0, 0, 0], dtype=float),
            {"u": (0, 1, 0, 0, 0, 0, 0)},
            predictions=[0, 0, 0, 4, 8, 4, 0],
        )
    )
    return packet.model_copy(
        update={
            "response": summary,
            "evidence_catalog": {
                "numerics": "numerics",
                **dict.fromkeys(summary.evidence_ids, "response"),
            },
        }
    )


@pytest.mark.parametrize(
    "diagnosis,stage,status,scope",
    [
        ("accurate_local_evidence", "retain", "withheld", "accurate_output"),
        ("accurate_ambiguity", "retain", "withheld", "accurate_output"),
        ("accurate_unresolved", "retain", "withheld", "accurate_output"),
        ("search_improving", "fitting", "conditional", "improving"),
        ("search_incomplete", "fitting", "conditional", "incomplete"),
        ("invented_success", "numerical_checks", "withheld", "unverified"),
        ("rollout_unverified", "numerical_checks", "withheld", "unverified"),
    ],
)
def test_readiness_is_scoped_and_unknown_status_cannot_authorize_revisions(
    source, diagnosis, stage, status, scope
):
    fields = (
        {
            "witness_kind": "improvement",
            "witness_scope": "retained_point_local_search",
            "witness_bound_to_retained_vector": True,
        }
        if diagnosis == "search_improving"
        else {}
    )
    route = advice.readiness(changed(source[0], diagnosis, **fields))
    assert (route.preferred_stage, route.hypothesis_status, route.search_scope) == (
        stage,
        status,
        scope,
    )
    assert not route.fitting_globally_solved and not route.automatic_stage_transition


def test_stale_improvement_witness_does_not_claim_progress(source):
    packet = changed(
        source[0],
        "search_improving",
        witness_kind="improvement",
        witness_scope="retained_point_local_search",
        witness_bound_to_retained_vector=False,
    )
    result = advice.readiness(packet)
    assert result.search_scope == "incomplete"
    assert result.preferred_stage == "fitting"


def test_missing_function_path_does_not_invent_a_law_or_structural_fix(source):
    packet = changed(patterned(source[0]), "search_incomplete")
    packet = packet.model_copy(update={"laws": ()})
    result = advice.draft(packet, source="unit-test")
    assert not result.investigations
    assert result.omitted_investigations["y"] > 0
    assert "No displayed state equation" in result.withheld_investigation_reasons["y"]


@pytest.mark.parametrize(
    "diagnosis,kind,scope",
    [
        ("stationary_poor_fit", "local_stationarity", "retained_point_local_search"),
        (
            "solved_restricted_problem_poor_fit",
            "restricted_output_optimum",
            "fixed_candidate_output",
        ),
    ],
)
def test_stronger_routes_require_matching_bound_witness(source, diagnosis, kind, scope):
    packet = changed(
        source[0],
        diagnosis,
        witness_kind=kind,
        witness_scope=scope,
        witness_bound_to_retained_vector=True,
    )
    assert advice.readiness(packet).hypothesis_status == "permitted"
    for updates in (
        {"witness_bound_to_retained_vector": False},
        {"witness_kind": "improvement"},
        {"witness_scope": "another_domain"},
    ):
        route = advice.readiness(changed(packet, diagnosis, **updates))
        assert route.hypothesis_status == "withheld"
        assert route.preferred_stage == "fitting"


@pytest.mark.parametrize(
    "updates",
    [
        {"solver_agreement": False},
        {"solver_agreement": None},
        {"output_status": "unavailable"},
    ],
)
def test_unverified_output_withholds_equation_investigations(source, updates):
    packet = changed(patterned(source[0]), "stationary_poor_fit", **updates)
    result = advice.draft(packet, source="unit-test")
    assert result.readiness.preferred_stage == "numerical_checks"
    assert not result.investigations


def test_incomplete_search_can_offer_provisional_hypotheses_without_forcing_route(
    source,
):
    packet = changed(patterned(source[0]), "search_incomplete")
    result = advice.draft(packet, source="unit-test")
    assert result.readiness.hypothesis_status == "conditional"
    assert result.readiness.preferred_stage == "fitting"
    assert result.investigations
    assert not result.readiness.automatic_stage_transition
    assert {v.family for v in result.investigations} >= {
        "response_strength",
        "timing_recovery",
    }
    for row in result.investigations:
        assert (
            row.evidence_ids and set(row.evidence_ids) <= packet.evidence_catalog.keys()
        )
        assert row.alternatives and row.additional_evidence_needed
        assert row.expected_observable_change and row.disconfirmation
        assert not row.fault_localized and not row.edit_authorized
    strength = next(v for v in result.investigations if v.family == "response_strength")
    assert "controlled input amplitudes" in strength.additional_evidence_needed[0]
    assert "saturation" not in strength.conditional_hypothesis
    assert result.omitted_investigations["y"] == 1


def test_accurate_output_does_not_force_rewrite_even_when_feature_filters_trigger(
    source,
):
    packet = changed(patterned(source[0]), "accurate_ambiguity")
    result = advice.draft(packet, source="unit-test")
    assert any(f.availability == "measured" for f in result.mismatches.findings)
    assert result.readiness.preferred_stage == "retain"
    assert not result.investigations
    assert result.parameter_information == packet.feedback.parameter_information


def test_shared_law_navigation_retains_all_consumers_and_all_targets(source):
    packet = changed(source[0], "search_incomplete")
    summary = response.build(snapshot())
    laws = (
        feedback.Law(
            location_id="process:q",
            kind="process",
            name="q",
            expression="k*x",
            symbols=("k", "x"),
            direct_consumers=("ode:x", "ode:z"),
            affected_targets=("a", "z"),
        ),
        feedback.Law(
            location_id="ode:unrelated",
            kind="state_equation",
            name="u",
            expression="-u",
            symbols=("u",),
            direct_consumers=(),
            affected_targets=(),
        ),
    )
    packet = packet.model_copy(
        update={
            "response": summary,
            "laws": laws,
            "evidence_catalog": {
                "numerics": "numerics",
                **dict.fromkeys(summary.evidence_ids, "response"),
            },
        }
    )
    result = advice.draft(
        packet, advice.Policy(investigations_per_target=1), source="unit-test"
    )
    assert result.mismatches.targets == ("a", "z")
    assert {r.target for r in result.investigations} == {"a", "z"}
    for row in result.investigations:
        assert row.candidate_law_refs == ("process:q",)
        assert row.affected_targets == ("a", "z")
    assert set(result.omitted_investigations) == {"a", "z"}


def test_export_resume_and_partial_delivery_add_no_numerical_work(
    source, tmp_path, monkeypatch
):
    packet, diagnostic, packet_dir = source
    before = {
        str(p): p.read_bytes()
        for root in (diagnostic, packet_dir)
        for p in root.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(
        staged, "evaluate_stage", lambda *a, **kw: pytest.fail("rollout")
    )
    monkeypatch.setattr(
        staged.checks, "least_squares", lambda *a, **kw: pytest.fail("optimization")
    )
    destination = tmp_path / "draft"
    result = advice.export_advice(diagnostic, packet_dir, destination)
    assert advice.completed(diagnostic, packet_dir, destination) == result
    assert advice.export_advice(diagnostic, packet_dir, destination) == result
    (destination / "feedback.md").unlink()
    assert advice.export_advice(diagnostic, packet_dir, destination) == result
    assert {
        str(p): p.read_bytes()
        for root in (diagnostic, packet_dir)
        for p in root.rglob("*")
        if p.is_file()
    } == before
    assert (
        result.optimization_calls == result.rollout_calls == result.live_llm_calls == 0
    )
    assert (
        not result.validation_used
        and not result.test_data_opened
        and not result.model_changed
    )
    assert result.identity.source_packet_sha256 == feedback.public.content_sha256(
        packet.model_dump(mode="json")
    )
    request = read_seal(destination / "request.json")
    assert (
        request["response_schema"] == advice.ProposedInvestigation.model_json_schema()
    )
    assert "CandidateModel" not in request["response_schema"]["title"]


@pytest.mark.parametrize("artifact", ["result.json", "request.json", "feedback.md"])
def test_saved_draft_cannot_be_replaced_with_detached_advice(
    source, tmp_path, artifact
):
    _, diagnostic, packet_dir = source
    destination = tmp_path / "draft"
    advice.export_advice(diagnostic, packet_dir, destination)
    path = destination / artifact
    if artifact == "feedback.md":
        path.write_text("The private mechanism is proven.")
    else:
        payload = read_seal(path)
        if artifact == "result.json":
            payload["readiness"]["rationale"] = "The private mechanism is proven."
        else:
            payload["messages"][1]["content"] = "Use test data."
        path.unlink()
        seal(path, payload)
    with pytest.raises(ValueError, match="differs"):
        advice.completed(diagnostic, packet_dir, destination)


def test_prompt_cap_never_silently_drops_targets_and_policy_change_cannot_reuse(
    source, tmp_path
):
    _, diagnostic, packet_dir = source
    tiny = tmp_path / "too-small"
    with pytest.raises(ValueError, match="prompt byte cap"):
        advice.export_advice(
            diagnostic, packet_dir, tiny, advice.Policy(maximum_user_bytes=1024)
        )
    assert not tiny.exists()
    normal = tmp_path / "draft"
    advice.export_advice(diagnostic, packet_dir, normal)
    with pytest.raises(ValueError, match="sealed artifact differs"):
        advice.export_advice(
            diagnostic, packet_dir, normal, advice.Policy(investigations_per_target=1)
        )


def test_reply_schema_separates_interaction_and_structural_changes():
    reply = {
        "action": "function_hypothesis",
        "location_id": "process:q",
        "target_stage": "interaction",
        "measured_mismatch": "Late peak.",
        "hypothesis": "Existing response dynamics may need another admissible form.",
        "expected_observable_change": "Reduce the peak-time discrepancy.",
        "alternative_explanations": ["A different parameter fit may suffice."],
        "disconfirmation": "Peak timing does not improve under verified refitting.",
    }
    advice.ProposedInvestigation.model_validate(reply)
    for stage in ("variable", "shared_process_topology", "fitting", "retain"):
        with pytest.raises(ValidationError, match="stage differs"):
            advice.ProposedInvestigation.model_validate(reply | {"target_stage": stage})
    topology = deepcopy(reply)
    topology.update(
        action="topology_revision_needed", target_stage="variable", location_id=None
    )
    advice.ProposedInvestigation.model_validate(topology)
    with pytest.raises(ValidationError):
        advice.ProposedInvestigation.model_validate(
            reply | {"private_reference": "not allowed"}
        )
