"""Optional joint-effect questions preserve numerical and scientific uncertainty."""

from copy import deepcopy

import pytest

from autoformalism.fitting import interaction_dependency_advice as dependency
from autoformalism.fitting import interaction_feedback as feedback
from autoformalism.fitting import interaction_response as response
from autoformalism.fitting import staged_feedback as staged
from autoformalism.schemas.candidate import ProcessSpec
from tests.test_fitting_feedback import fitted as fitted_fixture
from tests.test_interaction_advice import changed, patterned
from tests.test_interaction_response import snapshot

fitted = fitted_fixture


@pytest.fixture
def source(fitted, tmp_path):
    """Real verified evidence; pure projection mutations below are never published."""
    diagnostic, packet_dir = tmp_path / "diagnostic", tmp_path / "packet"
    staged.run(*fitted, diagnostic)
    return feedback.export_packet(diagnostic, packet_dir), diagnostic, packet_dir


def candidate(packet, expression="a*u-b*y", processes=()):
    """Supply a synthetic law for expression/routing unit tests, without refitting."""
    view = packet.model_view.model_copy(
        update={"processes": tuple(ProcessSpec.model_validate(p) for p in processes)}
    )
    law = feedback.Law(
        location_id="ode:y",
        kind="state_equation",
        name="y",
        expression=expression,
        symbols=(),
        direct_consumers=("output:y",),
        affected_targets=("y",),
    )
    return packet.model_copy(update={"model_view": view, "laws": (law,)})


def test_verified_entry_default_disabled_read_only_and_rejects_tampering(
    source, monkeypatch
):
    packet, diagnostic, packet_dir = source
    before = {str(p): p.read_bytes() for p in diagnostic.parent.rglob("*.json")}
    monkeypatch.setattr(
        staged, "evaluate_stage", lambda *a, **kw: pytest.fail("new numerical work")
    )
    result = dependency.review(diagnostic, packet_dir)
    assert result.status == "disabled" and not result.questions
    assert result.source_verification == "verified_snapshot"
    enabled = dependency.review(diagnostic, packet_dir, dependency.Policy(enabled=True))
    assert enabled.status == "inapplicable"  # a*u has only one dynamic argument.
    assert result.source_packet_sha256 == enabled.source_packet_sha256
    assert {str(p): p.read_bytes() for p in diagnostic.parent.rglob("*.json")} == before
    assert dependency.review(diagnostic, packet_dir) == result
    (packet_dir / "result.json").write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        dependency.review(diagnostic, packet_dir, dependency.Policy(enabled=True))
    assert packet.validation_used is False


def test_qualified_discrepancy_offers_question_not_diagnosis(source):
    packet = candidate(patterned(source[0]))
    before = deepcopy(packet.model_dump())
    result = dependency.draft(packet, dependency.Policy(enabled=True))
    assert result.status == "investigation_available"
    assert result.source_verification == "caller_required"
    assert len(result.questions) == 1
    question = result.questions[0]
    assert question.law_refs == ("ode:y",)
    assert set(question.evidence_ids) <= packet.evidence_catalog.keys()
    assert question.affected_targets == ("y",)
    assert not question.missing_dependency_diagnosed and not question.edit_authorized
    assert "No new argument" in question.question
    assert "predetermined equal budgets" in question.required_comparison
    assert "initialization" in " ".join(question.alternatives)
    assert "not global model adequacy" in result.limitations[0]
    assert not result.model_changed and not result.validation_used
    assert result.optimization_calls == result.rollout_calls == result.llm_calls == 0
    assert packet.model_dump() == before


@pytest.mark.parametrize(
    "diagnosis,updates",
    [
        ("accurate_local_evidence", {}),
        ("search_improving", {}),
        ("search_incomplete", {}),
        ("solved_restricted_problem_poor_fit", {"solver_agreement": False}),
        (
            "solved_restricted_problem_poor_fit",
            {"witness_bound_to_retained_vector": False},
        ),
        ("solved_restricted_problem_poor_fit", {"witness_scope": "other_candidate"}),
    ],
)
def test_absent_or_inapplicable_numerical_qualification_withholds(
    source, diagnosis, updates
):
    packet = changed(candidate(patterned(source[0])), diagnosis, **updates)
    result = dependency.draft(packet, dependency.Policy(enabled=True))
    assert result.status == "withheld" and not result.questions


def test_local_stationarity_is_scoped_not_a_global_failure(source):
    packet = changed(
        candidate(patterned(source[0])),
        "stationary_poor_fit",
        witness_kind="local_stationarity",
        witness_scope="retained_point_local_search",
        witness_bound_to_retained_vector=True,
    )
    result = dependency.draft(packet, dependency.Policy(enabled=True))
    assert result.status == "investigation_available"
    assert result.readiness.search_scope == "retained_point_stationary"
    assert "different fitting starts" in result.questions[0].alternatives[0]
    assert not result.readiness.fitting_globally_solved


@pytest.mark.parametrize(
    "expression,processes,status,arguments",
    [
        ("a*u-b*y", (), "additive_unary", ("u", "y")),
        ("a*(u-y)", (), "additive_unary", ("u", "y")),
        ("a*u*y", (), "joint_or_unresolved", ("u", "y")),
        ("max(u-y,0)", (), "joint_or_unresolved", ("u", "y")),
        ("a*z", (), "single_dynamic_source", ("z",)),
        ("q", ({"name": "q", "expression": "u-y"},), "additive_unary", ("u", "y")),
        (
            "tanh(q)",
            ({"name": "q", "expression": "u+y"},),
            "joint_or_unresolved",
            ("u", "y"),
        ),
        ("q", ({"name": "q", "expression": "u*y"},), "joint_or_unresolved", ("u", "y")),
        ("q", ({"name": "q", "expression": "q+u"},), "joint_or_unresolved", ()),
        ("unlisted+y", (), "joint_or_unresolved", ()),
        ("t+y", (), "joint_or_unresolved", ()),
        ("__import__('os')", (), "joint_or_unresolved", ()),
    ],
)
def test_effective_arguments_handle_composition_and_ode_leaves(
    source, expression, processes, status, arguments
):
    packet = candidate(patterned(source[0]), expression, processes)
    result = dependency.draft(packet, dependency.Policy(enabled=True))
    inspection = result.inspections[0]
    assert inspection.status == status
    assert inspection.dynamic_arguments == arguments
    assert bool(result.questions) == (status == "additive_unary")


def test_initial_channel_dependency_does_not_create_continuous_rhs_argument(source):
    packet = candidate(patterned(source[0]), "a*z")
    payload = packet.model_view.initialization_plan.model_dump(mode="json")
    payload["rules"]["z"] = {
        "initial": {
            "mode": "map",
            "expression": "s*y",
            "parameters": [{"name": "s", "role": "scale", "guess": 0.2}],
        }
    }
    plan = type(packet.model_view.initialization_plan).model_validate(payload)
    packet = packet.model_copy(
        update={
            "model_view": packet.model_view.model_copy(
                update={"initialization_plan": plan}
            )
        }
    )
    result = dependency.draft(packet, dependency.Policy(enabled=True))
    assert result.initialization_dependencies["z"] == ("y",)
    assert result.inspections[0].dynamic_arguments == ("z",)
    assert not result.questions


def test_fixed_covariate_does_not_make_unary_discharge_joint(source):
    packet = candidate(patterned(source[0]), "a*max(y-crest,0)/area")
    roles = packet.model_view.public_channel_roles.model_copy(
        update={"fixed_covariates": ("crest", "area")}
    )
    packet = packet.model_copy(
        update={
            "model_view": packet.model_view.model_copy(
                update={"public_channel_roles": roles}
            )
        }
    )
    result = dependency.draft(packet, dependency.Policy(enabled=True))
    assert result.inspections[0].dynamic_arguments == ("y",)
    assert result.status == "inapplicable" and not result.questions


def test_named_t_parameter_is_not_assumed_to_be_time(source):
    packet = candidate(patterned(source[0]), "t+y")
    parameter = packet.model_view.parameters[0].model_copy(update={"name": "t"})
    view = packet.model_view.model_copy(update={"parameters": (parameter,)})
    packet = packet.model_copy(update={"model_view": view})
    result = dependency.draft(packet, dependency.Policy(enabled=True))
    assert result.inspections[0].dynamic_arguments == ("y",)
    assert not result.questions


def test_no_measured_discrepancy_or_missing_citation_never_creates_question(source):
    packet = candidate(patterned(source[0]))
    policy = dependency.Policy(
        enabled=True,
        extraction={
            "normalized_difference_threshold": 1e9,
            "time_difference_fraction": 1.0,
            "window_nmse_threshold": 1e9,
        },
    )
    assert not dependency.draft(packet, policy).questions
    packet = packet.model_copy(update={"evidence_catalog": {}})
    with pytest.raises(ValueError, match="evidence reference is absent"):
        dependency.draft(packet, dependency.Policy(enabled=True))


def test_question_limit_preserves_all_shared_consumers_and_counts_omissions(source):
    packet = candidate(source[0])
    summary = response.build(snapshot())
    law = packet.laws[0].model_copy(update={"affected_targets": ("a", "z")})
    packet = packet.model_copy(
        update={
            "laws": (law,),
            "response": summary,
            "evidence_catalog": {
                "numerics": "numerics",
                **dict.fromkeys(summary.evidence_ids, "response"),
            },
        }
    )
    limited = dependency.draft(packet, dependency.Policy(enabled=True))
    full = dependency.draft(
        packet, dependency.Policy(enabled=True, maximum_questions=2)
    )
    assert len(limited.questions) == limited.omitted_questions == 1
    assert limited.questions == full.questions[:1]
    assert full.omitted_questions == 0
    assert {q.target for q in full.questions} == {"a", "z"}
    assert all(q.affected_targets == ("a", "z") for q in full.questions)
    assert limited == dependency.draft(packet, dependency.Policy(enabled=True))
