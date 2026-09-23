"""Equivalent explicit signs, version isolation, and scientific direction limits."""

import json

import pytest

from autoformalism.expressions import RestrictedParser
from autoformalism.judging import atomic_role_compatibility_assessments
from autoformalism.judging.hybrid import (
    FACTOR_SIGN_POLICY,
    LEGACY_SIGN_POLICY,
    _algebraic_expression_facts,
    build_atomic_evidence_plan,
    structural_facts,
)
from autoformalism.rebuttal import repair_scientific_judge as judge
from autoformalism.rebuttal.repair_judge_evidence import named_references
from autoformalism.schemas import AtomicJudgeResult, CandidateModel
from tests.test_search_hybrid_pair import _candidate


def facts(expression, policy=FACTOR_SIGN_POLICY):
    return _algebraic_expression_facts(
        expression,
        location="test",
        parser=RestrictedParser(),
        sign_policy=policy,
    )["top_level_additive_terms"]


@pytest.mark.parametrize(
    "source, equivalent, polarity, unsigned",
    [
        ("-k*x", "-(k*x)", "negative", "k * x"),
        ("-z/tau", "-(z/tau)", "negative", "z / tau"),
        ("z/(-tau)", "-(z/tau)", "negative", "z / tau"),
        ("(-k)*(-x)", "k*x", "positive", "k * x"),
        ("-((-k)*x)", "k*x", "positive", "k * x"),
        ("(-k)*(-x)/(-tau)", "-(k*x/tau)", "negative", "k * x / tau"),
        ("-2*x", "-(2*x)", "negative", "2 * x"),
        ("a-(-k*x)", "a+k*x", "positive", "k * x"),
        ("-k*(x-y)", "-(k*(x-y))", "negative", "k * (x - y)"),
    ],
)
def test_product_quotient_sign_parity(source, equivalent, polarity, unsigned):
    result = facts(source)
    assert result == facts(equivalent)
    assert result[-1]["polarity"] == polarity
    assert result[-1]["normalized_expression"] == unsigned


@pytest.mark.parametrize(
    "expression",
    [
        "tanh(-x)",
        "(-x)**2",
        "k*(x-y)",
        "1*(Tj-Tjf)",
        "max(0, x-crest)",
    ],
)
def test_internal_signs_and_exchange_directions_stay_opaque(expression):
    assert facts(expression) == facts(expression, LEGACY_SIGN_POLICY)


def test_legacy_default_and_occurrence_ids_are_preserved():
    a, b = _candidate("a", "-2*Gp"), _candidate("b", "-(2*Gp)")
    payload = a.model_dump(mode="json")
    legacy = build_atomic_evidence_plan(a, b)
    assert legacy == build_atomic_evidence_plan(a, b, sign_policy=LEGACY_SIGN_POLICY)
    assert [o.actual_polarity for o in legacy.occurrences] == ["positive", "negative"]
    new = build_atomic_evidence_plan(a, b, sign_policy=FACTOR_SIGN_POLICY)
    assert all(o.actual_polarity == "negative" for o in new.occurrences)
    assert all(o.unsigned_expression == "2 * Gp" for o in new.occurrences)
    assert new == build_atomic_evidence_plan(b, b, sign_policy=FACTOR_SIGN_POLICY)
    assert a.model_dump(mode="json") == payload
    assert "negative" not in json.dumps(new.prompt_payload())
    assert (
        new.prompt_payload()["schema_version"]
        != legacy.prompt_payload()["schema_version"]
    )


def test_process_terms_repeat_groups_and_named_references_use_same_policy():
    raw = _candidate("a", "p").model_dump(mode="json")
    raw["processes"] = [{"name": "p", "expression": "-2*Gp - (2*Gp)"}]
    model = CandidateModel.model_validate(raw)
    old = build_atomic_evidence_plan(model, model)
    new = build_atomic_evidence_plan(model, model, sign_policy=FACTOR_SIGN_POLICY)
    assert not old.repeat_candidates and len(new.repeat_candidates) == 2
    term = next(
        o
        for o in new.occurrences
        if o.candidate_side == "candidate_b" and o.equation_location == "process:p"
    )
    refs = named_references(
        model, model, term.occurrence_id, sign_policy=FACTOR_SIGN_POLICY
    )
    assert refs["named_references"][0]["actual_outer_polarity"] == "negative"
    assert refs["unresolved_reference_ids"] == []
    result = structural_facts(
        model,
        task_inputs=(),
        include_model_semantics=True,
        sign_policy=FACTOR_SIGN_POLICY,
    )
    assert FACTOR_SIGN_POLICY in result["schema_version"]
    assert (
        result["algebraic_expressions"]["process:p"]["top_level_additive_terms"][0][
            "polarity"
        ]
        == "negative"
    )


def test_true_wrong_direction_remains_a_failure():
    a, b = _candidate("a", "-2*Gp"), _candidate("b", "2*Gp")
    plan = build_atomic_evidence_plan(a, b, sign_policy=FACTOR_SIGN_POLICY)
    atomic = AtomicJudgeResult.model_validate(
        {
            "signed_occurrence_assessments": [
                {
                    "occurrence_id": o.occurrence_id,
                    "expected_direction": "negative_contribution",
                    "evidence": "Synthetic sink role.",
                }
                for o in plan.occurrences
            ],
            "repeated_contribution_assessments": [],
        }
    )
    roles = atomic_role_compatibility_assessments(atomic, plan)
    sink = next(r for r in roles if r.criterion.value == "sink_roles_consistent")
    assert sink.candidate_a.verdict.value == "pass"
    assert sink.candidate_b.verdict.value == "fail"


def test_protocol_is_opt_in_and_unknown_versions_fail():
    original = judge.judge_protocol()
    revised = judge.judge_protocol(sign_policy=FACTOR_SIGN_POLICY)
    assert revised.pop("sign_evidence_policy") == FACTOR_SIGN_POLICY
    assert revised == original
    with pytest.raises(ValueError, match="unknown"):
        facts("x", "invented")
    with pytest.raises(ValueError, match="unknown"):
        judge.judge_protocol(sign_policy="invented")
