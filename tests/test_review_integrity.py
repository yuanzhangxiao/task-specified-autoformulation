"""Explicit parameter repair and stable, versioned incumbent selection."""

import copy
import json
import math

import pytest

from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.revision_decision import parameter_aliases
from autoformalism.schemas import CandidateModel
from autoformalism.search import response_revision as response
from autoformalism.search import review_revision_multi as edits
from autoformalism.search.review_integrity import (
    PARAMETER_POLICY,
    selection_decision,
    selection_policy,
)
from scripts.smoke_review_multi import example, fixture
from tests.test_response_evidence import response_example
from tests.test_response_revision import client, reply


def patch(name="c", role="time_constant"):
    return {
        "hypothesis": "Change the response timescale.",
        "equations": [{"component": "q", "expression": f"(i-q)/{name}"}],
        "new_parameters": [{"name": name, "role": role}],
    }


@pytest.mark.parametrize("use_alias", [False, True])
def test_conflicting_existing_roles_fail_atomically_with_actionable_feedback(use_alias):
    bundle, packet, params, *_ = example()
    before = copy.deepcopy(bundle)
    base = CandidateModel.model_validate(bundle["initialization"]["base_candidate"])
    alias = parameter_aliases(base)["c"]
    name = alias if use_alias else "c"
    raw = patch(name)
    with pytest.raises(ValueError) as caught:
        edits.apply_edits(bundle, packet, raw, reject_existing_role_conflicts=True)
    feedback = edits.feedback(bundle, packet, params, raw, caught.value)
    assert feedback["code"] == "EXISTING_PARAMETER_ROLE_CONFLICT"
    assert feedback["details"] == {
        "referenced_name": name,
        "canonical_name": "c",
        "display_name": alias,
        "existing_role": "rate",
        "requested_role": "time_constant",
        "incumbent_unchanged": True,
    }
    assert "fresh name" in feedback["message"]
    assert bundle == before
    # The proposer explicitly resolves the ambiguity with an independent parameter.
    result = edits.apply_edits(
        bundle, packet, patch("tau_new"), reject_existing_role_conflicts=True
    )
    model = result["bundle"]["initialization"]["base_candidate"]
    assert next(p for p in model["parameters"] if p["name"] == "c") == next(
        p
        for p in before["initialization"]["base_candidate"]["parameters"]
        if p["name"] == "c"
    )
    assert (
        next(p for p in model["parameters"] if p["name"] == "tau_new")["role"]
        == "time_constant"
    )
    assert bundle == before


@pytest.mark.parametrize(
    "declaration", [[], [{"name": "c"}], [{"name": "c", "role": "rate"}]]
)
def test_existing_references_still_inherit_exact_declarations(declaration):
    bundle, packet, *_ = example()
    raw = patch()
    raw["new_parameters"] = declaration
    result = edits.apply_edits(bundle, packet, raw, reject_existing_role_conflicts=True)
    params = result["bundle"]["initialization"]["base_candidate"]["parameters"]
    assert next(p for p in params if p["name"] == "c")["role"] == "rate"
    assert (
        result["provenance"]["parameter_declaration_audit"]["existing_parameter_policy"]
        == PARAMETER_POLICY
    )


def test_canonical_name_that_looks_like_display_alias_is_not_a_fresh_parameter():
    bundle, packet, *_ = example()
    first = edits.apply_edits(bundle, packet, patch("par_004"))
    bundle = first["bundle"]
    raw = patch("par_004", "nonnegative_coefficient")
    with pytest.raises(ValueError, match="immutable role") as caught:
        edits.apply_edits(bundle, None, raw, reject_existing_role_conflicts=True)
    assert caught.value.details["canonical_name"] == "par_004"


def test_legacy_conflicting_declaration_remains_replayable():
    bundle, packet, *_ = example()
    result = edits.apply_edits(bundle, packet, patch())
    audit = result["provenance"]["parameter_declaration_audit"]
    assert "existing_parameter_policy" not in audit
    assert audit["inherited_declarations_ignored"][0]["effective_role"] == "rate"


@pytest.mark.parametrize(
    "old,new,choose,reason",
    [
        (
            (0.9778491472623487, 8),
            (0.9778491472283518, 10),
            False,
            "within_tolerance_keep_incumbent",
        ),
        ((0.5, 8), (0.5 - 1e-10, 8), False, "within_tolerance_keep_incumbent"),
        ((0.5, 8), (0.5 + 1e-10, 7), True, "within_tolerance_fewer_terms"),
        ((0.5, 8), (0.5, 8), False, "within_tolerance_keep_incumbent"),
        ((0.5, 8), (0.49, 10), True, "validation_improved"),
        ((0.5, 8), (0.51, 7), False, "validation_worse"),
        ((0.0, 8), (1e-10, 7), True, "within_tolerance_fewer_terms"),
        ((1e6, 8), (1e6 - 0.5, 9), False, "within_tolerance_keep_incumbent"),
        ((math.inf, math.inf), (0.5, 8), True, "no_eligible_incumbent"),
        ((0.5, 8), (math.inf, math.inf), False, "trial_ineligible"),
        ((math.inf, math.inf), (math.inf, math.inf), False, "trial_ineligible"),
        ((0.5, 8), (math.nan, 7), False, "trial_ineligible"),
        ((0.5, 8), (None, 7), False, "trial_ineligible"),
        ((0.5, 8), (-0.1, 7), False, "trial_ineligible"),
    ],
)
def test_validation_band_and_complexity_decisions(old, new, choose, reason):
    audit = selection_decision(old, new)
    assert audit["choose_trial"] is choose and audit["reason"] == reason
    json.dumps(audit, allow_nan=False)


def test_selection_uses_validation_not_training_and_preserves_legacy():
    def selected(score, terms, *, eligible=True):
        return {
            "certificate": {"eligible_for_development_selection": eligible},
            "fit": {
                "training": {"available": True, "normalized_mse": 0.1},
                "validation": {"available": True, "normalized_mse": score},
            },
            "bundle": {
                "candidate": {
                    "state_equations": [{"rhs": "+".join(["x"] * terms)}],
                    "processes": [],
                }
            },
        }

    old, new = selected(0.5, 2), selected(0.5 - 1e-11, 3)
    new["fit"]["training"]["normalized_mse"] = 1e-15
    plan = {"protocol": io.INTEGRITY_PROTOCOL}
    assert pipeline.select_candidate(plan, old, new)[0] is old
    assert pipeline.select_candidate({"protocol": io.RESPONSE_PROTOCOL}, old, new) == (
        new,
        None,
    )
    assert (
        pipeline.select_candidate(plan, old, selected(0.01, 1, eligible=False))[0]
        is old
    )
    new["fit"]["validation"]["normalized_mse"] = 0.49
    new["fit"]["training"]["normalized_mse"] = 100
    assert pipeline.select_candidate(plan, old, new)[0] is new


def test_versioned_import_and_cached_explicit_repair_preserve_public_requirements(
    tmp_path,
):
    source, root = tmp_path / "source", tmp_path / "continued"
    original = fixture(source, fitted_only=True)
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    plan = continuation.prepare(source, root, 2, 2, protocol=io.INTEGRITY_PROTOCOL)
    assert plan["continuation"]["parameter_declaration_policy"] == PARAMETER_POLICY
    assert plan["continuation"]["selection_policy"] == selection_policy()
    assert plan["cells"] == original["cells"]  # No extra scientific hints or gates.
    assert (
        continuation.prepare(source, root, 2, 2, protocol=io.INTEGRITY_PROTOCOL) == plan
    )
    tampered = copy.deepcopy(plan)
    tampered["continuation"]["selection_policy"]["relative_tolerance"] = 1
    with pytest.raises(ValueError, match="integrity policies differ"):
        continuation.verify_imports(root, tampered)
    task = plan["tasks"][0]
    calls = []

    def generation(url, body, timeout):
        shown = json.loads(body["messages"][1]["content"])
        calls.append(shown)
        assert shown["parameter_declaration_policy"]["policy"] == PARAMETER_POLICY
        if len(calls) == 1:
            return reply(patch())
        assert shown["retry_feedback"]["code"] == "EXISTING_PARAMETER_ROLE_CONFLICT"
        return reply(patch("tau_new"))

    subject = client(
        tmp_path / "calls",
        generation,
        lambda *_: {"count": 5000, "max_model_len": 32768},
    )
    parent = io.read_round(root, task, 0)
    evidence = response_example()[3]
    decision = response.propose(plan, task, parent, subject, evidence)
    assert decision["status"] == "committed" and len(calls) == 2
    assert response.propose(plan, task, parent, subject, evidence) == decision
    assert len(calls) == 2
    assert {str(p): p.read_bytes() for p in source.rglob("*.json")} == before
