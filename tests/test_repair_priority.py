"""Evidence routing under disagreement, missing provenance and bounded fit failure."""

from copy import deepcopy

import pytest

from autoformalism.rebuttal import repair_comparison as campaign
from autoformalism.rebuttal.repair_evidence import (
    Finding,
    decision_report,
    model_hash,
    numerical_findings,
)
from autoformalism.rebuttal.repair_priority import numerical_reliability
from tests.test_repair_comparison import candidate


def concern():
    return Finding(
        source="scientific_judge",
        stage="prefit",
        candidate_sha256=model_hash(candidate()),
        category="scientific_requirement",
        code="mechanism",
        component="m",
        certainty="advisory",
        observation="Potential mechanism defect",
        recheck="paired scientific review",
        evidence={"request_sha256": "review"},
    )


def paired_review():
    return {
        "status": "reviewed",
        "request_sha256": "review",
        "comparison": {
            "protocol_version": (
                "incumbent-hybrid-question-consensus-4-blinded-metadata"
            ),
            "request_hashes": ["a", "b", "c", "d"],
            "consensus_result": {
                "absolute_assessments": [
                    {
                        "criterion": "mechanism",
                        "subject_id": "m",
                        "candidate_b": {
                            "verdict": "fail",
                            "evidence": concern().observation,
                        },
                    }
                ]
            },
        },
    }


def fitted(*, converged=True):
    point = {"rate": 0.3}
    return {
        "status": "complete",
        "training": {"normalized_mse": 0.8, "failed_trajectories": []},
        "validation": {"normalized_mse": 0.9},
        "refinement": {
            "method": "feasibility_then_refinement",
            "parameters": point,
            "production_training_rollout_verified": True,
            "stages": [
                {
                    "mode": "sensitivity",
                    "result": {
                        "optimizer_native_success": converged,
                        "optimizer_success": converged,
                        "parameters": point,
                    },
                }
            ],
        },
    }


def report(fit, science=(), runtime=(), review=None):
    return decision_report(
        candidate(),
        list(runtime),
        list(science),
        numerical_findings(candidate(), fit),
        [],
        routing_policy="evidence_strength",
        fit=fit,
        scientific_review=review,
    )


def test_paired_absolute_concern_outranks_timeout_without_declaring_infeasibility():
    result = report({"status": "fit_failed"}, [concern()], review=paired_review())
    assert result["objective_category"] == "scientific_requirement"
    assert not result["numerical_reliability"]["global_infeasibility_established"]
    assert (
        result["numerical_reliability"]["search_opportunity"]["budget_exhausted"]
        is None
    )


def test_verified_selected_local_fit_outranks_unsupported_advice():
    result = report(fitted(), [concern()])
    assert result["objective_category"] == "prediction_refinement"
    assert result["numerical_reliability"]["support"]["strength"] == "corroborated"


def test_equal_support_keeps_both_in_focus():
    result = report(fitted(), [concern()], review=paired_review())
    assert result["objective_category"] == "joint_review"
    assert result["priority_evidence"]["selected_categories"] == [
        "prediction_refinement",
        "scientific_requirement",
    ]
    assert len(result["focus"]) == 2


def test_identical_cached_self_pair_is_not_extra_corroboration():
    review = paired_review()
    review["comparison"]["request_hashes"] = ["a", "b", "a", "b"]
    result = report(fitted(), [concern()], review=review)
    assert result["objective_category"] == "prediction_refinement"
    row = result["priority_evidence"]["ranked_findings"][0]
    assert row["support"]["strength"] == "observed"
    assert row["support"]["basis"] == ["same_requests_reused_across_orientations"]


def test_deterministic_blocker_precedes_both_and_unresolved_risk_does_not():
    blocker = Finding(
        source="runtime",
        stage="prefit",
        candidate_sha256=model_hash(candidate()),
        category="contract",
        code="UNAVAILABLE_SYMBOL",
        certainty="observed",
        blocking=True,
        observation="Unavailable forcing",
        recheck="compile",
    )
    result = report(fitted(), [concern()], [blocker], paired_review())
    assert result["objective_category"] == "runtime_contract"
    risk = blocker.model_copy(update={"blocking": False, "certainty": "unresolved"})
    assert (
        report(fitted(), runtime=[risk])["objective_category"]
        == "prediction_refinement"
    )


@pytest.mark.parametrize(
    "mutation", ["request", "missing", "verdict", "protocol", "evidence"]
)
def test_unverified_scientific_support_is_never_promoted(mutation):
    review = paired_review()
    pair = review["comparison"]
    assessment = pair["consensus_result"]["absolute_assessments"][0]["candidate_b"]
    if mutation == "request":
        review["request_sha256"] = "other"
    elif mutation == "missing":
        pair["request_hashes"].pop()
    elif mutation == "protocol":
        pair["protocol_version"] = "unknown"
    elif mutation == "evidence":
        assessment["evidence"] = "unrelated finding"
    else:
        assessment["verdict"] = "indeterminate"
    result = report(fitted(converged=False), [concern()], review=review)
    assert result["objective_category"] == "prediction_refinement"


def test_converged_other_point_does_not_certify_selected_point():
    fit = fitted()
    fit["refinement"]["stages"][0]["result"]["parameters"] = {"rate": 1.0}
    support = numerical_reliability(fit)
    assert not support["selected_optimizer_converged"]
    assert support["support"]["strength"] == "observed"


@pytest.mark.parametrize("mutation", ["pending", "flag", "metric", "failed"])
def test_complete_label_alone_does_not_verify_replay(mutation):
    fit = fitted()
    if mutation == "pending":
        fit["status"] = "verification_pending"
    elif mutation == "flag":
        fit["refinement"].pop("production_training_rollout_verified")
    elif mutation == "metric":
        fit["training"]["normalized_mse"] = float("nan")
    else:
        fit["training"]["failed_trajectories"] = ["train_0"]
    assert numerical_reliability(fit)["support"]["strength"] == "unresolved"


def test_augmented_failure_is_evidence_of_numerical_difficulty_at_trial_points():
    fit = {
        "status": "fit_failed",
        "refinement": {
            "state_integration_succeeded": True,
            "augmented_integration_failed": True,
            "actual_residual_calls": 8,
            "budget_exhausted": True,
        },
    }
    evidence = numerical_reliability(fit)
    assert evidence["support"]["strength"] == "observed"
    assert evidence["support"]["scope"] == "numerical_search"
    assert not evidence["selected_training_replay_verified"]
    assert not evidence["global_infeasibility_established"]


def test_validation_results_do_not_set_confidence_or_route():
    fit = fitted()
    changed = deepcopy(fit)
    changed["validation"] = {"normalized_mse": 1e9, "failed_trajectories": ["val_0"]}
    assert numerical_reliability(fit) == numerical_reliability(changed)


def test_stale_findings_remain_visible_but_cannot_set_current_priority():
    stale = concern().model_copy(update={"candidate_sha256": "older_model"})
    result = report(fitted(), [stale], review=paired_review())
    assert result["objective_category"] == "prediction_refinement"
    assert (
        result["priority_evidence"]["excluded_findings"][0]["reason"]
        == "different_candidate"
    )
    assert result["scientific_findings"][0]["candidate_sha256"] == "older_model"


def test_legacy_policy_stays_unchanged_and_unknown_policy_fails():
    model = candidate()
    args = (model, [], [concern()], numerical_findings(model, fitted()), [])
    assert decision_report(*args)["objective_category"] == "scientific_requirement"
    assert "priority_evidence" not in decision_report(*args)
    with pytest.raises(ValueError, match="unknown routing"):
        decision_report(*args, routing_policy="made_up")
    with pytest.raises(ValueError, match="protocol and routing"):
        campaign.RepairComparisonConfig(
            judge_revision="a" * 40, routing_policy="evidence_strength"
        )


def test_freeze_binds_new_policy_and_refuses_changed_resume(tmp_path, monkeypatch):
    inputs = {
        "tasks": [],
        "plan_sha256": "inputs",
        "config": {
            "model_settings": {},
            "serving_image_sha256": "image",
            "public_obligation_quote": "memory",
            "public_obligation_prompt_sha256": "prompt",
        },
    }
    monkeypatch.setattr(campaign, "freeze_inputs", lambda *args: inputs)
    monkeypatch.setattr(campaign, "_verified_plan", lambda *args: inputs)
    root = tmp_path / "run"
    plan = campaign.freeze(tmp_path / "source", root, "a" * 40, "evidence_strength")
    assert plan["schema_version"] == "repair-feedback-comparison-4"
    assert campaign.verify(root) == plan
    assert (
        campaign.freeze(tmp_path / "source", root, "a" * 40, "evidence_strength")
        == plan
    )
    with pytest.raises(ValueError, match="fresh output root"):
        campaign.freeze(tmp_path / "source", root, "a" * 40)
