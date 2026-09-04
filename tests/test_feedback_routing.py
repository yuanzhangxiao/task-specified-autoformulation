"""Stage-specific proposer feedback routing tests."""

from autoformalism.fitting import EvaluationMetrics, FitResult, OptimizationDiagnostic
from autoformalism.schemas import CandidateModel, ScientificJudgeResult
from autoformalism.search import (
    CandidateFeedbackEvidence,
    RevisionStage,
    TargetValidationMetric,
    evidence_from_completed_candidate,
    route_proposer_feedback,
)


def _evidence() -> CandidateFeedbackEvidence:
    return CandidateFeedbackEvidence(
        target_contract_failures=("Target U omits its required baseline term.",),
        graph_mechanism_failures=("Delayed action is not connected to U.",),
        annotation_failures=("The insulin-action edge lacks its tag.",),
        deterministic_validation_failures=("Undefined symbol bad_name.",),
        validation_metrics=(
            TargetValidationMetric(target="Gp", normalized_mse=2.0),
            TargetValidationMetric(target="U", normalized_mse=8.0),
        ),
        fit_failures=("The optimizer reached its evaluation limit.",),
        integration_failures=("The state trajectory became non-finite.",),
        scientific_missing_requirements=("Missing delayed response memory.",),
        scientific_actionable_edits=("Add a stable relaxation pathway.",),
    )


def test_feedback_is_disclosed_only_to_responsible_stage() -> None:
    routed = route_proposer_feedback(_evidence())

    topology = routed.for_stage(RevisionStage.TOPOLOGY)
    functional = routed.for_stage(RevisionStage.FUNCTIONAL_FORM)
    integrated = routed.for_stage(RevisionStage.INTEGRATED_REPAIR)

    topology_sources = {item["source"] for item in topology["items"]}
    functional_sources = {item["source"] for item in functional["items"]}
    integrated_sources = {item["source"] for item in integrated["items"]}
    assert topology_sources == {"public_target_contract", "graph_mechanism"}
    assert "scientific_judge" not in topology_sources
    assert functional_sources == {
        "mechanism_annotation",
        "deterministic_validator",
        "validation_metric",
        "numerical_fitter",
        "integrator",
    }
    assert integrated_sources == topology_sources | functional_sources | {
        "scientific_judge"
    }
    assert any(
        item["message"].startswith("Worst public validation target is U")
        for item in functional["items"]
    )


def test_feedback_budget_retains_blocking_items_first() -> None:
    routed = route_proposer_feedback(_evidence(), maximum_items=2)

    assert len(routed.items) == 2
    assert routed.omitted_item_count == 7
    assert {item.priority.value for item in routed.items} == {"blocking"}


def test_completed_candidate_evidence_selects_public_fit_and_judge_signals() -> None:
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "candidate_0",
            "parent_candidate_id": None,
            "states": [{"name": "x", "kind": "observed"}],
            "state_equations": [{"state": "x", "rhs": "-rate * x"}],
            "observation_mappings": [{"channel": "target", "expression": "x"}],
            "parameters": [{"name": "rate", "scope": "global", "role": "rate"}],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "target"}
            ],
        }
    )
    metrics = EvaluationMetrics(
        normalized_mse=3.0,
        per_target_normalized_mse={"target": 3.0},
        failed_trajectories=("trajectory_0",),
    )
    fit = FitResult(
        success=False,
        global_parameters={"rate": 1.0},
        global_initial_conditions={},
        training_trajectory_initial_conditions={},
        validation_trajectory_initial_conditions={},
        training_metrics=metrics,
        validation_metrics=metrics,
        diagnostics=(
            OptimizationDiagnostic(
                start_index=0,
                success=False,
                status=-2,
                message="evaluation limit",
                cost=3.0,
                function_evaluations=10,
                integration_failures=1,
                integration_failure_messages=("non-finite state",),
            ),
        ),
        best_start_index=0,
        target_scales={"target": 1.0},
        message="candidate rejected",
    )
    judge = ScientificJudgeResult.model_validate(
        {
            "category_scores": dict.fromkeys(
                (
                    "mechanistic_coherence",
                    "source_sink_balance_semantics",
                    "dynamic_plausibility",
                    "mechanism_coupling_task_sufficiency",
                    "nonredundancy_accounting",
                    "latent_state_complexity_justification",
                ),
                0.5,
            ),
            "missing_requirements": ["Missing delay."],
            "actionable_edits": [
                {
                    "target": "x",
                    "instruction": "Add stable memory.",
                    "priority": "required",
                }
            ],
        }
    )

    evidence = evidence_from_completed_candidate(candidate, fit, judge)
    routed = route_proposer_feedback(evidence)

    functional = routed.for_stage(RevisionStage.FUNCTIONAL_FORM)
    integrated = routed.for_stage(RevisionStage.INTEGRATED_REPAIR)
    assert any(item["code"] == "integration_failure" for item in functional["items"])
    assert any(
        item["message"] == "required/x: Add stable memory."
        for item in integrated["items"]
    )
