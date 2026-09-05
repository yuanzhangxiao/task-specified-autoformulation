"""Runtime-owned construction agenda tests."""

from autoformalism.search.construction_agenda import (
    select_construction_agenda,
    select_next_revision_stage,
)
from autoformalism.search.feedback_routing import (
    CandidateFeedbackEvidence,
    TargetValidationMetric,
    route_proposer_feedback,
)


def test_graph_mechanism_precedes_validation_error() -> None:
    feedback = route_proposer_feedback(
        CandidateFeedbackEvidence(
            graph_mechanism_failures=("Delayed pathway is disconnected.",),
            validation_metrics=(
                TargetValidationMetric(target="output", normalized_mse=9.0),
            ),
        )
    )

    assert select_next_revision_stage(feedback) == "topology"
    agenda = select_construction_agenda(
        stage="topology",
        feedback=feedback,
        allowed_requirement_ids=("delay", "input_response"),
        target_channels=("output",),
    )

    assert agenda.feedback_category == "graph_mechanism_failure"
    assert agenda.objective.value == "mechanism_repair"
    assert agenda.eligible_requirement_ids == ("delay", "input_response")
    assert len(agenda.feedback_items) == 1


def test_target_contract_precedes_graph_mechanism() -> None:
    feedback = route_proposer_feedback(
        CandidateFeedbackEvidence(
            target_contract_failures=("Target output is unmapped.",),
            graph_mechanism_failures=("Input pathway is disconnected.",),
        )
    )

    agenda = select_construction_agenda(
        stage="topology",
        feedback=feedback,
        allowed_requirement_ids=("input_response",),
        target_channels=("output",),
    )

    assert agenda.feedback_category == "target_contract_failure"
    assert agenda.objective.value == "target_path_repair"


def test_empty_feedback_creates_initial_construction_agenda() -> None:
    agenda = select_construction_agenda(
        stage="topology",
        feedback=route_proposer_feedback(CandidateFeedbackEvidence()),
        allowed_requirement_ids=("path_a", "path_b"),
        target_channels=("output",),
    )

    assert agenda.feedback_category == "initial_construction"
    assert agenda.feedback_items == ()
    assert agenda.objective.value == "initial_construction"
