"""Deterministic routing of search evidence to staged proposer calls."""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import Field

from autoformalism.fitting import FitResult
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    PredicateResult,
    evaluate_mechanisms,
)
from autoformalism.schemas import CandidateModel, JudgeAssessment
from autoformalism.schemas.base import NonEmptyText, StrictSchema
from autoformalism.targets import PublicTargetContract, evaluate_public_targets

FeedbackSource = Literal[
    "public_target_contract",
    "graph_mechanism",
    "mechanism_annotation",
    "deterministic_validator",
    "validation_metric",
    "numerical_fitter",
    "integrator",
    "scientific_judge",
]


class RevisionStage(str, Enum):
    """Provider-visible construction or revision stage."""

    TOPOLOGY = "topology"
    FUNCTIONAL_FORM = "functional_form"
    INTEGRATED_REPAIR = "integrated_repair"


class FeedbackRoute(str, Enum):
    """Part of model construction primarily responsible for one signal."""

    TOPOLOGY = "topology"
    FUNCTIONAL_FORM = "functional_form"
    NUMERICAL_FIT = "numerical_fit"
    INTEGRATED_REPAIR = "integrated_repair"


class FeedbackPriority(str, Enum):
    """Stable ordering class for routed evidence."""

    BLOCKING = "blocking"
    PRIMARY = "primary"
    ADVISORY = "advisory"


class TargetValidationMetric(StrictSchema):
    """One public validation error available to proposal selection."""

    target: NonEmptyText
    normalized_mse: float = Field(ge=0.0)


class CandidateFeedbackEvidence(StrictSchema):
    """Bounded typed evidence from deterministic, fit, and judge stages."""

    target_contract_failures: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=64
    )
    graph_mechanism_failures: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=64
    )
    annotation_failures: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=64
    )
    deterministic_validation_failures: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=64
    )
    validation_metrics: tuple[TargetValidationMetric, ...] = Field(
        default=(), max_length=64
    )
    fit_failures: tuple[NonEmptyText, ...] = Field(default=(), max_length=64)
    integration_failures: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=64
    )
    scientific_missing_requirements: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=64
    )
    scientific_actionable_edits: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=64
    )


class RoutedFeedbackItem(StrictSchema):
    """One concise signal assigned to its responsible revision layer."""

    route: FeedbackRoute
    priority: FeedbackPriority
    source: FeedbackSource
    code: NonEmptyText
    message: NonEmptyText


class RoutedProposerFeedback(StrictSchema):
    """Versioned feedback bundle with deterministic per-stage disclosure."""

    schema_version: Literal["routed-proposer-feedback-1"] = (
        "routed-proposer-feedback-1"
    )
    items: tuple[RoutedFeedbackItem, ...] = Field(default=(), max_length=64)
    omitted_item_count: int = Field(default=0, ge=0)

    def for_stage(
        self,
        stage: RevisionStage,
        *,
        include_integrated_repairs: bool = False,
    ) -> dict[str, object]:
        """Return only evidence appropriate for one proposer operation.

        A two-stage search may expose cross-cutting scientific-judge repairs to
        both construction calls while retaining the normal stage-local view for
        callers that do not opt in.
        """
        visible_routes = {
            RevisionStage.TOPOLOGY: {FeedbackRoute.TOPOLOGY},
            RevisionStage.FUNCTIONAL_FORM: {
                FeedbackRoute.FUNCTIONAL_FORM,
                FeedbackRoute.NUMERICAL_FIT,
            },
            RevisionStage.INTEGRATED_REPAIR: set(FeedbackRoute),
        }[stage]
        if include_integrated_repairs:
            visible_routes = visible_routes | {FeedbackRoute.INTEGRATED_REPAIR}
        visible = tuple(item for item in self.items if item.route in visible_routes)
        return {
            "schema_version": "stage-feedback-view-1",
            "stage": stage.value,
            "items": [item.model_dump(mode="json") for item in visible],
            "hidden_other_stage_item_count": len(self.items) - len(visible),
            "upstream_omitted_item_count": self.omitted_item_count,
        }


_PRIORITY_ORDER = {
    FeedbackPriority.BLOCKING: 0,
    FeedbackPriority.PRIMARY: 1,
    FeedbackPriority.ADVISORY: 2,
}


def route_proposer_feedback(
    evidence: CandidateFeedbackEvidence,
    *,
    maximum_items: int = 24,
) -> RoutedProposerFeedback:
    """Convert raw search evidence into a small, stage-local action record."""
    if maximum_items < 1:
        raise ValueError("maximum_items must be positive")
    items: list[RoutedFeedbackItem] = []

    def add_many(
        messages: tuple[str, ...],
        *,
        route: FeedbackRoute,
        priority: FeedbackPriority,
        source: FeedbackSource,
        code: str,
    ) -> None:
        for message in messages:
            items.append(
                RoutedFeedbackItem(
                    route=route,
                    priority=priority,
                    source=source,
                    code=code,
                    message=message,
                )
            )

    add_many(
        evidence.target_contract_failures,
        route=FeedbackRoute.TOPOLOGY,
        priority=FeedbackPriority.BLOCKING,
        source="public_target_contract",
        code="target_contract_failure",
    )
    add_many(
        evidence.graph_mechanism_failures,
        route=FeedbackRoute.TOPOLOGY,
        priority=FeedbackPriority.PRIMARY,
        source="graph_mechanism",
        code="graph_mechanism_failure",
    )
    add_many(
        evidence.annotation_failures,
        route=FeedbackRoute.FUNCTIONAL_FORM,
        priority=FeedbackPriority.ADVISORY,
        source="mechanism_annotation",
        code="annotation_repair",
    )
    add_many(
        evidence.deterministic_validation_failures,
        route=FeedbackRoute.FUNCTIONAL_FORM,
        priority=FeedbackPriority.BLOCKING,
        source="deterministic_validator",
        code="executable_contract_failure",
    )
    add_many(
        evidence.fit_failures,
        route=FeedbackRoute.NUMERICAL_FIT,
        priority=FeedbackPriority.PRIMARY,
        source="numerical_fitter",
        code="fit_failure",
    )
    add_many(
        evidence.integration_failures,
        route=FeedbackRoute.NUMERICAL_FIT,
        priority=FeedbackPriority.PRIMARY,
        source="integrator",
        code="integration_failure",
    )
    add_many(
        evidence.scientific_missing_requirements,
        route=FeedbackRoute.INTEGRATED_REPAIR,
        priority=FeedbackPriority.PRIMARY,
        source="scientific_judge",
        code="missing_scientific_requirement",
    )
    add_many(
        evidence.scientific_actionable_edits,
        route=FeedbackRoute.INTEGRATED_REPAIR,
        priority=FeedbackPriority.PRIMARY,
        source="scientific_judge",
        code="scientific_actionable_edit",
    )
    finite_metrics = [
        item
        for item in evidence.validation_metrics
        if math.isfinite(item.normalized_mse)
    ]
    if finite_metrics:
        worst = max(
            finite_metrics,
            key=lambda item: (item.normalized_mse, item.target),
        )
        items.append(
            RoutedFeedbackItem(
                route=FeedbackRoute.NUMERICAL_FIT,
                priority=FeedbackPriority.PRIMARY,
                source="validation_metric",
                code="worst_validation_target",
                message=(
                    f"Worst public validation target is {worst.target} with "
                    f"normalized MSE {worst.normalized_mse:.12g}."
                ),
            )
        )

    items.sort(
        key=lambda item: (
            _PRIORITY_ORDER[item.priority],
            item.route.value,
            item.source,
            item.code,
            item.message,
        )
    )
    retained = tuple(items[:maximum_items])
    return RoutedProposerFeedback(
        items=retained,
        omitted_item_count=len(items) - len(retained),
    )


def evidence_from_completed_candidate(
    candidate: CandidateModel,
    fit: FitResult,
    judge: JudgeAssessment | None,
    *,
    public_target_contract: PublicTargetContract | None = None,
    public_mechanism_spec: MechanismEvaluationSpec | None = None,
) -> CandidateFeedbackEvidence:
    """Extract stage-routable public evidence from one completed candidate."""
    target_failures: list[str] = []
    if public_target_contract is not None:
        target_evaluation = evaluate_public_targets(
            candidate, public_target_contract
        )
        target_failures.extend(
            f"{item.target_channel}/{item.predicate}: {item.evidence}"
            for item in target_evaluation.predicates
            if item.status == "failed"
        )

    graph_failures: list[str] = []
    annotation_failures: list[str] = []
    if public_mechanism_spec is not None:
        mechanism_evaluation = evaluate_mechanisms(
            candidate, public_mechanism_spec
        )
        graph_failures.extend(
            _render_mechanism_result(item.mechanism_id, item.status, item.predicates)
            for item in mechanism_evaluation.mechanism_results
            if item.status != "satisfied"
        )
        annotation_failures.extend(
            _render_mechanism_result(item.mechanism_id, item.status, item.predicates)
            for item in mechanism_evaluation.annotation_results
            if item.status != "satisfied"
        )
        annotation_failures.extend(
            f"{item.mechanism_id}: {item.evidence}; suggested components="
            f"{list(item.suggested_components)}"
            for item in mechanism_evaluation.annotation_repairs
        )

    fit_failures: list[str] = []
    if fit.message:
        fit_failures.append(fit.message)
    fit_failures.extend(
        diagnostic.message
        for diagnostic in fit.diagnostics
        if not diagnostic.success and diagnostic.message
    )
    integration_failures = sorted(
        {
            message
            for diagnostic in fit.diagnostics
            for message in diagnostic.integration_failure_messages
        }
    )
    integration_failures.extend(
        f"Training rollout failed for trajectory {name}."
        for name in fit.training_metrics.failed_trajectories
    )
    integration_failures.extend(
        f"Validation rollout failed for trajectory {name}."
        for name in fit.validation_metrics.failed_trajectories
    )

    return CandidateFeedbackEvidence(
        target_contract_failures=tuple(target_failures),
        graph_mechanism_failures=tuple(graph_failures),
        annotation_failures=tuple(annotation_failures),
        validation_metrics=tuple(
            TargetValidationMetric(target=target, normalized_mse=value)
            for target, value in sorted(
                fit.validation_metrics.per_target_normalized_mse.items()
            )
            if math.isfinite(value) and value >= 0.0
        ),
        fit_failures=tuple(dict.fromkeys(fit_failures)),
        integration_failures=tuple(dict.fromkeys(integration_failures)),
        scientific_missing_requirements=(
            () if judge is None else tuple(judge.missing_requirements)
        ),
        scientific_actionable_edits=(
            ()
            if judge is None
            else tuple(
                f"{item.priority.value}/{item.target}: {item.instruction}"
                for item in judge.actionable_edits
            )
        ),
    )


def _render_mechanism_result(
    mechanism_id: str,
    status: str,
    predicates: tuple[PredicateResult, ...],
) -> str:
    evidence = [
        item.evidence
        for item in predicates
        if item.status != "satisfied"
    ]
    return f"{mechanism_id}/{status}: {'; '.join(evidence)}"
