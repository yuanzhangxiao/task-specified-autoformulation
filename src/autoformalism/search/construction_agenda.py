"""Runtime-owned priority agenda for incremental model construction."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from autoformalism.schemas import ConstructionObjective
from autoformalism.schemas.base import Identifier, NonEmptyText, StrictSchema
from autoformalism.search.feedback_routing import (
    FeedbackPriority,
    FeedbackRoute,
    RoutedFeedbackItem,
    RoutedProposerFeedback,
)

AgendaStage = Literal["topology", "functional_form"]


class ConstructionAgenda(StrictSchema):
    """One deterministic feedback category exposed to a proposer decision."""

    schema_version: Literal["construction-agenda-1"] = "construction-agenda-1"
    stage: AgendaStage
    objective: ConstructionObjective
    feedback_category: NonEmptyText
    priority: FeedbackPriority | None = None
    feedback_items: tuple[RoutedFeedbackItem, ...] = Field(default=(), max_length=64)
    eligible_requirement_ids: tuple[Identifier, ...] = Field(default=(), max_length=32)
    eligible_target_channels: tuple[Identifier, ...] = Field(default=(), max_length=64)


_CATEGORY_ORDER = {
    "target_contract_failure": 0,
    "graph_mechanism_failure": 1,
    "missing_scientific_requirement": 1,
    "executable_contract_failure": 2,
    "scientific_actionable_edit": 3,
    "missing_relaxation": 4,
    "fit_failure": 5,
    "integration_failure": 5,
    "inactive_target_dynamics": 6,
    "worst_validation_target": 7,
    "parameter_boundary_contact": 8,
    "annotation_function_mismatch": 9,
    "annotation_repair": 10,
}

_OBJECTIVE_BY_CATEGORY = {
    "target_contract_failure": ConstructionObjective.TARGET_PATH_REPAIR,
    "graph_mechanism_failure": ConstructionObjective.MECHANISM_REPAIR,
    "missing_scientific_requirement": ConstructionObjective.MECHANISM_REPAIR,
    "executable_contract_failure": ConstructionObjective.FUNCTION_REPAIR,
    "scientific_actionable_edit": ConstructionObjective.FUNCTION_REPAIR,
    "missing_relaxation": ConstructionObjective.MECHANISM_REPAIR,
    "fit_failure": ConstructionObjective.NUMERICAL_REPAIR,
    "integration_failure": ConstructionObjective.NUMERICAL_REPAIR,
    "inactive_target_dynamics": ConstructionObjective.FUNCTION_REPAIR,
    "worst_validation_target": ConstructionObjective.FUNCTION_REPAIR,
    "parameter_boundary_contact": ConstructionObjective.NUMERICAL_REPAIR,
    "annotation_function_mismatch": ConstructionObjective.FUNCTION_REPAIR,
    "annotation_repair": ConstructionObjective.FUNCTION_REPAIR,
}

_PRIORITY_TIE_BREAK = {
    FeedbackPriority.BLOCKING: 0,
    FeedbackPriority.PRIMARY: 1,
    FeedbackPriority.ADVISORY: 2,
}


def select_construction_agenda(
    *,
    stage: AgendaStage,
    feedback: RoutedProposerFeedback,
    allowed_requirement_ids: tuple[str, ...],
    target_channels: tuple[str, ...],
) -> ConstructionAgenda:
    """Select exactly one stage-local feedback category deterministically.

    Scientific target and graph obligations intentionally precede validation
    error.  The provider chooses the particular items and one or more public
    mechanism/target anchors only after this category has been fixed.
    """
    visible_routes = (
        {FeedbackRoute.TOPOLOGY, FeedbackRoute.INTEGRATED_REPAIR}
        if stage == "topology"
        else {
            FeedbackRoute.FUNCTIONAL_FORM,
            FeedbackRoute.NUMERICAL_FIT,
            FeedbackRoute.INTEGRATED_REPAIR,
        }
    )
    visible = tuple(item for item in feedback.items if item.route in visible_routes)
    if not visible:
        return ConstructionAgenda(
            stage=stage,
            objective=ConstructionObjective.INITIAL_CONSTRUCTION,
            feedback_category="initial_construction",
            eligible_requirement_ids=_unique(allowed_requirement_ids),
            eligible_target_channels=_unique(target_channels),
        )

    grouped = _group_feedback(visible)
    category = _select_category(grouped)
    selected = tuple(grouped[category])
    priority = min(
        (item.priority for item in selected),
        key=_PRIORITY_TIE_BREAK.__getitem__,
    )
    return ConstructionAgenda(
        stage=stage,
        objective=_OBJECTIVE_BY_CATEGORY.get(
            category, ConstructionObjective.FUNCTION_REPAIR
        ),
        feedback_category=category,
        priority=priority,
        feedback_items=selected,
        eligible_requirement_ids=_unique(allowed_requirement_ids),
        eligible_target_channels=_unique(target_channels),
    )


def select_next_revision_stage(
    feedback: RoutedProposerFeedback,
) -> AgendaStage | None:
    """Choose the responsible stage for the globally highest-priority category.

    This is the controller-level decision that ensures graph-mechanism repair
    precedes validation-error repair.  ``select_construction_agenda`` then
    chooses exactly one category within that responsible stage.
    """
    if not feedback.items:
        return None
    grouped = _group_feedback(feedback.items)
    category = _select_category(grouped)
    routes = {item.route for item in grouped[category]}
    if FeedbackRoute.TOPOLOGY in routes:
        return "topology"
    if category == "missing_scientific_requirement":
        return "topology"
    return "functional_form"


def _group_feedback(
    items: tuple[RoutedFeedbackItem, ...],
) -> dict[str, list[RoutedFeedbackItem]]:
    grouped: dict[str, list[RoutedFeedbackItem]] = {}
    for item in items:
        grouped.setdefault(item.code, []).append(item)
    return grouped


def _select_category(
    grouped: dict[str, list[RoutedFeedbackItem]],
) -> str:
    return min(
        grouped,
        key=lambda code: (
            _CATEGORY_ORDER.get(code, 100),
            min(_PRIORITY_TIE_BREAK[item.priority] for item in grouped[code]),
            code,
        ),
    )


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = [
    "ConstructionAgenda",
    "select_construction_agenda",
    "select_next_revision_stage",
]
