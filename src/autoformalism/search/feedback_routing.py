"""Deterministic routing of search evidence to staged proposer calls."""

from __future__ import annotations

import ast
import math
from collections.abc import Iterable
from enum import Enum
from typing import Literal

from pydantic import Field

from autoformalism.expressions import ModelValidationError, RestrictedParser
from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY
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
    "dynamic_structure",
    "parameter_identifiability",
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
    annotation_function_advisories: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=64
    )
    dynamic_structure_advisories: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=64
    )
    deterministic_validation_failures: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=64
    )
    validation_metrics: tuple[TargetValidationMetric, ...] = Field(
        default=(), max_length=64
    )
    fit_failures: tuple[NonEmptyText, ...] = Field(default=(), max_length=64)
    parameter_boundary_advisories: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=64
    )
    inactive_dynamics_advisories: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=64
    )
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
        evidence.annotation_function_advisories,
        route=FeedbackRoute.FUNCTIONAL_FORM,
        priority=FeedbackPriority.ADVISORY,
        source="mechanism_annotation",
        code="annotation_function_mismatch",
    )
    add_many(
        evidence.dynamic_structure_advisories,
        route=FeedbackRoute.TOPOLOGY,
        priority=FeedbackPriority.ADVISORY,
        source="dynamic_structure",
        code="missing_relaxation",
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
        evidence.parameter_boundary_advisories,
        route=FeedbackRoute.NUMERICAL_FIT,
        priority=FeedbackPriority.ADVISORY,
        source="parameter_identifiability",
        code="parameter_boundary_contact",
    )
    add_many(
        evidence.inactive_dynamics_advisories,
        route=FeedbackRoute.NUMERICAL_FIT,
        priority=FeedbackPriority.ADVISORY,
        source="numerical_fitter",
        code="inactive_target_dynamics",
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
    boundary_advisories, inactive_advisories = _fit_parameter_advisories(
        candidate,
        fit,
    )

    return CandidateFeedbackEvidence(
        target_contract_failures=tuple(target_failures),
        graph_mechanism_failures=tuple(graph_failures),
        annotation_failures=tuple(annotation_failures),
        annotation_function_advisories=(
            _annotation_function_advisories(candidate)
        ),
        dynamic_structure_advisories=_dynamic_structure_advisories(candidate),
        validation_metrics=tuple(
            TargetValidationMetric(target=target, normalized_mse=value)
            for target, value in sorted(
                fit.validation_metrics.per_target_normalized_mse.items()
            )
            if math.isfinite(value) and value >= 0.0
        ),
        fit_failures=tuple(dict.fromkeys(fit_failures)),
        parameter_boundary_advisories=boundary_advisories,
        inactive_dynamics_advisories=inactive_advisories,
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


def _fit_parameter_advisories(
    candidate: CandidateModel,
    fit: FitResult,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return nonblocking boundary-contact and target-inactivity evidence."""
    parameter_names = {item.name for item in candidate.parameters}
    lower_contacts = _normalized_parameter_contacts(
        candidate,
        (
            contact
            for diagnostic in fit.diagnostics
            for contact in diagnostic.parameters_at_lower_bound
        ),
    )
    upper_contacts = _normalized_parameter_contacts(
        candidate,
        (
            contact
            for diagnostic in fit.diagnostics
            for contact in diagnostic.parameters_at_upper_bound
        ),
    )
    boundary: list[str] = []
    if lower_contacts:
        boundary.append(
            "Fitted parameters contact their runtime lower boundary: "
            f"{list(lower_contacts)}. Treat this as possible scale or "
            "identifiability evidence, not as standalone structural invalidity."
        )
    if upper_contacts:
        boundary.append(
            "Fitted parameters contact their runtime upper boundary: "
            f"{list(upper_contacts)}. Treat this as possible scale or "
            "identifiability evidence, not as standalone structural invalidity."
        )

    target_parameters = _target_path_parameters(candidate)
    positive_parameters = {
        item.name
        for item in candidate.parameters
        if item.domain.value in {"positive", "nonnegative"}
    }
    inactive: tuple[str, ...] = ()
    if (
        target_parameters
        and target_parameters <= parameter_names
        and target_parameters <= positive_parameters
        and target_parameters <= set(lower_contacts)
    ):
        inactive = (
            "Every positive/nonnegative fitted parameter on a target-generating "
            f"dependency path contacts its lower runtime boundary: "
            f"{sorted(target_parameters)}. The fitted target response may have "
            "collapsed toward inactive dynamics; compare its trajectory with a "
            "persistence or null prediction before retaining the structure.",
        )
    return tuple(boundary), inactive


def _normalized_parameter_contacts(
    candidate: CandidateModel,
    contacts: Iterable[str],
) -> tuple[str, ...]:
    """Normalize legacy ``parameter:name`` optimizer labels to model names."""
    parameter_names = {item.name for item in candidate.parameters}
    normalized = {
        name
        for raw in contacts
        if (name := str(raw).removeprefix("parameter:")) in parameter_names
    }
    return tuple(sorted(normalized))


def _target_path_parameters(candidate: CandidateModel) -> set[str]:
    """Collect parameters recursively upstream of public target mappings."""
    parser = RestrictedParser()
    definitions = {
        item.state: item.rhs for item in candidate.state_equations
    } | {item.name: item.expression for item in candidate.processes}
    parameter_names = {item.name for item in candidate.parameters}
    frontier: list[str] = []
    found: set[str] = set()
    for index, mapping in enumerate(candidate.observation_mappings):
        symbols = _safe_symbols(
            parser,
            mapping.expression,
            f"observation_mapping:{index}",
        )
        found.update(symbols & parameter_names)
        frontier.extend(symbols & definitions.keys())
    visited: set[str] = set()
    while frontier:
        symbol = frontier.pop()
        if symbol in visited:
            continue
        visited.add(symbol)
        symbols = _safe_symbols(parser, definitions[symbol], f"definition:{symbol}")
        found.update(symbols & parameter_names)
        frontier.extend(symbols & definitions.keys() - visited)
    return found


def _dynamic_structure_advisories(candidate: CandidateModel) -> tuple[str, ...]:
    """Find factual missing-self-regulation patterns without rejecting a model."""
    parser = RestrictedParser()
    state_names = {item.name for item in candidate.states}
    mechanisms = {item.name: item.mechanisms for item in candidate.states}
    dependencies = {
        item.state: (
            _safe_symbols(parser, item.rhs, f"state_equation:{item.state}")
            & state_names
        )
        for item in candidate.state_equations
    }
    advisories: list[str] = []
    memory_tokens = (
        "memory",
        "delay",
        "stock",
        "storage",
        "reservoir",
        "compartment",
        "persistent",
        "feedback",
    )
    for state in sorted(state_names):
        if state in dependencies.get(state, set()):
            continue
        matching_tags = sorted(
            tag
            for tag in mechanisms.get(state, ())
            if any(token in tag.lower() for token in memory_tokens)
        )
        if matching_tags:
            advisories.append(
                f"State {state} is tagged {matching_tags} but its RHS has no "
                "self-dependent term. This may be an intentional accumulator; "
                "otherwise add or justify a removal/relaxation timescale."
            )

    seen_cycles: set[frozenset[str]] = set()
    reachability = {
        state: _reachable_states(state, dependencies) for state in state_names
    }
    for state in sorted(state_names):
        component = frozenset(
            other
            for other in state_names
            if other in reachability[state] and state in reachability[other]
        )
        if len(component) < 2 or component in seen_cycles:
            continue
        seen_cycles.add(component)
        if all(item not in dependencies.get(item, set()) for item in component):
            advisories.append(
                f"Coupled state cycle {sorted(component)} has no direct "
                "self-dependent term on any member. Verify an explicit "
                "stabilizing removal/relaxation mechanism rather than assuming "
                "the cycle is dynamically stable."
            )
    return tuple(advisories)


def _reachable_states(
    start: str,
    dependencies: dict[str, frozenset[str] | set[str]],
) -> set[str]:
    """Return state dependencies reachable from one state."""
    reached: set[str] = set()
    frontier = list(dependencies.get(start, ()))
    while frontier:
        state = frontier.pop()
        if state in reached:
            continue
        reached.add(state)
        frontier.extend(dependencies.get(state, ()))
    return reached


def _annotation_function_advisories(
    candidate: CandidateModel,
) -> tuple[str, ...]:
    """Flag a claimed nonlinear mechanism when the whole model is affine."""
    claimed = sorted(
        f"{item.name}:{tag}"
        for item in (*candidate.states, *candidate.processes)
        for tag in item.mechanisms
        if "nonlinear" in tag.lower()
    )
    if not claimed or _candidate_has_syntactic_nonlinearity(candidate):
        return ()
    return (
        "Mechanism annotations claim nonlinear behavior at "
        f"{claimed}, but no state equation, process, or target mapping contains "
        "syntactically nonlinear dependence on a non-parameter symbol. Revise "
        "the functional form or correct the annotation.",
    )


def _candidate_has_syntactic_nonlinearity(candidate: CandidateModel) -> bool:
    """Conservatively identify nonlinear dependence on states or inputs."""
    parser = RestrictedParser()
    parameter_names = {item.name for item in candidate.parameters}
    expressions = [item.rhs for item in candidate.state_equations]
    expressions.extend(item.expression for item in candidate.processes)
    expressions.extend(item.expression for item in candidate.observation_mappings)
    for index, expression in enumerate(expressions):
        try:
            tree = parser.parse(expression, location=f"expression:{index}").tree
        except ModelValidationError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _has_nonparameter_symbol(
                node,
                parameter_names,
            ):
                return True
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and all(
                _has_nonparameter_symbol(branch, parameter_names)
                for branch in (node.left, node.right)
            ):
                return True
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Div)
                and _has_nonparameter_symbol(
                    node.right,
                    parameter_names,
                )
            ):
                return True
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Pow)
                and _has_nonparameter_symbol(node.left, parameter_names)
                and not _is_literal_one(node.right)
            ):
                return True
    return False


def _has_nonparameter_symbol(node: ast.AST, parameter_names: set[str]) -> bool:
    return any(
        isinstance(child, ast.Name)
        and child.id not in parameter_names
        and child.id not in APPROVED_FUNCTION_ARITY
        for child in ast.walk(node)
    )


def _is_literal_one(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
        and float(node.value) == 1.0
    )


def _safe_symbols(
    parser: RestrictedParser,
    expression: str,
    location: str,
) -> frozenset[str]:
    """Return symbols for valid expressions and no inferred fact otherwise."""
    try:
        return parser.parse(expression, location=location).symbols
    except ModelValidationError:
        return frozenset()
