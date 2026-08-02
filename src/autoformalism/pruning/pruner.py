"""Contribution-based pruning of complete additive expression terms."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from autoformalism.data import DatasetSplit
from autoformalism.expressions import (
    CompiledModel,
    ModelValidationError,
    ParsedExpression,
    RestrictedParser,
    compile_candidate,
)
from autoformalism.fitting import FitConfig, FitResult, fit_candidate
from autoformalism.fitting.simulation import (
    causal_interval_state,
    simulate_trajectory,
    trajectory_forcing,
)
from autoformalism.pruning.models import (
    PruningCandidateResult,
    PruningConfig,
    PruningResult,
    TermContribution,
)
from autoformalism.schemas import CandidateModel


@dataclass(frozen=True)
class _Term:
    term_id: str
    location: str
    index: int
    expression: str
    parsed: ParsedExpression
    parameters: tuple[str, ...]


def prune_candidate(
    model: CompiledModel,
    training: DatasetSplit,
    validation: DatasetSplit,
    *,
    fit_config: FitConfig | None = None,
    pruning_config: PruningConfig | None = None,
    unpruned_fit: FitResult | None = None,
) -> PruningResult:
    """Measure whole terms, refit generated supports, and select the simplest."""
    fitting = fit_config or FitConfig()
    settings = pruning_config or PruningConfig()
    baseline = unpruned_fit or fit_candidate(model, training, validation, fitting)
    if not baseline.success or baseline.validation_metrics.failed_trajectories:
        raise ValueError("unpruned model must have a successful numerical fit")

    terms = _extract_terms(model)
    contributions = _measure_contributions(
        model,
        training,
        baseline,
        terms,
        settings.contribution_epsilon,
        fitting,
    )
    thresholds = _thresholds(
        contributions,
        settings.threshold_epsilon,
        settings.maximum_normalized_contribution,
    )
    baseline_ids = tuple(item.term_id for item in terms)
    results: list[PruningCandidateResult] = [
        PruningCandidateResult(
            threshold=0.0,
            retained_term_ids=baseline_ids,
            removed_term_ids=(),
            removed_parameters=(),
            candidate=model.validated.candidate,
            fit_result=baseline,
            accepted=True,
            rejection_reason=None,
        )
    ]
    seen_supports = {baseline_ids}
    contribution_map = {
        contribution.term_id: contribution.normalized_rms
        for contribution in contributions
    }
    protected_terms = _protected_terms(model, terms, settings)
    protected_locations = _target_producing_locations(model) | _mechanism_locations(
        model
    )
    for threshold in thresholds:
        retained = tuple(
            item.term_id
            for item in terms
            if contribution_map[item.term_id] >= threshold
            or item.term_id in protected_terms
        )
        if settings.require_target_dynamics and any(
            not any(
                term.term_id in retained and term.location == location
                for term in terms
            )
            for location in protected_locations
        ):
            continue
        if retained in seen_supports:
            continue
        seen_supports.add(retained)
        results.append(
            _evaluate_support(
                model,
                training,
                validation,
                terms,
                retained,
                threshold,
                fitting,
                baseline.validation_metrics.normalized_mse,
                settings.validation_mse_tolerance,
            )
        )
        if settings.support_strategy == "single_support":
            break

    eligible = [
        result
        for result in results
        if result.accepted and result.candidate is not None and result.fit_result
    ]
    selected = min(
        eligible,
        key=lambda result: (
            len(result.retained_term_ids),
            len(result.candidate.parameters),
            result.fit_result.validation_metrics.normalized_mse,
            result.threshold,
        ),
    )
    assert selected.candidate is not None
    assert selected.fit_result is not None
    persistence_training_mse = _persistence_mse(training, baseline.target_scales)
    persistence_validation_mse = _persistence_mse(
        validation, baseline.target_scales
    )
    return PruningResult(
        unpruned_candidate=model.validated.candidate,
        unpruned_fit=baseline,
        contributions=contributions,
        thresholds=thresholds,
        candidates=tuple(results),
        selected_candidate=selected.candidate,
        selected_fit=selected.fit_result,
        selected_threshold=selected.threshold,
        selected_removed_terms=selected.removed_term_ids,
        selected_removed_parameters=selected.removed_parameters,
        contribution_by_term=contribution_map,
        persistence_training_mse=persistence_training_mse,
        persistence_validation_mse=persistence_validation_mse,
    )


def _protected_terms(
    model: CompiledModel,
    terms: Sequence[_Term],
    settings: PruningConfig,
) -> frozenset[str]:
    """Keep task inputs from disappearing solely because they are sparse."""
    if not settings.preserve_external_input_terms:
        return frozenset()
    external = set(model.validated.context.external_inputs)
    return frozenset(
        term.term_id for term in terms if external & set(term.parsed.symbols)
    )


def _target_producing_locations(model: CompiledModel) -> frozenset[str]:
    """Return dynamic/process locations needed by target observations."""
    processes = model.validated.process_expressions
    states = model.validated.equation_expressions
    pending: list[str] = []
    for channel in model.validated.context.targets:
        expression = model.validated.observation_expressions.get(channel)
        if expression is not None:
            pending.extend(expression.symbols)
    locations: set[str] = set()
    visited: set[str] = set()
    while pending:
        symbol = pending.pop()
        if symbol in visited:
            continue
        visited.add(symbol)
        if symbol in states:
            locations.add(f"equation:{symbol}")
            pending.extend(states[symbol].symbols)
        elif symbol in processes:
            locations.add(f"process:{symbol}")
            pending.extend(processes[symbol].symbols)
    return frozenset(locations)


def _mechanism_locations(model: CompiledModel) -> frozenset[str]:
    """Keep at least one term in every task-mechanism component."""
    candidate = model.validated.candidate
    state_locations = {
        f"equation:{state.name}" for state in candidate.states if state.mechanisms
    }
    process_locations = {
        f"process:{process.name}"
        for process in candidate.processes
        if process.mechanisms
    }
    return frozenset(state_locations | process_locations)


def _persistence_mse(
    split: DatasetSplit, target_scales: Mapping[str, float]
) -> float:
    """Compute the causal one-step persistence baseline using train scales."""
    scales = dict(target_scales)  # FitResult exposes an immutable mapping.
    residuals: list[np.ndarray] = []
    for trajectory in split.trajectories:
        for target, scale in scales.items():
            values = np.asarray(trajectory.targets[target], dtype=float)
            if len(values) > 1:
                residuals.append((values[1:] - values[:-1]) / float(scale))
    if not residuals:
        raise ValueError("persistence baseline needs at least two target samples")
    joined = np.concatenate(residuals)
    return float(np.mean(joined**2))


def _extract_terms(model: CompiledModel) -> tuple[_Term, ...]:
    parser = RestrictedParser()
    parameter_names = set(model.parameter_names)
    result: list[_Term] = []
    expressions = (
        tuple(
            (f"process:{item.name}", item.expression)
            for item in model.validated.candidate.processes
        )
        + tuple(
            (f"equation:{item.state}", item.rhs)
            for item in model.validated.candidate.state_equations
        )
    )
    for location, source in expressions:
        root = parser.parse(source, location=location).tree.body
        for index, node in enumerate(_additive_terms(root)):
            expression = ast.unparse(ast.fix_missing_locations(node))
            parsed = parser.parse(expression, location=f"{location}:term:{index}")
            result.append(
                _Term(
                    term_id=f"{location}:term:{index}",
                    location=location,
                    index=index,
                    expression=expression,
                    parsed=parsed,
                    parameters=tuple(
                        sorted(parameter_names & set(parsed.symbols))
                    ),
                )
            )
    return tuple(result)


def _additive_terms(node: ast.expr, sign: int = 1) -> tuple[ast.expr, ...]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _additive_terms(node.left, sign) + _additive_terms(node.right, sign)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return _additive_terms(node.left, sign) + _additive_terms(node.right, -sign)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _additive_terms(node.operand, -sign)
    if sign < 0:
        return (ast.UnaryOp(op=ast.USub(), operand=node),)
    return (node,)


def _measure_contributions(
    model: CompiledModel,
    training: DatasetSplit,
    fit: FitResult,
    terms: Sequence[_Term],
    epsilon: float,
    fit_config: FitConfig,
) -> tuple[TermContribution, ...]:
    values: dict[str, list[float]] = {item.term_id: [] for item in terms}
    totals: dict[str, list[float]] = {
        item.location: [] for item in terms
    }
    terms_by_location: dict[str, list[_Term]] = {}
    for term in terms:
        terms_by_location.setdefault(term.location, []).append(term)

    for trajectory in training.trajectories:
        initials = {
            **fit.global_initial_conditions,
            **fit.training_trajectory_initial_conditions.get(
                trajectory.trajectory_id, {}
            ),
        }
        simulation = simulate_trajectory(
            model, trajectory, fit.global_parameters, initials, fit_config
        )
        if not simulation.success or simulation.states is None:
            raise ValueError(
                f"cannot measure terms on trajectory {trajectory.trajectory_id}: "
                f"{simulation.message}"
            )
        full_expressions = {
            **{
                f"process:{name}": expression
                for name, expression in model.validated.process_expressions.items()
            },
            **{
                f"equation:{name}": expression
                for name, expression in model.validated.equation_expressions.items()
            },
        }
        for index in range(len(trajectory.time) - 1):
            start_time = float(trajectory.time[index])
            end_time = float(trajectory.time[index + 1])
            forcing = trajectory_forcing(model, trajectory, causal_index=index)
            start_state = causal_interval_state(
                model, trajectory, simulation.states[:, index], index
            )
            end_state = simulation.states[:, index + 1]
            # Include interval interiors so sparse inputs are not declared
            # irrelevant merely because they vanish at sampled boundaries.
            samples = (
                (start_time, start_state),
                (
                    0.5 * (start_time + end_time),
                    0.5 * (start_state + end_state),
                ),
                (end_time, end_state),
            )
            for time, state in samples:
                for location, location_terms in terms_by_location.items():
                    totals[location].append(
                        model.evaluate_expression(
                            full_expressions[location],
                            time,
                            state,
                            fit.global_parameters,
                            forcing,
                        )
                    )
                    for term in location_terms:
                        values[term.term_id].append(
                            model.evaluate_expression(
                                term.parsed,
                                time,
                                state,
                                fit.global_parameters,
                                forcing,
                            )
                        )

    result: list[TermContribution] = []
    for term in terms:
        term_rms = _rms(values[term.term_id])
        total_rms = max(_rms(totals[term.location]), epsilon)
        result.append(
            TermContribution(
                term_id=term.term_id,
                location=term.location,
                expression=term.expression,
                normalized_rms=term_rms / total_rms,
                parameters=term.parameters,
            )
        )
    return tuple(result)


def _rms(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("term contribution contains nonfinite values")
    return float(np.sqrt(np.mean(array**2)))


def _thresholds(
    contributions: Sequence[TermContribution],
    epsilon: float,
    maximum: float,
) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                threshold
                for contribution in contributions
                for threshold in (
                    contribution.normalized_rms
                    + max(epsilon, abs(contribution.normalized_rms) * epsilon),
                )
                if threshold <= maximum
            }
        )
    )


def _evaluate_support(
    model: CompiledModel,
    training: DatasetSplit,
    validation: DatasetSplit,
    terms: Sequence[_Term],
    retained: tuple[str, ...],
    threshold: float,
    fit_config: FitConfig,
    baseline_validation_mse: float,
    tolerance: float,
) -> PruningCandidateResult:
    retained_set = set(retained)
    removed = tuple(item.term_id for item in terms if item.term_id not in retained_set)
    try:
        candidate, removed_parameters = _candidate_for_support(
            model.validated.candidate,
            terms,
            retained_set,
            threshold,
        )
        compiled = compile_candidate(candidate, model.validated.context)
        fitted = fit_candidate(compiled, training, validation, fit_config)
        if not fitted.success:
            raise ValueError("refit failed")
        if (
            fitted.training_metrics.failed_trajectories
            or fitted.validation_metrics.failed_trajectories
        ):
            raise ValueError("refit contains failed trajectories")
        accepted = (
            fitted.validation_metrics.normalized_mse
            <= baseline_validation_mse + tolerance
        )
        reason = None if accepted else "validation MSE exceeds configured tolerance"
        return PruningCandidateResult(
            threshold,
            retained,
            removed,
            removed_parameters,
            candidate,
            fitted,
            accepted,
            reason,
        )
    except (ModelValidationError, ArithmeticError, TypeError, ValueError) as exc:
        return PruningCandidateResult(
            threshold,
            retained,
            removed,
            (),
            None,
            None,
            False,
            str(exc),
        )


def _candidate_for_support(
    candidate: CandidateModel,
    terms: Sequence[_Term],
    retained: set[str],
    threshold: float,
) -> tuple[CandidateModel, tuple[str, ...]]:
    by_location: dict[str, list[_Term]] = {}
    for term in terms:
        by_location.setdefault(term.location, []).append(term)
    process_updates = tuple(
        process.model_copy(
            update={
                "expression": _support_expression(
                    by_location[f"process:{process.name}"], retained
                )
            }
        )
        for process in candidate.processes
    )
    equation_updates = tuple(
        equation.model_copy(
            update={
                "rhs": _support_expression(
                    by_location[f"equation:{equation.state}"], retained
                )
            }
        )
        for equation in candidate.state_equations
    )
    sources = (
        tuple(item.expression for item in process_updates)
        + tuple(item.rhs for item in equation_updates)
        + tuple(item.expression for item in candidate.observation_mappings)
    )
    parser = RestrictedParser()
    referenced = {
        symbol
        for index, source in enumerate(sources)
        for symbol in parser.parse(source, location=f"pruned:{index}").symbols
    }
    retained_parameters = tuple(
        parameter
        for parameter in candidate.parameters
        if parameter.name in referenced
    )
    removed_parameters = tuple(
        parameter.name
        for parameter in candidate.parameters
        if parameter.name not in referenced
    )
    constraints = tuple(
        constraint
        for constraint in candidate.constraints
        if constraint.subject not in removed_parameters
    )
    support_key = "\n".join(sorted(retained)).encode()
    suffix = hashlib.sha256(support_key).hexdigest()[:12]
    updated = candidate.model_copy(
        update={
            "candidate_id": f"{candidate.candidate_id}_pruned_{suffix}"[:128],
            "parent_candidate_id": candidate.candidate_id,
            "change_summary": (
                f"Removed {len(terms) - len(retained)} complete additive terms."
            ),
            "processes": process_updates,
            "state_equations": equation_updates,
            "parameters": retained_parameters,
            "constraints": constraints,
        }
    )
    return CandidateModel.model_validate(updated.model_dump()), removed_parameters


def _support_expression(terms: Sequence[_Term], retained: set[str]) -> str:
    kept = [term.parsed.tree.body for term in terms if term.term_id in retained]
    if not kept:
        return "0"
    expression = kept[0]
    for term in kept[1:]:
        expression = ast.BinOp(left=expression, op=ast.Add(), right=term)
    return ast.unparse(ast.fix_missing_locations(expression))
