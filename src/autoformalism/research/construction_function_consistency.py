"""Conservative affine-domain training conflicts, without fitting or rollout.

The finite proof uses declared parameter domains, actual initializer graphs and
authenticated complete forcing. It never substitutes measured state histories
for latent trajectories. Unsupported expressions or missing evidence abstain.
"""

from __future__ import annotations

import ast
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from math import inf, isfinite, nextafter
from typing import Any

from autoformalism.expressions import CompiledModel, compile_candidate
from autoformalism.fitting.initialization_graph import apply_initialization_graph
from autoformalism.research.construction_revision_consistency import (
    TARGET_COMPARISON_POLICY,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.search.topology_compatibility import (
    CompleteTrajectory,
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-function-domain-consistency-1"
Number = Fraction | float


class _Unsupported(ValueError):
    """An expression is outside this finite proof, not an invalid model."""


@dataclass(frozen=True)
class _Interval:
    """Exact rational endpoints, infinities and excluded zero boundaries."""

    low: Number
    high: Number
    low_open: bool = False
    high_open: bool = False

    def __post_init__(self) -> None:
        for endpoint in (self.low, self.high):
            if (
                isinstance(endpoint, Fraction)
                and max(
                    endpoint.numerator.bit_length(), endpoint.denominator.bit_length()
                )
                > 4096
            ):
                raise _Unsupported("exact interval arithmetic exceeded its bit budget")

    @property
    def zero(self) -> bool:
        return self.low == self.high == 0

    @property
    def excludes_zero(self) -> bool:
        return (
            self.low > 0
            or (self.low == 0 and self.low_open)
            or self.high < 0
            or (self.high == 0 and self.high_open)
        )

    def __neg__(self) -> _Interval:
        return _Interval(-self.high, -self.low, self.high_open, self.low_open)

    def __add__(self, other: _Interval) -> _Interval:
        return _Interval(
            self.low + other.low,
            self.high + other.high,
            self.low_open or other.low_open,
            self.high_open or other.high_open,
        )

    def __mul__(self, other: _Interval) -> _Interval:
        # Endpoint limits use 0*infinity=0; no floating arithmetic on finite bounds.
        endpoints = [
            Fraction(0) if a == 0 or b == 0 else a * b
            for a in (self.low, self.high)
            for b in (other.low, other.high)
        ]
        low, high = min(endpoints), max(endpoints)
        excludes = self.excludes_zero and other.excludes_zero
        return _Interval(low, high, low == 0 and excludes, high == 0 and excludes)

    def reciprocal(self) -> _Interval:
        if not self.excludes_zero:
            raise _Unsupported("a denominator can be zero under its declared domain")

        def inverse(value: Number, *, low: bool) -> Number:
            if value in (-inf, inf):
                return Fraction(0)
            if value == 0:
                return -inf if low else inf
            return Fraction(1) / value

        return _Interval(
            inverse(self.high, low=True),
            inverse(self.low, low=False),
            self.high == inf or self.high_open,
            self.low == -inf or self.low_open,
        )


_ZERO = _Interval(Fraction(0), Fraction(0))
_ONE = _Interval(Fraction(1), Fraction(1))
_REAL = _Interval(-inf, inf)
_NONNEGATIVE = _Interval(Fraction(0), inf)


@dataclass
class _Affine:
    """Affine dependence on states/forcing, with interval parameter coefficients."""

    constant: _Interval
    coefficients: dict[str, _Interval]

    def add(self, other: _Affine) -> _Affine:
        coefficients = {
            name: self.coefficients.get(name, _ZERO)
            + other.coefficients.get(name, _ZERO)
            for name in self.coefficients.keys() | other.coefficients.keys()
        }
        return _Affine(
            self.constant + other.constant,
            {name: value for name, value in coefficients.items() if not value.zero},
        )

    def scale(self, value: _Interval) -> _Affine:
        return _Affine(
            self.constant * value,
            {
                name: product
                for name, coefficient in self.coefficients.items()
                if not (product := coefficient * value).zero
            },
        )


def _affine(node: ast.AST, symbols: dict[str, _Affine]) -> _Affine:
    """Interpret only already-restricted syntax; never execute expression text."""
    if isinstance(node, ast.Expression):
        return _affine(node.body, symbols)
    if isinstance(node, ast.Constant):
        value = Fraction(node.value)
        return _Affine(_Interval(value, value), {})
    if isinstance(node, ast.Name):
        if node.id not in symbols:
            raise _Unsupported(f"no supported expansion for {node.id}")
        return symbols[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _affine(node.operand, symbols)
        return value.scale(-_ONE) if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _affine(node.left, symbols), _affine(node.right, symbols)
        if isinstance(node.op, ast.Add):
            return left.add(right)
        if isinstance(node.op, ast.Sub):
            return left.add(right.scale(-_ONE))
        if isinstance(node.op, ast.Mult):
            if not left.coefficients:
                return right.scale(left.constant)
            if not right.coefficients:
                return left.scale(right.constant)
        if isinstance(node.op, ast.Div) and not right.coefficients:
            return left.scale(right.constant.reciprocal())
        if isinstance(node.op, ast.Pow) and not left.coefficients:
            power = right.constant
            if (
                not right.coefficients
                and power.low == power.high
                and power.low.denominator == 1
                and abs(power.low) <= 16
            ):
                exponent = int(power.low)
                base = left.constant.reciprocal() if exponent < 0 else left.constant
                result = _ONE
                for _ in range(abs(exponent)):
                    result = result * base
                return _Affine(result, {})
    raise _Unsupported("expression is outside the affine rational proof grammar")


def _range(expression: _Affine, values: dict[str, _Interval]) -> _Interval:
    """Conservative expression bounds; oversized exact arithmetic abstains."""
    try:
        result = expression.constant
        for name, coefficient in expression.coefficients.items():
            result += coefficient * values.get(name, _REAL)
        return result
    except _Unsupported:
        return _REAL


def _public_ranges(
    row: CompleteTrajectory,
    evidence: TrainingCompatibilityEvidence,
    *,
    initial: bool,
) -> dict[str, _Interval]:
    """Bound complete forcing; target histories are available at time zero only."""
    result = {}
    groups = (
        (row.inputs, row.auxiliaries, row.targets)
        if initial
        else (
            row.inputs,
            row.auxiliaries,
        )
    )
    for group in groups:
        for name, raw in group.items():
            error = evidence.contract.absolute_error_bounds.get(name)
            interpolation = evidence.contract.interpolation.get(name)
            if error is None or (
                not initial
                and interpolation
                not in {
                    "linear_complete_knots",
                    "previous_complete_knots",
                }
            ):
                continue
            values = [Fraction(raw[0])] if initial else list(map(Fraction, raw))
            result[name] = _Interval(
                min(values) - Fraction(error), max(values) + Fraction(error)
            )
    for name, value in row.covariates.items():
        error = evidence.contract.absolute_error_bounds.get(name)
        if error is not None:
            result[name] = _Interval(
                Fraction(value) - Fraction(error), Fraction(value) + Fraction(error)
            )
    result[evidence.context.time_symbol] = _Interval(
        Fraction(row.time[0]), Fraction(row.time[0] if initial else row.time[-1])
    )
    return result


def _invariant(
    equations: dict[str, _Affine],
    initials: dict[str, _Interval],
    forcing: dict[str, _Interval],
) -> set[str]:
    """Greatest proven nonnegative subsystem, with each self-term zero at boundary."""
    candidates = {name for name in equations if initials.get(name, _REAL).low >= 0}
    while True:
        values = {**forcing, **dict.fromkeys(candidates, _NONNEGATIVE)}
        rejected = {
            name
            for name in candidates
            if _range(equations[name], {**values, name: _ZERO}).low < 0
        }
        if not rejected:
            return candidates
        candidates -= rejected


def _render(value: Number, *, lower: bool) -> float | str:
    """Outward-round finite exact bounds for display; proofs retain rationals."""
    if value in (-inf, inf):
        return "-infinity" if value < 0 else "infinity"
    try:
        rendered = float(value)
    except OverflowError:
        return str(value)
    if not isfinite(rendered):
        return str(value)
    if lower and Fraction(rendered) > value:
        return nextafter(rendered, -inf)
    if not lower and Fraction(rendered) < value:
        return nextafter(rendered, inf)
    return rendered


def _witness(
    row: CompleteTrajectory, target: str, slope: Number, error: float, direction: int
) -> dict[str, Any] | None:
    """Find a violated integrated slope bound in linear time using exact numbers."""
    if slope in (-inf, inf):
        return None
    times = list(map(Fraction, row.time))
    values = [direction * Fraction(value) for value in row.targets[target]]
    best_prior, best_index = values[0] - slope * times[0], 0
    best = None
    for index in range(1, len(times)):
        excess = slope * times[index] - values[index] + best_prior - 2 * Fraction(error)
        if excess > 0:
            guard = Fraction(TARGET_COMPARISON_POLICY["absolute_tolerance"]) + Fraction(
                TARGET_COMPARISON_POLICY["relative_tolerance"]
            ) * max(abs(values[best_index]), abs(values[index]))
            if excess > guard and (best is None or excess > best[0]):
                best = excess, best_index, index, guard
        candidate = values[index] - slope * times[index]
        if candidate > best_prior:
            best_prior, best_index = candidate, index
    if best is None:
        return None
    excess, first, last, guard = best
    bound = slope * (times[last] - times[first])
    delta = Fraction(row.targets[target][last]) - Fraction(row.targets[target][first])
    return {
        "trajectory": row.trajectory_id,
        "sample_indices": [first, last],
        "times": [row.time[first], row.time[last]],
        "observations": [row.targets[target][first], row.targets[target][last]],
        "absolute_error_bound": error,
        "observed_change_interval": [
            _render(delta - 2 * Fraction(error), lower=True),
            _render(delta + 2 * Fraction(error), lower=False),
        ],
        "required_change": {
            "comparison": "at_least" if direction == 1 else "at_most",
            "value": _render(direction * bound, lower=direction == 1),
        },
        "contradiction_margin_lower_bound": _render(excess, lower=True),
        "target_numerical_guard": _render(guard, lower=False),
    }


def assess(problem: PrefixFitProblem, evidence: dict) -> dict[str, Any]:
    """Prove only qualified affine monotonicity conflicts in sealed training data.

    Candidate constraints and numerical guesses never substitute for dynamical
    invariance. Legacy numerical parameter bounds are ignored (a safe relaxation).
    Non-identity readouts and unsupported formulas abstain rather than pass.
    """
    if not isinstance(problem, PrefixFitProblem):
        raise ValueError("typed PrefixFitProblem required")
    problem = PrefixFitProblem.model_validate(problem.model_dump(mode="json"))
    training = TrainingCompatibilityEvidence.model_validate(evidence)
    validate_evidence(training)
    if training.context != problem.context:
        raise ValueError("training evidence context differs from problem context")
    if problem.context.fitted_initialization or problem.context.lagged_targets:
        raise ValueError("unlowered open-loop problem required")
    if any(p.scope.value != "global" for p in problem.base_candidate.parameters):
        raise ValueError("dynamics parameters must be globally shared")
    original = compile_candidate(problem.base_candidate, problem.context)
    model, _, _ = apply_initialization_graph(original, problem.graph)
    required = (
        "dependencies_exhaustive",
        "shared_parameters",
        "open_loop",
        "reset_impulse_free",
        "uniqueness_assumed",
    )
    missing = [
        name for name in required if getattr(training.contract, name) is not True
    ]
    checks: list[dict[str, Any]] = []
    if missing:
        checks.append(
            {
                "property": "candidate_class_contract",
                "status": "deferred",
                "unavailable": missing,
            }
        )
    else:
        checks = _checks(problem, original, model, training)
    conflicts = [item for item in checks if item["status"] == "conditional_conflict"]
    deferred = [item for item in checks if item["status"] == "deferred"]
    return {
        "protocol": PROTOCOL,
        "status": "conflict"
        if conflicts
        else "deferred"
        if deferred
        else "no_conflict_found",
        "child_problem_sha256": content_hash(problem.model_dump(mode="json")),
        "evidence_sha256": training.sha256,
        "checks": checks,
        "conflicts": conflicts,
        "deferred": deferred,
        "assumptions": [
            "Finite domain-admissible constant parameters and well-posed affine "
            "ODEs on the witnessed interval.",
            "Actual initializer domains and complete authenticated forcing apply; "
            "no resets or impulses occur.",
            "Contradictions concern simultaneous reproduction within declared "
            "observation errors, not useful approximate fitting.",
        ],
        "coverage": "Affine rational expressions, nonnegative invariant subsets "
        "and direct-state output slope bounds only; absence of a conflict is "
        "not scientific acceptance.",
        "target_comparison_policy": dict(TARGET_COMPARISON_POLICY),
        "contract_provenance": training.contract.provenance,
        "scientific_correctness_established": False,
        "parameter_fitting_performed": False,
        "numerical_rollout_performed": False,
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }


def prompt_view(report: dict[str, Any]) -> dict[str, Any]:
    """Keep one witnessed conflict per target; full proofs remain in the receipt."""
    selected = {}
    for conflict in report["conflicts"]:
        target = conflict["target"]
        if target not in selected:
            witness = conflict["witnesses"][0]
            selected[target] = {
                key: conflict[key]
                for key in (
                    "target",
                    "state",
                    "model_relation",
                    "proven_nonnegative_states",
                    "derivative_interval",
                    "repair_feedback",
                )
            } | {
                "witness": witness,
                "complete_forcing_ranges": conflict["proof_inputs"][
                    "complete_forcing_intervals"
                ],
            }
    return {
        "protocol": report["protocol"],
        "status": report["status"],
        "assumptions": report["assumptions"],
        "coverage": report["coverage"],
        "status_counts": dict(Counter(item["status"] for item in report["checks"])),
        "conflicts": [selected[target] for target in sorted(selected)],
        "full_conflict_count": len(report["conflicts"]),
        "evidence_sha256": report["evidence_sha256"],
        "child_problem_sha256": report["child_problem_sha256"],
    }


def _checks(
    problem: PrefixFitProblem,
    original: CompiledModel,
    model: CompiledModel,
    training: TrainingCompatibilityEvidence,
) -> list[dict[str, Any]]:
    """Build immutable equation proofs and apply each to individual training rows."""
    parameters = {
        item.name: _Affine(
            _Interval(Fraction(0), inf, item.domain.value == "positive")
            if item.domain.value != "real"
            else _REAL,
            {},
        )
        for item in model.validated.candidate.parameters
    }
    symbols = {
        **parameters,
        **{
            name: _Affine(_ZERO, {name: _ONE})
            for name in (
                *model.state_names,
                *problem.context.forcing_channels,
                problem.context.time_symbol,
            )
        },
    }
    for name in model.validated.process_order:
        with suppress(_Unsupported):
            symbols[name] = _affine(
                model.validated.process_expressions[name].tree, symbols
            )
    equations, unavailable = {}, {}
    for name, expression in model.validated.equation_expressions.items():
        try:
            equations[name] = _affine(expression.tree, symbols)
        except _Unsupported as exc:
            unavailable[name] = str(exc)
    direct = {
        channel: state
        for state, channel in original.direct_state_observation_channels.items()
    }
    checks = []
    for row in training.trajectories:
        # A modeled auxiliary owns its generated trajectory. Its public recording
        # may initialize that state, but must never be treated as ongoing forcing.
        forcing = {
            name: interval
            for name, interval in _public_ranges(row, training, initial=False).items()
            if name not in model.state_names
        }
        initial_public = _public_ranges(row, training, initial=True)
        initial_symbols = {
            **parameters,
            **{name: _Affine(value, {}) for name, value in initial_public.items()},
        }
        initials = {}
        for spec in model.validated.candidate.initial_conditions:
            try:
                if spec.fixed_value is not None:
                    initials[spec.state] = _Interval(
                        Fraction(spec.fixed_value), Fraction(spec.fixed_value)
                    )
                else:
                    initials[spec.state] = _affine(
                        model.validated.initial_condition_expressions[spec.state].tree,
                        initial_symbols,
                    ).constant
            except _Unsupported:
                initials[spec.state] = _REAL
        positive = _invariant(equations, initials, forcing)
        values = {**forcing, **dict.fromkeys(positive, _NONNEGATIVE)}
        for target in problem.context.targets:
            state = direct.get(target)
            check = {
                "property": "affine_state_derivative_bound",
                "target": target,
                "trajectory": row.trajectory_id,
                "status": "deferred",
            }
            checks.append(check)
            error = training.contract.absolute_error_bounds.get(target)
            if state not in equations or error is None or len(row.time) < 2:
                check["unavailable"] = [
                    unavailable.get(
                        state,
                        "direct-state readout, known target error and at least "
                        "two times required",
                    )
                ]
                continue
            derivative = _range(equations[state], values)
            check.update(
                state=state,
                model_relation=f"d({state})/dt = "
                f"{model.validated.equation_expressions[state].source}",
                proven_nonnegative_states=sorted(positive),
                derivative_interval=[
                    _render(derivative.low, lower=True),
                    _render(derivative.high, lower=False),
                ],
                parameter_domains={
                    item.name: item.domain.value
                    for item in model.validated.candidate.parameters
                },
                witnesses=[],
            )
            for slope, direction in ((derivative.low, 1), (-derivative.high, -1)):
                witness = _witness(row, target, slope, error, direction)
                if witness:
                    check["witnesses"].append(witness)
            if check["witnesses"]:
                check["status"] = "conditional_conflict"
                check["proof_inputs"] = {
                    "initial_intervals": {
                        name: [
                            _render(value.low, lower=True),
                            _render(value.high, lower=False),
                        ]
                        for name, value in sorted(initials.items())
                    },
                    "complete_forcing_intervals": {
                        name: [
                            _render(value.low, lower=True),
                            _render(value.high, lower=False),
                        ]
                        for name, value in sorted(forcing.items())
                    },
                    "state_equations": {
                        name: expression.source
                        for name, expression in sorted(
                            model.validated.equation_expressions.items()
                        )
                    },
                    "process_definitions": {
                        name: expression.source
                        for name, expression in sorted(
                            model.validated.process_expressions.items()
                        )
                    },
                    "invariance_rule": "At each nonnegative subsystem boundary, "
                    "the state's self-term vanishes and the remaining derivative "
                    "is nonnegative for every admitted parameter and forcing.",
                }
                check["repair_feedback"] = (
                    "The assembled differential equation and declared domains "
                    "force a change incompatible with these training measurements. "
                    "Revise the scientifically intended function, sign, dependency, "
                    "parameter domain or initializer where justified. "
                    "Preserve required public mechanisms; removing this diagnostic "
                    "or acknowledging the limitation does not resolve the conflict."
                )
            elif derivative.low != -inf or derivative.high != inf:
                check["status"] = "no_conflict_found"
            else:
                check["unavailable"] = [
                    "no finite derivative bound is proved from the actual "
                    "initializers, parameter domains and authenticated complete forcing"
                ]
    return checks
