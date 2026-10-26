"""Finite exact affine-family necessary conditions, without fitting or simulation.

The checks use actual process expansions, closed dynamic ancestors and scoped
initializers. Exact time-zero reproduction is essential: no error bound for an
approximate forecast is inferred from a superposition contradiction.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations, islice
from math import isfinite
from typing import Any

from autoformalism.expressions import CompiledModel, compile_candidate
from autoformalism.fitting.initialization_graph import apply_initialization_graph
from autoformalism.research.construction_function_consistency import (
    _ONE,
    _REAL,
    _ZERO,
    _Affine,
    _affine,
    _Interval,
    _Unsupported,
)
from autoformalism.research.construction_initial_family_consistency import (
    LIMITS as INITIAL_READOUT_LIMITS,
)
from autoformalism.research.construction_initial_family_consistency import (
    _Affine as _InitialAffine,
)
from autoformalism.research.construction_initial_family_consistency import (
    _Budget as _InitialBudget,
)
from autoformalism.research.construction_initial_family_consistency import (
    _InitialView,
)
from autoformalism.research.construction_initial_family_consistency import (
    _interpret as _initial_expression,
)
from autoformalism.research.construction_initial_family_consistency import (
    _Unsupported as _InitialUnsupported,
)
from autoformalism.research.construction_revision_consistency import (
    TARGET_COMPARISON_POLICY,
    _forcing_equal,
    _initial_key,
    _initial_sources,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.search.topology_compatibility import (
    CompleteTrajectory,
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-affine-family-consistency-1"
SYMBOLIC_INITIAL_POLICY = "shared-symbolic-initial-readout-1"
WEIGHTED_SYMBOLIC_POLICY = "weighted-shared-symbolic-initials-1"
COMBINED_SYMBOLIC_POLICY = "combined-symbolic-map-readout-initials-1"
MAX_TRIPLES = 256
MAX_TIMES = 64
MAX_STATES = 32
MAX_TRAJECTORIES = 32
MAX_SAMPLES = 8192
MAX_BITS = 4096


def _bounded(value: Fraction) -> Fraction:
    """Abort rather than silently approximate oversized rational elimination."""
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise _Unsupported("exact elimination exceeded its bit budget")
    return value


def _display(value: Fraction) -> float:
    """Refuse a nonfinite display instead of corrupting an exact certificate."""
    try:
        result = float(value)
    except OverflowError as exc:
        raise _Unsupported("witness display exceeds finite numeric range") from exc
    if not isfinite(result):
        raise _Unsupported("witness display exceeds finite numeric range")
    return result


def _rref(
    rows: list[list[Fraction]], width: int
) -> tuple[list[list[Fraction]], list[int]]:
    """Bounded exact row reduction; no fitted coefficients or rank tolerances."""
    matrix = [list(row) for row in rows]
    pivot_columns: list[int] = []
    for column in range(width):
        pivot = next(
            (i for i in range(len(pivot_columns), len(matrix)) if matrix[i][column]),
            None,
        )
        if pivot is None:
            continue
        index = len(pivot_columns)
        matrix[index], matrix[pivot] = matrix[pivot], matrix[index]
        divisor = matrix[index][column]
        matrix[index] = [_bounded(value / divisor) for value in matrix[index]]
        for i, row in enumerate(matrix):
            if i != index and row[column]:
                multiplier = row[column]
                matrix[i] = [
                    _bounded(a - multiplier * b)
                    for a, b in zip(row, matrix[index], strict=True)
                ]
        pivot_columns.append(column)
    return matrix, pivot_columns


def _null_vectors(rows: list[list[Fraction]], width: int) -> list[list[Fraction]]:
    matrix, pivots = _rref(rows, width)
    result = []
    for free in set(range(width)) - set(pivots):
        vector = [Fraction(0)] * width
        vector[free] = Fraction(1)
        for index, pivot in enumerate(pivots):
            vector[pivot] = -matrix[index][free]
        scale = max(map(abs, vector))
        result.append([_bounded(value / scale) for value in vector])
    return result


def _exact(interval: _Interval) -> Fraction:
    if not isinstance(interval.low, Fraction) or interval.low != interval.high:
        raise _Unsupported("an initial constraint has an unknown shared coefficient")
    return interval.low


def _numeric(expression: _Affine) -> tuple[Fraction, dict[str, Fraction]]:
    return _exact(expression.constant), {
        name: _exact(value) for name, value in expression.coefficients.items()
    }


def _public_initial(
    row: CompleteTrajectory, training: TrainingCompatibilityEvidence
) -> dict[str, _Affine]:
    values = {training.context.time_symbol: row.time[0], **row.covariates}
    for group in (row.targets, row.inputs, row.auxiliaries):
        values.update(
            (name, data[0])
            for name, data in group.items()
            if training.contract.absolute_error_bounds.get(name) == 0
        )
    # Covariates also need exact conditioning, even though they have one value.
    values = {
        name: value
        for name, value in values.items()
        if name == training.context.time_symbol
        or training.contract.absolute_error_bounds.get(name) == 0
    }
    return {
        name: _Affine(_Interval(Fraction(value), Fraction(value)), {})
        for name, value in values.items()
    }


def _known_initials(model, row, training, symbols, outputs) -> dict[str, Fraction]:
    """Deduce numeric state initials from actual maps and exact algebraic readouts.

    Unknown maps/root parameters are relaxed. Numeric output equations can pin a
    root (for example y(0)=a(0)+z(0)); unpinned roots remain unknown.
    """
    states = list(model.state_names)
    public = _public_initial(row, training)
    constraints = []
    initial_symbols = {**symbols, **public}
    for spec in model.validated.candidate.initial_conditions:
        try:
            expression = (
                _Affine(
                    _Interval(Fraction(spec.fixed_value), Fraction(spec.fixed_value)),
                    {},
                )
                if spec.fixed_value is not None
                else _affine(
                    model.validated.initial_condition_expressions[spec.state].tree,
                    initial_symbols,
                )
            )
            constant, coefficients = _numeric(expression)
            if coefficients:
                continue  # Lowered initials may only use public time-zero values.
            constraints.append(
                [Fraction(name == spec.state) for name in states] + [constant]
            )
        except _Unsupported:
            continue
    for channel, expression in outputs.items():
        if (
            channel not in row.targets
            or training.contract.absolute_error_bounds.get(channel) != 0
        ):
            continue
        try:
            constant, coefficients = _numeric(expression)
            for name, coefficient in coefficients.items():
                if name not in states:
                    constant += coefficient * _exact(public[name].constant)
            constraints.append(
                [coefficients.get(name, Fraction(0)) for name in states]
                + [Fraction(row.targets[channel][0]) - constant]
            )
        except (_Unsupported, KeyError):
            continue
    matrix, pivots = _rref(constraints, len(states))
    if any(not any(row[:-1]) and row[-1] for row in matrix):
        raise _Unsupported(
            "time-zero constraints are inconsistent; use the initializer-family check"
        )
    free = set(range(len(states))) - set(pivots)
    return {
        states[pivot]: matrix[i][-1]
        for i, pivot in enumerate(pivots)
        if not any(matrix[i][column] for column in free)
    }


def _readout_initials(
    model,
    row: CompleteTrajectory,
    training: TrainingCompatibilityEvidence,
    numeric: dict[str, Fraction],
    shared_parameters: set[str],
) -> dict[str, _InitialAffine]:
    """Infer affine shared values by isolating one numerically weighted state.

    Substitution uses only already established exact initials. A remaining state
    may be solved only with a nonzero rational coefficient; unknown-parameter
    division and products of unresolved coordinates are unsupported. In
    particular, ``y=a+x+c*b`` pins ``x=y-a-c*b`` when a/b are known, even though
    its value is not numeric. Trajectory root coordinates are never shared here.
    """
    values = {name: _InitialAffine(value) for name, value in numeric.items()}
    states = set(model.state_names)
    public = _public_initial(row, training)
    budget = _InitialBudget()

    def resolve(name: str) -> _InitialAffine:
        if name in states:
            return values.get(
                name, _InitialAffine(coefficients={name: Fraction(1)})
            )
        if name in model.validated.process_expressions:
            return _initial_expression(
                model.validated.process_expressions[name].tree, resolve, budget
            )
        if name in shared_parameters:
            return _InitialAffine(coefficients={name: Fraction(1)})
        if name in public:
            return _InitialAffine(_exact(public[name].constant))
        raise _InitialUnsupported(f"unavailable exact initial symbol: {name}")

    for _ in states:
        changed = False
        for channel, parsed in model.validated.observation_expressions.items():
            if (
                channel not in row.targets
                or training.contract.absolute_error_bounds.get(channel) != 0
            ):
                continue
            try:
                expression = _initial_expression(parsed.tree, resolve, budget)
                unknown = states & expression.coefficients.keys()
                if len(unknown) != 1:
                    continue
                state = next(iter(unknown))
                coefficient = expression.coefficients[state]
                value = _InitialAffine(
                    budget.check(
                        Fraction(row.targets[channel][0]) - expression.constant
                    ),
                    {
                        name: -value
                        for name, value in expression.coefficients.items()
                        if name != state
                    },
                ).scale(budget.check(1 / coefficient), budget)
                if len(value.coefficients) > INITIAL_READOUT_LIMITS["coordinates"]:
                    raise _InitialUnsupported(
                        "initial readout exceeds shared-coordinate limit"
                    )
                values[state] = value
                changed = True
            except (_InitialUnsupported, _Unsupported):
                # A proof limit or unsupported expression cannot create equality.
                if budget.remaining < 0:
                    return values
        if not changed:
            break
    return values


def _common_readout_initial(state, rows, readout_initials):
    """Return the identical shared expression only when all rows establish it."""
    values = [readout_initials.get(row.trajectory_id, {}).get(state) for row in rows]
    if values[0] is None or any(value != values[0] for value in values[1:]):
        return None
    value = values[0]
    return {
        "source": "exact initial readout with shared symbolic parameters",
        "policy": SYMBOLIC_INITIAL_POLICY,
        "constant_exact": str(value.constant),
        "shared_coefficients_exact": {
            name: str(coefficient)
            for name, coefficient in sorted(value.coefficients.items())
        },
    }


def _shared_map_initials(
    model: CompiledModel,
    rows: Sequence[CompleteTrajectory],
    training: TrainingCompatibilityEvidence,
    roots: set[str],
) -> dict[str, dict[str, _InitialAffine]]:
    """Expand actual initial maps into shared-coordinate affine expressions.

    These expressions may differ between trajectories. A later nullspace check
    cancels every coefficient independently, so its weights cannot depend on a
    fitted parameter. Relevant trajectory roots, uncertain public readings and
    products/divisions of unknown shared coordinates remain unsupported.
    """
    result = {}
    for row in rows:
        budget = _InitialBudget()
        view = _InitialView(model, roots, training, row, budget)
        values = {}
        for state in sorted(model.state_names):
            try:
                value = view.symbol(state)
                if len(value.coefficients) > INITIAL_READOUT_LIMITS["coordinates"]:
                    continue
                values[state] = value
            except _InitialUnsupported:
                # A failed recursive expansion must not poison another state's
                # independent proof. Cached completed expressions remain valid.
                view.visiting.clear()
                if budget.remaining < 0:
                    break
        result[row.trajectory_id] = values
    return result


def _weighted_initial_constraints(
    state: str,
    rows: Sequence[CompleteTrajectory],
    map_initials: dict[str, dict[str, _InitialAffine]],
    *, policy: str = WEIGHTED_SYMBOLIC_POLICY,
) -> tuple[list[list[Fraction]], dict[str, Any]] | None:
    """Return exact rows whose nullspace cancels one shared symbolic initial."""
    values = [map_initials.get(row.trajectory_id, {}).get(state) for row in rows]
    if any(value is None for value in values):
        return None
    names = sorted({name for value in values for name in value.coefficients})
    if len(names) > INITIAL_READOUT_LIMITS["coordinates"]:
        return None
    constraints = [[value.constant for value in values]] + [
        [value.coefficients.get(name, Fraction(0)) for value in values]
        for name in names
    ]
    return constraints, {
        "source": (
            "actual initializer maps combined with exact observed initial readouts"
            if policy == COMBINED_SYMBOLIC_POLICY
            else "actual initializer maps with shared symbolic parameters"
        ),
        "policy": policy,
        "constant_exact": [str(value.constant) for value in values],
        "shared_coefficients_exact": {
            name: [str(value.coefficients.get(name, Fraction(0))) for value in values]
            for name in names
        },
        "required_cancellation": "weights cancel the constant and every shared "
        "parameter coefficient separately; no parameter values are fitted",
    }


class _RootInitialView(_InitialView):
    """Keep trajectory roots distinct until exact readouts constrain them.

    Root names are algebraic unknowns within one trajectory, never shared
    coordinates. The caller discards every final initial expression still
    containing one. Initializer chains are already lowered by the compiler.
    """

    def initial_symbol(self, name: str) -> _InitialAffine:
        if name in self.roots:
            return _InitialAffine(coefficients={name: Fraction(1)})
        return super().initial_symbol(name)


def _root_reduction(
    matrix: list[list[Fraction]], width: int, budget: _InitialBudget
) -> tuple[list[list[Fraction]], list[int]]:
    """Eliminate only rational root coefficients, leaving shared symbols free."""
    matrix = [list(row) for row in matrix]
    pivots = []
    for column in range(width):
        pivot = next(
            (i for i in range(len(pivots), len(matrix)) if matrix[i][column]), None
        )
        if pivot is None:
            continue
        index = len(pivots)
        matrix[index], matrix[pivot] = matrix[pivot], matrix[index]
        divisor = matrix[index][column]
        matrix[index] = [budget.check(v / divisor) for v in matrix[index]]
        for i, row in enumerate(matrix):
            if i != index and row[column]:
                multiplier = row[column]
                matrix[i] = [
                    budget.check(a - multiplier * b)
                    for a, b in zip(row, matrix[index], strict=True)
                ]
        pivots.append(column)
    return matrix, pivots


def _combined_initial_row(
    model: CompiledModel, row: CompleteTrajectory,
    training: TrainingCompatibilityEvidence, roots: set[str],
) -> dict[str, _InitialAffine]:
    """Substitute exact initial readouts into affine lowered initializer maps.

    All shared parameters remain symbolic. Rational elimination may express one
    trajectory root in other roots; only final state initials free of every root
    qualify for cross-trajectory cancellation. Unknown-parameter pivots and
    products of unknown coordinates are outside this bounded proof.
    """
    budget = _InitialBudget()
    view = _RootInitialView(model, roots, training, row, budget)
    values, constraints = {}, []
    if len(model.validated.observation_expressions) > INITIAL_READOUT_LIMITS[
        "coordinates"
    ]:
        raise _InitialUnsupported("initial readout count exceeds coordinate limit")
    for state in sorted(model.state_names):
        try:
            values[state] = view.symbol(state)
        except _InitialUnsupported:
            view.visiting.clear()
            if budget.remaining < 0:
                raise
    for channel, parsed in sorted(model.validated.observation_expressions.items()):
        if (
            channel not in row.targets
            or training.contract.absolute_error_bounds.get(channel) != 0
        ):
            continue
        try:
            expression = _initial_expression(parsed.tree, view.symbol, budget)
            constraints.append(
                expression.add(
                    _InitialAffine(-Fraction(row.targets[channel][0])), budget
                )
            )
        except _InitialUnsupported:
            view.visiting.clear()
            if budget.remaining < 0:
                raise
    names = set().union(
        *(v.coefficients for v in [*values.values(), *constraints])
    )
    if len(names) > INITIAL_READOUT_LIMITS["coordinates"]:
        raise _InitialUnsupported("combined initial coordinates exceed limit")
    root_names, shared_names = sorted(names & roots), sorted(names - roots)
    if not set(shared_names) <= set(model.parameter_names) - roots:
        raise _InitialUnsupported("unclassified combined initial coordinate")
    matrix = [
        [value.coefficients.get(name, Fraction(0)) for name in root_names]
        + [-value.constant]
        + [-value.coefficients.get(name, Fraction(0)) for name in shared_names]
        for value in constraints
    ]
    reduced, pivots = _root_reduction(matrix, len(root_names), budget)
    substitutions = {}
    for index, pivot in enumerate(pivots):
        line = reduced[index]
        coefficients = {
            name: -line[i]
            for i, name in enumerate(root_names)
            if i != pivot and line[i]
        }
        coefficients.update(
            (name, line[len(root_names) + 1 + i])
            for i, name in enumerate(shared_names)
            if line[len(root_names) + 1 + i]
        )
        substitutions[root_names[pivot]] = _InitialAffine(
            line[len(root_names)], coefficients
        )
    qualified = {}
    for state, initial in values.items():
        value = _InitialAffine(initial.constant)
        for name, coefficient in sorted(initial.coefficients.items()):
            term = substitutions.get(name, _InitialAffine(coefficients={name: 1}))
            value = value.add(term.scale(coefficient, budget), budget)
        if not roots.intersection(value.coefficients):
            qualified[state] = value
    return qualified


def _combined_initials(
    model: CompiledModel, rows: Sequence[CompleteTrajectory],
    training: TrainingCompatibilityEvidence, roots: set[str],
) -> dict[str, dict[str, _InitialAffine]]:
    """Return only completed row certificates; proof exhaustion never guesses."""
    result = {}
    for row in rows:
        try:
            result[row.trajectory_id] = _combined_initial_row(
                model, row, training, roots
            )
        except _InitialUnsupported:
            result[row.trajectory_id] = {}
    return result


def _closed_states(
    output: _Affine, equations: dict[str, _Affine], state_names: set[str]
) -> tuple[set[str], set[str]]:
    pending = set(output.coefficients)
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        if name in state_names:
            if name not in equations:
                raise _Unsupported(f"non-affine or unsupported state equation: {name}")
            pending.update(equations[name].coefficients.keys() - visited)
    return visited & state_names, visited - state_names


def _exact_initial_key(state, row, problem, observed, training):
    public, roots = _initial_sources(state, problem, observed)
    if roots or any(
        name != problem.context.time_symbol
        and training.contract.absolute_error_bounds.get(name) != 0
        for name in public
    ):
        return None
    return _initial_key(state, row, problem, observed)


def _common_rows(rows, forcing, training) -> bool:
    if len({row.time[0] for row in rows}) != 1:
        return False
    start, end = rows[0].time[0], min(row.time[-1] for row in rows)
    return all(
        _forcing_equal(rows[0], row, forcing, training.contract, start, end)
        for row in rows[1:]
    )


def _times(rows) -> list[float]:
    common = sorted(set(rows[0].time).intersection(*(row.time for row in rows[1:])))
    if len(common) <= MAX_TIMES:
        return common
    # Preserve early samples and spread the remaining bounded witness budget.
    early = common[: MAX_TIMES // 2]
    return sorted(
        set(
            early
            + [
                common[(len(common) - 1) * i // (MAX_TIMES // 2 - 1)]
                for i in range(MAX_TIMES // 2)
            ]
        )
    )


def _guard(values) -> Fraction:
    return _bounded(
        Fraction(TARGET_COMPARISON_POLICY["absolute_tolerance"])
        + Fraction(TARGET_COMPARISON_POLICY["relative_tolerance"])
        * max(map(abs, values), default=Fraction(0))
    )


def _superposition(
    rows, target, states, initials, problem, observed, training, readout_initials,
    *, map_initials=None, map_initial_policy=WEIGHTED_SYMBOLIC_POLICY,
):
    constraints = [[Fraction(1)] * 3]
    known = {}
    for state in sorted(states):
        values = [initials[row.trajectory_id].get(state) for row in rows]
        if all(value is not None for value in values):
            constraints.append(values)
            known[state] = [_display(value) for value in values]
            continue
        common = _common_readout_initial(state, rows, readout_initials)
        if common is not None:
            known[state] = common
            continue
        weighted = _weighted_initial_constraints(
            state, rows, map_initials or {}, policy=map_initial_policy
        )
        if weighted is not None:
            coefficient_rows, known[state] = weighted
            constraints.extend(coefficient_rows)
            continue
        keys = [
            _exact_initial_key(state, row, problem, observed, training) for row in rows
        ]
        if keys[0] is None or not all(key == keys[0] for key in keys):
            return None
        # A shared unknown constant cancels because the weights sum to zero.
        known[state] = "same from actual initializer graph"
    for weights in _null_vectors(constraints, 3):
        for time in _times(rows):
            indices = [row.time.index(time) for row in rows]
            values = [
                Fraction(row.targets[target][index])
                for row, index in zip(rows, indices, strict=True)
            ]
            defect = _bounded(
                sum(
                    (
                        weight * value
                        for weight, value in zip(weights, values, strict=True)
                    ),
                    Fraction(0),
                )
            )
            guard = _bounded(_guard(values) * sum(map(abs, weights)))
            if abs(defect) > guard:
                return {
                    "trajectories": [row.trajectory_id for row in rows],
                    "time": time,
                    "sample_indices": indices,
                    "observations": [_display(value) for value in values],
                    "weights_exact": [str(weight) for weight in weights],
                    "weights": [_display(weight) for weight in weights],
                    "weighted_observation_defect_exact": str(defect),
                    "weighted_observation_defect": _display(defect),
                    "target_numerical_guard": _display(guard),
                    "initial_state_values": known,
                    "required_identity": "weighted observed output sum = 0 "
                    "at every common time",
                    "scope": "Requires exact time-zero reproduction. No bound on "
                    "approximate forecast error or NMSE is claimed.",
                }
    return None


def _rank_one(
    rows, target, states, initials, problem, observed, training, readout_initials
):
    varying = []
    common_readouts = {}
    for state in sorted(states):
        values = [initials[row.trajectory_id].get(state) for row in rows]
        if all(value is not None for value in values) and len(set(values)) == 1:
            continue
        common = _common_readout_initial(state, rows, readout_initials)
        if common is not None:
            common_readouts[state] = common
            continue
        keys = [
            _exact_initial_key(state, row, problem, observed, training) for row in rows
        ]
        if keys[0] is not None and all(key == keys[0] for key in keys):
            continue
        varying.append(state)
    if len(varying) != 1:
        return None
    for first, second in combinations(_times(rows), 2):
        values = [
            [Fraction(row.targets[target][row.time.index(time)]) for row in rows]
            for time in (first, second)
        ]
        difference = [[line[1] - line[0], line[2] - line[0]] for line in values]
        a, b = difference[0]
        c, d = difference[1]
        determinant = _bounded(a * d - b * c)
        # Perturb every recorded output by the guard; a difference changes by
        # at most 2g, so |delta(ad-bc)| <= 2g*sum(|entries|)+8g^2.
        guard = _guard([value for line in values for value in line])
        determinant_guard = _bounded(
            2 * guard * sum(map(abs, (a, b, c, d))) + 8 * guard * guard
        )
        if abs(determinant) > determinant_guard:
            return {
                "trajectories": [row.trajectory_id for row in rows],
                "times": [first, second],
                "observations": [
                    [_display(value) for value in line] for line in values
                ],
                "difference_matrix_exact": [
                    [str(value) for value in line] for line in difference
                ],
                "difference_matrix": [
                    [_display(value) for value in line] for line in difference
                ],
                "determinant_exact": str(determinant),
                "determinant": _display(determinant),
                "determinant_numerical_guard": _display(determinant_guard),
                "possibly_different_initial_states": varying,
                **(
                    {"common_initial_readout_values": common_readouts}
                    if common_readouts
                    else {}
                ),
                "required_identity": "trajectory-difference matrix has rank at most "
                "one; every 2 by 2 determinant is zero",
                "scope": "Allows arbitrary trajectory-specific values of the one "
                "varying initial state; no fitting or initial-root equality "
                "is assumed.",
            }
    return None


def _checks(
    problem, model, training, *, symbolic_initials=False,
    weighted_symbolic_initials=False, combined_symbolic_initials=False,
    roots=frozenset(),
):
    rows_to_check = training.trajectories[:MAX_TRAJECTORIES]
    if any(len(row.time) > MAX_SAMPLES for row in rows_to_check):
        raise _Unsupported("complete forcing sample count exceeds the proof budget")
    parameters = {
        item.name: _Affine(_REAL, {}) for item in model.validated.candidate.parameters
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
    # Positive parameter domains only establish denominators are nonzero. They
    # are not used as fitted coefficient values or state positivity assumptions.
    for item in model.validated.candidate.parameters:
        if item.domain.value == "positive":
            symbols[item.name] = _Affine(_Interval(Fraction(0), float("inf"), True), {})
    unavailable = {}
    for name in model.validated.process_order:
        try:
            symbols[name] = _affine(
                model.validated.process_expressions[name].tree, symbols
            )
        except _Unsupported as exc:
            unavailable[name] = str(exc)
    equations, outputs = {}, {}
    for collection, parsed in (
        (equations, model.validated.equation_expressions),
        (outputs, model.validated.observation_expressions),
    ):
        for name, expression in parsed.items():
            try:
                collection[name] = _affine(expression.tree, symbols)
            except _Unsupported as exc:
                unavailable[name] = str(exc)
    observed = compile_candidate(
        problem.base_candidate, problem.context
    ).direct_state_observation_channels
    initials = {
        row.trajectory_id: _known_initials(model, row, training, symbols, outputs)
        for row in rows_to_check
    }
    readout_initials = (
        {
            row.trajectory_id: _readout_initials(
                model,
                row,
                training,
                initials[row.trajectory_id],
                {parameter.name for parameter in problem.base_candidate.parameters},
            )
            for row in rows_to_check
        }
        if symbolic_initials
        else {}
    )
    map_initials = (
        _shared_map_initials(model, rows_to_check, training, roots)
        if weighted_symbolic_initials or combined_symbolic_initials
        else {}
    )
    if combined_symbolic_initials:
        combined = _combined_initials(model, rows_to_check, training, roots)
        # An unsupported readout cannot erase an already established map proof.
        map_initials = {
            row.trajectory_id: {
                **map_initials[row.trajectory_id], **combined[row.trajectory_id]
            }
            for row in rows_to_check
        }
    result = []
    for target in problem.context.targets:
        for kind, test in (
            ("affine_initial_superposition", _superposition),
            ("linear_initial_difference_rank_one", _rank_one),
        ):
            check = {
                "property": kind,
                "target": target,
                "status": "deferred",
                "witnesses": [],
            }
            result.append(check)
            if training.contract.absolute_error_bounds.get(target) != 0:
                check["unavailable"] = [
                    "exact target observations are required; nonzero/unknown "
                    "error is not treated as exact"
                ]
                continue
            if target not in outputs:
                check["unavailable"] = [
                    unavailable.get(target, "unsupported affine readout")
                ]
                continue
            try:
                states, forcing = _closed_states(
                    outputs[target], equations, set(model.state_names)
                )
                forcing.discard(problem.context.time_symbol)
                if any(
                    training.contract.absolute_error_bounds.get(name) != 0
                    or (
                        name not in problem.context.fixed_covariates
                        and training.contract.interpolation.get(name)
                        not in {"linear_complete_knots", "previous_complete_knots"}
                    )
                    for name in forcing
                ):
                    raise _Unsupported(
                        "exact complete forcing and authenticated interpolation "
                        "are required"
                    )
                if not states:
                    raise _Unsupported("no dynamic state in the output closure")
                common_count = 0
                for rows in islice(combinations(rows_to_check, 3), MAX_TRIPLES):
                    if not _common_rows(rows, forcing, training):
                        continue
                    common_count += 1
                    witness = test(
                        rows,
                        target,
                        states,
                        initials,
                        problem,
                        observed,
                        training,
                        readout_initials,
                        **(
                            {
                                "map_initials": map_initials,
                                "map_initial_policy": (
                                    COMBINED_SYMBOLIC_POLICY
                                    if combined_symbolic_initials
                                    else WEIGHTED_SYMBOLIC_POLICY
                                ),
                            }
                            if test == _superposition
                            else {}
                        ),
                    )
                    if witness:
                        check.update(
                            status="conditional_conflict",
                            witnesses=[witness],
                            closed_states=sorted(states),
                            complete_common_forcing=sorted(forcing),
                            model_relations={
                                name: model.validated.equation_expressions[name].source
                                for name in sorted(states)
                            },
                            output_relation=model.validated.observation_expressions[
                                target
                            ].source,
                        )
                        if kind == "affine_initial_superposition":
                            check["repair_feedback"] = (
                                "These shared affine equations preserve the displayed "
                                "initial-state combination, but the recorded output "
                                "violates it. Changing only shared linear coefficients "
                                "cannot remove this identity. Revise the relevant "
                                "dependence/function family or a scientifically "
                                "justified initialization rule; adding losses alone "
                                "is not a remedy. A nonlinear change must still pass "
                                "all other checks."
                            )
                        else:
                            check["repair_feedback"] = (
                                "Only one initial-state coordinate can vary in this "
                                "closed linear subsystem, so trajectory differences "
                                "must have one common time shape. Data require more "
                                "than one. Revise the missing dependence or function "
                                "family, or justify an additional independent "
                                "initial-state direction. Different values of the "
                                "existing single root alone cannot fix this conflict."
                            )
                        break
                if check["status"] == "deferred":
                    check["unavailable"] = [
                        "no qualified contradiction in the bounded common-forcing/"
                        "initial-state witness search; this is not "
                        "scientific acceptance"
                    ]
                    check["common_forcing_triples_examined"] = common_count
            except _Unsupported as exc:
                check["unavailable"] = [str(exc)]
    return result


def assess(
    problem: PrefixFitProblem, evidence: dict, *, symbolic_initials: bool = False,
    weighted_symbolic_initials: bool = False,
    combined_symbolic_initials: bool = False,
) -> dict[str, Any]:
    """Audit affine conflicts with separately opt-in symbolic initial proofs."""
    if not isinstance(problem, PrefixFitProblem):
        raise ValueError("typed PrefixFitProblem required")
    problem = PrefixFitProblem.model_validate(problem.model_dump(mode="json"))
    training = TrainingCompatibilityEvidence.model_validate(evidence)
    validate_evidence(training)
    if training.context != problem.context:
        raise ValueError("training evidence context differs from problem context")
    if problem.context.fitted_initialization or problem.context.lagged_targets:
        raise ValueError("unlowered open-loop problem required")
    if any(item.scope.value != "global" for item in problem.base_candidate.parameters):
        raise ValueError("dynamics parameters must be globally shared")
    original = compile_candidate(problem.base_candidate, problem.context)
    model, _, initialization = apply_initialization_graph(original, problem.graph)
    missing = [
        name
        for name in (
            "dependencies_exhaustive",
            "shared_parameters",
            "open_loop",
            "reset_impulse_free",
            "uniqueness_assumed",
        )
        if getattr(training.contract, name) is not True
    ]
    try:
        if len(model.state_names) > MAX_STATES:
            raise _Unsupported("state count exceeds the bounded exact family proof")
        checks = (
            [
                {
                    "property": "candidate_class_contract",
                    "status": "deferred",
                    "unavailable": missing,
                }
            ]
            if missing
            else _checks(
                problem, model, training, symbolic_initials=symbolic_initials,
                weighted_symbolic_initials=weighted_symbolic_initials,
                combined_symbolic_initials=combined_symbolic_initials,
                roots=set(initialization["trajectory_parameter_names"]),
            )
        )
    except _Unsupported as exc:
        checks = [
            {
                "property": "affine_family_proof",
                "status": "deferred",
                "unavailable": [str(exc)],
            }
        ]
    conflicts = [check for check in checks if check["status"] == "conditional_conflict"]
    deferred = [check for check in checks if check["status"] == "deferred"]
    return {
        "protocol": PROTOCOL,
        **(
            {"symbolic_initial_policy": SYMBOLIC_INITIAL_POLICY}
            if symbolic_initials
            else {}
        ),
        **(
            {"weighted_symbolic_initial_policy": WEIGHTED_SYMBOLIC_POLICY}
            if weighted_symbolic_initials
            else {}
        ),
        **(
            {"combined_symbolic_initial_policy": COMBINED_SYMBOLIC_POLICY}
            if combined_symbolic_initials
            else {}
        ),
        "status": "conflict" if conflicts else "deferred",
        "child_problem_sha256": content_hash(problem.model_dump(mode="json")),
        "evidence_sha256": training.sha256,
        "checks": checks,
        "conflicts": conflicts,
        "deferred": deferred,
        "assumptions": [
            "Shared finite parameters and unique deterministic open-loop solutions "
            "on the common interval; no resets or impulses.",
            "Conditioning and target observations used here have authenticated "
            "absolute error zero. Forcing functions match exactly, with authenticated "
            "complete-knot interpolation.",
            "Superposition requires exact reproduction of initial observations. "
            "Small allowed initial mismatches with unbounded amplification do not "
            "give an approximate forecast or NMSE bound.",
        ],
        "coverage": "Finite affine rational closed-subsystem superposition and "
        "one-direction initial-response checks; unsupported or "
        "unwitnessed cases defer.",
        "witness_budget": {
            "triples_per_target_check": MAX_TRIPLES,
            "times_per_triple": MAX_TIMES,
            "maximum_states": MAX_STATES,
            "trajectories": MAX_TRAJECTORIES,
            "complete_samples_per_row": MAX_SAMPLES,
            "rational_bits": MAX_BITS,
            **(
                {"symbolic_initial_readout": dict(INITIAL_READOUT_LIMITS)}
                if symbolic_initials
                else {}
            ),
            **(
                {"weighted_symbolic_initials": dict(INITIAL_READOUT_LIMITS)}
                if weighted_symbolic_initials
                else {}
            ),
            **(
                {"combined_symbolic_initials": dict(INITIAL_READOUT_LIMITS)}
                if combined_symbolic_initials
                else {}
            ),
        },
        "target_comparison_policy": dict(TARGET_COMPARISON_POLICY),
        "scientific_correctness_established": False,
        "parameter_fitting_performed": False,
        "numerical_rollout_performed": False,
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }


def prompt_view(report: dict[str, Any]) -> dict[str, Any]:
    """Show a concise exact witness and actionable direction for each conflict."""
    return {
        key: report[key]
        for key in (
            "protocol",
            "status",
            "assumptions",
            "coverage",
            "child_problem_sha256",
            "evidence_sha256",
        )
    } | {
        **(
            {"symbolic_initial_policy": report["symbolic_initial_policy"]}
            if "symbolic_initial_policy" in report
            else {}
        ),
        **(
            {
                "weighted_symbolic_initial_policy": report[
                    "weighted_symbolic_initial_policy"
                ]
            }
            if "weighted_symbolic_initial_policy" in report
            else {}
        ),
        **(
            {
                "combined_symbolic_initial_policy": report[
                    "combined_symbolic_initial_policy"
                ]
            }
            if "combined_symbolic_initial_policy" in report
            else {}
        ),
        "status_counts": dict(Counter(item["status"] for item in report["checks"])),
        "conflicts": report["conflicts"],
        "full_conflict_count": len(report["conflicts"]),
    }
