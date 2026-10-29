"""Finite exact affine-family necessary conditions, without fitting or simulation.

The checks use actual process expansions, closed dynamic ancestors and scoped
initializers. Exact time-zero reproduction is essential: no error bound for an
approximate forecast is inferred from a superposition contradiction.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
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
COMMON_ZERO_FORCING_POLICY = "certified-common-zero-forcing-affine-1"
DEFINED_ZERO_FORCING_POLICY = "defined-common-zero-forcing-affine-1"
SCALED_READOUT_POLICY = "shared-scaled-initial-readout-1"
MAX_ZERO_RESTRICTIONS = 8
MAX_TRIPLES = 256
MAX_TIMES = 64
MAX_STATES = 32
MAX_TRAJECTORIES = 32
MAX_SAMPLES = 8192
MAX_BITS = 4096


class _DefinedZeroQuotients(ast.NodeTransformer):
    """Reduce only parameter-domain zero quotients, retaining their obligations.

    This is a conditional proof transformation, never a candidate rewrite. The
    complementary denominator-zero stratum has no defined literal quotient.
    State/time denominators and unsupported operands are deliberately excluded.
    """

    def __init__(self, symbols, parameter_only: dict[str, set[str]]) -> None:
        self.symbols = symbols
        self.parameter_only = parameter_only
        self.obligations: list[dict[str, Any]] = []

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if not isinstance(node.op, ast.Div):
            return node
        try:
            numerator = _affine(node.left, self.symbols)
            denominator = _affine(node.right, self.symbols)
        except _Unsupported:
            return node
        names = {item.id for item in ast.walk(node.right) if isinstance(item, ast.Name)}
        if (
            numerator.coefficients
            or not numerator.constant.zero
            or denominator.coefficients
            or denominator.constant.excludes_zero
            or denominator.constant.zero
            or not names <= self.parameter_only.keys()
        ):
            return node
        self.obligations.append({
            "restricted_quotient": ast.unparse(node),
            "restricted_denominator": ast.unparse(node.right),
            "global_parameters": sorted(set().union(
                *(self.parameter_only[name] for name in names)
            )),
            "defined_case": "The displayed denominator is nonzero; this literal "
            "zero-numerator quotient equals zero.",
            "undefined_case": "The displayed denominator equals zero; this "
            "required literal quotient is undefined. No removable-singularity "
            "continuation is declared or inferred.",
        })
        return ast.copy_location(ast.Constant(value=0), node)


def _conditional_affine(
    parsed, symbols, parameter_only, zero_forcing, *, admission_guard: bool,
    component: str,
) -> tuple[_Affine, list[dict[str, Any]]]:
    """Interpret a restricted expression and preserve conditional domain proof."""
    if not admission_guard or not zero_forcing:
        return _affine(parsed.tree, symbols), []
    tree = _ZeroNames(set(zero_forcing)).visit(deepcopy(parsed.tree))
    transform = _DefinedZeroQuotients(symbols, parameter_only)
    restricted = transform.visit(tree)
    value = _affine(restricted, symbols)
    return value, [
        {"component": component, "original_expression": parsed.source, **entry}
        for entry in transform.obligations
    ]


def _domain_cases(obligations: list[dict[str, Any]]) -> dict[str, Any]:
    """Certify the union of defined and undefined parameter strata explicitly."""
    unique = {content_hash(entry): entry for entry in obligations}
    return {
        "policy": DEFINED_ZERO_FORCING_POLICY,
        "obligations": [unique[key] for key in sorted(unique)],
        "defined_parameter_stratum": "All displayed denominators are nonzero. "
        "The restricted affine identity applies to every shared parameter choice "
        "in this stratum; no parameters are fitted or fixed.",
        "complementary_stratum": "At least one required literal denominator "
        "equals zero. That expression is undefined on the certified schedule; "
        "these choices cannot supply defined solutions of the declared model.",
        "candidate_parameter_domains_modified": False,
        "singularities_filled": False,
        "scope": "Exact reproduction only. No approximate-error or NMSE bound.",
    }


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


def _scaled_readout_row(model, row, training, numeric, roots, affine_outputs):
    """Recognize numeric-offset readouts of one shared gain times one state.

    This bounded polynomial expansion records products; it does not divide by
    an unknown gain, solve for a fitted initial, or rewrite the candidate. Only
    rational arithmetic and monomials consisting of one global parameter and
    one unresolved state are supported. Unknown offsets and additional free
    states deliberately remain outside this proof.
    """
    states = set(model.state_names)
    parameters = {
        p.name: p for p in model.validated.candidate.parameters
        if p.scope.value == "global" and p.name not in roots
    }
    public = _public_initial(row, training)
    result = {}
    for channel, parsed in sorted(model.validated.observation_expressions.items()):
        if (
            channel not in row.targets
            or channel not in affine_outputs
            or training.contract.absolute_error_bounds.get(channel) != 0
        ):
            continue
        budget = _InitialBudget()
        processes_used, public_used, initials_used = {}, {}, {}

        def clean(poly, budget=budget):
            if len(poly) > INITIAL_READOUT_LIMITS["coordinates"]:
                raise _InitialUnsupported("scaled readout monomial limit")
            return {key: budget.check(value) for key, value in poly.items() if value}

        def interpret(node, budget=budget):
            budget.check(Fraction(0))
            budget.depth += 1
            try:
                if budget.depth > INITIAL_READOUT_LIMITS["expansion_depth"]:
                    raise _InitialUnsupported("scaled readout expansion depth limit")
                return interpret_node(node)
            finally:
                budget.depth -= 1

        def interpret_node(
            node, budget=budget, processes_used=processes_used,
            public_used=public_used, initials_used=initials_used,
        ):
            if isinstance(node, ast.Expression):
                return interpret(node.body)
            if isinstance(node, ast.Constant):
                return clean({(): Fraction(node.value)})
            if isinstance(node, ast.Name):
                name = node.id
                if name in states:
                    if name in numeric:
                        initials_used[name] = str(numeric[name])
                        return clean({(): numeric[name]})
                    return {(("state", name),): Fraction(1)}
                if name in model.validated.process_expressions:
                    expression = model.validated.process_expressions[name]
                    processes_used[name] = expression.source
                    return interpret(expression.tree)
                if name in parameters:
                    return {(("parameter", name),): Fraction(1)}
                if name in public:
                    value = _exact(public[name].constant)
                    public_used[name] = str(value)
                    return clean({(): value})
                raise _InitialUnsupported("unsupported scaled initial symbol")
            if isinstance(node, ast.UnaryOp):
                sign = -1 if isinstance(node.op, ast.USub) else 1
                return clean({key: sign * value for key, value in interpret(
                    node.operand
                ).items()})
            if isinstance(node, ast.BinOp):
                # Evaluate both operands even when multiplication later yields
                # zero; unsupported or undefined expressions cannot disappear.
                left, right = interpret(node.left), interpret(node.right)
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    sign = -1 if isinstance(node.op, ast.Sub) else 1
                    return clean({
                        key: left.get(key, Fraction(0))
                        + sign * right.get(key, Fraction(0))
                        for key in left.keys() | right.keys()
                    })
                if isinstance(node.op, ast.Mult):
                    product = {}
                    for first, a in left.items():
                        for second, b in right.items():
                            key = tuple(sorted(first + second))
                            if len(key) > 2 or (
                                len(key) == 2
                                and [kind for kind, _ in key] != ["parameter", "state"]
                            ):
                                raise _InitialUnsupported(
                                    "unsupported scaled readout product"
                                )
                            product[key] = budget.check(
                                product.get(key, Fraction(0)) + budget.check(a * b)
                            )
                    return clean(product)
                if (
                    isinstance(node.op, ast.Div)
                    and set(right) == {()} and right[()]
                ):
                    return clean({
                        key: value / right[()] for key, value in left.items()
                    })
            raise _InitialUnsupported("unsupported scaled readout expression")

        try:
            polynomial = interpret(parsed.tree)
            terms = {key: value for key, value in polynomial.items() if key}
            if len(terms) != 1:
                continue
            monomial, scale = next(iter(terms.items()))
            if len(monomial) != 2 or [kind for kind, _ in monomial] != [
                "parameter", "state"
            ]:
                continue
            gain, state = monomial[0][1], monomial[1][1]
            offset = polynomial.get((), Fraction(0))
            observed = budget.check(Fraction(row.targets[channel][0]))
            residual = budget.check(observed - offset)
            result.setdefault(state, []).append({
                "channel": channel,
                "gain": gain,
                "gain_domain": parameters[gain].domain.value,
                "rational_scale_exact": str(scale),
                "observed_initial_exact": str(observed),
                "numeric_offset_exact": str(offset),
                "scaled_initial_residual_exact": str(budget.check(residual / scale)),
                "zero_gain_residual_exact": str(residual),
                "zero_gain_numerical_guard_exact": str(_guard([observed, offset])),
                "original_readout": parsed.source,
                "expanded_process_relations": processes_used,
                "exact_public_initials": public_used,
                "known_state_initials_exact": initials_used,
            })
        except (_InitialUnsupported, _Unsupported):
            continue
    return result


def _scaled_initial_constraint(state, rows, scaled_initials):
    """Cancel weighted initials without evaluating or dividing by their gain."""
    values = [scaled_initials.get(row.trajectory_id, {}).get(state, []) for row in rows]
    common = set.intersection(*(
        {(v["channel"], v["gain"], v["rational_scale_exact"]) for v in line}
        for line in values
    ))
    for channel, gain, scale in sorted(common):
        entries = [next(
            v for v in line
            if (v["channel"], v["gain"], v["rational_scale_exact"])
            == (channel, gain, scale)
        ) for line in values]
        exclusion = next((
            {"trajectory_id": row.trajectory_id, "sample_index": 0,
             "residual_exact": entry["zero_gain_residual_exact"],
             "numerical_guard_exact": entry["zero_gain_numerical_guard_exact"]}
            for row, entry in zip(rows, entries, strict=True)
            if abs(Fraction(entry["zero_gain_residual_exact"]))
            > Fraction(entry["zero_gain_numerical_guard_exact"])
        ), None)
        positive = all(entry["gain_domain"] == "positive" for entry in entries)
        if not positive and exclusion is None:
            continue  # k=0 can leave a genuinely free initial state.
        residuals = [Fraction(v["scaled_initial_residual_exact"]) for v in entries]
        return residuals, {
            "policy": SCALED_READOUT_POLICY,
            "state": state,
            "readout_channel": channel,
            "shared_gain": gain,
            "rational_scale_exact": scale,
            "zero_gain_case": {
                "status": "excluded_by_declared_positive_domain" if positive
                else "contradicts_exact_recorded_initial_readout",
                "witness": exclusion,
            },
            "nonzero_gain_case": "For every remaining shared nonzero gain k, "
            "the exact readouts imply k*z_i(0)=r_i. Weights cancelling r_i "
            "therefore cancel z_i(0). No value of k or 1/k is computed or fitted.",
            "rows": [
                {"trajectory_id": row.trajectory_id, **entry}
                for row, entry in zip(rows, entries, strict=True)
            ],
            "required_cancellation": "sum(weights_i * scaled_initial_residual_i) = 0",
            "original_affine_closure_required": True,
            "candidate_or_parameter_domains_modified": False,
            "scope": "Exact initial and trajectory reproduction only; no NMSE bound.",
        }
    return None


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
    scaled_initials=None,
):
    constraints = [[Fraction(1)] * 3]
    known = {}
    scaled_constraints = []
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
        scaled = _scaled_initial_constraint(state, rows, scaled_initials or {})
        if scaled is not None:
            residuals, certificate = scaled
            constraints.append(residuals)
            known[state] = {
                "source": "exact scaled readout with a shared nonzero gain",
                "policy": SCALED_READOUT_POLICY,
                "shared_gain": certificate["shared_gain"],
                "scaled_initial_residuals_exact": list(map(str, residuals)),
            }
            scaled_constraints.append(certificate)
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
                    **({"scaled_initial_readout_constraints": scaled_constraints}
                       if scaled_constraints else {}),
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
    zero_forcing=frozenset(),
    admission_guard=False,
    scaled_readout_initials=False,
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
    # Only the opt-in wrapper supplies channels certified zero on every complete
    # schedule in this restricted group. Eager parsing still checks denominators
    # and unsupported operands; zero never short-circuits an undefined expression.
    for name in zero_forcing:
        symbols[name] = _Affine(_ZERO, {})
    parameter_only = {
        item.name: {item.name}
        for item in model.validated.candidate.parameters
        if item.name not in roots and item.scope.value == "global"
    }
    parameter_only.update((name, set()) for name in zero_forcing)
    process_domains: dict[str, list[dict[str, Any]]] = {}
    equation_domains: dict[str, list[dict[str, Any]]] = {}
    output_domains: dict[str, list[dict[str, Any]]] = {}

    def parse(expression, component):
        value, own = _conditional_affine(
            expression, symbols, parameter_only, zero_forcing,
            admission_guard=admission_guard, component=component,
        )
        inherited = [
            entry for name in sorted(expression.symbols)
            for entry in process_domains.get(name, [])
        ]
        return value, inherited + own

    unavailable = {}
    for name in model.validated.process_order:
        try:
            parsed = model.validated.process_expressions[name]
            symbols[name], process_domains[name] = parse(parsed, name)
            if parsed.symbols <= parameter_only.keys():
                parameter_only[name] = set().union(
                    *(parameter_only[symbol] for symbol in parsed.symbols)
                )
        except _Unsupported as exc:
            unavailable[name] = str(exc)
    equations, outputs = {}, {}
    for collection, domains, parsed in (
        (equations, equation_domains, model.validated.equation_expressions),
        (outputs, output_domains, model.validated.observation_expressions),
    ):
        for name, expression in parsed.items():
            try:
                collection[name], domains[name] = parse(expression, name)
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
    scaled_initials = (
        {
            row.trajectory_id: _scaled_readout_row(
                model, row, training, initials[row.trajectory_id], roots, outputs
            )
            for row in rows_to_check
        }
        if scaled_readout_initials else {}
    )
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
                                "scaled_initials": scaled_initials,
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
                        if "scaled_initial_readout_constraints" in witness:
                            check["scaled_initial_readout_constraints"] = witness[
                                "scaled_initial_readout_constraints"
                            ]
                        # Initial readouts can pin hidden coordinates even when
                        # they lie outside the selected target's dynamic closure.
                        # Preserve every conditional obligation used by any
                        # interpreted output, as well as the closed equations.
                        obligations = [
                            entry for state in sorted(states)
                            for entry in equation_domains.get(state, [])
                        ] + [
                            entry for name in sorted(output_domains)
                            for entry in output_domains[name]
                        ]
                        if obligations:
                            check["defined_parameter_domain_cases"] = _domain_cases(
                                obligations
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


def _original_target_forcing(model, target: str) -> set[str]:
    """Find actual supplied dependencies before any zero substitution."""
    pending = list(model.validated.observation_expressions[target].symbols)
    visited, forcing = set(), set()
    expressions = {
        **model.validated.process_expressions, **model.validated.equation_expressions
    }
    available = set(model.validated.context.forcing_channels) - expressions.keys()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        if name in expressions:
            pending.extend(expressions[name].symbols - visited)
        elif name in available:
            forcing.add(name)
    return forcing


def _zero_schedules(row, training, available: set[str]) -> set[str]:
    """Certify only supplied complete zero histories, never measured states."""
    if len(row.time) > MAX_SAMPLES:
        return set()
    result = set()
    for name in sorted(available):
        if (
            training.contract.absolute_error_bounds.get(name) != 0
            or training.contract.interpolation.get(name)
            not in {"linear_complete_knots", "previous_complete_knots"}
        ):
            continue
        values = row.inputs.get(name, row.auxiliaries.get(name))
        if values is not None and all(value == 0 for value in values):
            result.add(name)
    return result


class _ZeroNames(ast.NodeTransformer):
    """Display literal certified substitutions without modifying saved syntax."""

    def __init__(self, names: set[str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.names:
            return ast.copy_location(ast.Constant(value=0), node)
        return node


def _zero_display(parsed, names: set[str]) -> str:
    return ast.unparse(_ZeroNames(names).visit(deepcopy(parsed.tree)))


def _zero_certificate(problem, training, rows, names: set[str]) -> dict[str, Any]:
    """Bind the local restriction to complete schedules and declared domains."""
    return {
        "policy": COMMON_ZERO_FORCING_POLICY,
        "channels": sorted(names),
        "interval": [rows[0].time[0], min(row.time[-1] for row in rows)],
        "schedules": {
            name: {
                "interpolation": training.contract.interpolation[name],
                "absolute_error_bound": 0.0,
                "trajectories": {
                    row.trajectory_id: {
                        "knots": len(row.time),
                        "start": row.time[0], "end": row.time[-1],
                        "schedule_sha256": content_hash({
                            "time": list(row.time),
                            "values": list(row.inputs.get(
                                name, row.auxiliaries.get(name)
                            )),
                        }),
                    }
                    for row in rows
                },
            }
            for name in sorted(names)
        },
        "parameter_domains": {
            p.name: p.domain.value for p in problem.base_candidate.parameters
        },
        "denominator_check": "Every divided expression in the restricted affine "
        "closure retains a domain excluding zero; a zero numerator does not "
        "erase an undefined denominator.",
        "global_affinity_claim": False,
    }


def _zero_forcing_checks(problem, model, training, checks, **options):
    """Retry deferred targets on bounded groups with certified common zeros.

    General nonzero common-input transformations remain outside this correction.
    Old conflicts are retained, and the original evidence/compiled model are
    never modified. At most eight restricted groups receive the existing bounded
    triple search. Their certificates name the original supplied dependencies.
    """
    missing = {
        check["target"] for check in checks if check.get("target")
    } - {
        check["target"] for check in checks if check["status"] == "conditional_conflict"
    }
    if not missing:
        return checks
    dependencies = {
        target: _original_target_forcing(model, target) for target in sorted(missing)
    }
    available = set().union(*dependencies.values())
    rows = training.trajectories[:MAX_TRAJECTORIES]
    zero = {
        row.trajectory_id: _zero_schedules(row, training, available) for row in rows
    }
    restrictions = {
        tuple(sorted(set.intersection(*(zero[row.trajectory_id] for row in triple))))
        for triple in islice(combinations(rows, 3), MAX_TRIPLES)
    } - {()}
    for names_tuple in sorted(restrictions, key=lambda v: (-len(v), v))[
        :MAX_ZERO_RESTRICTIONS
    ]:
        names = set(names_tuple)
        group = tuple(row for row in rows if names <= zero[row.trajectory_id])
        if len(group) < 3:
            continue
        # This internal subset retains provenance of the already validated parent
        # evidence; it is not presented as a newly authenticated evidence packet.
        restricted = training.model_copy(update={"trajectories": group})
        try:
            prospective = _checks(
                problem, model, restricted, zero_forcing=names, **options
            )
        except _Unsupported:
            continue
        for fresh in prospective:
            target = fresh["target"]
            if fresh["status"] != "conditional_conflict" or target not in missing:
                continue
            used = names & dependencies[target]
            if not used:
                continue
            if options.get("admission_guard", False) or options.get(
                "scaled_readout_initials", False
            ):
                # Exact initial readouts from other targets may also use a
                # restricted input outside this target's dynamic closure.
                # Certify every substituted schedule, not just the target path.
                used = names
            witnesses = fresh["witnesses"]
            witness_rows = [
                next(row for row in group if row.trajectory_id == identity)
                for identity in witnesses[0]["trajectories"]
            ]
            fresh["common_zero_forcing"] = _zero_certificate(
                problem, training, witness_rows, used
            )
            if "defined_parameter_domain_cases" in fresh:
                fresh["common_zero_forcing"]["denominator_check"] = (
                    "Each displayed domain obligation is split explicitly: "
                    "nonzero denominators yield the restricted affine family; "
                    "zero denominators make required literal expressions "
                    "undefined. No singularity is silently cancelled or filled."
                )
            fresh["complete_common_forcing"] = sorted(
                set(fresh["complete_common_forcing"]) | used
            )
            fresh["restricted_model_relations"] = {
                name: _zero_display(model.validated.equation_expressions[name], used)
                for name in fresh["closed_states"]
            }
            fresh["restricted_process_relations"] = {
                name: _zero_display(parsed, used)
                for name, parsed in sorted(model.validated.process_expressions.items())
            }
            fresh["repair_feedback"] = (
                "On the certified zero-input schedules, these equations reduce "
                "to an affine state family and preserve the displayed identity. "
                "A nonlinear response solely to a supplied channel that is zero "
                "throughout these schedules cannot resolve this witness. Revise "
                "a relevant state dependence or justified initialization; memory "
                "states themselves are not asserted zero. This is a restriction "
                "to the witnessed schedules, not a global linearity claim."
            )
            if "defined_parameter_domain_cases" in fresh:
                fresh["repair_feedback"] += (
                    " A denominator that may be zero does not evade this "
                    "witness: every defined parameter choice still obeys the "
                    "identity, while a zero denominator leaves the literal "
                    "model undefined. Changing only that parameter's sign or "
                    "bounds cannot repair the displayed identity."
                )
            for i, existing in enumerate(checks):
                if (existing.get("target"), existing["property"]) == (
                    target, fresh["property"]
                ) and existing["status"] == "deferred":
                    checks[i] = fresh
                    break
        missing -= {
            c["target"] for c in checks if c["status"] == "conditional_conflict"
        }
        if not missing:
            break
    return checks


def assess(
    problem: PrefixFitProblem, evidence: dict, *, symbolic_initials: bool = False,
    weighted_symbolic_initials: bool = False,
    combined_symbolic_initials: bool = False,
    common_zero_forcing: bool = False,
    admission_guard: bool = False,
    scaled_readout_initials: bool = False,
) -> dict[str, Any]:
    """Audit affine conflicts with separately opt-in symbolic initial proofs."""
    if not isinstance(problem, PrefixFitProblem):
        raise ValueError("typed PrefixFitProblem required")
    if admission_guard and not common_zero_forcing:
        raise ValueError("admission_guard requires common_zero_forcing")
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
                admission_guard=admission_guard,
                scaled_readout_initials=scaled_readout_initials,
            )
        )
        if common_zero_forcing and not missing:
            checks = _zero_forcing_checks(
                problem, model, training, checks,
                symbolic_initials=symbolic_initials,
                weighted_symbolic_initials=weighted_symbolic_initials,
                combined_symbolic_initials=combined_symbolic_initials,
                roots=set(initialization["trajectory_parameter_names"]),
                admission_guard=admission_guard,
                scaled_readout_initials=scaled_readout_initials,
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
        **(
            {"common_zero_forcing_policy": COMMON_ZERO_FORCING_POLICY}
            if common_zero_forcing else {}
        ),
        **(
            {"defined_zero_forcing_policy": DEFINED_ZERO_FORCING_POLICY}
            if admission_guard else {}
        ),
        **(
            {"scaled_readout_initial_policy": SCALED_READOUT_POLICY}
            if scaled_readout_initials else {}
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
            **(
                {"common_zero_forcing": {
                    "maximum_restrictions": MAX_ZERO_RESTRICTIONS,
                    "triples_per_restriction": MAX_TRIPLES,
                }} if common_zero_forcing else {}
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
        **(
            {"common_zero_forcing_policy": report["common_zero_forcing_policy"]}
            if "common_zero_forcing_policy" in report else {}
        ),
        **(
            {"defined_zero_forcing_policy": report["defined_zero_forcing_policy"]}
            if "defined_zero_forcing_policy" in report else {}
        ),
        **(
            {"scaled_readout_initial_policy": report["scaled_readout_initial_policy"]}
            if "scaled_readout_initial_policy" in report else {}
        ),
        "status_counts": dict(Counter(item["status"] for item in report["checks"])),
        "conflicts": report["conflicts"],
        "full_conflict_count": len(report["conflicts"]),
    }
