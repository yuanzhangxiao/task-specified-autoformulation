"""Conservative validation for graph-meta-model parameter linearity."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    ValidationDiagnostic,
)
from autoformalism.expressions.observability import infer_effective_observability
from autoformalism.expressions.parser import ParsedExpression
from autoformalism.expressions.validation import ValidatedCandidate


@dataclass(frozen=True)
class ParameterLinearityReport:
    """Certified parameter usage for one validated candidate."""

    parameter_names: tuple[str, ...]
    rhs_parameters: frozenset[str]
    state_rhs_parameters: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class ProfiledLatentBasisParameterizationReport:
    """Certified variable-projection partition for a latent-basis candidate."""

    parameter_names: tuple[str, ...]
    affine_parameter_names: tuple[str, ...]
    latent_shape_parameter_names: tuple[str, ...]
    reciprocal_transformations: tuple[ReciprocalParameterTransformation, ...]
    state_rhs_parameters: Mapping[str, frozenset[str]]
    observation_parameters: Mapping[str, frozenset[str]]
    effective_observed_state_names: frozenset[str]
    effective_latent_state_names: frozenset[str]
    runtime_inferred_observed_state_names: frozenset[str]

    @property
    def outer_parameter_names(self) -> tuple[str, ...]:
        """Return all externally optimized parameters.

        ``latent_shape_parameter_names`` is retained as a compatibility alias
        for existing frozen artifacts; outer parameters can now also arise from
        nonlinear observation mappings.
        """
        return self.latent_shape_parameter_names


@dataclass(frozen=True)
class ReciprocalParameterTransformation:
    """One certified positive reciprocal coordinate ``k = 1 / tau``."""

    parameter_name: str
    coordinate_name: str
    coordinate_lower: float
    coordinate_upper: float
    coordinate_start_lower: float
    coordinate_start_upper: float


def validate_gmm_parameterization(
    validated: ValidatedCandidate,
) -> ParameterLinearityReport:
    """Require every optimized parameter to enter the expanded RHS affinely.

    Algebraic processes are expanded recursively. A parameter may multiply any
    parameter-free basis function, but may not multiply another parameterized
    expression, appear in a denominator or exponent, or occur inside a named
    nonlinear function. Parameters in observation or initialization expressions
    are also rejected because they are not graph-edge weights in Eq. (10).
    """
    parameters = tuple(item.name for item in validated.candidate.parameters)
    parameter_set = frozenset(parameters)
    processes = validated.process_expressions
    diagnostics: list[ValidationDiagnostic] = []
    rhs_parameters: set[str] = set()
    state_rhs_parameters: dict[str, frozenset[str]] = {}

    cache: dict[str, frozenset[str]] = {}

    def process_dependencies(name: str) -> frozenset[str]:
        cached = cache.get(name)
        if cached is not None:
            return cached
        dependencies = analyze(
            processes[name].tree.body,
            location=f"process:{name}",
        )
        cache[name] = dependencies
        return dependencies

    def analyze(node: ast.AST, *, location: str) -> frozenset[str]:
        if isinstance(node, ast.Constant):
            return frozenset()
        if isinstance(node, ast.Name):
            if node.id in parameter_set:
                return frozenset({node.id})
            if node.id in processes:
                return process_dependencies(node.id)
            return frozenset()
        if isinstance(node, ast.UnaryOp):
            return analyze(node.operand, location=location)
        if isinstance(node, ast.BinOp):
            left = analyze(node.left, location=location)
            right = analyze(node.right, location=location)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return left | right
            if isinstance(node.op, ast.Mult):
                if left and right:
                    diagnostics.append(
                        ValidationDiagnostic(
                            "NONLINEAR_PARAMETER_PRODUCT",
                            location,
                            "optimized parameters occur on both sides of a product",
                        )
                    )
                return left | right
            if isinstance(node.op, ast.Div):
                if right:
                    diagnostics.append(
                        ValidationDiagnostic(
                            "PARAMETER_IN_DENOMINATOR",
                            location,
                            "optimized parameters may not occur in a denominator",
                        )
                    )
                return left | right
            if isinstance(node.op, ast.Pow):
                if right:
                    diagnostics.append(
                        ValidationDiagnostic(
                            "PARAMETER_IN_EXPONENT",
                            location,
                            "optimized parameters may not occur in an exponent",
                        )
                    )
                if left and not _is_literal_one(node.right):
                    diagnostics.append(
                        ValidationDiagnostic(
                            "NONLINEAR_PARAMETER_POWER",
                            location,
                            "a parameterized expression may only have power one",
                        )
                    )
                return left | right
        if isinstance(node, ast.Call):
            dependencies = frozenset().union(
                *(analyze(argument, location=location) for argument in node.args)
            )
            if dependencies:
                function = node.func.id if isinstance(node.func, ast.Name) else "call"
                diagnostics.append(
                    ValidationDiagnostic(
                        "PARAMETER_IN_NONLINEAR_FUNCTION",
                        location,
                        f"optimized parameters may not occur inside {function}()",
                    )
                )
            return dependencies
        raise AssertionError(f"unexpected restricted AST node: {type(node).__name__}")

    for state_name, expression in validated.equation_expressions.items():
        used = analyze(
            expression.tree.body, location=f"state_equation:{state_name}"
        )
        state_rhs_parameters[state_name] = used
        rhs_parameters.update(used)

    for kind, expressions in (
        ("observation_mapping", validated.observation_expressions),
        ("initial_condition", validated.initial_condition_expressions),
    ):
        for name, expression in expressions.items():
            used = _expanded_parameter_symbols(
                expression,
                parameter_set=parameter_set,
                processes=processes,
            )
            if used:
                diagnostics.append(
                    ValidationDiagnostic(
                        "PARAMETER_OUTSIDE_RHS",
                        f"{kind}:{name}",
                        "optimized graph weights may occur only in state RHSs: "
                        + ", ".join(sorted(used)),
                    )
                )

    missing = sorted(parameter_set - rhs_parameters)
    if missing:
        diagnostics.append(
            ValidationDiagnostic(
                "PARAMETER_NOT_IN_RHS",
                "candidate",
                "every optimized parameter must weight an RHS basis function: "
                + ", ".join(missing),
            )
        )
    if diagnostics:
        raise ModelValidationError(tuple(sorted(diagnostics)))
    return ParameterLinearityReport(
        parameters,
        frozenset(rhs_parameters),
        MappingProxyType(dict(state_rhs_parameters)),
    )


def validate_fixed_latent_basis_parameterization(
    validated: ValidatedCandidate,
) -> ParameterLinearityReport:
    """Certify affine weights with parameter-free latent-basis dynamics.

    Observed-state derivatives may identify affine graph weights. Latent states
    remain genuinely unobserved: their trajectories are generated from their
    proposer-specified, parameter-free dynamics and fixed initialization rather
    than supplied by an oracle.
    """
    report = validate_gmm_parameterization(validated)
    latent_names = infer_effective_observability(validated).latent_state_names
    diagnostics: list[ValidationDiagnostic] = []
    for state_name in sorted(latent_names):
        used = report.state_rhs_parameters[state_name]
        if used:
            diagnostics.append(
                ValidationDiagnostic(
                    "PARAMETER_IN_FIXED_LATENT_BASIS",
                    f"state_equation:{state_name}",
                    "fixed latent-basis dynamics cannot contain fitted "
                    f"parameters: {', '.join(sorted(used))}",
                )
            )
    initials = {
        item.state: item for item in validated.candidate.initial_conditions
    }
    for state_name in sorted(latent_names):
        initial = initials.get(state_name)
        if initial is None or (
            initial.fixed_value is None and initial.expression is None
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    "FIXED_LATENT_INITIAL_REQUIRED",
                    f"initial_condition:{state_name}",
                    "fixed latent-basis fitting requires a fixed_value or "
                    "parameter-free analytic initial condition",
                )
            )
    if diagnostics:
        raise ModelValidationError(tuple(diagnostics))
    return report


def validate_profiled_latent_basis_parameterization(
    validated: ValidatedCandidate,
) -> ProfiledLatentBasisParameterizationReport:
    """Partition fitted parameters into outer latent shapes and affine weights.

    A parameter used by any effectively latent-state RHS is an outer shape
    parameter. Parameters used non-affinely in an observed RHS or observation
    mapping are also outer parameters. All remaining parameters are inner
    weights and must enter state RHSs and observation mappings affinely
    conditional on the outer values. Latent initial values remain
    proposer-owned and parameter-free, so the fit never consumes latent oracle
    values or derivatives.
    """
    parameters = tuple(item.name for item in validated.candidate.parameters)
    parameter_set = frozenset(parameters)
    observability = infer_effective_observability(validated)
    latent_names = observability.latent_state_names
    state_rhs_parameters = {
        state_name: _expanded_parameter_symbols(
            expression,
            parameter_set=parameter_set,
            processes=validated.process_expressions,
        )
        for state_name, expression in validated.equation_expressions.items()
    }
    observation_parameters = {
        channel: _expanded_parameter_symbols(
            expression,
            parameter_set=parameter_set,
            processes=validated.process_expressions,
        )
        for channel, expression in validated.observation_expressions.items()
    }
    outer_parameters = frozenset().union(
        *(state_rhs_parameters[name] for name in latent_names)
    )
    used_parameters = frozenset().union(
        *state_rhs_parameters.values(),
        *observation_parameters.values(),
    )
    while True:
        candidate_affine = parameter_set - outer_parameters
        non_affine = _intrinsically_non_affine_parameters(
            validated,
            candidate_parameter_set=candidate_affine,
        )
        if non_affine <= outer_parameters:
            break
        outer_parameters |= non_affine
    affine_parameters = parameter_set - outer_parameters
    diagnostics: list[ValidationDiagnostic] = []
    if not latent_names:
        diagnostics.append(
            ValidationDiagnostic(
                "PROFILED_LATENT_STATE_REQUIRED",
                "candidate",
                "profiled latent-basis fitting requires at least one latent state",
            )
        )
    if not outer_parameters:
        diagnostics.append(
            ValidationDiagnostic(
                "LATENT_SHAPE_PARAMETER_REQUIRED",
                "candidate",
                "profiled latent-basis fitting requires at least one outer "
                "latent-shape or nonlinear observation parameter",
            )
        )
    if not affine_parameters:
        diagnostics.append(
            ValidationDiagnostic(
                "AFFINE_WEIGHT_PARAMETER_REQUIRED",
                "candidate",
                "profiled latent-basis fitting requires at least one conditional "
                "affine RHS or observation-mapping weight",
            )
        )

    missing = sorted(parameter_set - used_parameters)
    if missing:
        diagnostics.append(
            ValidationDiagnostic(
                "PARAMETER_NOT_IN_RHS",
                "candidate",
                "every optimized parameter must occur in a state RHS or "
                "observation mapping: "
                + ", ".join(missing),
            )
        )
    for name, expression in validated.initial_condition_expressions.items():
        used = _expanded_parameter_symbols(
            expression,
            parameter_set=parameter_set,
            processes=validated.process_expressions,
        )
        if used:
            diagnostics.append(
                ValidationDiagnostic(
                    "PARAMETER_IN_PROFILED_INITIAL",
                    f"initial_condition:{name}",
                    "profiled fitting does not yet support parameters in "
                    "initial-condition expressions: "
                    + ", ".join(sorted(used)),
                )
            )

    diagnostics.extend(
        _affine_subset_diagnostics(
            validated,
            affine_parameter_set=affine_parameters,
        )
    )
    reciprocal_transformations = certify_reciprocal_transformations(
        validated,
        eligible_parameter_names=outer_parameters,
    )
    initials = {
        item.state: item for item in validated.candidate.initial_conditions
    }
    for state_name in sorted(latent_names):
        initial = initials.get(state_name)
        if initial is None or (
            initial.fixed_value is None and initial.expression is None
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    "FIXED_LATENT_INITIAL_REQUIRED",
                    f"initial_condition:{state_name}",
                    "profiled latent-basis fitting requires a fixed_value or "
                    "parameter-free analytic initial condition",
                )
            )
    if diagnostics:
        raise ModelValidationError(tuple(sorted(diagnostics)))
    return ProfiledLatentBasisParameterizationReport(
        parameter_names=parameters,
        affine_parameter_names=tuple(
            name for name in parameters if name in affine_parameters
        ),
        latent_shape_parameter_names=tuple(
            name for name in parameters if name in outer_parameters
        ),
        reciprocal_transformations=reciprocal_transformations,
        state_rhs_parameters=MappingProxyType(dict(state_rhs_parameters)),
        observation_parameters=MappingProxyType(dict(observation_parameters)),
        effective_observed_state_names=observability.observed_state_names,
        effective_latent_state_names=observability.latent_state_names,
        runtime_inferred_observed_state_names=(
            observability.runtime_inferred_observed_state_names
        ),
    )


def certify_reciprocal_transformations(
    validated: ValidatedCandidate,
    *,
    eligible_parameter_names: frozenset[str] | None = None,
) -> tuple[ReciprocalParameterTransformation, ...]:
    """Certify positive parameters whose RHS use is affine in ``1 / p``.

    The certificate is intentionally narrow. Every occurrence must be the
    complete right operand of division, such as ``x / tau``. After replacing
    each such division by multiplication with the reciprocal coordinate, the
    expanded state RHSs must be affine in that coordinate. Bounds and start
    ranges must be strictly positive, making the mapping one-to-one.

    This certificate describes RHS parameterization only. A reciprocal used in
    latent dynamics still changes the integrated latent path and therefore is
    not automatically eligible for an inner linear solve under partial
    observation.
    """
    eligible = (
        frozenset(item.name for item in validated.candidate.parameters)
        if eligible_parameter_names is None
        else eligible_parameter_names
    )
    expressions = (
        *validated.equation_expressions.values(),
        *validated.process_expressions.values(),
    )
    transformations: list[ReciprocalParameterTransformation] = []
    for spec in validated.candidate.parameters:
        name = spec.name
        if (
            name not in eligible
            or spec.bounds is None
            or spec.initialization_range is None
            or spec.bounds.lower <= 0.0
        ):
            continue
        if any(
            not _all_parameter_occurrences_are_direct_divisors(
                expression.tree.body, name
            )
            for expression in expressions
        ):
            continue
        if not any(
            _tree_contains_name(expression.tree.body, name)
            for expression in expressions
        ):
            continue
        if not _expanded_rhs_is_affine_in_reciprocal(validated, name):
            continue
        start = spec.initialization_range
        transformations.append(
            ReciprocalParameterTransformation(
                parameter_name=name,
                coordinate_name=f"reciprocal:{name}",
                coordinate_lower=1.0 / spec.bounds.upper,
                coordinate_upper=1.0 / spec.bounds.lower,
                coordinate_start_lower=1.0 / start.upper,
                coordinate_start_upper=1.0 / start.lower,
            )
        )
    return tuple(transformations)


def _all_parameter_occurrences_are_direct_divisors(
    node: ast.AST,
    parameter_name: str,
) -> bool:
    if isinstance(node, ast.Name):
        return node.id != parameter_name
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if not _all_parameter_occurrences_are_direct_divisors(
            node.left, parameter_name
        ):
            return False
        if isinstance(node.right, ast.Name) and node.right.id == parameter_name:
            return True
        return _all_parameter_occurrences_are_direct_divisors(
            node.right, parameter_name
        )
    return all(
        _all_parameter_occurrences_are_direct_divisors(child, parameter_name)
        for child in ast.iter_child_nodes(node)
    )


def _tree_contains_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id == name
        for child in ast.walk(node)
    )


def _expanded_rhs_is_affine_in_reciprocal(
    validated: ValidatedCandidate,
    parameter_name: str,
) -> bool:
    """Return whether expanded RHSs are affine after ``1 / p -> k``."""
    processes = validated.process_expressions
    cache: dict[str, bool] = {}

    def process_dependency(name: str) -> bool:
        cached = cache.get(name)
        if cached is not None:
            return cached
        dependency = analyze(processes[name].tree.body)
        cache[name] = dependency
        return dependency

    def analyze(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return False
        if isinstance(node, ast.Name):
            if node.id == parameter_name:
                raise ValueError("uncertified direct parameter occurrence")
            return process_dependency(node.id) if node.id in processes else False
        if isinstance(node, ast.UnaryOp):
            return analyze(node.operand)
        if isinstance(node, ast.BinOp):
            if (
                isinstance(node.op, ast.Div)
                and isinstance(node.right, ast.Name)
                and node.right.id == parameter_name
            ):
                if analyze(node.left):
                    raise ValueError("reciprocal coordinates multiply")
                return True
            left = analyze(node.left)
            right = analyze(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return left or right
            if isinstance(node.op, ast.Mult):
                if left and right:
                    raise ValueError("reciprocal coordinates multiply")
                return left or right
            if isinstance(node.op, ast.Div):
                if right:
                    raise ValueError("reciprocal coordinate is in a denominator")
                return left
            if isinstance(node.op, ast.Pow):
                if right or (left and not _is_literal_one(node.right)):
                    raise ValueError("reciprocal coordinate has nonlinear power")
                return left
        if isinstance(node, ast.Call):
            if any(analyze(argument) for argument in node.args):
                raise ValueError("reciprocal coordinate is in a nonlinear call")
            return False
        raise AssertionError(f"unexpected restricted AST node: {type(node).__name__}")

    try:
        for expression in validated.equation_expressions.values():
            analyze(expression.tree.body)
    except ValueError:
        return False
    return True


def _intrinsically_non_affine_parameters(
    validated: ValidatedCandidate,
    *,
    candidate_parameter_set: frozenset[str],
) -> frozenset[str]:
    """Find candidate inner parameters that must instead be optimized outside.

    Parameters in denominators, powers, or nonlinear calls are promoted first.
    Once those symbols are treated as fixed outer coordinates, products among
    the remaining candidate weights are promoted. This ordering preserves a
    useful partition such as ``gain * sigmoid(shape * z)``: ``shape`` is outer
    while ``gain`` remains an affine observation weight.
    """
    processes = validated.process_expressions

    def scan(*, mark_products: bool) -> frozenset[str]:
        cache: dict[str, frozenset[str]] = {}
        non_affine: set[str] = set()

        def process_dependencies(name: str) -> frozenset[str]:
            cached = cache.get(name)
            if cached is not None:
                return cached
            dependencies = analyze(processes[name].tree.body)
            cache[name] = dependencies
            return dependencies

        def analyze(node: ast.AST) -> frozenset[str]:
            if isinstance(node, ast.Constant):
                return frozenset()
            if isinstance(node, ast.Name):
                if node.id in candidate_parameter_set:
                    return frozenset({node.id})
                if node.id in processes:
                    return process_dependencies(node.id)
                return frozenset()
            if isinstance(node, ast.UnaryOp):
                return analyze(node.operand)
            if isinstance(node, ast.BinOp):
                left = analyze(node.left)
                right = analyze(node.right)
                if isinstance(node.op, ast.Mult) and mark_products and left and right:
                    non_affine.update(left | right)
                elif isinstance(node.op, ast.Div):
                    non_affine.update(right)
                elif isinstance(node.op, ast.Pow):
                    non_affine.update(right)
                    if not _is_literal_one(node.right):
                        non_affine.update(left)
                return left | right
            if isinstance(node, ast.Call):
                dependencies = frozenset().union(
                    *(analyze(argument) for argument in node.args)
                )
                non_affine.update(dependencies)
                return dependencies
            raise AssertionError(
                f"unexpected restricted AST node: {type(node).__name__}"
            )

        for expression in (
            *validated.equation_expressions.values(),
            *validated.observation_expressions.values(),
        ):
            analyze(expression.tree.body)
        return frozenset(non_affine)

    intrinsic = scan(mark_products=False)
    return intrinsic if intrinsic else scan(mark_products=True)


def _affine_subset_diagnostics(
    validated: ValidatedCandidate,
    *,
    affine_parameter_set: frozenset[str],
) -> tuple[ValidationDiagnostic, ...]:
    """Check conditional affinity while treating outer parameters as basis data."""
    processes = validated.process_expressions
    diagnostics: list[ValidationDiagnostic] = []
    cache: dict[str, frozenset[str]] = {}

    def process_dependencies(name: str) -> frozenset[str]:
        cached = cache.get(name)
        if cached is not None:
            return cached
        dependencies = analyze(
            processes[name].tree.body,
            location=f"process:{name}",
        )
        cache[name] = dependencies
        return dependencies

    def analyze(node: ast.AST, *, location: str) -> frozenset[str]:
        if isinstance(node, ast.Constant):
            return frozenset()
        if isinstance(node, ast.Name):
            if node.id in affine_parameter_set:
                return frozenset({node.id})
            if node.id in processes:
                return process_dependencies(node.id)
            return frozenset()
        if isinstance(node, ast.UnaryOp):
            return analyze(node.operand, location=location)
        if isinstance(node, ast.BinOp):
            left = analyze(node.left, location=location)
            right = analyze(node.right, location=location)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return left | right
            if isinstance(node.op, ast.Mult):
                if left and right:
                    diagnostics.append(
                        ValidationDiagnostic(
                            "NONLINEAR_PARAMETER_PRODUCT",
                            location,
                            "affine weights occur on both sides of a product",
                        )
                    )
                return left | right
            if isinstance(node.op, ast.Div):
                if right:
                    diagnostics.append(
                        ValidationDiagnostic(
                            "PARAMETER_IN_DENOMINATOR",
                            location,
                            "an affine weight may not occur in a denominator",
                        )
                    )
                return left | right
            if isinstance(node.op, ast.Pow):
                if right:
                    diagnostics.append(
                        ValidationDiagnostic(
                            "PARAMETER_IN_EXPONENT",
                            location,
                            "an affine weight may not occur in an exponent",
                        )
                    )
                if left and not _is_literal_one(node.right):
                    diagnostics.append(
                        ValidationDiagnostic(
                            "NONLINEAR_PARAMETER_POWER",
                            location,
                            "an affine-weight expression may only have power one",
                        )
                    )
                return left | right
        if isinstance(node, ast.Call):
            dependencies = frozenset().union(
                *(analyze(argument, location=location) for argument in node.args)
            )
            if dependencies:
                function = node.func.id if isinstance(node.func, ast.Name) else "call"
                diagnostics.append(
                    ValidationDiagnostic(
                        "PARAMETER_IN_NONLINEAR_FUNCTION",
                        location,
                        f"an affine weight may not occur inside {function}()",
                    )
                )
            return dependencies
        raise AssertionError(f"unexpected restricted AST node: {type(node).__name__}")

    for state_name, expression in validated.equation_expressions.items():
        analyze(expression.tree.body, location=f"state_equation:{state_name}")
    for channel, expression in validated.observation_expressions.items():
        analyze(expression.tree.body, location=f"observation_mapping:{channel}")
    return tuple(diagnostics)


def _expanded_parameter_symbols(
    expression: ParsedExpression,
    *,
    parameter_set: frozenset[str],
    processes: Mapping[str, ParsedExpression],
) -> frozenset[str]:
    """Collect parameter symbols through process references."""
    seen: set[str] = set()

    def collect(node: ast.AST) -> set[str]:
        found: set[str] = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Name):
                continue
            if child.id in parameter_set:
                found.add(child.id)
            elif child.id in processes and child.id not in seen:
                seen.add(child.id)
                found.update(collect(processes[child.id].tree.body))
        return found

    return frozenset(collect(expression.tree.body))


def _is_literal_one(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
        and float(node.value) == 1.0
    )
