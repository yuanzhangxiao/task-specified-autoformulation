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
from autoformalism.expressions.parser import ParsedExpression
from autoformalism.expressions.validation import ValidatedCandidate


@dataclass(frozen=True)
class ParameterLinearityReport:
    """Certified parameter usage for one validated candidate."""

    parameter_names: tuple[str, ...]
    rhs_parameters: frozenset[str]
    state_rhs_parameters: Mapping[str, frozenset[str]]


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
    latent_names = {
        item.name
        for item in validated.candidate.states
        if item.kind.value == "latent"
    }
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
