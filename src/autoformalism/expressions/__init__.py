"""Restricted expression parsing, validation, and numerical compilation."""

from autoformalism.expressions.compiler import (
    CompiledModel,
    PiecewiseLinearForcing,
    compile_candidate,
)
from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    RuntimeExpressionError,
    ValidationDiagnostic,
)
from autoformalism.expressions.parameter_linearity import (
    ParameterLinearityReport,
    validate_fixed_latent_basis_parameterization,
    validate_gmm_parameterization,
)
from autoformalism.expressions.parser import ParsedExpression, RestrictedParser
from autoformalism.expressions.validation import (
    CandidateValidator,
    ValidatedCandidate,
    ValidationContext,
    repair_protected_declarations,
)

__all__ = [
    "CandidateValidator",
    "CompiledModel",
    "ModelValidationError",
    "ParameterLinearityReport",
    "ParsedExpression",
    "PiecewiseLinearForcing",
    "RestrictedParser",
    "RuntimeExpressionError",
    "ValidatedCandidate",
    "ValidationContext",
    "ValidationDiagnostic",
    "compile_candidate",
    "repair_protected_declarations",
    "validate_fixed_latent_basis_parameterization",
    "validate_gmm_parameterization",
]
