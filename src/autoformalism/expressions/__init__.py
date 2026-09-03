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
from autoformalism.expressions.observability import (
    EffectiveObservability,
    infer_effective_observability,
)
from autoformalism.expressions.parameter_linearity import (
    ParameterLinearityReport,
    ProfiledLatentBasisParameterizationReport,
    ReciprocalParameterTransformation,
    certify_reciprocal_transformations,
    validate_fixed_latent_basis_parameterization,
    validate_gmm_parameterization,
    validate_profiled_latent_basis_parameterization,
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
    "EffectiveObservability",
    "ModelValidationError",
    "ParameterLinearityReport",
    "ParsedExpression",
    "PiecewiseLinearForcing",
    "ProfiledLatentBasisParameterizationReport",
    "ReciprocalParameterTransformation",
    "RestrictedParser",
    "RuntimeExpressionError",
    "ValidatedCandidate",
    "ValidationContext",
    "ValidationDiagnostic",
    "certify_reciprocal_transformations",
    "compile_candidate",
    "infer_effective_observability",
    "repair_protected_declarations",
    "validate_fixed_latent_basis_parameterization",
    "validate_gmm_parameterization",
    "validate_profiled_latent_basis_parameterization",
]
