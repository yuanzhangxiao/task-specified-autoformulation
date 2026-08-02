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
    "ParsedExpression",
    "PiecewiseLinearForcing",
    "RestrictedParser",
    "RuntimeExpressionError",
    "ValidatedCandidate",
    "ValidationContext",
    "ValidationDiagnostic",
    "compile_candidate",
    "repair_protected_declarations",
]
