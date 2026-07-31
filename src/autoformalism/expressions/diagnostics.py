"""Stable diagnostics for candidate validation and runtime evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ValidationDiagnostic:
    """One deterministic validation failure."""

    code: str
    location: str
    message: str


class ModelValidationError(Exception):
    """Raised with all deterministically ordered candidate diagnostics."""

    def __init__(self, diagnostics: tuple[ValidationDiagnostic, ...]) -> None:
        self.diagnostics = tuple(sorted(diagnostics))
        summary = "; ".join(
            f"{item.code} at {item.location}: {item.message}"
            for item in self.diagnostics
        )
        super().__init__(summary)


class RuntimeExpressionError(Exception):
    """Raised when valid syntax encounters invalid numerical runtime data."""

