"""Typed LLM boundary failures and repair-safe diagnostics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class LLMFailureCategory(str, Enum):
    """Stable failure classes used for retry and experiment accounting."""

    PROVIDER = "provider"
    REPAIRABLE_CONTRACT = "repairable_contract"


class RepairDiagnosticCode(str, Enum):
    """Contract-only diagnostics that may be returned to the proposer."""

    RESPONSE_VALIDATION = "response_validation"
    POST_SCHEMA_VALIDATION = "post_schema_validation"


@dataclass(frozen=True)
class RepairDiagnostic:
    """Bounded public-contract feedback safe for a repair attempt."""

    code: RepairDiagnosticCode
    message: str

    def __post_init__(self) -> None:
        normalized = re.sub(r"\s+", " ", self.message).strip()[:2000]
        if not normalized:
            raise ValueError("repair diagnostic message must not be empty")
        object.__setattr__(self, "message", normalized)


class LLMError(Exception):
    """Base error for structured LLM calls."""


class LLMProviderError(LLMError):
    """Provider failure with an explicit retry classification."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.category = LLMFailureCategory.PROVIDER


class LLMResponseError(LLMError):
    """A response could not be validated against the required schema."""

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        raw_response: object | None = None,
        diagnostic_code: RepairDiagnosticCode = (
            RepairDiagnosticCode.RESPONSE_VALIDATION
        ),
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.category = LLMFailureCategory.REPAIRABLE_CONTRACT
        self.repair_diagnostics = (RepairDiagnostic(diagnostic_code, message),)

    def repair_prompt(self) -> str:
        """Return bounded contract feedback without provider payloads or secrets."""
        diagnostics = "\n".join(
            f"- [{item.code.value}] {item.message}"
            for item in self.repair_diagnostics
        )
        return (
            "Repair only the executable response contract. Preserve the scientific "
            "hypothesis except where a listed contract correction requires a local "
            "notation or declaration change. Return one complete corrected object.\n"
            f"{diagnostics}"
        )


class LLMCacheError(LLMError):
    """A cache entry is malformed or cannot be persisted."""


class LLMCacheMissError(LLMCacheError):
    """A cache-only client could not find the requested response."""
