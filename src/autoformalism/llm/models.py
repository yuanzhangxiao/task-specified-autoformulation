"""Typed provider-neutral LLM contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from autoformalism.schemas import CandidateModel, JudgeResult

StructuredT = TypeVar("StructuredT", bound=BaseModel)


@dataclass(frozen=True)
class TokenUsage:
    """Provider token accounting when available."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMCallResult(Generic[StructuredT]):
    """Validated result plus reproducibility and performance metadata."""

    request_hash: str
    parsed: StructuredT
    raw_response: dict[str, object]
    cache_hit: bool
    attempts: int
    latency_ms: float | None
    usage: TokenUsage | None

    @property
    def logical_calls(self) -> int:
        """One validated client operation, independent of repair attempts."""
        return 1

    @property
    def provider_attempts(self) -> int:
        """Actual provider generations; cache hits require no provider attempt."""
        return self.attempts

    @property
    def repair_attempts(self) -> int:
        """Provider generations after the initial logical-call attempt."""
        return max(0, self.attempts - 1)


class LLMClient(Protocol):
    """Provider-neutral proposer and judge interface."""

    def propose(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMCallResult[CandidateModel]:
        """Request one candidate conforming to ``CandidateModel``."""
        ...

    def judge(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMCallResult[JudgeResult]:
        """Request one assessment conforming to ``JudgeResult``."""
        ...
