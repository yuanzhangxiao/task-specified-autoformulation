"""Deterministic mock client for offline tests."""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Any

from autoformalism.llm.models import LLMCallResult
from autoformalism.schemas import CandidateModel, ScientificJudgeResult


class MockLLMClient:
    """Queue-backed mock implementing separate proposer and judge methods."""

    def __init__(
        self,
        *,
        proposer_responses: list[CandidateModel | dict[str, Any]] | None = None,
        judge_responses: list[ScientificJudgeResult | dict[str, Any]] | None = None,
    ) -> None:
        self._proposer_responses = deque(proposer_responses or [])
        self._judge_responses = deque(judge_responses or [])
        self.calls: list[dict[str, str]] = []

    def propose(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMCallResult[CandidateModel]:
        """Return the next validated proposer response."""
        if not self._proposer_responses:
            raise AssertionError("no mock proposer response remains")
        self.calls.append(
            {
                "role": "proposer",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        parsed = CandidateModel.model_validate(self._proposer_responses.popleft())
        return self._result("proposer", system_prompt, user_prompt, parsed)

    def judge(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMCallResult[ScientificJudgeResult]:
        """Return the next validated judge response."""
        if not self._judge_responses:
            raise AssertionError("no mock judge response remains")
        self.calls.append(
            {
                "role": "judge",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        parsed = ScientificJudgeResult.model_validate(
            self._judge_responses.popleft()
        )
        return self._result("judge", system_prompt, user_prompt, parsed)

    @staticmethod
    def _result(
        role: str,
        system_prompt: str,
        user_prompt: str,
        parsed: CandidateModel | ScientificJudgeResult,
    ) -> LLMCallResult[Any]:
        content = "\0".join((role, system_prompt, user_prompt))
        request_hash = hashlib.sha256(content.encode()).hexdigest()
        raw_response = {"mock": True, "parsed": parsed.model_dump(mode="json")}
        return LLMCallResult(
            request_hash=request_hash,
            parsed=parsed,
            raw_response=raw_response,
            cache_hit=False,
            attempts=1,
            latency_ms=0.0,
            usage=None,
        )
