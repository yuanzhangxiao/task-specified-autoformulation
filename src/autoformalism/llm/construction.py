"""Preserve complete construction context within a verified serving window."""

from autoformalism.llm.response_revision import (
    CONTEXT_TOKENS,
    MAX_INPUT_TOKENS,
    SAFETY_TOKENS,
    PromptPreflightError,
    ServingTokenCounter,
    diagnostic_transport,
)
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.repair_comparison import BudgetedRepairClient
from autoformalism.staged_topology import content_hash


class ConstructionClient(ServingTokenCounter, BudgetedRepairClient):
    """Reuse existing call budgets/cache; never truncate scientific requirements."""

    def __init__(self, *, token_transport=None, **kwargs):
        super().__init__(**kwargs)
        self.token_transport = token_transport or diagnostic_transport

    def call(self, *, system, user, response_model, step, attempt):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        count = self._count(messages)
        limit = min(
            MAX_INPUT_TOKENS,
            CONTEXT_TOKENS - self.settings.max_output_tokens - SAFETY_TOKENS,
        )
        audit = {
            "input_tokens": count,
            "input_limit": limit,
            "output_allowance": self.settings.max_output_tokens,
            "context_tokens": CONTEXT_TOKENS,
            "scientific_context_truncated": False,
            "status": "ready" if count <= limit else "context_limit_exceeded",
        }
        key = content_hash([step, attempt, messages])
        atomic_json(self.directory / "preflight" / f"{key}.json", audit)
        if count > limit:
            raise PromptPreflightError(
                f"complete construction context uses {count} tokens; limit={limit}; "
                "no generation was sent and no context was removed"
            )
        return super().call(
            system=system,
            user=user,
            response_model=response_model,
            step=step,
            attempt=attempt,
        )
