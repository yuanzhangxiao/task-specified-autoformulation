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
        self.budget_diagnostics_enabled = False
        self.last_budget_diagnostic = None
        self._context_preflight = None

    def budget_diagnostic(
        self, key: str, body: dict, charged: int, reservation: int
    ) -> None:
        """Save exact admission arithmetic even when generation never starts."""
        if not self.budget_diagnostics_enabled:
            return
        token_limit = self.settings.maximum_total_tokens
        request_limit = self.settings.maximum_requests
        reasons = []
        if len(self.records) >= request_limit:
            reasons.append("request_limit")
        if charged + reservation > token_limit:
            reasons.append("token_reservation")
        diagnostic = {
            "request_hash": key,
            "status": "blocked" if reasons else "ready",
            "blocking_reasons": reasons,
            "charged_before_call": charged,
            "active_token_ceiling": token_limit,
            "available_before_reservation": max(0, token_limit - charged),
            "request_bytes": reservation - self.settings.max_output_tokens,
            "output_allowance": self.settings.max_output_tokens,
            "reservation": reservation,
            "reservation_shortfall": max(0, charged + reservation - token_limit),
            "reservation_basis": "conservative_request_bytes_plus_output_allowance",
            "requests_before_call": len(self.records),
            "active_request_ceiling": request_limit,
            "context_preflight": self._context_preflight,
            "schema_name": body["response_format"]["json_schema"]["name"],
            "interpretation": "Budget admission is separate from context capacity. "
            "A blocked request starts no generation and consumes no reservation. "
            "Reported usage replaces a started request's reservation; unknown "
            "usage remains conservatively charged.",
        }
        atomic_json(self.directory / "budget_preflight" / f"{key}.json", diagnostic)
        self.last_budget_diagnostic = diagnostic

    def call(self, *, system, user, response_model, step, attempt):
        self.last_budget_diagnostic = None
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
        self._context_preflight = audit
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
