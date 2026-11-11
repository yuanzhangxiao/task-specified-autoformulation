"""Write the call log the existing accounting rule already knows how to read.

`phase_b_d3.accounting` counts physical requests including retries, separates
cache hits, sums observed tokens, and -- the part that makes it fair -- counts
requests whose usage the provider did not report, so a method cannot look
cheap because its usage went unrecorded.

Only the D3 campaign produced that log. The vendored campaigns counted their
own requests instead, in their own shapes, which cannot be compared with it.
They write this format now so one rule covers every method that calls a model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def usage_tokens(payload: Any) -> int | None:
    """Total tokens a provider reported, under the names the APIs use.

    Returns None when the provider reported nothing, which the accounting
    rule records as an unknown-usage request rather than as zero.
    """
    usage = None
    if isinstance(payload, dict):
        usage = payload.get("usage")
    else:
        usage = getattr(payload, "usage", None)
    if usage is None:
        return None
    if not isinstance(usage, dict):
        usage = {
            name: getattr(usage, name, None)
            for name in ("total_tokens", "input_tokens", "output_tokens",
                         "prompt_tokens", "completion_tokens")
        }
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    parts = [
        usage.get(name)
        for name in ("input_tokens", "output_tokens",
                     "prompt_tokens", "completion_tokens")
    ]
    counted = [item for item in parts if isinstance(item, int)]
    return sum(counted) if counted else None


class CallLog:
    """Append-only record of one task's provider calls.

    Appends are line-atomic: a task that dies mid-campaign leaves a readable
    log rather than a truncated final line that would stop the whole report.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, event: dict) -> None:
        line = json.dumps(event, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def response(
        self,
        payload: Any,
        *,
        attempts: int = 1,
        finish_reasons: list[str] | None = None,
        cache_hit: bool = False,
    ) -> None:
        """Record one successful call and whatever usage it reported.

        ``finish_reasons``, when given, records why each choice stopped, so how
        often a token limit cut a reply off can be counted afterwards.
        ``cache_hit`` marks an answer a gateway replayed rather than generated;
        the accounting rule counts it apart from the requests that cost work.
        """
        tokens = usage_tokens(payload)
        event = {
            "event": "llm_response",
            "provider_attempts": int(attempts),
            "cache_hit": bool(cache_hit),
            "usage": {} if tokens is None else {"total_tokens": tokens},
        }
        if finish_reasons is not None:
            event["finish_reasons"] = list(finish_reasons)
        self._append(event)

    def failure(self, reason: str, *, attempts: int = 1) -> None:
        """A failed call still cost the provider work, so it is counted."""
        self._append(
            {
                "event": "llm_failure",
                "provider_attempts": int(attempts),
                "cache_hit": False,
                "usage": {},
                "reason": reason[:200],
            }
        )
