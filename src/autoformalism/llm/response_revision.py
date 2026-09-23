"""Bounded response prompts with cached serving-tokenizer preflight."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from autoformalism.llm.review_revision import RevisionClient
from autoformalism.llm.staged_topology import DeferredCall, atomic_json
from autoformalism.search.response_evidence import compact_user
from autoformalism.staged_topology import content_hash

CONTEXT_TOKENS = 32768
MAX_INPUT_TOKENS = 24000
SAFETY_TOKENS = 512


class PromptPreflightError(ValueError):
    """The complete immutable task cannot be safely delivered within this window."""


def diagnostic_transport(url: str, body: dict, timeout: float) -> dict:
    """Save a bounded HTTP response body, not credentials or response headers."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(8192).decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {error.code}: {detail}") from None
    if not isinstance(value, dict):
        raise ValueError("provider response is not an object")
    return value


class ResponseRevisionClient(RevisionClient):
    """Tokenization is logged separately and never charged as model generation."""

    def __init__(self, *, token_transport=None, **kwargs):
        kwargs.setdefault("transport", diagnostic_transport)
        super().__init__(**kwargs)
        self.token_transport = token_transport or diagnostic_transport

    def _count(self, messages: list[dict]) -> int:
        body = {
            "model": self.settings.model,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {
                "reasoning_effort": self.settings.reasoning_effort
            },
        }
        identity = {
            "policy": "response-token-preflight-1",
            "namespace": self.namespace,
            "model_revision": self.settings.model_revision,
            "body": body,
        }
        key = content_hash(identity)
        path = self.directory / "tokenization" / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["request"] != identity:
                raise PromptPreflightError("tokenization cache identity differs")
        else:
            if not self.can_start():
                raise DeferredCall("allocation draining before token preflight")
            saved = {"request": identity}
            try:
                reply = self.token_transport(self.base_url + "/tokenize", body, 60)
                count = reply.get("count")
                if type(count) is not int or count <= 0:
                    raise ValueError("tokenizer did not return a positive count")
                if reply.get("max_model_len") != CONTEXT_TOKENS:
                    raise ValueError("serving context differs from frozen 32768 tokens")
                saved.update(count=count, max_model_len=reply["max_model_len"])
            except (ValueError, OSError, TimeoutError) as error:
                saved["error"] = str(error)[:8192]
            atomic_json(path, saved)
        if "error" in saved:
            raise PromptPreflightError(saved["error"])
        return saved["count"]

    def call(self, *, system, user, response_model, step, attempt):
        """Keep all target overviews and contracts; reduce examples before failing."""
        payload = json.loads(user)
        limit = min(
            MAX_INPUT_TOKENS,
            CONTEXT_TOKENS - self.settings.max_output_tokens - SAFETY_TOKENS,
        )
        checks, seen = [], set()
        for per_target, samples in ((3, True), (3, False), (2, False), (1, False)):
            packed = compact_user(payload, per_target=per_target, samples=samples)
            encoded = json.dumps(packed, sort_keys=True, separators=(",", ":"))
            if encoded in seen:
                continue
            seen.add(encoded)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": encoded},
            ]
            count = self._count(messages)
            checks.append(
                {"per_target": per_target, "samples": samples, "input_tokens": count}
            )
            if count <= limit:
                record = super().call(
                    system=system,
                    user=encoded,
                    response_model=response_model,
                    step=step,
                    attempt=attempt,
                )
                audit = {
                    "protocol": "response-token-preflight-1",
                    "checks": checks,
                    "input_tokens": count,
                    "input_limit": limit,
                    "output_allowance": self.settings.max_output_tokens,
                    "context_tokens": CONTEXT_TOKENS,
                }
                if "prompt_preflight" in record and record["prompt_preflight"] != audit:
                    raise ValueError("cached prompt packing differs")
                record["prompt_preflight"] = audit
                atomic_json(self.directory / f"{record['request_hash']}.json", record)
                return record
        atomic_json(
            self.directory / "preflight" / f"{step}-{attempt}.json",
            {
                "status": "request_preflight_failed",
                "checks": checks,
                "input_limit": limit,
                "physical_generation_requests": 0,
            },
        )
        raise PromptPreflightError(
            "Model/public contract plus one response example per target exceeds "
            f"the {limit}-token input allowance; no generation was sent."
        )
