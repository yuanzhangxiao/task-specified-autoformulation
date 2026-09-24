"""Authenticated Jetstream proxy transport for an isolated judge pilot."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from autoformalism.llm.config import LLMConfig
from autoformalism.llm.exceptions import LLMProviderError
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.llm.vllm import VLLMClient

BASE_URL = "https://llm.jetstream-cloud.org/api"
ENDPOINT = BASE_URL + "/chat/completions"
MODEL = "gpt-oss-120b"
POLICY = "jetstream-authenticated-proxy-1"
MAX_CALLS = 80  # Two seeds, two orientations, two stages, ten attempts.


def _redact(value: object, key: str) -> object:
    """Scrub decoded JSON too, so escaped characters cannot preserve a credential."""
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]")
    if isinstance(value, list):
        return [_redact(item, key) for item in value]
    if isinstance(value, dict):
        return {
            name.replace(key, "[REDACTED]"): _redact(item, key)
            for name, item in value.items()
        }
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the credential to a redirected endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", {}, None)


class JetstreamTransport:
    """Record every physical attempt; keep authentication out of artifacts."""

    def __init__(self, directory: Path, key_supplier: Callable[[], str]) -> None:
        self.directory = directory
        self._key_supplier = key_supplier
        self._key: str | None = None
        self._opener = urllib.request.build_opener(_NoRedirect())

    def __call__(self, url: str, body: dict, timeout: float) -> dict:
        """Translate only endpoint/model alias, retaining all judge request fields."""
        if url != BASE_URL + "/v1/chat/completions":
            raise ValueError("unexpected Jetstream transport endpoint")
        if body.get("model") != "openai/gpt-oss-120b":
            raise ValueError("unexpected Jetstream judge model")
        if self._key is None:
            self._key = self._key_supplier().strip()
        if not self._key or any(c in self._key for c in "\r\n"):
            raise ValueError("a nonempty, single-line Jetstream key is required")
        self.directory.mkdir(parents=True, exist_ok=True)
        index = len(list(self.directory.glob("call-*")))
        if index >= MAX_CALLS:
            raise LLMProviderError("physical request cap exhausted", retryable=False)
        work = self.directory / f"call-{index:03d}"
        work.mkdir()
        payload = {**body, "model": MODEL}
        atomic_json(work / "request.json", {"endpoint": ENDPOINT, "body": payload})
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + self._key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        atomic_json(work / "started.json", {"utc_seconds": time.time()})
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                status = response.status
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            status = error.code
        except (OSError, TimeoutError) as error:
            atomic_json(
                work / "result.json",
                {
                    "status": "transport_error",
                    "error_type": type(error).__name__,
                    "seconds": time.monotonic() - started,
                },
            )
            raise LLMProviderError(
                f"Jetstream transport failed ({type(error).__name__})", retryable=True
            ) from None
        # A server can echo submitted headers; scrub the credential even from errors.
        raw = raw.replace(self._key, "[REDACTED]")
        try:
            value = _redact(json.loads(raw), self._key)
        except ValueError:
            value = None
        atomic_json(
            work / "result.json",
            {
                "status": "response",
                "http_status": status,
                "seconds": time.monotonic() - started,
                "response": value,
                "non_json_body": raw if value is None else None,
            },
        )
        if status != 200:
            raise LLMProviderError(
                f"Jetstream HTTP {status}; see recorded response",
                retryable=status == 429 or status >= 500,
            )
        if not isinstance(value, dict) or value.get("model") != MODEL:
            raise LLMProviderError(
                "Jetstream response is not JSON for the requested model alias",
                retryable=False,
            )
        return value


class JetstreamJudgeClient(VLLMClient):
    """Reuse the existing schema parser, repairs, cache and accounting."""

    def _hashable_provider_options(self) -> dict[str, object]:
        return {
            **super()._hashable_provider_options(),
            "transport_policy": POLICY,
            "wire_endpoint": ENDPOINT,
            "wire_model": MODEL,
            "served_model_revision": None,
        }


def client(config: LLMConfig, transport: JetstreamTransport) -> VLLMClient:
    """Preserve the frozen scientific settings while substituting only transport."""
    return JetstreamJudgeClient(
        model=config.model,
        cache_directory=config.cache_directory,
        log_path=config.log_path,
        base_url=BASE_URL,
        reasoning_effort=config.vllm_reasoning_effort,
        timeout_seconds=config.timeout_seconds,
        max_output_tokens=config.max_output_tokens,
        temperature=config.vllm_temperature,
        seed=config.vllm_seed,
        transport=transport,
        max_attempts=config.max_attempts,
        initial_backoff_seconds=config.initial_backoff_seconds,
        max_backoff_seconds=config.max_backoff_seconds,
        jitter_fraction=config.jitter_fraction,
    )
