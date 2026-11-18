"""Logged local-cluster judge transport with the existing atomic wire contract."""

from __future__ import annotations

import ipaddress
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from autoformalism.llm import jetstream
from autoformalism.llm.config import LLMConfig
from autoformalism.llm.exceptions import LLMProviderError
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.llm.vllm import VLLMClient

POLICY = "local-cluster-judge-1"
MODEL = "openai/gpt-oss-120b"
REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"


def checked_url(value: str) -> str:
    """Allow only loopback/private addresses or explicitly named Delta nodes."""
    parts = urllib.parse.urlsplit(value)
    host = parts.hostname or ""
    try:
        address = ipaddress.ip_address(host)
        local = address.is_loopback or (
            address.is_private and not address.is_unspecified
        )
    except ValueError:
        local = host == "localhost" or host.endswith(".delta.ncsa.illinois.edu")
    if (
        not local
        or parts.scheme != "http"
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or parts.path not in ("", "/")
        or not parts.port
    ):
        raise ValueError("explicit local/Delta HTTP judge host and port required")
    return value.rstrip("/")


class LocalJudgeTransport:
    """Record every physical request; never use proxy services or redirects."""

    def __init__(
        self, directory: Path, base_url: str, key_supplier: Callable[[], str]
    ) -> None:
        self.directory = directory
        self.base_url = checked_url(base_url)
        self._key_supplier = key_supplier
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), jetstream._NoRedirect()
        )

    def __call__(self, url: str, body: dict, timeout: float) -> dict:
        """Use identical requested atomic identifiers and scientific settings."""
        if url != self.base_url + "/v1/chat/completions" or body.get("model") != MODEL:
            raise ValueError("local judge endpoint or requested model differs")
        payload = jetstream.wire_payload(body)
        payload["model"] = MODEL
        key = self._key_supplier().strip()
        if not key or any(c in key for c in "\r\n"):
            raise ValueError("a local service credential is required")
        self.directory.mkdir(parents=True, exist_ok=True)
        number = len(list(self.directory.glob("call-*")))
        if number >= jetstream.MAX_CALLS:
            raise LLMProviderError("physical request cap exhausted", retryable=False)
        work = self.directory / f"call-{number:03d}"
        work.mkdir()
        atomic_json(work / "request.json", {"endpoint": url, "body": payload})
        atomic_json(work / "started.json", {"utc_seconds": time.time()})
        started = time.monotonic()
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw, status = (
                    response.read().decode("utf-8", errors="replace"),
                    response.status,
                )
        except urllib.error.HTTPError as error:
            raw, status = error.read().decode("utf-8", errors="replace"), error.code
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
                "local judge transport failed", retryable=True
            ) from None
        raw = raw.replace(key, "[REDACTED]")
        try:
            value = jetstream._redact(json.loads(raw), key)
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
                f"local judge HTTP {status}", retryable=status == 429 or status >= 500
            )
        if not isinstance(value, dict) or value.get("model") != MODEL:
            raise LLMProviderError(
                "local judge returned a different model", retryable=False
            )
        return value


class LocalJudgeClient(VLLMClient):
    """Keep hosted and local critic cache identities distinct."""

    def _hashable_provider_options(self) -> dict[str, object]:
        return {
            **super()._hashable_provider_options(),
            "transport_policy": POLICY,
            "output_contract": jetstream.OUTPUT_CONTRACT,
            "served_model_revision": REVISION,
        }


def client(config: LLMConfig, transport: LocalJudgeTransport) -> VLLMClient:
    """Preserve all scientific settings, caching, parsing, and attempt limits."""
    return LocalJudgeClient(
        model=config.model,
        cache_directory=config.cache_directory,
        log_path=config.log_path,
        base_url=transport.base_url,
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
