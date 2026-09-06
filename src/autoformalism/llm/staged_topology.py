"""Exact-schema, checkpointed transport for the staged topology pilot."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash

Transport = Callable[[str, dict[str, Any], float], dict[str, Any]]


class StagedModelSettings(StrictSchema):
    """Frozen inference settings; one attempt means one physical request."""

    model: str = "openai/gpt-oss-20b"
    model_revision: str = "unspecified"
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_output_tokens: int = Field(default=8192, ge=128, le=32768)
    timeout_seconds: float = Field(default=180, gt=0, le=1200)
    maximum_requests: int = Field(default=64, ge=1, le=128)
    maximum_total_tokens: int = Field(default=262144, ge=256)
    attempts_per_step: int = Field(default=3, ge=1, le=5)


class DeferredCall(RuntimeError):
    """No new request may start before the allocation deadline or after a signal."""


def atomic_json(path: Path, payload: object) -> None:
    """Durably replace one checkpoint on its own filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def http_transport(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Issue one local vLLM request without hidden automatic retries."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.load(response)
    if not isinstance(raw, dict):
        raise ValueError("provider response is not an object")
    return raw


def strict_provider_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Require all object fields without widening enums, bounds, or array limits."""
    result = json.loads(json.dumps(schema))

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("default", None)
            if value.get("type") == "object":
                value["additionalProperties"] = False
                value["required"] = list(value.get("properties", {}))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result)
    return result


class StagedTopologyClient:
    """Persist every physical outcome, including failures, before local validation.

    A request interrupted without a recorded response becomes an explicit
    uncertain outcome on resume. It is never silently resent under the same key.
    The controller may spend its next bounded attempt on that stage.
    """

    def __init__(
        self,
        *,
        settings: StagedModelSettings,
        base_url: str,
        directory: Path,
        namespace: str,
        seed: int,
        transport: Transport = http_transport,
        can_start: Callable[[], bool] = lambda: True,
    ) -> None:
        self.settings = settings
        self.base_url = base_url.rstrip("/")
        self.directory = directory
        self.namespace = namespace
        self.seed = seed
        self.transport = transport
        self.can_start = can_start
        self.records: list[dict[str, Any]] = []

    def call(
        self,
        *,
        system: str,
        user: str,
        response_model: type[StrictSchema],
        step: str,
        attempt: int,
    ) -> dict[str, Any]:
        """Return one cached physical outcome; validation stays with the caller."""
        body = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "reasoning_effort": self.settings.reasoning_effort,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_output_tokens,
            "seed": int(content_hash([self.seed, step, attempt])[:8], 16) % (2**31),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": strict_provider_schema(
                        response_model.model_json_schema()
                    ),
                },
            },
        }
        identity = {
            "protocol": "scientific-staged-topology-1",
            "namespace": self.namespace,
            "settings": self.settings.model_dump(mode="json"),
            "body": body,
        }
        key = content_hash(identity)
        path = self.directory / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["request"] != identity:
                raise ValueError("cached request identity mismatch")
            if record["status"] == "inflight":
                record.update(
                    status="uncertain", error="interrupted before response checkpoint"
                )
                atomic_json(path, record)
            self.records.append(record)
            return record
        if not self.can_start():
            raise DeferredCall("allocation is draining before the next provider call")
        if len(self.records) >= self.settings.maximum_requests:
            raise ValueError("total provider request budget exhausted")
        charged = sum(item.get("budget_charge", 0) for item in self.records)
        reservation = len(json.dumps(body).encode()) + self.settings.max_output_tokens
        if charged + reservation > self.settings.maximum_total_tokens:
            raise ValueError("total token budget cannot accommodate the next request")
        record = {
            "request_hash": key,
            "request": identity,
            "endpoint": self.base_url,
            "status": "inflight",
            "step": step,
            "attempt": attempt,
            "budget_charge": reservation,
            "budget_charge_basis": "conservative_request_bytes_plus_output_allowance",
            "observed_total_tokens": None,
        }
        atomic_json(path, record)
        started = time.monotonic()
        try:
            raw = self.transport(
                f"{self.base_url}/v1/chat/completions",
                body,
                self.settings.timeout_seconds,
            )
            record.update(status="responded", raw_response=raw)
            usage = raw.get("usage", {})
            total = usage.get("total_tokens") if isinstance(usage, dict) else None
            if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
                record["budget_charge"] = total
                record["budget_charge_basis"] = "provider_total_tokens"
                record["observed_total_tokens"] = total
        except (OSError, ValueError, TimeoutError) as exc:
            record.update(
                status="provider_failure", error=f"{type(exc).__name__}: {exc}"
            )
        record["latency_seconds"] = time.monotonic() - started
        atomic_json(path, record)
        self.records.append(record)
        return record


def visible_response(record: dict[str, Any]) -> object:
    """Read only final visible JSON; never promote a reasoning channel to output."""
    if record["status"] != "responded":
        raise ValueError(record.get("error", record["status"]))
    raw = record["raw_response"]
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("provider must return exactly one choice")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"incomplete provider response: {choice.get('finish_reason')}")
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("provider returned no final content")
    return json.loads(content)
