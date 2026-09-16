"""Three cached revision attempts, with per-call limits and observed usage only."""

from __future__ import annotations

import json
import time

from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedTopologyClient,
    atomic_json,
    strict_provider_schema,
)
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.staged_topology import content_hash


class RevisionClient(StagedTopologyClient):
    """No cumulative token gate; an uncertain call consumes its physical attempt."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.records = []
        for path in sorted(self.directory.glob("*.json")):
            record = json.loads(path.read_text())
            if (
                record["request"]["namespace"] != self.namespace
                or content_hash(record["request"]) != path.stem
                or record["request_hash"] != path.stem
            ):
                raise ValueError("revision cache provenance differs")
            self.records.append(record)

    def call(self, *, system, user, response_model, step, attempt):
        """Persist one request/response without estimating tokens from request bytes."""
        if not 0 <= attempt < 3:
            raise RepairBudgetExceeded("three revision attempts exhausted")
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
            "protocol": "three-attempt-revision-no-cumulative-cap-1",
            "namespace": self.namespace,
            "settings": {
                **self.settings.model_dump(mode="json"),
                "maximum_requests": 3,
                "attempts_per_step": 3,
                "maximum_total_tokens": None,
            },
            "body": body,
        }
        key = content_hash(identity)
        path = self.directory / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["request"] != identity or record["request_hash"] != key:
                raise ValueError("cached revision identity differs")
            if record["status"] == "inflight":
                record.update(
                    status="uncertain", error="interrupted before response checkpoint"
                )
                atomic_json(path, record)
        else:
            if not self.can_start():
                raise DeferredCall(
                    "allocation is draining before the next provider call"
                )
            if len(self.records) >= 3:
                raise RepairBudgetExceeded("three physical revision requests exhausted")
            if any(r["attempt"] == attempt for r in self.records):
                raise ValueError("revision attempt already belongs to another request")
            record = {
                "request_hash": key,
                "request": identity,
                "endpoint": self.base_url,
                "status": "inflight",
                "step": step,
                "attempt": attempt,
                "request_bytes": len(json.dumps(body).encode()),
                "budget_charge": 0,
                "budget_charge_basis": "usage_unknown_no_token_estimate",
                "observed_total_tokens": None,
                "cumulative_token_limit": None,
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
                if (
                    isinstance(total, int)
                    and not isinstance(total, bool)
                    and total >= 0
                ):
                    record.update(
                        budget_charge=total,
                        observed_total_tokens=total,
                        budget_charge_basis="provider_total_tokens",
                    )
            except (OSError, ValueError, TimeoutError) as error:
                record.update(
                    status="provider_failure", error=f"{type(error).__name__}: {error}"
                )
            record["latency_seconds"] = time.monotonic() - started
            atomic_json(path, record)
        self.records = [r for r in self.records if r["request_hash"] != key] + [record]
        return record
