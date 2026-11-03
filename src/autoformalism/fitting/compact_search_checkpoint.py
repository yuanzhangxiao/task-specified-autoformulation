"""Opt-in compact search progress, without changing objective or reservations.

Full operation results remain authoritative. Compact progress retains a complete
best endpoint for interrupted-stage recovery, while repeated immutable evidence
is represented by content digests. The owning campaign must retain completed
setup/qualification receipts and must never renew an interrupted reservation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from time import monotonic
from typing import Literal

from pydantic import Field

from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.base import FiniteFloat, StrictSchema


class SearchCheckpointPolicy(StrictSchema):
    """Throttle routine progress only; every better incumbent remains durable."""

    protocol: Literal["compact-search-progress-1"] = "compact-search-progress-1"
    interval_seconds: FiniteFloat = Field(default=30, gt=0, le=300)
    history_limit: int = Field(default=8, ge=1, le=32)


_STATIC_FIELDS = frozenset(
    {
        "root_profile_certificate",
        "certificate",
        "optimizer_start",
        "coordinate_chart",
        "outer_coordinate_chart",
        "unoptimized_control",
        "initial_root_profile",
    }
)
_HISTORY_FIELDS = ("accepted_history", "accepted_shared_iteration_history")


class SearchProgress:
    """Persist forced scientific checkpoints and bounded periodic diagnostics."""

    def __init__(self, policy: SearchCheckpointPolicy, checkpoint: Callable):
        if not isinstance(policy, SearchCheckpointPolicy):
            raise TypeError("typed SearchCheckpointPolicy required")
        self.policy, self.checkpoint = policy, checkpoint
        self.last_written: float | None = None
        self.stats = {
            "protocol": policy.protocol,
            "requests": 0,
            "skipped_routine_requests": 0,
            "writes_started": 0,
            "writes_completed": 0,
            "writes_failed": 0,
            "forced_writes": 0,
            "serialized_bytes_completed": 0,
            "maximum_snapshot_bytes": 0,
            "scope": (
                "Progress callback writes only; final operation sealing is separate. "
                "Persisted counters predate their own completed write. Bytes use "
                "compact JSON encoding of the exact callback payload."
            ),
        }

    def attach(self, result: dict) -> None:
        """Expose mutable accounting in the full terminal result."""
        result["checkpoint_policy"] = self.policy.model_dump(mode="json")
        result["checkpoint_persistence"] = self.stats

    def _compact(self, result: dict, event: str) -> dict:
        snapshot = {
            k: deepcopy(v) for k, v in result.items() if k not in _STATIC_FIELDS
        }
        references = {
            k: content_sha256(result[k])
            for k in sorted(_STATIC_FIELDS)
            if result.get(k) is not None
        }
        snapshot["compact_progress"] = {
            "protocol": self.policy.protocol,
            "event": event,
            "omitted_static_sha256": references,
            "full_terminal_required_for_complete_evidence": True,
            "scope": (
                "Complete best endpoint retained; static bodies are not duplicated. "
                "Retain campaign setup/qualification and full terminal receipts."
            ),
        }
        for name in _HISTORY_FIELDS:
            if name in snapshot:
                snapshot["compact_progress"][name + "_total"] = len(snapshot[name])
                snapshot[name] = snapshot[name][-self.policy.history_limit :]
        return snapshot

    def _write(self, result: dict, event: str, *, forced: bool) -> None:
        self.stats["writes_started"] += 1
        self.stats["forced_writes"] += int(forced)
        try:
            snapshot = self._compact(result, event)
            size = len(
                json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
            )
            self.checkpoint(snapshot)
        except BaseException:
            self.stats["writes_failed"] += 1
            raise
        self.stats["writes_completed"] += 1
        self.stats["serialized_bytes_completed"] += size
        self.stats["maximum_snapshot_bytes"] = max(
            self.stats["maximum_snapshot_bytes"], size
        )
        self.last_written = monotonic()

    def save(self, result: dict, budget, *, force=False, event="progress") -> None:
        """Skip only routine persistence; numerical checks remain with the caller."""
        self.stats["requests"] += 1
        if (
            not force
            and self.last_written is not None
            and monotonic() - self.last_written < self.policy.interval_seconds
        ):
            self.stats["skipped_routine_requests"] += 1
            return
        result["timing"] = budget.snapshot()
        budget.callback(lambda: self._write(result, event, forced=force))

    def final(self, snapshot: dict) -> None:
        """Forced final payload; caller already measures this callback's time."""
        self.stats["requests"] += 1
        snapshot["checkpoint_persistence"] = self.stats
        self._write(snapshot, "terminal", forced=True)
