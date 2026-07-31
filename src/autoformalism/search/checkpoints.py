"""Atomic JSON checkpoints with run-fingerprint verification."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, incompatible, or malformed."""


class CheckpointStore:
    """Write and read one atomic artifact after every controller stage."""

    def __init__(self, directory: Path, fingerprint: str) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        metadata = self.directory / "run.json"
        if metadata.exists():
            existing = json.loads(metadata.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != fingerprint:
                raise CheckpointError("checkpoint fingerprint does not match this run")
        else:
            self._atomic_write(metadata, {"fingerprint": fingerprint})

    def load_round(self, round_index: int) -> dict[str, Any] | None:
        """Load the latest stage for a round."""
        path = self.directory / f"round_{round_index:04d}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != self.fingerprint:
            raise CheckpointError("round checkpoint fingerprint mismatch")
        return payload

    def save_round(self, round_index: int, payload: dict[str, Any]) -> None:
        """Atomically replace the latest stage for a round."""
        self._atomic_write(
            self.directory / f"round_{round_index:04d}.json",
            {**payload, "fingerprint": self.fingerprint},
        )

    def load_final(self) -> dict[str, Any] | None:
        """Load a completed final evaluation when present."""
        path = self.directory / "final.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != self.fingerprint:
            raise CheckpointError("final checkpoint fingerprint mismatch")
        return payload

    def save_final(self, payload: dict[str, Any]) -> None:
        """Atomically checkpoint the one-time final evaluation."""
        self._atomic_write(
            self.directory / "final.json",
            {**payload, "fingerprint": self.fingerprint},
        )

    def claim_test_access(self) -> None:
        """Atomically claim the one permitted test access or fail closed."""
        path = self.directory / "test_access.claim"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise CheckpointError(
                "test access was already started without a completed result; "
                "refusing to access test again"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(self.fingerprint)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
