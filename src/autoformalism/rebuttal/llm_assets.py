"""Audit content-addressed LLM cache entries without contacting providers."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class LLMCacheRecord(BaseModel):
    """One valid cache file and the metadata needed for safe deduplication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    request_hash: str
    provider: str
    model: str
    parsed_digest: str
    raw_digest: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


class LLMCacheAudit(BaseModel):
    """Deterministic summary of cache coverage, corruption, and conflicts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_file_count: int
    valid_cache_file_count: int
    unique_request_count: int
    duplicate_copy_count: int
    conflicting_request_count: int
    metadata_only_conflict_count: int
    semantic_conflict_count: int
    malformed_files: tuple[str, ...]
    conflicting_hashes: tuple[str, ...]
    metadata_only_conflicting_hashes: tuple[str, ...]
    semantic_conflicting_hashes: tuple[str, ...]
    providers: dict[str, int]
    models: dict[str, int]
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    records: tuple[LLMCacheRecord, ...]


class LLMCacheResolution(BaseModel):
    """Outcome of deterministic cache canonicalization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolved_request_count: int
    excluded_semantic_conflict_count: int
    excluded_semantic_hashes: tuple[str, ...]
    selected_sources: dict[str, str]


def audit_llm_caches(roots: tuple[Path, ...]) -> LLMCacheAudit:
    """Audit all files below directories named ``llm_cache`` or ``cache``.

    Token totals count each request hash once. Identical copies are harmless;
    copies with different response payloads are reported as conflicts.
    """
    paths = _cache_paths(roots)
    malformed: list[str] = []
    records: list[LLMCacheRecord] = []
    for path in paths:
        try:
            records.append(_record(path))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            malformed.append(str(path))

    by_hash: dict[str, list[LLMCacheRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.request_hash].append(record)
    metadata_conflicts = tuple(
        sorted(
            request_hash
            for request_hash, copies in by_hash.items()
            if len({copy.parsed_digest for copy in copies}) == 1
            and len({copy.raw_digest for copy in copies}) > 1
        )
    )
    semantic_conflicts = tuple(
        sorted(
            request_hash
            for request_hash, copies in by_hash.items()
            if len({copy.parsed_digest for copy in copies}) > 1
        )
    )
    conflicts = tuple(sorted((*metadata_conflicts, *semantic_conflicts)))
    representatives = tuple(
        min(copies, key=lambda item: item.path) for _, copies in sorted(by_hash.items())
    )
    return LLMCacheAudit(
        cache_file_count=len(paths),
        valid_cache_file_count=len(records),
        unique_request_count=len(by_hash),
        duplicate_copy_count=len(records) - len(by_hash),
        conflicting_request_count=len(conflicts),
        metadata_only_conflict_count=len(metadata_conflicts),
        semantic_conflict_count=len(semantic_conflicts),
        malformed_files=tuple(sorted(malformed)),
        conflicting_hashes=conflicts,
        metadata_only_conflicting_hashes=metadata_conflicts,
        semantic_conflicting_hashes=semantic_conflicts,
        providers=dict(
            sorted(Counter(item.provider for item in representatives).items())
        ),
        models=dict(sorted(Counter(item.model for item in representatives).items())),
        total_input_tokens=sum(item.input_tokens or 0 for item in representatives),
        total_output_tokens=sum(item.output_tokens or 0 for item in representatives),
        total_tokens=sum(item.total_tokens or 0 for item in representatives),
        records=tuple(sorted(records, key=lambda item: item.path)),
    )


def resolve_llm_caches(audit: LLMCacheAudit, destination: Path) -> LLMCacheResolution:
    """Copy one canonical entry per safe hash and exclude semantic conflicts.

    Identical parsed responses are interchangeable for replay, even when raw
    provider metadata differs. A semantic conflict is never selected implicitly.
    """
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    excluded = set(audit.semantic_conflicting_hashes)
    by_hash: dict[str, list[LLMCacheRecord]] = defaultdict(list)
    for record in audit.records:
        by_hash[record.request_hash].append(record)
    selected: dict[str, str] = {}
    for request_hash, copies in sorted(by_hash.items()):
        if request_hash in excluded:
            continue
        source = min(copies, key=lambda item: item.path)
        target = destination / f"{request_hash}.json"
        shutil.copyfile(source.path, target)
        selected[request_hash] = source.path
    return LLMCacheResolution(
        resolved_request_count=len(selected),
        excluded_semantic_conflict_count=len(excluded),
        excluded_semantic_hashes=tuple(sorted(excluded)),
        selected_sources=selected,
    )


def _cache_paths(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved.is_file() and resolved.suffix == ".json":
            paths.add(resolved)
            continue
        for path in resolved.rglob("*.json"):
            if any(parent.name in {"llm_cache", "cache"} for parent in path.parents):
                paths.add(path)
    return tuple(sorted(paths))


def _record(path: Path) -> LLMCacheRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("cache payload must be an object")
    request_hash = str(payload["request_hash"])
    if path.stem != request_hash:
        raise ValueError("cache filename does not match request hash")
    parsed = payload["parsed_response"]
    raw = payload["raw_response"]
    usage = payload.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise TypeError("usage must be an object or null")
    return LLMCacheRecord(
        path=str(path),
        request_hash=request_hash,
        provider=str(payload["provider"]),
        model=str(payload["model"]),
        parsed_digest=_stable_hash(parsed),
        raw_digest=_stable_hash(raw),
        input_tokens=_optional_int(usage, "input_tokens"),
        output_tokens=_optional_int(usage, "output_tokens"),
        total_tokens=_optional_int(usage, "total_tokens"),
    )


def _optional_int(payload: dict[str, Any] | None, key: str) -> int | None:
    if payload is None or payload.get(key) is None:
        return None
    value = int(payload[key])
    if value < 0:
        raise ValueError("token counts must be nonnegative")
    return value


def _stable_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
