#!/usr/bin/env python3
"""Audit requested and observed raw-agent tool budgets without new API calls."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from autoformalism.baselines.raw_data_agent import RawAgentArtifact

FIELDS = (
    "run",
    "provider",
    "model",
    "benchmark_id",
    "tier",
    "repetition",
    "requested_max_tool_calls",
    "provider_reported_max_tool_calls",
    "artifact_tool_call_count",
    "raw_code_interpreter_items",
    "raw_unique_code_interpreter_ids",
    "raw_code_interpreter_statuses",
    "raw_processed_code_interpreter_calls",
    "raw_nonterminal_code_interpreter_records",
    "limit_exceeded",
    "artifact_count_disagrees_with_processed_count",
    "raw_cache_available",
)

_TERMINAL_TOOL_STATUSES = frozenset({"completed", "incomplete", "failed"})


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _raw_response(run: Path, artifact: RawAgentArtifact) -> dict[str, object] | None:
    cache_path = run / "cache" / f"{artifact.request_hash}.json"
    if not cache_path.is_file():
        return None
    payload = _read_json(cache_path)
    raw = payload.get("raw_response")
    return raw if isinstance(raw, dict) else None


def _code_items(raw: dict[str, object] | None) -> tuple[dict[str, object], ...]:
    if raw is None or not isinstance(raw.get("output"), list):
        return ()
    return tuple(
        item
        for item in raw["output"]
        if isinstance(item, dict) and item.get("type") == "code_interpreter_call"
    )


def audit_run(run: Path) -> dict[str, object]:
    """Return one offline budget audit row for a completed provider response."""
    config = _read_json(run / "run_config.json")
    artifact = RawAgentArtifact.model_validate_json(
        (run / "agent_result.json").read_text(encoding="utf-8")
    )
    raw = _raw_response(run, artifact)
    items = _code_items(raw)
    statuses = Counter(
        str(item.get("status", "missing")) for item in items
    )
    identifiers = {
        str(item["id"])
        for item in items
        if isinstance(item.get("id"), str) and item["id"]
    }
    agent_config = config.get("agent_config")
    requested = (
        agent_config.get("max_tool_calls")
        if isinstance(agent_config, dict)
        else None
    )
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise ValueError(f"run has no integer max_tool_calls: {run}")
    raw_limit = None if raw is None else raw.get("max_tool_calls")
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
        raw_limit = None
    processed = sum(statuses[status] for status in _TERMINAL_TOOL_STATUSES)
    nonterminal = len(items) - processed
    authoritative_count = processed if raw is not None else artifact.tool_call_count
    return {
        "run": run.name,
        "provider": config["provider"],
        "model": config["model"],
        "benchmark_id": config["benchmark_id"],
        "tier": config["tier"],
        "repetition": config["repetition"],
        "requested_max_tool_calls": requested,
        "provider_reported_max_tool_calls": raw_limit,
        "artifact_tool_call_count": artifact.tool_call_count,
        "raw_code_interpreter_items": len(items),
        "raw_unique_code_interpreter_ids": len(identifiers),
        "raw_code_interpreter_statuses": json.dumps(statuses, sort_keys=True),
        "raw_processed_code_interpreter_calls": processed,
        "raw_nonterminal_code_interpreter_records": nonterminal,
        "limit_exceeded": authoritative_count > requested,
        "artifact_count_disagrees_with_processed_count": (
            raw is not None and artifact.tool_call_count != processed
        ),
        "raw_cache_available": raw is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    rows = [
        audit_run(run)
        for run in sorted(root.iterdir())
        if run.is_dir()
        and (run / "run_config.json").is_file()
        and (run / "agent_result.json").is_file()
    ]
    output = root / "tool_budget_audit.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "raw-data-agent-tool-budget-audit-2",
        "run_count": len(rows),
        "runs_exceeding_requested_limit": sum(
            bool(row["limit_exceeded"]) for row in rows
        ),
        "runs_with_raw_cache": sum(
            bool(row["raw_cache_available"]) for row in rows
        ),
        "runs_with_nonterminal_tool_records": sum(
            bool(row["raw_nonterminal_code_interpreter_records"])
            for row in rows
        ),
        "runs_with_legacy_count_disagreement": sum(
            bool(row["artifact_count_disagrees_with_processed_count"])
            for row in rows
        ),
        "output": str(output),
    }
    summary_path = root / "tool_budget_audit.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
