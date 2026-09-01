#!/usr/bin/env python3
"""Freeze all GPT-5.6 raw-agent sources before sealed deterministic evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal.final_evaluation_pilot import (
    FrozenPilotSource,
    validate_hidden_audit,
)
from autoformalism.rebuttal.raw_agent_deterministic_evaluation import (
    freeze_raw_agent_sources,
    load_raw_agent_evaluation_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-protocol-config", type=Path, required=True)
    parser.add_argument("--prompt-refresh-config", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--refresh-root", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--hidden-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    plan = load_raw_agent_evaluation_plan(args.config)
    audit_sha256 = validate_hidden_audit(
        args.hidden_audit, plan.hidden_contract_audit
    )
    requests, sources, cells = freeze_raw_agent_sources(
        plan,
        full_protocol_config=args.full_protocol_config,
        prompt_refresh_config=args.prompt_refresh_config,
        historical_root=args.historical_root,
        refresh_root=args.refresh_root,
        public_data_root=args.public_data_root,
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    requests_path = output_root / "source_adapter_requests.jsonl"
    _write_or_validate(
        requests_path,
        "".join(item.model_dump_json() + "\n" for item in requests),
    )
    resource_path = output_root / "source_resource_ledger.jsonl"
    _write_or_validate(
        resource_path,
        "".join(
            json.dumps(_resource_row(item), sort_keys=True, separators=(",", ":"))
            + "\n"
            for item in sources
        ),
    )
    manifest = {
        "schema_version": "phase-b-raw-agent-deterministic-evaluation-freeze-1",
        "status": "frozen_before_test_or_private_evaluation",
        "method_id": plan.method_id,
        "plan_path": str(args.config.expanduser().resolve()),
        "plan_sha256": _sha256(args.config),
        "hidden_contract_audit_path": str(args.hidden_audit.expanduser().resolve()),
        "hidden_contract_audit_sha256": audit_sha256,
        "cell_count": len(cells),
        "repetition_count": len(plan.repetitions),
        "source_count": len(sources),
        "available_source_count": sum(
            item.artifact_status == "available" for item in sources
        ),
        "missing_source_count": sum(
            item.artifact_status == "missing" for item in sources
        ),
        "expected_source_count": len(cells) * len(plan.repetitions),
        "sources": [item.model_dump(mode="json") for item in sources],
        "source_adapter_requests_sha256": _sha256(requests_path),
        "source_resource_ledger_sha256": _sha256(resource_path),
        "postfreeze_shard_count": plan.postfreeze_shard_count,
        "hidden_shard_count": plan.hidden_shard_count,
        "selection_frozen": True,
        "test_data_opened": False,
        "private_reference_opened_for_candidate_selection": False,
        "parameter_refit_applied": False,
        "weighted_overall_score_defined": False,
        "qualitative_llm_requested": False,
    }
    manifest_path = output_root / "raw_agent_freeze_manifest.json"
    _write_or_validate(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    _write_or_validate(
        manifest_path.with_name(f"{manifest_path.name}.sha256"),
        f"{_sha256(manifest_path)}  {manifest_path.name}\n",
    )
    print(
        f"froze {len(sources)} GPT-5.6 sources before sealed evaluation; "
        f"available={manifest['available_source_count']} "
        f"missing={manifest['missing_source_count']}"
    )


def _write_or_validate(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"existing frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _resource_row(source: FrozenPilotSource) -> dict[str, object]:
    path = Path(source.source_path) / "status.json"
    payload: dict[str, object] = {}
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            payload = value
    usage = payload.get("usage")
    token_usage = usage if isinstance(usage, dict) else {}
    return {
        "schema_version": "phase-b-raw-agent-resource-row-1",
        "request_id": source.request_id,
        "benchmark_id": source.benchmark_id,
        "tier": source.tier,
        "repetition": source.repetition,
        "resource_status": "available" if payload else "unavailable",
        "agent_latency_seconds": payload.get("agent_latency_seconds"),
        "tool_call_count": payload.get("tool_call_count"),
        "requested_max_tool_calls": payload.get("requested_max_tool_calls"),
        "input_tokens": token_usage.get("input_tokens"),
        "output_tokens": token_usage.get("output_tokens"),
        "total_tokens": token_usage.get("total_tokens"),
        "monetary_cost_usd": None,
        "monetary_cost_status": "not_provider_reported",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
