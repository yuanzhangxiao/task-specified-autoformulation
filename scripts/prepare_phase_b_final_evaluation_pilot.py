#!/usr/bin/env python3
"""Freeze the prespecified two-cell Phase-B final-evaluation pilot sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal.final_evaluation_pilot import (
    freeze_pilot_sources,
    load_pilot_plan,
    validate_hidden_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--autoformalism-root", type=Path, required=True)
    parser.add_argument("--raw-agent-root", type=Path, required=True)
    parser.add_argument("--hidden-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    plan = load_pilot_plan(args.config)
    audit_sha256 = validate_hidden_audit(
        args.hidden_audit, plan.hidden_contract_audit
    )
    requests, sources = freeze_pilot_sources(
        plan,
        autoformalism_root=args.autoformalism_root,
        raw_agent_root=args.raw_agent_root,
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    requests_path = output_root / "source_adapter_requests.jsonl"
    _write_text_atomic(
        requests_path,
        "".join(item.model_dump_json() + "\n" for item in requests),
    )
    manifest = {
        "schema_version": "phase-b-final-evaluation-pilot-freeze-1",
        "status": "frozen_before_test_or_private_evaluation",
        "development_only": plan.development_only,
        "plan_path": str(args.config.expanduser().resolve()),
        "plan_sha256": _sha256(args.config),
        "hidden_contract_audit_path": str(args.hidden_audit.expanduser().resolve()),
        "hidden_contract_audit_sha256": audit_sha256,
        "source_count": len(sources),
        "expected_source_count": (
            len(plan.cells) * len(plan.repetitions) * len(plan.methods)
        ),
        "sources": [item.model_dump(mode="json") for item in sources],
        "source_adapter_requests_sha256": _sha256(requests_path),
        "selection_frozen": True,
        "test_data_opened": False,
        "private_reference_opened_for_candidate_selection": False,
        "weighted_overall_score_defined": False,
        "qualitative_llm_requested": False,
    }
    manifest_path = output_root / "pilot_freeze_manifest.json"
    _write_text_atomic(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        manifest_path.with_name(f"{manifest_path.name}.sha256"),
        f"{_sha256(manifest_path)}  {manifest_path.name}\n",
    )
    print(
        f"froze {len(sources)} planned sources before sealed evaluation in "
        f"{output_root}"
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
