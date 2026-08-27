"""Offline budget-audit tests for raw-data frontier-agent artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from autoformalism.baselines.raw_data_agent import (
    RawAgentArtifact,
    RawAgentProvider,
)
from autoformalism.schemas import (
    CandidateModel,
    ProposerCandidateV2,
    enrich_proposal_v2,
)
from scripts.audit_raw_data_agent_budget import audit_run


def _candidate() -> tuple[ProposerCandidateV2, CandidateModel]:
    compact = ProposerCandidateV2.model_validate(
        {
            "schema_version": "2",
            "candidate_id": "candidate",
            "states": [
                {
                    "name": "x",
                    "kind": "observed",
                    "observed_channel": "x",
                    "rhs": "-x",
                }
            ],
            "algebraics": [],
            "parameters": [],
        }
    )
    return compact, enrich_proposal_v2(compact, ("x",))


def test_audit_detects_provider_tool_limit_excess(tmp_path: Path) -> None:
    compact, candidate = _candidate()
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    config = {
        "provider": "openai",
        "model": "gpt-test",
        "benchmark_id": "cell",
        "tier": "easy",
        "repetition": 0,
        "agent_config": {"max_tool_calls": 2},
    }
    (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    artifact = RawAgentArtifact(
        request_hash="hash",
        provider=RawAgentProvider.OPENAI,
        model="gpt-test",
        repetition=0,
        latency_seconds=1.0,
        tool_call_count=3,
        compact_candidate=compact,
        candidate=candidate,
        raw_response_sha256="0" * 64,
    )
    (run / "agent_result.json").write_text(
        artifact.model_dump_json(), encoding="utf-8"
    )
    raw = {
        "max_tool_calls": 2,
        "output": [
            {"id": "call-1", "type": "code_interpreter_call", "status": "completed"},
            {"id": "call-2", "type": "code_interpreter_call", "status": "completed"},
            {"id": "call-3", "type": "code_interpreter_call", "status": "incomplete"},
        ],
    }
    (run / "cache" / "hash.json").write_text(
        json.dumps({"artifact": artifact.model_dump(mode="json"), "raw_response": raw}),
        encoding="utf-8",
    )

    row = audit_run(run)

    assert row["provider_reported_max_tool_calls"] == 2
    assert row["raw_code_interpreter_items"] == 3
    assert row["raw_unique_code_interpreter_ids"] == 3
    assert row["limit_exceeded"] is True
