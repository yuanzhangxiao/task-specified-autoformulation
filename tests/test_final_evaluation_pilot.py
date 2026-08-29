"""Tests for the frozen two-cell final-evaluation pilot planner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from autoformalism.rebuttal.final_evaluation_pilot import (
    FinalEvaluationPilotPlan,
    freeze_pilot_sources,
    validate_hidden_audit,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(audit_sha256: str) -> FinalEvaluationPilotPlan:
    return FinalEvaluationPilotPlan.model_validate(
        {
            "schema_version": "phase-b-final-evaluation-pilot-plan-1",
            "status": "frozen_before_autoformalism_pilot_runs",
            "purpose": "test pilot",
            "development_only": True,
            "cells": [{"benchmark_id": "benchmark", "tier": "easy"}],
            "repetitions": [0, 1],
            "methods": ["autoformalism", "raw_data_agent:gpt-5.6-sol"],
            "hidden_contract_audit": {
                "schema_version": "phase-b-hidden-subspace-contract-audit-2",
                "sha256": audit_sha256,
                "required_status": "pass",
            },
            "endpoints": [
                "source_completion",
                "runtime_validity",
                "public_mechanism_compliance",
                "sealed_target_nmse",
                "hidden_response_subspace_nmse",
                "intervention_behavior",
                "model_complexity",
            ],
            "weighted_overall_score_defined": False,
            "qualitative_llm_requested": False,
        }
    )


def test_freeze_pilot_sources_builds_exact_cross_product(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    _write_json(
        audit,
        {
            "schema_version": "phase-b-hidden-subspace-contract-audit-2",
            "status": "pass",
        },
    )
    digest = _sha256(audit)
    audit.with_name("audit.json.sha256").write_text(
        f"{digest}  audit.json\n", encoding="utf-8"
    )
    plan = _plan(digest)
    auto_root = tmp_path / "auto"
    raw_root = tmp_path / "raw"
    for repetition in plan.repetitions:
        _write_json(
            auto_root
            / "runs"
            / f"benchmark_easy_seed{repetition}"
            / "summary.json",
            {
                "benchmark_id": "benchmark",
                "tier": "easy",
                "seed": repetition,
                "selection_policy": "incumbent_relative_hybrid",
                "evaluation_stage": "development_selection_frozen",
            },
        )
        run = raw_root / f"openai_gpt-5-6-sol_benchmark_easy_rep{repetition}"
        _write_json(
            run / "run_config.json",
            {
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "benchmark_id": "benchmark",
                "tier": "easy",
                "repetition": repetition,
            },
        )
        _write_json(run / "candidate.json", {"candidate": repetition})
        _write_json(run / "evaluation.json", {"evaluation": repetition})

    assert validate_hidden_audit(audit, plan.hidden_contract_audit) == digest
    requests, sources = freeze_pilot_sources(
        plan, autoformalism_root=auto_root, raw_agent_root=raw_root
    )

    assert len(requests) == 4
    assert len(sources) == 4
    assert len({item.request_id for item in requests}) == 4
    assert {item.source_kind for item in requests} == {
        "autoformalism",
        "raw_data_agent",
    }
    raw_sources = [item for item in sources if item.source_kind == "raw_data_agent"]
    assert all(len(item.artifact_sha256) == 3 for item in raw_sources)
    assert all(item.artifact_status == "available" for item in sources)


def test_freeze_pilot_sources_retains_missing_method_outcomes(tmp_path: Path) -> None:
    plan = _plan("0" * 64)

    requests, sources = freeze_pilot_sources(
        plan,
        autoformalism_root=tmp_path / "auto",
        raw_agent_root=tmp_path / "raw",
    )

    assert len(requests) == 4
    assert all(item.expected_benchmark_id == "benchmark" for item in requests)
    assert all(item.artifact_status == "missing" for item in sources)
    assert sum(len(item.missing_artifacts) for item in sources) == 8


def test_hidden_audit_requires_exact_companion_digest(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    _write_json(
        audit,
        {
            "schema_version": "phase-b-hidden-subspace-contract-audit-2",
            "status": "pass",
        },
    )
    digest = _sha256(audit)
    plan = _plan(digest)
    audit.with_name("audit.json.sha256").write_text(
        f"{'f' * 64}  audit.json\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="companion digest is invalid"):
        validate_hidden_audit(audit, plan.hidden_contract_audit)
