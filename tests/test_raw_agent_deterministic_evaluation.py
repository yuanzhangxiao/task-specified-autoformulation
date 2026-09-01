"""Tests for the full GPT-5.6 deterministic evaluation source freeze."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from autoformalism.data import BenchmarkRegistry
from autoformalism.rebuttal.raw_agent_deterministic_evaluation import (
    RawAgentDeterministicEvaluationPlan,
    freeze_raw_agent_sources,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "phase_b_raw_agent_deterministic_evaluation_v1.json"
FULL = ROOT / "configs" / "raw_data_agent_fitted_model_full_v1.json"
REFRESH = ROOT / "configs" / "raw_data_agent_fitted_model_prompt_v3_refresh_v1.json"
JOBS = tuple(
    ROOT / "scripts" / "hpc" / name
    for name in (
        "phase_b_raw_agent_eval_prepare.slurm",
        "phase_b_raw_agent_eval_postfreeze.slurm",
        "phase_b_raw_agent_eval_postfreeze_merge.slurm",
        "phase_b_raw_agent_eval_hidden.slurm",
        "phase_b_raw_agent_eval_finalize.slurm",
        "submit_phase_b_raw_agent_deterministic_evaluation.sh",
    )
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _plan(full: Path, refresh: Path) -> RawAgentDeterministicEvaluationPlan:
    return RawAgentDeterministicEvaluationPlan.model_validate(
        {
            "schema_version": (
                "phase-b-raw-agent-deterministic-evaluation-plan-1"
            ),
            "status": "frozen_before_test_or_private_evaluation",
            "method_id": "raw_data_agent:gpt-5.6-sol",
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "full_protocol_config_sha256": _sha256(full),
            "prompt_refresh_config_sha256": _sha256(refresh),
            "hidden_contract_audit": {
                "schema_version": "phase-b-hidden-subspace-contract-audit-2",
                "sha256": "0" * 64,
                "required_status": "pass",
            },
            "repetitions": [0, 1],
            "postfreeze_shard_count": 2,
            "hidden_shard_count": 2,
            "endpoints": ["source_completion", "sealed_target_nmse"],
            "weighted_overall_score_defined": False,
            "qualitative_llm_requested": False,
            "parameter_refit_applied": False,
            "test_data_opened": False,
        }
    )


def _run(
    root: Path,
    *,
    benchmark_id: str,
    tier: str,
    repetition: int,
    prompt_sha256: str,
) -> None:
    run = root / (
        f"openai_gpt-5-6-sol_{benchmark_id}_{tier}_rep{repetition}"
    )
    _write_json(
        run / "run_config.json",
        {
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "benchmark_id": benchmark_id,
            "tier": tier,
            "repetition": repetition,
            "agent_config": {"output_contract": "fitted_model"},
            "public_input_hashes": {"proposer_prompt.txt": prompt_sha256},
            "parameter_refit_applied": False,
            "test_data_opened": False,
        },
    )
    _write_json(run / "candidate.json", {"candidate": repetition})
    _write_json(run / "evaluation.json", {"evaluation": repetition})


def test_repository_plan_covers_exact_full_suite() -> None:
    plan = RawAgentDeterministicEvaluationPlan.model_validate_json(
        PLAN.read_text(encoding="utf-8")
    )
    full = json.loads(FULL.read_text(encoding="utf-8"))
    configured = {item["benchmark_id"] for item in full["benchmarks"]}
    registered = {
        item
        for item in BenchmarkRegistry().identifiers()
        if item.startswith("phase_b_")
    }

    assert configured == registered
    assert len(configured) == 40
    assert plan.repetitions == (0, 1, 2)
    assert plan.full_protocol_config_sha256 == _sha256(FULL)
    assert plan.prompt_refresh_config_sha256 == _sha256(REFRESH)
    assert plan.weighted_overall_score_defined is False
    assert plan.qualitative_llm_requested is False


def test_source_freeze_composes_refreshed_and_historical_runs(tmp_path: Path) -> None:
    full = tmp_path / "full.json"
    refresh = tmp_path / "refresh.json"
    cells = [
        {"benchmark_id": "unchanged", "tier": "easy"},
        {"benchmark_id": "changed", "tier": "hard"},
    ]
    _write_json(
        full,
        {
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "output_contract": "fitted_model",
            "repetitions": 2,
            "benchmarks": cells,
        },
    )
    public = tmp_path / "public"
    prompt_hashes = {}
    for cell in cells:
        prompt = (
            public
            / "phase_b_v1"
            / cell["benchmark_id"]
            / "proposer_prompt.txt"
        )
        prompt.parent.mkdir(parents=True)
        prompt.write_text(f"prompt for {cell['benchmark_id']}\n", encoding="utf-8")
        prompt_hashes[cell["benchmark_id"]] = _sha256(prompt)
    _write_json(
        refresh,
        {
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "output_contract": "fitted_model",
            "repetitions": 2,
            "benchmarks": [
                {
                    "benchmark_id": "changed",
                    "tier": "hard",
                    "public_prompt_sha256": prompt_hashes["changed"],
                }
            ],
        },
    )
    historical = tmp_path / "historical"
    refreshed = tmp_path / "refreshed"
    for repetition in range(2):
        _run(
            historical,
            benchmark_id="unchanged",
            tier="easy",
            repetition=repetition,
            prompt_sha256=prompt_hashes["unchanged"],
        )
        _run(
            refreshed / "runs",
            benchmark_id="changed",
            tier="hard",
            repetition=repetition,
            prompt_sha256=prompt_hashes["changed"],
        )

    requests, sources, frozen_cells = freeze_raw_agent_sources(
        _plan(full, refresh),
        full_protocol_config=full,
        prompt_refresh_config=refresh,
        historical_root=historical,
        refresh_root=refreshed,
        public_data_root=public,
    )

    assert len(requests) == 4
    assert len(sources) == 4
    assert len(frozen_cells) == 2
    assert all(item.artifact_status == "available" for item in sources)
    by_benchmark = {item.benchmark_id: item.source_kind for item in sources}
    assert by_benchmark == {
        "unchanged": "raw_data_agent:historical_unchanged_prompt",
        "changed": "raw_data_agent:prompt_v3_refresh",
    }


def test_source_freeze_rejects_stale_prompt_hash(tmp_path: Path) -> None:
    full = tmp_path / "full.json"
    refresh = tmp_path / "refresh.json"
    _write_json(
        full,
        {
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "output_contract": "fitted_model",
            "repetitions": 2,
            "benchmarks": [{"benchmark_id": "unchanged", "tier": "easy"}],
        },
    )
    _write_json(
        refresh,
        {
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "output_contract": "fitted_model",
            "repetitions": 2,
            "benchmarks": [],
        },
    )
    prompt = (
        tmp_path
        / "public"
        / "phase_b_v1"
        / "unchanged"
        / "proposer_prompt.txt"
    )
    prompt.parent.mkdir(parents=True)
    prompt.write_text("current prompt\n", encoding="utf-8")
    for repetition in range(2):
        _run(
            tmp_path / "historical",
            benchmark_id="unchanged",
            tier="easy",
            repetition=repetition,
            prompt_sha256="f" * 64,
        )

    with pytest.raises(ValueError, match="different public prompt"):
        freeze_raw_agent_sources(
            _plan(full, refresh),
            full_protocol_config=full,
            prompt_refresh_config=refresh,
            historical_root=tmp_path / "historical",
            refresh_root=tmp_path / "refreshed",
            public_data_root=tmp_path / "public",
        )


def test_raw_agent_evaluation_chain_is_cpu_only_and_syntactically_valid() -> None:
    for path in JOBS:
        text = path.read_text(encoding="utf-8")
        assert "--gres" not in text
        assert "--gpus" not in text
        subprocess.run(["bash", "-n", str(path)], check=True)
    assert "#SBATCH --array=0-23%12" in JOBS[1].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-23%12" in JOBS[3].read_text(encoding="utf-8")
