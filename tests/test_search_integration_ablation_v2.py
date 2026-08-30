"""Tests for the prompt-v3 deterministic-target search integration plan."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from autoformalism.benchmarks import phase_b_public_spec, render_phase_b_prompts
from autoformalism.rebuttal.search_integration_ablation import (
    build_search_integration_tasks,
    freeze_search_integration_plan,
    load_search_integration_plan,
)
from scripts.write_phase_b_search_task_runtime import write_runtime_record

CONFIG = Path("configs/phase_b_search_integration_ablation_v2.json")
OVERLAY_CONFIG = Path("configs/phase_b_public_prompt_overlay_v3.json")
CONTRACT_ROOT = Path("configs/target_eval/phase_b_v1")
SEARCH_SLURM = Path(
    "scripts/hpc/phase_b_search_integration_ablation_v2_120b.slurm"
)
EVALUATION_SLURM = Path(
    "scripts/hpc/phase_b_search_ablation_evaluation_v2.slurm"
)
SUBMIT = Path(
    "scripts/hpc/submit_phase_b_search_integration_ablation_v2.sh"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_fixture(tmp_path: Path) -> tuple[Path, Path]:
    public = tmp_path / "public"
    contracts = tmp_path / "contracts"
    (contracts / "specs").mkdir(parents=True)
    (contracts / "manifest.json").write_bytes(
        (CONTRACT_ROOT / "manifest.json").read_bytes()
    )
    specifications = (
        phase_b_public_spec(
            "dalla_man",
            "easy",
            "named",
            task="T2",
            dynamics="canonical",
            data_root=Path("data_raw"),
        ),
        phase_b_public_spec(
            "alien_device",
            "hard",
            "opaque",
            task=None,
            dynamics="canonical",
            data_root=Path("data_raw"),
        ),
    )
    for specification in specifications:
        benchmark_id = specification.benchmark_id
        cell = public / "phase_b_v1" / benchmark_id
        cell.mkdir(parents=True)
        prompt = render_phase_b_prompts(specification)[0]
        (cell / "proposer_prompt.txt").write_text(prompt, encoding="utf-8")
        (cell / "manifest.json").write_text(
            json.dumps(
                {
                    "benchmark_id": benchmark_id,
                    "status": "production_registered",
                    "test_sealed": True,
                }
            ),
            encoding="utf-8",
        )
        source_contract = CONTRACT_ROOT / "specs" / f"{benchmark_id}.json"
        (contracts / "specs" / f"{benchmark_id}.json").write_bytes(
            source_contract.read_bytes()
        )
    plan = load_search_integration_plan(CONFIG)
    assert plan.public_input_contract is not None
    (public / "prompt_overlay_manifest.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "suite_version": "phase_b_v1",
                "cell_count": 40,
                "non_proposer_files_byte_identical": True,
                "target_contract_manifest_sha256": (
                    plan.public_input_contract.target_contract_manifest_sha256
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return public, contracts


def test_v2_plan_freezes_exact_public_inputs_and_resource_policy(
    tmp_path: Path,
) -> None:
    public, contracts = _public_fixture(tmp_path)
    frozen = tmp_path / "frozen"
    manifest = freeze_search_integration_plan(
        CONFIG,
        frozen,
        public_data_root=public,
        target_contract_root=contracts,
        prompt_overlay_config_path=OVERLAY_CONFIG,
    )
    plan = load_search_integration_plan(CONFIG)

    assert len(build_search_integration_tasks(plan)) == 12
    assert manifest["public_input_validation"]["cell_count"] == 2
    assert manifest["public_input_validation"]["test_data_opened"] is False
    assert manifest["resource_accounting_schema_version"] == (
        "phase-b-search-resource-ledger-1"
    )
    assert plan.resource_accounting is not None
    assert plan.resource_accounting.logical_cached_usage_counted is True

    first = plan.cells[0]
    prompt = public / "phase_b_v1" / first.benchmark_id / "proposer_prompt.txt"
    prompt.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="public prompt differs"):
        freeze_search_integration_plan(
            CONFIG,
            tmp_path / "other",
            public_data_root=public,
            target_contract_root=contracts,
            prompt_overlay_config_path=OVERLAY_CONFIG,
        )


def test_v2_launchers_pin_overlay_contract_and_accounting() -> None:
    for path in (SEARCH_SLURM, EVALUATION_SLURM, SUBMIT):
        subprocess.run(["bash", "-n", str(path)], check=True)

    search = SEARCH_SLURM.read_text(encoding="utf-8")
    for required in (
        "public-prompt-v3",
        "phase_b_search_integration_ablation_v2.json",
        "--public-target-contract",
        "public_target_contract_sha256",
        "prompt_overlay_manifest_sha256",
        "write_phase_b_search_task_runtime.py",
        "/usr/bin/time",
        "--no-judge",
        "--require-initial-proposer-cache-hit",
        'AF_VLLM_PORT:=8000',
        "search-integration-ablation-v3",
    ):
        assert required in search
    assert "SLURM_JOB_ID %" not in search
    assert "API_KEY" not in search
    assert "--evaluate-test" not in search

    evaluation = EVALUATION_SLURM.read_text(encoding="utf-8")
    assert "slurm_accounting.psv" in evaluation
    assert "summarize_phase_b_search_integration_ablation.py" in evaluation
    assert "evaluate_phase_b_hidden_subspace.py" in evaluation

    submit = SUBMIT.read_text(encoding="utf-8")
    assert "--array=0-5%2" in submit
    assert "--array=6-11%2" in submit
    assert "submission_manifest.json" in submit
    assert "--public-data-root" in submit
    assert "--target-contract-root" in submit
    assert "AF_HIDDEN_AUDIT" in submit
    assert "sha256sum -c" in submit
    assert '--output="${AF_REPO_ROOT}/logs/' in submit
    assert "required_cache_hit_no_provider_fallback" in submit
    assert "phase-b-search-integration-submission-2" in submit


def test_task_runtime_records_allocation_and_process_metrics(tmp_path: Path) -> None:
    process = tmp_path / "time.txt"
    process.write_text(
        "elapsed_seconds=12.5\n"
        "user_cpu_seconds=4.0\n"
        "system_cpu_seconds=1.5\n"
        "max_rss_kib=2048\n",
        encoding="utf-8",
    )
    gpu = tmp_path / "gpu.txt"
    gpu.write_text("NVIDIA A40, 595.71.05, 46068 MiB\n", encoding="utf-8")
    output = tmp_path / "runtime.json"
    record = write_runtime_record(
        output=output,
        task_index=0,
        arm_id="paired_question_consensus",
        benchmark_id="benchmark",
        tier="easy",
        repetition=0,
        started_epoch_seconds=100.0,
        finished_epoch_seconds=460.0,
        exit_code=0,
        allocated_cpus=16,
        allocated_gpus=4,
        gpu_inventory_path=gpu,
        process_time_path=process,
    )

    assert record["task_elapsed_wall_seconds"] == 360.0
    assert record["allocated_gpu_hours"] == 0.4
    assert record["allocated_cpu_core_hours"] == 1.6
    assert record["search_process_elapsed_seconds"] == 12.5
    assert record["search_process_max_rss_kib"] == 2048
    assert record["monetary_cost_usd"] is None
    assert json.loads(output.read_text()) == record


def test_v2_hashes_match_committed_contract_assets() -> None:
    plan = load_search_integration_plan(CONFIG)
    assert plan.public_input_contract is not None
    assert _sha256(OVERLAY_CONFIG) == (
        plan.public_input_contract.prompt_overlay_config_sha256
    )
    assert _sha256(CONTRACT_ROOT / "manifest.json") == (
        plan.public_input_contract.target_contract_manifest_sha256
    )
    for cell in plan.cells:
        assert _sha256(CONTRACT_ROOT / "specs" / f"{cell.benchmark_id}.json") == (
            cell.public_target_contract_sha256
        )
