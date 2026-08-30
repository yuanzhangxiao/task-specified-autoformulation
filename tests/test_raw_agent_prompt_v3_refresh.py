"""Tests for the bounded GPT-5.6 public-prompt v3 refresh."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.prepare_raw_data_agent_prompt_v3_refresh import (
    freeze_prompt_v3_refresh,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/raw_data_agent_fitted_model_prompt_v3_refresh_v1.json"
OVERLAY_CONFIG = ROOT / "configs/phase_b_public_prompt_overlay_v3.json"
SOURCE_CONFIG = ROOT / "configs/raw_data_agent_fitted_model_full_v1.json"
JOB = ROOT / "scripts/hpc/phase_b_raw_data_agent_prompt_v3_refresh.slurm"
SUBMIT = ROOT / "scripts/hpc/submit_phase_b_raw_data_agent_prompt_v3_refresh.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_refresh_is_exactly_the_ten_v3_changed_cells() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    overlay = json.loads(OVERLAY_CONFIG.read_text(encoding="utf-8"))
    expected = {
        benchmark_id: revision["expected_revised_prompt_sha256"]
        for revision in overlay["revisions"]
        for benchmark_id in revision["benchmark_ids"]
    }
    configured = {
        item["benchmark_id"]: item["public_prompt_sha256"]
        for item in config["benchmarks"]
    }

    assert configured == expected
    assert len(configured) == 10
    assert config["repetitions"] == 3
    assert config["model"] == "gpt-5.6-sol"
    assert config["output_contract"] == "fitted_model"
    assert config["source_full_protocol_config_sha256"] == _sha256(SOURCE_CONFIG)
    assert config["prompt_overlay_config_sha256"] == _sha256(OVERLAY_CONFIG)
    assert config["test_data_opened"] is False


def test_refresh_freeze_binds_prompt_and_development_splits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"source":true}\n', encoding="utf-8")
    overlay_config = tmp_path / "overlay-config.json"
    overlay_config.write_text('{"overlay":true}\n', encoding="utf-8")
    public = tmp_path / "public"
    benchmark_id = "phase_b_changed_cell"
    cell = public / "phase_b_v1" / benchmark_id
    cell.mkdir(parents=True)
    (cell / "proposer_prompt.txt").write_text(
        "Revised prompt.\n", encoding="utf-8"
    )
    (cell / "train.csv").write_text(
        "trajectory_id,time,y\na,0,1\n", encoding="utf-8"
    )
    (cell / "validation.csv").write_text(
        "trajectory_id,time,y\nb,0,2\n", encoding="utf-8"
    )
    overlay_manifest = {
        "status": "ready",
        "suite_version": "phase_b_v1",
        "non_proposer_files_byte_identical": True,
        "target_contract_manifest_sha256": "target-contract",
        "changed_benchmark_ids": [benchmark_id],
        "cells": [
            {
                "benchmark_id": benchmark_id,
                "changed": True,
                "overlay_prompt_sha256": _sha256(
                    cell / "proposer_prompt.txt"
                ),
            }
        ],
    }
    manifest_path = public / "prompt_overlay_manifest.json"
    manifest_path.write_text(
        json.dumps(overlay_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": (
                    "raw-data-agent-fitted-model-prompt-v3-refresh-1"
                ),
                "status": "frozen_before_refresh_calls",
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "output_contract": "fitted_model",
                "source_full_protocol_config_sha256": _sha256(source),
                "prompt_overlay_config_sha256": _sha256(overlay_config),
                "prompt_overlay_manifest_sha256": _sha256(manifest_path),
                "target_contract_manifest_sha256": "target-contract",
                "benchmarks": [
                    {
                        "benchmark_id": benchmark_id,
                        "tier": "easy",
                        "public_prompt_sha256": _sha256(
                            cell / "proposer_prompt.txt"
                        ),
                    }
                ],
                "repetitions": 3,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    frozen = tmp_path / "frozen"
    result = freeze_prompt_v3_refresh(
        config_path=config,
        output_root=frozen,
        public_data_root=public,
        overlay_config_path=overlay_config,
        source_full_config_path=source,
    )

    assert result["cell_count"] == 1
    assert result["task_count"] == 3
    assert result["test_data_opened"] is False
    tasks = [
        json.loads(line)
        for line in (frozen / "task_plan.jsonl").read_text().splitlines()
    ]
    assert [item["repetition"] for item in tasks] == [0, 1, 2]
    assert all("test" not in item for item in tasks)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        freeze_prompt_v3_refresh(
            config_path=config,
            output_root=frozen,
            public_data_root=public,
            overlay_config_path=overlay_config,
            source_full_config_path=source,
        )


def test_refresh_launchers_are_fail_closed_and_do_not_persist_secrets() -> None:
    for path in (JOB, SUBMIT):
        subprocess.run(["bash", "-n", str(path)], check=True)
    job = JOB.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-29%4" in job
    assert "public-prompt-v3" in job
    assert "sha256sum -c" in job
    assert "proposer_prompt.txt:public_prompt_sha256" in job
    assert "test.csv" not in job
    assert "Paste OPENAI_API_KEY" in submit
    assert "raw-data-agent-prompt-v3-refresh-submission-1" in submit
    assert "OPENAI_API_KEY" not in CONFIG.read_text(encoding="utf-8")
