"""Static checks for the isolated vLLM GPT-OSS deployment smoke test."""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path("scripts/hpc/vllm_gpt_oss_smoke.slurm")


def test_vllm_smoke_script_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_vllm_smoke_script_pins_and_isolates_deployment() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "vllm/vllm-openai:v0.27.1" in text
    assert "openai/gpt-oss-20b" in text
    assert "--gpus-per-node=1" in text
    assert '"reasoning_effort": "low"' in text
    assert '"type": "json_schema"' in text
    assert "--max-model-len 16384" in text
    assert "AF_HF_HOME" in text
    assert "AF_VLLM_IMAGE" in text
