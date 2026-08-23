"""Static checks for the frozen vLLM low-versus-high judge pilot."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

CONFIG = Path("configs/hybrid_judge_vllm_reasoning_pilot_v1.json")
SCRIPT = Path("scripts/hpc/phase_b_hybrid_judge_vllm_reasoning.slurm")


def test_vllm_reasoning_pilot_is_frozen_on_the_difficult_pairs() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert payload["reasoning_efforts"] == ["low", "high"]
    assert payload["planned_logical_calls_per_reasoning_effort"] == 40
    assert payload["planned_logical_calls_total"] == 80
    assert payload["candidate_order_policy"] == "both_orientations"
    assert len(payload["selected_pair_ids"]) == 4
    assert payload["transport"] == "vllm_openai_chat_json_schema"
    assert payload["provider_retry"]["max_attempts"] == 10


def test_vllm_reasoning_pilot_script_is_valid_and_self_contained() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")

    assert "#SBATCH --gpus-per-node=1" in text
    assert "#SBATCH --array=0-7" in text
    assert '0) readonly reasoning_effort="low"' in text
    assert '1) readonly reasoning_effort="high"' in text
    assert 'apptainer build --sandbox "${runtime_image}"' in text
    assert '--judge-models "vllm:${AF_LOCAL_MODEL}"' in text
    assert '--vllm-reasoning-effort "${reasoning_effort}"' in text
    assert '--max-model-len 32768' in text
    assert "--pair-ids" in text
    assert "frozen pilot requires five repetitions and four shards" in text
    assert "frozen pilot requires ten attempts and seed base 9000" in text
