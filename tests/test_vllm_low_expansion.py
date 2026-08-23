"""Static checks for the frozen six-pair vLLM-low expansion."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

CONFIG = Path("configs/hybrid_judge_vllm_low_expansion_v1.json")
SCRIPT = Path("scripts/hpc/phase_b_hybrid_judge_vllm_low_expansion.slurm")


def test_vllm_low_expansion_reuses_four_pairs_and_adds_six() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    stress = json.loads(
        Path("configs/hybrid_judge_vllm_reasoning_pilot_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["reasoning_effort"] == "low"
    assert payload["reused_logical_calls"] == 40
    assert payload["planned_new_logical_calls"] == 60
    assert payload["planned_final_logical_calls"] == 100
    assert len(payload["selected_pair_ids"]) == 6
    assert payload["final_analysis"]["pair_count"] == 10
    assert payload["final_analysis"]["merge_duplicate_policy"] == "error"
    assert not (set(payload["selected_pair_ids"]) & set(stress["selected_pair_ids"]))
    all_pair_ids = set(payload["selected_pair_ids"]) | set(
        stress["selected_pair_ids"]
    )
    assert len(all_pair_ids) == 10


def test_vllm_low_expansion_script_is_valid_and_frozen() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")

    assert "#SBATCH --gpus-per-node=1" in text
    assert "#SBATCH --array=0-5" in text
    assert 'apptainer build --sandbox "${runtime_image}"' in text
    assert '--judge-models "vllm:${AF_LOCAL_MODEL}"' in text
    assert "--vllm-reasoning-effort low" in text
    assert "frozen expansion requires five repetitions and six shards" in text
    assert "--pair-ids" in text
