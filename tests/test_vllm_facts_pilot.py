"""Static checks for the frozen algebraic-facts judge pilot."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

CONFIG = Path("configs/hybrid_judge_vllm_facts_pilot_v1.json")
SCRIPT = Path("scripts/hpc/phase_b_hybrid_judge_vllm_facts_pilot.slurm")
RUNNER = Path("scripts/run_hybrid_judge.py")


def test_facts_pilot_changes_only_prompt_and_structural_facts() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert payload["status"] == "frozen_matched_development_ablation"
    assert payload["judge_model"] == "vllm:openai/gpt-oss-20b"
    assert payload["selected_mutations"] == [
        "duplicated_gp_flux",
        "wrong_meal_sink",
    ]
    assert len(payload["selected_pair_ids"]) == 4
    assert payload["planned_logical_calls"] == 40
    assert payload["reasoning_effort"] == "low"
    assert payload["temperature"] == 0.2
    assert payload["seed_base"] == 9000
    assert payload["repetitions"] == 5
    assert payload["scoring"] == {
        "partial_tiebreak_weight": 0.05,
        "comparative_weight": 0.25,
        "tie_threshold": 0.05,
    }
    assert payload["matched_control"][
        "same_model_reasoning_seeds_orders_and_scoring"
    ]
    assert payload["treatment"]["hybrid_judge_protocol_version"] == (
        "hybrid-judge-protocol-2"
    )
    assert payload["treatment"]["structural_facts_schema_version"] == (
        "structural-facts-2"
    )


def test_facts_pilot_script_and_manifest_contract_are_frozen() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    script = SCRIPT.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "#SBATCH --gpus-per-node=1" in script
    assert "#SBATCH --array=0-3" in script
    assert 'apptainer build --sandbox "${runtime_image}"' in script
    assert '--judge-models "vllm:${AF_LOCAL_MODEL}"' in script
    assert "--vllm-reasoning-effort low" in script
    assert "--vllm-temperature 0.2" in script
    assert "five repetitions and four shards" in script
    assert "hybrid_judge_protocol_version" in runner
    assert "structural_facts_schema_version" in runner
