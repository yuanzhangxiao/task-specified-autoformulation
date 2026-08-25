"""Static checks for the self-contained Delta hybrid-search smoke job."""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path("scripts/hpc/phase_b_hybrid_search_smoke_120b.slurm")


def test_hybrid_search_smoke_script_is_valid_and_self_contained() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text(encoding="utf-8")

    for required in (
        "#SBATCH --account=bibo-delta-gpu",
        "#SBATCH --partition=gpuA40x4",
        ': "${AF_PROJECT:=/projects/bibo/${af_user}}"',
        ': "${AF_WORK:=/work/hdd/bibo/${af_user}}"',
        ': "${AF_PUBLIC_DATA_ROOT:=${AF_PROJECT}/phase_b/inputs/public}"',
        "--selection-policy incumbent_relative_hybrid",
        "--development-only",
        "--beam-size 1",
        "--hybrid-science-weight",
        "--resume",
        "configs/hybrid_search_objective_pilot_v1.json",
        'vllm serve "${AF_LOCAL_MODEL}"',
        '--proposer-model "vllm:${AF_LOCAL_MODEL}"',
        '--judge-model "vllm:${AF_LOCAL_MODEL}"',
    ):
        assert required in source

    assert "API_KEY" not in source
    assert "--evaluate-test" not in source
