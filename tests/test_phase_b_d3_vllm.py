"""Local vLLM provider support and batched submission for the D3 campaign."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.list_phase_b_d3_task_indices import task_indices

CONFIG = Path("configs/phase_b_d3_full_v1.json")
HPC = Path("scripts/hpc")


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_indices_follow_plan_order_and_filter_by_tier() -> None:
    """prepare() builds rows as cell-major, repetition-minor."""
    config = _config()
    repetitions = len(config["repetitions"])
    assert task_indices(config, "all") == tuple(
        range(len(config["cells"]) * repetitions)
    )
    easy = task_indices(config, "easy")
    hard = task_indices(config, "hard")
    assert len(easy) + len(hard) == len(task_indices(config, "all"))
    assert not set(easy) & set(hard)
    for position, cell in enumerate(config["cells"]):
        expected = set(range(position * repetitions, (position + 1) * repetitions))
        selected = set(easy if cell["tier"] == "easy" else hard)
        assert expected <= selected


def test_full_matrix_splits_into_sixty_easy_and_sixty_hard() -> None:
    config = _config()
    assert len(task_indices(config, "easy")) == 60
    assert len(task_indices(config, "hard")) == 60


def test_provider_is_recorded_and_bound_to_its_endpoint(monkeypatch) -> None:
    """A resume must not silently switch backends."""
    pytest.importorskip("torch")
    from autoformalism.rebuttal.phase_b_d3 import _endpoint, environment_identity

    monkeypatch.setenv("AF_VLLM_BASE_URL", "http://127.0.0.1:31337")
    assert _endpoint("vllm") == "http://127.0.0.1:31337"
    assert _endpoint("openai").startswith("https://")
    hosted = environment_identity("openai")
    local = environment_identity("vllm")
    assert hosted["provider"] == "openai"
    assert local["provider"] == "vllm"
    assert hosted["provider_endpoint_sha256"] != local["provider_endpoint_sha256"]


def test_prepare_rejects_an_unknown_provider(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from autoformalism.rebuttal.phase_b_d3 import prepare

    with pytest.raises(ValueError, match="unsupported D3 provider"):
        prepare(CONFIG, tmp_path, tmp_path, "gpt-oss-120b", "anthropic")


def test_cli_exposes_the_provider_choice() -> None:
    text = Path("scripts/phase_b_d3.py").read_text(encoding="utf-8")
    assert '"--provider"' in text
    assert 'choices=("openai", "vllm")' in text
    assert "args.provider" in text


def test_batch_job_defaults_to_one_gpu_and_resumes_after_a_failure() -> None:
    path = HPC / "phase_b_d3_vllm_batch.slurm"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text(encoding="utf-8")
    # the GPU request is supplied by the submitter: ACES and Delta differ
    assert "#SBATCH --gpus-per-node" not in text
    assert "#SBATCH --gres" not in text
    assert "AF_TENSOR_PARALLEL_SIZE:=1" in text
    # a single 80 GB card holds MXFP4 weights plus a 32k KV cache
    assert "AF_VLLM_MAX_MODEL_LEN:=32768" in text
    # one model load per batch, not per task
    assert "vllm serve" in text
    # CPU fitting dominates, so tasks overlap against the served model
    assert "AF_TASK_CONCURRENCY:=8" in text
    assert '--max-num-seqs "${AF_TASK_CONCURRENCY}"' in text
    assert "fit_threads=$(( ${SLURM_CPUS_PER_TASK:-8} / AF_TASK_CONCURRENCY ))" in text
    # a failed task must not abandon the rest of the batch
    assert "failures=$((failures + 1))" in text
    assert "export AF_VLLM_BASE_URL" in text
    # the plan is frozen in-job, serialised so only the first batch pays
    assert "flock" in text and "phase_b_d3.py freeze" in text
    assert "--provider vllm" in text


def test_submission_batches_by_tier_and_records_a_ledger() -> None:
    path = HPC / "submit_phase_b_d3_vllm.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text(encoding="utf-8")
    assert "AF_TIER:=easy" in text
    assert "list_phase_b_d3_task_indices.py" in text
    assert "submission_ledger.jsonl" in text
    # the account must match the job type and the GPU flag differs per site
    assert 'AF_CLUSTER:=delta' in text
    assert 'AF_ACCOUNT:=bibo-delta-gpu' in text
    assert 'AF_GPU_REQUEST:=--gpus-per-node=4' in text
    assert 'AF_GPU_REQUEST:=--gres=gpu:h100:1' in text
    assert "${AF_GPU_REQUEST}" in text
    # no CPU job: ACES rejects CPU work submitted under a GPU account
    assert "AF_CPU_PARTITION" not in text
    assert "partition=cpu" not in text
