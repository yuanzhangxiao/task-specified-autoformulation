#!/usr/bin/env python3
"""Submit the single frozen candidate-15 revision to two ACES H100 GPUs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from scripts import harmony_vocab
from scripts.submit_structural_revision_aces import (
    MODELS,
    _digest,
    _read,
    _seal,
    _submit_job,
    file_hash,
    snapshot_identity,
)

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "candidate-15-postmeal-revision-1"
EXECUTION_PROTOCOL = "candidate-15-insulin-revision-aces-execution-1"
SETTINGS = {
    "model": MODELS["120b"]["model"],
    "model_revision": "loopback_alias_revision_unverified",
    "maximum_requests": 1,
    "attempts_per_step": 1,
    "max_output_tokens": 4096,
    "maximum_total_tokens": 32768,
    "temperature": 0.2,
    "reasoning_effort": "low",
    "timeout_seconds": 180.0,
}
ENTRYPOINTS = (
    "scripts/propose_meal_insulin_revision.py",
    "scripts/submit_meal_insulin_revision_aces.py",
    "scripts/hpc/start_meal_insulin_revision_aces.sh",
    "scripts/hpc/run_meal_insulin_revision_aces.sh",
    "scripts/submit_structural_revision_aces.py",
    "scripts/harmony_vocab.py",
)


def _proposer_load(root: Path) -> tuple[dict, dict]:
    """Verify the frozen request and current proposer source through its owner."""
    from scripts import propose_meal_insulin_revision as proposer

    return proposer._load(root)


def _proposer_status(root: Path) -> dict:
    """Read a validated terminal receipt without requiring model assets or CUDA."""
    from scripts import propose_meal_insulin_revision as proposer

    return proposer.status(root)


def verify_inputs(root: Path) -> dict:
    """Require the exact training-only one-call 120B request before scheduling."""
    plan, _ = _proposer_load(root)
    if (
        plan.get("protocol") != PROTOCOL
        or plan.get("model_size") != "120b"
        or plan.get("settings") != SETTINGS
        or plan.get("seed") != 0
        or plan.get("automatic_followup") is not False
        or plan.get("test_data_opened") is not False
        or type(plan.get("optimizer_calls")) is not int
        or plan["optimizer_calls"] != 0
    ):
        raise ValueError("requires one 120B call, 4096 output tokens and no fitting")
    return plan


def _assets() -> tuple[Path, dict]:
    """Require the existing offline Python, serving image and pinned snapshot."""
    for name in ("AF_PYTHON", "AF_VLLM_IMAGE", "AF_HF_HOME"):
        if not os.environ.get(name):
            raise ValueError(f"set {name} before submission")
    image = Path(os.environ["AF_VLLM_IMAGE"])
    if not image.is_file() or not Path(os.environ["AF_PYTHON"]).is_file():
        raise ValueError("Python or serving image is missing")
    return image, snapshot_identity(Path(os.environ["AF_HF_HOME"]), "120b")


def execution_identity(plan: dict, image_sha256: str, snapshot: dict) -> dict:
    """Bind the request to pinned serving assets, launcher code and resources."""
    model = MODELS["120b"]
    return {
        "protocol": EXECUTION_PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "model_size": "120b",
        "model": model["model"],
        "model_revision": model["revision"],
        "platform": "aces-h100x2",
        "image_sha256": image_sha256,
        "snapshot": snapshot,
        "harmony_vocabulary": harmony_vocab.verify(
            Path(os.environ.get("AF_HARMONY_ASSETS", REPO / "assets/harmony"))
        ),
        "entrypoint_sha256": {name: file_hash(REPO / name) for name in ENTRYPOINTS},
        "served_context_tokens": 32768,
        "maximum_output_tokens": 4096,
        "tensor_parallel_size": 2,
        "maximum_sequences": 1,
        "gpu_memory_utilization": 0.90,
        "generation_settings": SETTINGS,
        "maximum_generation_requests": 1,
        "maximum_charged_tokens": 32768,
        "worker_seconds": 3000,
        "scheduler_walltime_seconds": 3600,
        "cpus_per_task": 16,
        "memory_gib": 128,
        "vllm_version": "0.27.1",
        "compiler_cache_policy": "node-local-job-temporary-1",
        "server_binding": "127.0.0.1",
        "optimizer_calls": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "model_promotion": False,
    }


def verify(root: Path, *, check_assets: bool = False) -> tuple[dict, dict]:
    """Recheck source identity and optionally actual offline assets before CUDA."""
    plan = verify_inputs(root)
    execution = _read(root / "execution.json")
    image_hash, snapshot = execution["image_sha256"], execution["snapshot"]
    if check_assets:
        image, snapshot = _assets()
        image_hash = file_hash(image)
    expected = execution_identity(plan, image_hash, snapshot)
    if execution != {**expected, "artifact_sha256": _digest(expected)}:
        raise ValueError("insulin revision execution identity differs")
    return plan, execution


def bind_server(root: Path, runtime_record: Path, endpoint: str) -> dict:
    """Freeze the observed pinned runtime before the sole physical request."""
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("server requires an unauthenticated loopback HTTP endpoint")
    _, execution = verify(root)
    runtime = _read(runtime_record)
    keys = (
        "artifact_sha256",
        "model_size",
        "model",
        "model_revision",
        "image_sha256",
        "served_context_tokens",
        "tensor_parallel_size",
        "maximum_sequences",
    )
    if runtime.get("serving_identity") != {key: execution[key] for key in keys}:
        raise ValueError("actual serving identity differs from the sealed request")
    names = runtime.get("gpu_names", [])
    memory = runtime.get("gpu_memory_bytes", [])
    drivers = runtime.get("gpu_driver", [])
    if (
        type(runtime.get("visible_gpus")) is not int
        or runtime["visible_gpus"] != 2
        or runtime.get("cuda_kernel_passed") is not True
        or not isinstance(names, list)
        or len(names) != 2
        or any(not isinstance(name, str) or "H100" not in name for name in names)
        or not isinstance(memory, list)
        or len(memory) != 2
        or any(type(size) is not int or size < 75 * 1024**3 for size in memory)
        or not isinstance(drivers, list)
        or not drivers
        or any(not isinstance(driver, str) or not driver for driver in drivers)
        or runtime.get("vllm") != "0.27.1"
        or not isinstance(runtime.get("torch"), str)
        or not runtime["torch"]
        or not isinstance(runtime.get("cuda"), str)
        or not runtime["cuda"]
    ):
        raise ValueError("runtime requires two working H100 80GB GPUs and pinned vLLM")
    harmony = runtime.get("harmony", {})
    if (
        {key: harmony.get(key) for key in ("protocol", "filename", "sha256")}
        != execution["harmony_vocabulary"]
        or harmony.get("harmony_render_passed") is not True
        or not isinstance(harmony.get("openai_harmony"), str)
        or not harmony["openai_harmony"]
        or type(harmony.get("rendered_tokens")) is not int
        or harmony["rendered_tokens"] <= 0
        or harmony.get("generation_requests") != 0
    ):
        raise ValueError("runtime requires the pinned offline Harmony rendering probe")
    with urlopen(endpoint.rstrip("/") + "/v1/models", timeout=15) as response:
        served = json.load(response).get("data", [])
    if (
        len(served) != 1
        or served[0].get("id") != execution["model"]
        or served[0].get("max_model_len") != execution["served_context_tokens"]
    ):
        raise ValueError("actual local model alias or context differs")
    return _seal(
        root / "serving_runtime.json",
        {key: value for key, value in runtime.items() if key != "artifact_sha256"},
    )


def submit(root: Path) -> dict:
    """Submit at most once; preserve uncertain scheduler intent for manual review."""
    root = root.resolve()
    plan = verify_inputs(root)
    if (root / "result.json").exists():
        return {**_proposer_status(root), "already_complete": True}
    if (root / "attempt.json").exists() and not (
        root / "submission/manifest.json"
    ).exists():
        raise ValueError(
            "generation already reserved; inspect or replay the saved call"
        )
    image, snapshot = _assets()
    identity = execution_identity(plan, file_hash(image), snapshot)
    os.environ.update(AF_REPO_ROOT=str(REPO), AF_OUTPUT_ROOT=str(root))
    root.mkdir(parents=True, exist_ok=True)
    with (root / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        execution = _seal(root / "execution.json", identity)
        directory = root / "submission"
        manifest = directory / "manifest.json"
        if manifest.exists():
            value = _read(manifest)
            if (
                value.get("execution_sha256") != execution["artifact_sha256"]
                or not isinstance(value.get("job"), str)
                or not value["job"].isdecimal()
            ):
                raise ValueError("submission identity differs")
            return value
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(
                "existing submission intent; inspect scheduler/receipt before "
                f"recovery: {directory}"
            )
        directory.mkdir(exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        argv = [
            "sbatch",
            "--parsable",
            "--nodes=1",
            "--ntasks=1",
            "--export=ALL",
            "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
            "--partition=gpu",
            "--gres=gpu:h100:2",
            "--cpus-per-task=16",
            "--mem=128G",
            "--time=01:00:00",
            "--signal=B:TERM@300",
            "--exclude=ac042",
            "--job-name=phasec-insulin-revision-120b",
            f"--output={root}/logs/propose-%j.out",
            f"--error={root}/logs/propose-%j.err",
            str(REPO / "scripts/hpc/run_meal_insulin_revision_aces.sh"),
        ]
        job = _submit_job(directory, argv)
        return _seal(
            manifest,
            {
                "execution_sha256": execution["artifact_sha256"],
                "model_size": "120b",
                "job": job,
                "maximum_generation_requests": 1,
                "automatic_followup": False,
                "optimizer_calls": 0,
            },
        )


def main() -> None:
    """Submit the sealed request or verify its frozen ACES execution identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--check-assets", action="store_true")
    args = parser.parse_args()
    if args.check_assets and not args.verify_only:
        parser.error("--check-assets requires --verify-only")
    result = (
        verify(args.root.resolve(), check_assets=args.check_assets)[1]
        if args.verify_only
        else submit(args.root)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
