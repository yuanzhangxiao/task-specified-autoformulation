#!/usr/bin/env python3
"""Submit one bounded saved function-construction campaign on an ACES H100."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import construction_functions as campaign
from scripts import harmony_vocab
from scripts.saved_initialization_aces import file_hash
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "saved-construction-functions-1"
EXECUTION_PROTOCOL = "construction-functions-aces-execution-1"
MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
WORKER_SECONDS = 5400
ENTRYPOINTS = (
    "scripts/construction_functions.py",
    "scripts/submit_construction_functions_aces.py",
    "scripts/submit_review_continuation.py",
    "scripts/saved_initialization_aces.py",
    "scripts/harmony_vocab.py",
    "scripts/hpc/start_construction_functions_aces.sh",
    "scripts/hpc/run_construction_functions_aces.sh",
)
BUDGETS = {
    "maximum_calls_per_case": 96,
    "maximum_charged_tokens_per_case": 2097152,
    "maximum_output_tokens": 8192,
    "attempts_per_step": 2,
}


def verify_inputs(root: Path) -> dict:
    """Require the sealed construction-only plan and its pinned serving limits."""
    plan = campaign.verify(root)
    settings = plan.get("settings", {})
    cases = plan.get("cases")
    if (
        plan.get("protocol") != PROTOCOL
        or not isinstance(cases, list)
        or not 1 <= len(cases) <= 8
        or settings.get("model") != MODEL
        or settings.get("model_revision") != REVISION
        or settings.get("max_output_tokens") != 8192
        or settings.get("maximum_requests") != BUDGETS["maximum_calls_per_case"]
        or settings.get("maximum_total_tokens")
        != BUDGETS["maximum_charged_tokens_per_case"]
        or plan.get("budgets") != BUDGETS
        or any(type(value) is not int for value in plan.get("budgets", {}).values())
        or plan.get("automatic_followup") is not False
        or plan.get("test_data_opened") is not False
        or type(plan.get("optimizer_calls")) is not int
        or plan["optimizer_calls"] != 0
    ):
        raise ValueError("requires the bounded ACES saved function-construction plan")
    return plan


def execution_identity(plan: dict, image_sha256: str) -> dict:
    """Freeze the serving assets, resources, and worker independently of history."""
    return {
        "protocol": EXECUTION_PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "platform": "aces-h100x1",
        "model": MODEL,
        "model_revision": REVISION,
        "image_sha256": image_sha256,
        "harmony_vocabulary": harmony_vocab.verify(
            Path(os.environ.get("AF_HARMONY_ASSETS", REPO / "assets/harmony"))
        ),
        "entrypoint_sha256": {name: file_hash(REPO / name) for name in ENTRYPOINTS},
        "served_context_tokens": 32768,
        "maximum_output_tokens": 8192,
        "tensor_parallel_size": 1,
        "maximum_sequences": 1,
        "gpu_memory_utilization": 0.90,
        "maximum_cases": len(plan["cases"]),
        "maximum_generation_requests": len(plan["cases"])
        * BUDGETS["maximum_calls_per_case"],
        "maximum_charged_tokens": len(plan["cases"])
        * BUDGETS["maximum_charged_tokens_per_case"],
        "worker_seconds": WORKER_SECONDS,
        "scheduler_walltime_seconds": 7200,
        "cpus_per_task": 8,
        "memory_gib": 64,
        "compiler_cache_policy": "node-local-job-temporary-1",
        "server_binding": "127.0.0.1",
        "optimizer_calls": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "model_promotion": False,
    }


def verify(root: Path) -> tuple[dict, dict]:
    """Reject execution drift before starting a server or making model requests."""
    plan = verify_inputs(root)
    execution = sealed_read(root / "execution.json")
    expected = execution_identity(plan, execution["image_sha256"])
    if execution != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("function-construction execution identity differs")
    return plan, execution


def _assets() -> Path:
    """Require local Python, image, and the immutable offline model snapshot."""
    for key in ("AF_PYTHON", "AF_VLLM_IMAGE", "AF_HF_HOME"):
        if not os.environ.get(key):
            raise ValueError(f"set {key} before submission")
    image = Path(os.environ["AF_VLLM_IMAGE"])
    snapshot = (
        Path(os.environ["AF_HF_HOME"])
        / "hub/models--openai--gpt-oss-20b/snapshots"
        / REVISION
        / "config.json"
    )
    if not Path(os.environ["AF_PYTHON"]).is_file() or not image.is_file():
        raise ValueError("Python or serving image is missing")
    if not snapshot.is_file():
        raise ValueError("pinned offline model snapshot is missing")
    return image


def submit(root: Path, wave: str = "functions-1") -> dict:
    """Preserve durable scheduler intent and never repeat an uncertain submission."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", wave):
        raise ValueError("wave must be a simple identifier")
    root = root.resolve()
    plan = verify_inputs(root)
    image = _assets()
    identity = execution_identity(plan, file_hash(image))
    os.environ.update(AF_REPO_ROOT=str(REPO), AF_OUTPUT_ROOT=str(root))
    with public._lock(root / "scheduler"):
        execution = sealed_write(root / "execution.json", identity)
        directory = root / "submissions" / wave
        manifest = directory / "manifest.json"
        submission_identity = {
            "execution_sha256": execution["artifact_sha256"],
            "wave": wave,
        }
        if manifest.exists():
            result = json.loads(manifest.read_text())
            if any(
                result.get(key) != value for key, value in submission_identity.items()
            ):
                raise ValueError("submission identity differs")
            return result
        if (directory / "propose.intent.json").exists():
            raise ValueError(
                f"inspect uncertain submission before retrying: {directory}"
            )
        directory.mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        options = [
            "--kill-on-invalid-dep=yes",
            "--nodes=1",
            "--ntasks=1",
            "--export=ALL",
            "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
            "--partition=gpu",
            "--gres=gpu:h100:1",
            "--cpus-per-task=8",
            "--mem=64G",
            "--time=02:00:00",
            "--signal=B:TERM@300",
            "--exclude=ac042",
            "--job-name=phasec-functions",
            f"--output={root}/logs/functions-%j.out",
            f"--error={root}/logs/functions-%j.err",
        ]
        job = submit_job(
            directory,
            "propose",
            options,
            REPO / "scripts/hpc/run_construction_functions_aces.sh",
            "propose",
            0,
        )
        result = {
            **submission_identity,
            "job": job,
            "automatic_followup": False,
            "maximum_cases": execution["maximum_cases"],
            "maximum_generation_requests": execution["maximum_generation_requests"],
            "maximum_charged_tokens": execution["maximum_charged_tokens"],
        }
        public._write(manifest, result)
        return result


def main() -> None:
    """Verify a frozen execution locally, or explicitly submit its bounded job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wave", default="functions-1")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = (
        verify(args.root.resolve())[1]
        if args.verify_only
        else submit(args.root.resolve(), args.wave)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
