#!/usr/bin/env python3
"""Submit one bounded training-evidence model revision on an ACES H100."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import construction_feedback_revision as campaign
from scripts import harmony_vocab
from scripts.saved_initialization_aces import file_hash
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "construction-feedback-revision-1"
EXECUTION_PROTOCOL = "construction-revision-aces-execution-1"
MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
WORKER_SECONDS = 5400
ENTRYPOINTS = (
    "scripts/construction_feedback_revision.py",
    "scripts/submit_construction_revision_aces.py",
    "scripts/hpc/start_construction_revision_aces.sh",
    "scripts/hpc/run_construction_revision_aces.sh",
    "scripts/hpc/run_construction_functions_aces.sh",
    "scripts/construction_functions.py",
    "scripts/submit_review_continuation.py",
    "scripts/saved_initialization_aces.py",
    "scripts/harmony_vocab.py",
)
SETTINGS = {
    "model": MODEL,
    "model_revision": REVISION,
    "maximum_requests": 3,
    "attempts_per_step": 3,
    "max_output_tokens": 8192,
    "maximum_total_tokens": 98304,
}


def verify_inputs(root: Path) -> dict:
    """Require the exact 24-episode bounded plan before a serving allocation."""
    plan = campaign.verify(root)
    if (
        plan["protocol"] not in {PROTOCOL, campaign.V2_PROTOCOL}
        or len(plan["cases"]) != 24
        or any(plan["settings"].get(k) != v for k, v in SETTINGS.items())
        or plan["optimizer_calls"] != 0
        or plan["automatic_followup"] is not False
    ):
        raise ValueError("requires all eight parents and three bounded revision seeds")
    if plan["protocol"] == campaign.V2_PROTOCOL and (
        plan.get("policies") != campaign.V2_POLICIES
        or not re.fullmatch(
            r"[0-9a-f]{64}", plan.get("consistency_evidence_sha256", "")
        )
    ):
        raise ValueError("v2 requires frozen formatting and consistency policies")
    return plan


def execution_identity(plan: dict, image_sha256: str) -> dict:
    """Freeze the serving assets, resources, source and revision-only worker."""
    count = len(plan["cases"])
    identity = {
        "protocol": EXECUTION_PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "input_bundle_sha256": plan["input_bundle_sha256"],
        "runtime_source_sha256": plan["runtime_source_sha256"],
        "case_ids": [case["case_id"] for case in plan["cases"]],
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
        "maximum_revision_cases": count,
        "maximum_generation_requests": count * SETTINGS["maximum_requests"],
        "maximum_charged_tokens": count * SETTINGS["maximum_total_tokens"],
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
    if plan["protocol"] == campaign.V2_PROTOCOL:
        identity.update(
            protocol="construction-revision-aces-execution-2",
            revision_protocol=plan["protocol"],
            acceptance_policies=plan["policies"],
            consistency_evidence_sha256=plan["consistency_evidence_sha256"],
        )
    return identity


def verify(root: Path) -> tuple[dict, dict]:
    """Reject execution drift before starting the server or requesting a formula."""
    plan = verify_inputs(root)
    execution = sealed_read(root / "execution.json")
    expected = execution_identity(plan, execution["image_sha256"])
    if execution != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("construction revision execution identity differs")
    return plan, execution


def _assets() -> Path:
    """Require local Python, image and the immutable offline model snapshot."""
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


def submit(root: Path, wave: str = "revision-1") -> dict:
    """Record one bounded job; never bypass existing intent through another wave."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", wave):
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
            result = public._read(manifest)
            if any(
                result.get(key) != value for key, value in submission_identity.items()
            ):
                raise ValueError("submission identity differs")
            return result
        previous = sorted((root / "submissions").glob("*/*"))
        if previous:
            raise ValueError(
                "inspect existing submission before retrying; another wave cannot "
                f"bypass the one-job bound: {root / 'submissions'}"
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
            "--job-name="
            + (
                "phasec-revision-checks"
                if plan["protocol"] == campaign.V2_PROTOCOL
                else "phasec-fit-feedback"
            ),
            f"--output={root}/logs/revision-%j.out",
            f"--error={root}/logs/revision-%j.err",
        ]
        job = submit_job(
            directory,
            "propose",
            options,
            REPO / "scripts/hpc/run_construction_revision_aces.sh",
            "propose",
            0,
        )
        result = {
            **submission_identity,
            "job": job,
            "automatic_followup": False,
            "maximum_cases": execution["maximum_cases"],
            "maximum_revision_cases": execution["maximum_revision_cases"],
            "maximum_generation_requests": execution["maximum_generation_requests"],
            "maximum_charged_tokens": execution["maximum_charged_tokens"],
        }
        public._write(manifest, result)
        return result


def main() -> None:
    """Verify locally, or explicitly submit the bounded revision-only job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wave", default="revision-1")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = (
        verify(args.root.resolve())[1]
        if args.verify_only
        else submit(args.root, args.wave)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
