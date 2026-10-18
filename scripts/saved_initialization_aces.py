#!/usr/bin/env python3
"""Run a sealed, at-most-three-case initializer retest on an ACES local server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import tarfile
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import saved_topology_repair as campaign
from autoformalism.staged_topology import content_hash
from scripts import harmony_vocab

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "saved-topology-initialization-3"
EQUALITY_PROTOCOL = "saved-topology-initialization-4"
EVIDENCE_PROTOCOL = "saved-topology-initialization-5"
BOOKKEEPING_PROTOCOL = "saved-topology-initialization-6"
PLATFORM = "aces-h100x1"
MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
ENTRYPOINTS = (
    "scripts/saved_initialization_aces.py",
    "scripts/submit_saved_initialization_aces.py",
    "scripts/hpc/run_saved_initialization_aces.sh",
    "scripts/hpc/start_saved_initialization_aces.sh",
    "scripts/harmony_vocab.py",
)


def file_hash(path: Path) -> str:
    """Hash an image or entrypoint without loading a large container into RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inputs(root: Path) -> dict:
    """Reject a different platform, model, protocol, or expanded task roster."""
    plan = campaign.verify(root)
    settings = plan["settings"]
    if (
        plan["protocol"]
        not in {PROTOCOL, EQUALITY_PROTOCOL, EVIDENCE_PROTOCOL, BOOKKEEPING_PROTOCOL}
        or plan.get("execution_platform") != PLATFORM
        or not 1 <= len(plan["tasks"]) <= (3 if plan["protocol"] == PROTOCOL else 2)
        or len({task["task_id"] for task in plan["tasks"]}) != len(plan["tasks"])
        or settings["model"] != MODEL
        or settings["model_revision"] != REVISION
        or settings["maximum_requests"] != 10
        or settings["maximum_total_tokens"] != 262144
        or plan.get("automatic_followup") is not False
    ):
        raise ValueError("requires a bounded ACES initializer-v3/v4/v5/v6 retest")
    if plan.get("provider_recovery_policy"):
        from autoformalism.research.initialization_provider_recovery import (
            validate_plan,
        )

        validate_plan(plan)
    return plan


def execution_identity(plan: dict, image_sha256: str) -> dict:
    """Keep ACES caches separate and pin every serving-relevant launcher option."""
    if (
        plan.get("provider_recovery_policy")
        and image_sha256 != plan["recovery_source"]["execution"]["image_sha256"]
    ):
        raise ValueError("recovery requires the original serving image")
    return {
        "protocol": "aces-saved-initialization-execution-2",
        "plan_sha256": plan["artifact_sha256"],
        "platform": PLATFORM,
        "model": MODEL,
        "model_revision": REVISION,
        "image_sha256": image_sha256,
        "harmony_vocabulary": harmony_vocab.verify(
            Path(os.environ.get("AF_HARMONY_ASSETS", REPO / "assets/harmony"))
        ),
        "entrypoint_sha256": {name: file_hash(REPO / name) for name in ENTRYPOINTS},
        "served_context_tokens": 32768,
        "tensor_parallel_size": 1,
        "maximum_sequences": 1,
        "gpu_memory_utilization": 0.90,
        "compiler_cache_policy": "node-local-job-temporary-1",
        "server_binding": "127.0.0.1",
        "test_data_opened": False,
        "automatic_followup": False,
    }


def prepare(inputs: Path, root: Path, image: Path) -> dict:
    """Freeze the imported plan and backend identity without a generation call."""
    plan = verify_inputs(inputs)
    execution = execution_identity(plan, file_hash(image))
    if (root / "results").exists() and not (root / "execution.json").exists():
        raise ValueError(
            "existing results lack ACES execution identity; use a fresh root"
        )
    sealed_write(
        root / "plan.json", {k: v for k, v in plan.items() if k != "artifact_sha256"}
    )
    return sealed_write(root / "execution.json", execution)


def verify(root: Path) -> tuple[dict, dict]:
    """Verify source, frozen scientific inputs and immutable serving configuration."""
    plan = verify_inputs(root)
    execution = sealed_read(root / "execution.json")
    expected = execution_identity(plan, execution["image_sha256"])
    if execution != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("ACES execution identity differs")
    return plan, execution


def bind_server(root: Path, runtime_record: Path, base_url: str) -> None:
    """Require one tested H100 and a matching model before any generation request."""
    url = urlparse(base_url)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or url.username is not None
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise ValueError("server must be an unauthenticated loopback HTTP endpoint")
    runtime = json.loads(runtime_record.read_text())
    if (
        runtime.get("visible_gpus") != 1
        or not runtime.get("cuda_kernel_passed")
        or "H100" not in runtime.get("gpu_name", "")
        or runtime.get("vllm") != "0.27.1"
    ):
        raise ValueError("runtime needs one working H100 and vLLM 0.27.1")
    harmony = runtime.get("harmony", {})
    if (
        harmony.get("protocol") != harmony_vocab.PROTOCOL
        or harmony.get("filename") != harmony_vocab.VOCAB_FILENAME
        or harmony.get("sha256") != harmony_vocab.VOCAB_SHA256
        or harmony.get("harmony_render_passed") is not True
        or not isinstance(harmony.get("openai_harmony"), str)
        or not harmony["openai_harmony"]
        or type(harmony.get("rendered_tokens")) is not int
        or harmony["rendered_tokens"] <= 0
        or harmony.get("generation_requests") != 0
    ):
        raise ValueError("runtime requires the pinned offline Harmony render preflight")
    with urlopen(base_url.rstrip("/") + "/v1/models", timeout=15) as response:
        served = json.load(response)
    if MODEL not in [row.get("id") for row in served.get("data", [])]:
        raise ValueError("local server model differs")
    sealed_write(root / "serving_runtime.json", runtime)


def run(root: Path, base_url: str, runtime_record: Path, seconds: int = 9000) -> dict:
    """Continue cached episodes with their original caps; signals drain safely."""
    plan, _ = verify(root)
    bind_server(root, runtime_record, base_url)
    draining = False
    deadline = time.monotonic() + seconds

    def stop(signum, frame):
        nonlocal draining
        draining = True

    handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for task in plan["tasks"]:
            try:
                outcome = campaign.run_episode(
                    root,
                    plan,
                    task,
                    base_url=base_url,
                    can_start=lambda: not draining
                    and time.monotonic() + 200 < deadline,
                )
            except DeferredCall:
                break
            campaign.report(root, plan)
            if outcome and outcome.get("status") == "provider_unavailable":
                break
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        summary = campaign.report(root, plan)
    return summary


def inspect(root: Path) -> dict:
    """Package reports, exact calls and launch records, without loading test data."""
    plan, _ = verify(root)
    summary = campaign.report(root, plan)
    archive = root / "inspection.tar.gz"
    included = 0
    with tarfile.open(archive, "w:gz") as output:
        for name in (
            "plan.json",
            "execution.json",
            "serving_runtime.json",
            "summary.json",
            "SUMMARY.md",
            "REPORT.html",
            "results",
            "runtime",
            "logs",
            "submissions",
        ):
            path = root / name
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            for item in paths:
                if item.is_symlink():
                    raise ValueError("inspection does not follow symlinks")
                if item.is_file():
                    output.add(
                        item, arcname=str(item.relative_to(root)), recursive=False
                    )
                    included += 1
    return {
        "archive": str(archive),
        "files": included,
        "groups": summary.get("groups"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "freeze-recovery",
            "freeze-equality",
            "freeze-evidence",
            "freeze-bookkeeping",
            "verify",
            "run",
            "report",
            "inspect",
        ),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--image", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--runtime-record", type=Path)
    args = parser.parse_args()
    with _lock(args.root / "campaign-lock"):
        if args.command == "freeze-bookkeeping":
            if args.source is None or not args.task_id:
                parser.error("freeze-bookkeeping requires --source and --task-id")
            from autoformalism.research.initialization_bookkeeping_retest import freeze

            result = freeze(args.source, args.root, args.task_id)
        elif args.command == "freeze-evidence":
            if args.source is None or not args.task_id:
                parser.error("freeze-evidence requires --source and --task-id")
            from autoformalism.research.initialization_evidence_retest import freeze

            result = freeze(args.source, args.root, args.task_id)
        elif args.command == "freeze-equality":
            if args.source is None or not args.task_id:
                parser.error("freeze-equality requires --source and --task-id")
            from autoformalism.research.initialization_equality_retest import freeze

            result = freeze(args.source, args.root, args.task_id)
        elif args.command == "freeze-recovery":
            if args.source is None:
                parser.error("freeze-recovery requires --source")
            from autoformalism.research.initialization_provider_recovery import freeze

            result = freeze(args.source, args.root)
        elif args.command == "prepare":
            if args.inputs is None or args.image is None:
                parser.error("prepare requires --inputs and --image")
            result = prepare(args.inputs, args.root, args.image)
        elif args.command == "run":
            if args.base_url is None or args.runtime_record is None:
                parser.error("run requires --base-url and --runtime-record")
            result = run(args.root, args.base_url, args.runtime_record)
        elif args.command == "inspect":
            result = inspect(args.root)
        else:
            plan, execution = verify(args.root)
            result = (
                campaign.report(args.root, plan)
                if args.command == "report"
                else execution
            )
    print(json.dumps(result, indent=2))
    if args.command == "run" and any(
        row.get("status") == "provider_unavailable" for row in result.get("rows", [])
    ):
        parser.exit(
            1, "Provider unavailable; saved report retained, no automatic retry.\n"
        )


if __name__ == "__main__":
    main()
