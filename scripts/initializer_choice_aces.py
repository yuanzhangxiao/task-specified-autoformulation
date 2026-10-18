#!/usr/bin/env python3
"""Run a separately bounded initializer-choice diagnostic on one ACES H100."""

from __future__ import annotations

import argparse
import json
import os
import signal
import tarfile
import time
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import initializer_choice_diagnostic as diagnostic
from autoformalism.research import initializer_choice_inputs as inputs
from autoformalism.staged_topology import content_hash
from scripts import harmony_vocab
from scripts.saved_initialization_aces import bind_server, file_hash

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "saved-initializer-choice-diagnostic-1"
PLATFORM = "aces-h100x1"
MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
MAXIMUM_CALLS = 6
MAXIMUM_TOKENS = 196608
BUDGETS = {
    "maximum_choice_calls_per_case": 2,
    "maximum_choice_tokens_per_case": 65536,
    "completion_probe_calls": 2,
    "completion_probe_tokens_per_call": 32768,
    "maximum_calls": MAXIMUM_CALLS,
    "maximum_tokens": MAXIMUM_TOKENS,
}
ENTRYPOINTS = (
    "scripts/initializer_choice_aces.py",
    "scripts/submit_initializer_choice_aces.py",
    "scripts/submit_review_continuation.py",
    "scripts/hpc/run_initializer_choice_aces.sh",
    "scripts/hpc/start_initializer_choice_aces.sh",
    "scripts/saved_initialization_aces.py",
    "scripts/harmony_vocab.py",
)


def verify_inputs(root: Path) -> dict:
    """Require the explicit new diagnostic, pinned model and at most two cases."""
    plan = inputs.verify(root)
    settings = plan["parent_plan"]["settings"]
    if (
        plan["protocol"] != PROTOCOL
        or plan.get("execution_platform") != PLATFORM
        or not 1 <= len(plan["cases"]) <= 2
        or plan.get("budgets") != BUDGETS
        or settings["model"] != MODEL
        or settings["model_revision"] != REVISION
        or settings["max_output_tokens"] != 8192
        or plan.get("automatic_followup") is not False
        or plan.get("test_data_opened") is not False
    ):
        raise ValueError("requires the bounded ACES initializer-choice diagnostic")
    return plan


def execution_identity(plan: dict, image_sha256: str) -> dict:
    """Bind local serving and immutable helper code independently of older runs."""
    return {
        "protocol": "aces-initializer-choice-execution-1",
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
        "maximum_output_tokens": 8192,
        "tensor_parallel_size": 1,
        "maximum_sequences": 1,
        "gpu_memory_utilization": 0.90,
        "maximum_generation_requests": MAXIMUM_CALLS,
        "maximum_charged_tokens": MAXIMUM_TOKENS,
        "compiler_cache_policy": "node-local-job-temporary-1",
        "server_binding": "127.0.0.1",
        "test_data_opened": False,
        "automatic_followup": False,
        "model_promotion": False,
    }


def prepare(source: Path, root: Path, image: Path) -> dict:
    """Freeze new diagnostic inputs without borrowing another run's cache."""
    plan = verify_inputs(source)
    execution = execution_identity(plan, file_hash(image))
    if (
        any((root / name).exists() for name in ("results", "calls"))
        and not (root / "execution.json").exists()
    ):
        raise ValueError(
            "existing results lack diagnostic execution identity; use a fresh root"
        )
    sealed_write(
        root / "plan.json", {k: v for k, v in plan.items() if k != "artifact_sha256"}
    )
    return sealed_write(root / "execution.json", execution)


def verify(root: Path) -> tuple[dict, dict]:
    """Verify the frozen scientific inputs and unchanged execution configuration."""
    plan = verify_inputs(root)
    execution = sealed_read(root / "execution.json")
    expected = execution_identity(plan, execution["image_sha256"])
    if execution != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("initializer-choice execution identity differs")
    return plan, execution


def run(root: Path, base_url: str, runtime_record: Path, seconds: int = 3600) -> dict:
    """Let independent frozen units finish without retrying unavailable outcomes."""
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
        diagnostic.run(
            root,
            plan,
            base_url=base_url,
            can_start=lambda: not draining and time.monotonic() + 200 < deadline,
        )
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        summary = diagnostic.report(root, plan)
    return summary


def run_exit_code(summary: dict) -> int:
    """Scientific disagreement is a result; unavailable or pending output is not."""
    return int(
        summary.get("status") != "complete"
        or summary.get("availability_status") != "available"
        or summary.get("status_counts", {}).get("provider_unavailable", 0) > 0
    )


def inspect(root: Path) -> dict:
    """Package exact responses, reports and operational receipts for review."""
    plan, _ = verify(root)
    summary = diagnostic.report(root, plan)
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
            "calls",
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
        "status": summary.get("status"),
        "availability_status": summary.get("availability_status"),
        "status_counts": summary.get("status_counts"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "verify", "run", "report", "inspect")
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--runtime-record", type=Path)
    args = parser.parse_args()
    with _lock(args.root / "campaign-lock"):
        if args.command == "prepare":
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
                diagnostic.report(args.root, plan)
                if args.command == "report"
                else execution
            )
    print(json.dumps(result, indent=2))
    if args.command == "run":
        raise SystemExit(run_exit_code(result))


if __name__ == "__main__":
    main()
