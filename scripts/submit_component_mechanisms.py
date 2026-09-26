#!/usr/bin/env python3
"""Submit a saved-component-model assessment on ACES CPUs, with durable receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read

if __package__:
    from .submit_shared_process_pilot import source_commit, submit_job
else:
    from submit_shared_process_pilot import source_commit, submit_job

REPO = Path(__file__).resolve().parents[1]


def submit(source: Path, root: Path, round_index: int | None = None) -> dict:
    """One snapshot, one model array, one report; never advance the search."""
    source, root = source.resolve(), root.resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("source and assessment roots must be separate")
    old = sealed_read(source / "plan.json")
    if old["protocol"] != "final-component-campaign-1":
        raise ValueError("unsupported source campaign")
    if round_index is not None and not 0 <= round_index < old["config"]["rounds"]:
        raise ValueError("round outside campaign")
    commit = source_commit(REPO)
    python = os.environ["AF_PYTHON"]
    if not Path(python).is_file():
        raise ValueError("AF_PYTHON is missing")
    if os.environ.get("AF_COMMIT", commit) != commit:
        raise ValueError("commit differs")
    identity = {
        "commit": commit,
        "source": str(source),
        "root": str(root),
        "source_plan_sha256": old["artifact_sha256"],
        "round": round_index,
        "python": python,
        "account": os.environ.get("AF_ACCOUNT", "156264627414"),
    }
    os.environ.update(
        AF_REPO_ROOT=str(REPO),
        AF_SOURCE_ROOT=str(source),
        AF_OUTPUT_ROOT=str(root),
        AF_COMMIT=commit,
        AF_SNAPSHOT_ROUND="" if round_index is None else str(round_index),
        PYTHONDONTWRITEBYTECODE="1",
    )
    with public._lock(root / "scheduler"):
        manifest = root / "submission.json"
        if manifest.exists():
            saved = public._read(manifest)
            if saved["identity"] != identity:
                raise ValueError("submission identity differs")
            return saved
        intent = root / "submission-intent"
        if intent.exists():
            raise ValueError(
                "partial submission: inspect saved receipts before retrying"
            )
        intent.mkdir()
        public._write(intent / "identity.json", identity)
        (root / "logs").mkdir(exist_ok=True)
        jobs = {}

        def queue(stage, extra):
            jobs[stage] = submit_job(
                intent,
                stage,
                [
                    "--kill-on-invalid-dep=yes",
                    "--account=" + identity["account"],
                    "--partition=cpu",
                    "--nodes=1",
                    "--ntasks=1",
                    "--cpus-per-task=1",
                    "--export=ALL",
                    f"--job-name=ours-mechanism-{stage}",
                    f"--output={root}/logs/{stage}-%A_%a.out",
                    f"--error={root}/logs/{stage}-%A_%a.err",
                    *extra,
                ],
                REPO / "scripts/hpc/run_component_mechanisms_aces.sh",
                stage,
                0,
            )
            return jobs[stage]

        prepare = queue("prepare", ["--mem=16G", "--time=00:30:00"])
        array = queue(
            "assess",
            [
                f"--dependency=afterok:{prepare}",
                f"--array=0-{len(old['tasks']) - 1}%8",
                "--mem=8G",
                "--time=03:00:00",
            ],
        )
        queue(
            "report", [f"--dependency=afterany:{array}", "--mem=8G", "--time=00:15:00"]
        )
        value = {
            "identity": identity,
            "jobs": jobs,
            "models": len(old["tasks"]),
            "llm_calls": 0,
            "optimizer_calls": 0,
            "gpus": 0,
            "test_data_opened": False,
            "automatic_followup": False,
        }
        public._write(manifest, value)
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--round", type=int, help="Default: latest retained at snapshot"
    )
    args = parser.parse_args()
    print(json.dumps(submit(args.source, args.root, args.round), indent=2))
