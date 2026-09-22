#!/usr/bin/env python3
"""Submit a CPU-only saved-repair gate, fit array and report with durable receipts."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import basin_saved_fit as io
from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.submit_basin_repair_pilot import source_commit, submit_job


def submit(source: Path, gate: Path, root: Path) -> dict:
    source, gate, root = (p.resolve() for p in (source, gate, root))
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in (source, gate)):
        raise ValueError("output must be separate")
    commit = source_commit(io.REPO)
    if os.environ.get("AF_COMMIT", commit) != commit:
        raise ValueError("pinned commit differs")
    if not Path(os.environ["AF_PYTHON"]).is_file():
        raise ValueError("AF_PYTHON is missing")
    old = sealed_read(source / "plan.json")
    prior = sealed_read(gate / "summary.json")
    identity = {
        "commit": commit,
        "source": str(source),
        "gate": str(gate),
        "root": str(root),
        "source_plan_sha256": old["artifact_sha256"],
        "gate_sha256": prior["artifact_sha256"],
    }
    os.environ.update(
        AF_REPO_ROOT=str(io.REPO),
        AF_SOURCE_ROOT=str(source),
        AF_GATE_ROOT=str(gate),
        AF_OUTPUT_ROOT=str(root),
        AF_COMMIT=commit,
        PYTHONDONTWRITEBYTECODE="1",
    )
    with public._lock(root / "scheduler"):
        manifest = root / "submission_manifest.json"
        if manifest.exists():
            value = public._read(manifest)
            if value["identity"] != identity:
                raise ValueError("submission identity differs")
            return value
        directory = root / "submission-intent"
        try:
            directory.mkdir()
        except FileExistsError:
            raise ValueError(
                "partial/uncertain submission; "
                "inspect receipts and jobs before recovery"
            ) from None
        public._write(directory / "identity.json", identity)
        (root / "logs").mkdir(exist_ok=True)
        jobs = {}

        def queue(stage, options):
            jobs[stage] = submit_job(
                directory,
                stage,
                [
                    "--kill-on-invalid-dep=yes",
                    "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
                    "--partition=cpu",
                    "--nodes=1",
                    "--ntasks=1",
                    "--cpus-per-task=1",
                    "--export=ALL",
                    f"--job-name=basin-saved-{stage}",
                    f"--output={root}/logs/{stage}-%A_%a.out",
                    f"--error={root}/logs/{stage}-%A_%a.err",
                    *options,
                ],
                io.REPO / "scripts/hpc/run_basin_saved_fit_aces.sh",
                stage,
                0,
            )
            return jobs[stage]

        prepare = queue("prepare", ["--mem=8G", "--time=00:30:00"])
        fit = queue(
            "fit",
            [
                f"--dependency=afterok:{prepare}",
                f"--array=0-{len(old['tasks']) - 1}%4",
                "--mem=16G",
                "--time=00:40:00",
            ],
        )
        queue("report", [f"--dependency=afterany:{fit}", "--mem=4G", "--time=00:15:00"])
        value = {
            "protocol": io.PROTOCOL,
            "identity": identity,
            "jobs": jobs,
            "array_tasks": len(old["tasks"]),
            "llm_calls": 0,
            "gpus": 0,
            "automatic_followup": False,
        }
        public._write(manifest, value)
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(submit(args.source, args.gate, args.root), indent=2))
