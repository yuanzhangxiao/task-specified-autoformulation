#!/usr/bin/env python3
"""Submit only pruning and serialized final reviews against a frozen checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def load_submitter(repo: Path):
    """Import the original trusted runtime, never copy it into the new checkout."""
    if "autoformalism" in sys.modules:
        raise ValueError("use a fresh Python process for the frozen runtime")
    sys.path.insert(0, str(repo / "src"))
    path = repo / "scripts/submit_component_campaign.py"
    spec = importlib.util.spec_from_file_location("frozen_submitter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.io.REPO.resolve() != repo.resolve():
        raise ValueError("imported a different runtime")
    return module


def submit(runtime, root: Path, wave: str, pruning: int = 4) -> dict:
    """Keep original worker budgets; uncertain scheduler replies require inspection."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", wave) or not 1 <= pruning <= 8:
        raise ValueError("require a simple wave and one to eight pruning workers")
    plan = runtime.campaign.verify(root)
    runtime.io.require_open(root)
    runtime.checked_authorization(root, plan)
    if plan["config"]["platform"] != "aces-h100x1":
        raise ValueError("this closeout is for the original ACES campaign")
    last = plan["config"]["rounds"] - 1
    if any(runtime.io.read_round(root, task, last) is None for task in plan["tasks"]):
        raise ValueError("search is incomplete; closeout cannot replace search workers")
    if not os.environ.get("AF_JETSTREAM_API_KEY"):
        raise ValueError("set AF_JETSTREAM_API_KEY without putting it in arguments")
    source = runtime.support("submit_shared_process_pilot")
    commit = source.source_commit(runtime.io.REPO)
    if os.environ.get("AF_COMMIT") != commit:
        raise ValueError("AF_COMMIT must identify the original runtime")
    if not Path(os.environ.get("AF_PYTHON", "")).is_file():
        raise ValueError("set AF_PYTHON to the original working interpreter")
    os.environ.update(
        AF_REPO_ROOT=str(runtime.io.REPO),
        AF_OUTPUT_ROOT=str(root),
        AF_SITE="aces",
        AF_WORKER_SECONDS=str(plan["config"]["wall_seconds"]),
    )
    signature = {
        "plan_sha256": plan["artifact_sha256"],
        "commit": commit,
        "wave": wave,
        "workers": {"prune": pruning, "critic": 1},
        "excluded_node": "ac042",
    }
    directory = root / "submissions" / wave
    with runtime.public._lock(directory):
        identity = directory / "identity.json"
        if identity.exists() and runtime.public._read(identity) != signature:
            raise ValueError("allocation identity differs")
        manifest = directory / "manifest.json"
        if manifest.exists():
            value = runtime.public._read(manifest)
            if any(value.get(k) != v for k, v in signature.items()):
                raise ValueError("manifest differs")
            return value
        # A partial wave may have its own queued jobs; never issue new requests
        # until that ambiguity has been inspected by the operator.
        queue = subprocess.check_output(
            ["squeue", "--noheader", "--user", os.environ["USER"], "--format=%j"],
            text=True,
        )
        if any(n.startswith("component-") for n in queue.splitlines()):
            raise ValueError(
                "component jobs still queued/running; inspect before closeout"
            )
        runtime.public._write(identity, signature)
        (root / "logs").mkdir(exist_ok=True)
        worker = runtime.io.REPO / "scripts/hpc/run_component_campaign.sh"
        jobs = {}
        for stage, count in signature["workers"].items():
            options = [
                "--kill-on-invalid-dep=yes",
                "--account=" + runtime.account("aces", stage),
                "--partition=" + runtime.partition("aces", False),
                "--nodes=1",
                "--ntasks=1",
                "--export=ALL",
                "--cpus-per-task=1",
                "--exclude=ac042",
                "--mem=16G" if stage == "prune" else "--mem=4G",
                "--time=06:30:00",
                "--signal=B:TERM@300",
                f"--array=0-{count - 1}",
                f"--job-name=component-{stage}",
                f"--output={root}/logs/{wave}-{stage}-%A_%a.out",
                f"--error={root}/logs/{wave}-{stage}-%A_%a.err",
            ]
            intent, job = directory / f"{stage}.intent.json", directory / f"{stage}.id"
            expected = {
                "argv": ["sbatch", "--parsable", *options, str(worker), stage, "0"]
            }
            if intent.exists():
                if runtime.public._read(intent) != expected or not job.exists():
                    raise ValueError(
                        f"uncertain/different submission; inspect {intent}"
                    )
                jobs[stage] = job.read_text().strip()
                if not jobs[stage].isdecimal():
                    raise ValueError("invalid saved job ID")
            else:
                jobs[stage] = source.submit_job(
                    directory, stage, options, worker, stage, 0
                )
        value = {
            **signature,
            "jobs": jobs,
            "automatic_test_access": False,
            "automatic_resubmission": False,
        }
        runtime.public._write(manifest, value)
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--wave", default="endpoint-closeout-1")
    parser.add_argument("--pruning", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            submit(
                load_submitter(args.repo.resolve()),
                args.root.resolve(),
                args.wave,
                args.pruning,
            ),
            indent=2,
        )
    )
