#!/usr/bin/env python3
"""Submit persistent ACES services once per named allocation wave."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import component_campaign as campaign
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal.component_critic import checked_authorization


def support(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def account(site, stage):
    key = "AF_GPU_ACCOUNT" if stage == "propose" else "AF_CPU_ACCOUNT"
    default = (
        "156264627414"
        if site == "aces"
        else ("bibo-delta-gpu" if stage == "propose" else "bibo-delta-cpu")
        if site == "delta"
        else None
    )
    value = os.environ.get(key) or default
    if not value:
        raise ValueError("set " + key)
    return value


def partition(site, gpu):
    key = "AF_GPU_PARTITION" if gpu else "AF_CPU_PARTITION"
    default = (
        ("gpu" if gpu else "cpu")
        if site == "aces"
        else ("gpuA40x4" if gpu else "cpu")
        if site == "delta"
        else None
    )
    value = os.environ.get(key) or default
    if not value:
        raise ValueError("set " + key)
    return value


def gpu_resource(site):
    value = os.environ.get("AF_GPU_GRES") or (
        "gpu:h100:1" if site == "aces" else "gpu:1" if site == "delta" else None
    )
    if not value:
        raise ValueError("set AF_GPU_GRES for one H100 on Koa")
    return value


def submit(root, wave, proposers=8, critics=4, fits=32, pruning=8):
    """Repeat calls reuse job IDs; ambiguous scheduler replies never auto-resubmit."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", wave):
        raise ValueError("wave must be a simple identifier")
    if any(n not in range(1, 129) for n in (proposers, critics, fits, pruning)):
        raise ValueError("worker counts must be between 1 and 128")
    plan = campaign.verify(root)
    io.require_open(root)
    checked_authorization(root, plan)
    for key in (
        "AF_PYTHON",
        "AF_VLLM_IMAGE",
        "AF_HF_HOME",
        "AF_COMPUTE_CACHE_ROOT",
        "AF_IPC_TMP_ROOT",
        "AF_JETSTREAM_API_KEY",
    ):
        if not os.environ.get(key):
            raise ValueError(f"set {key}")
    site = plan["config"]["platform"].split("-")[0]
    source = support("submit_shared_process_pilot")
    commit = source.source_commit(io.REPO)
    if os.environ.get("AF_COMMIT", commit) != commit:
        raise ValueError("commit differs")
    os.environ.update(
        AF_REPO_ROOT=str(io.REPO),
        AF_OUTPUT_ROOT=str(root),
        AF_COMMIT=commit,
        AF_SITE=site,
        AF_WORKER_SECONDS=str(plan["config"]["wall_seconds"]),
    )
    signature = {
        "plan_sha256": plan["artifact_sha256"],
        "commit": commit,
        "wave": wave,
        "workers": {
            "propose": proposers,
            "critic": critics,
            "fit": fits,
            "prune": pruning,
        },
    }
    directory = root / "submissions" / wave
    worker = io.REPO / "scripts/hpc/run_component_campaign.sh"
    with public._lock(directory):
        identity = directory / "identity.json"
        if identity.exists() and public._read(identity) != signature:
            raise ValueError(
                "allocation wave differs; use a new wave for new resources"
            )
        public._write(identity, signature)
        (root / "logs").mkdir(exist_ok=True)
        jobs = {}
        for stage, count in signature["workers"].items():
            opts = [
                "--kill-on-invalid-dep=yes",
                "--account=" + account(site, stage),
                "--nodes=1",
                "--ntasks=1",
                "--export=ALL",
                f"--array=0-{count - 1}",
                f"--job-name=component-{stage}",
                f"--output={root}/logs/{wave}-{stage}-%A_%a.out",
                f"--error={root}/logs/{wave}-{stage}-%A_%a.err",
                "--time=06:30:00",
                "--signal=B:TERM@300",
            ]
            if stage == "propose":
                opts += [
                    "--partition=" + partition(site, True),
                    "--gres=" + gpu_resource(site),
                    "--cpus-per-task=8",
                    "--mem=64G",
                ]
            else:
                opts += [
                    "--partition=" + partition(site, False),
                    "--cpus-per-task=1",
                    "--mem=16G" if stage in {"fit", "prune"} else "--mem=4G",
                ]
            # Store no environment values or credentials in manifests.
            intent = directory / f"{stage}.intent.json"
            ident = directory / f"{stage}.id"
            expected = {
                "argv": ["sbatch", "--parsable", *opts, str(worker), stage, "0"]
            }
            if intent.exists():
                if public._read(intent) != expected:
                    raise ValueError("submission command differs for " + stage)
                if not ident.exists() or not ident.read_text().strip().isdecimal():
                    raise ValueError(
                        "uncertain scheduler reply; inspect " + str(intent)
                    )
                jobs[stage] = ident.read_text().strip()
            else:
                jobs[stage] = source.submit_job(
                    directory, stage, opts, worker, stage, 0
                )
        value = {
            **signature,
            "jobs": jobs,
            "automatic_test_access": False,
            "automatic_resubmission": False,
        }
        public._write(directory / "manifest.json", value)
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wave", default="wave1")
    parser.add_argument("--proposers", type=int, default=8)
    parser.add_argument("--critics", type=int, default=4)
    parser.add_argument("--fits", type=int, default=32)
    parser.add_argument("--pruning", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            submit(
                args.root.resolve(),
                args.wave,
                args.proposers,
                args.critics,
                args.fits,
                args.pruning,
            ),
            indent=2,
        )
    )
