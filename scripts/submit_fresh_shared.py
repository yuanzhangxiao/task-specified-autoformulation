#!/usr/bin/env python3
"""Submit fresh shared search sequentially, then one bounded CPU pruning pass."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import review_deadline_io as io


def _support(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scheduler_record = _support("recover_review_continuation_submission").scheduler_record
_support_module = _support("submit_shared_process_pilot")
source_commit, submit_job = _support_module.source_commit, _support_module.submit_job


def submit(root: Path, index: int, adopt: dict | None = None) -> dict:
    """Resume confirmed submissions; verify uncertain jobs before adoption."""
    root = root.resolve()
    io.require_open(root)
    plan = io.verify(root)
    rounds = plan["config"]["rounds"]
    if plan["protocol"] != io.FRESH_PROTOCOL or index not in range(rounds):
        raise ValueError("requires a fresh shared campaign search round")
    repo = io.REPO
    commit = source_commit(repo)
    if os.environ.get("AF_COMMIT", commit) != commit:
        raise ValueError("pinned commit changed")
    for key in (
        "AF_PYTHON",
        "AF_VLLM_IMAGE",
        "AF_HF_HOME",
        "AF_COMPUTE_CACHE_ROOT",
        "AF_IPC_TMP_ROOT",
    ):
        if not os.environ.get(key):
            raise ValueError(f"set {key}")
    if not all(Path(os.environ[k]).is_file() for k in ("AF_PYTHON", "AF_VLLM_IMAGE")):
        raise ValueError("Python or serving image is missing")
    os.environ.update(
        AF_REPO_ROOT=str(repo), AF_OUTPUT_ROOT=str(root), AF_COMMIT=commit
    )
    adopt = adopt or {}
    with public._lock(root / "scheduler"):
        manifest = root / f"submission-round-{index}.json"
        if manifest.exists():
            value = public._read(manifest)
            if (
                value["commit"] != commit
                or value["plan_sha256"] != plan["artifact_sha256"]
            ):
                raise ValueError("submission identity differs")
            return value
        previous = {}
        if index:
            for task in plan["tasks"]:
                result = io.read_round(root, task, index - 1)
                if (
                    result is None
                    or result["task"] != task
                    or result["round"] != index - 1
                ):
                    raise ValueError(f"previous round missing: {task['task_id']}")
                if result.get("proposal_status") in {
                    "provider_request_failed",
                    "request_preflight_failed",
                }:
                    raise ValueError(
                        "previous round delivery failed; inspect before continuation"
                    )
                previous[task["task_id"]] = result["artifact_sha256"]
        directory = root / "submission-intent" / f"round-{index}"
        directory.mkdir(parents=True, exist_ok=True)
        prerequisites = {
            "plan": plan["artifact_sha256"],
            "commit": commit,
            "results": previous,
        }
        before = directory / "prerequisites.json"
        if before.exists() and public._read(before) != prerequisites:
            raise ValueError("submission prerequisites changed")
        public._write(before, prerequisites)
        (root / "logs").mkdir(exist_ok=True)
        worker = repo / "scripts/hpc/run_fresh_shared_aces.sh"
        seen = set()

        def queue(stage, at, options):
            key = f"{stage}-{at}"
            seen.add(key)
            opts = [
                "--kill-on-invalid-dep=yes",
                "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
                "--nodes=1",
                "--ntasks=1",
                "--export=ALL",
                f"--job-name=shared-multi-{key}",
                f"--output={root}/logs/{key}-%A_%a.out",
                f"--error={root}/logs/{key}-%A_%a.err",
                *options,
            ]
            argv = ["sbatch", "--parsable", *opts, str(worker), stage, str(at)]
            intent, ident = directory / f"{key}.intent.json", directory / f"{key}.id"
            if intent.exists():
                if public._read(intent) != {"argv": argv}:
                    raise ValueError(f"saved submission command differs for {key}")
                if ident.exists():
                    job = ident.read_text().strip()
                    if not job.isdecimal() or (key in adopt and adopt[key] != job):
                        raise ValueError("confirmed job identity differs")
                    return job
                if key not in adopt:
                    raise ValueError(
                        f"Unconfirmed {key}; inspect its receipt and scheduler. "
                        f"Resume with --adopt {key}=JOB_ID only for the matching job."
                    )
                job = adopt[key]
                record = scheduler_record(job, argv, since=intent.stat().st_mtime)
                public._write(directory / f"{key}.adoption.json", record)
                ident.write_text(job + "\n")
                return job
            if key in adopt:
                raise ValueError(f"cannot adopt {key} without saved intent")
            return submit_job(directory, key, opts, worker, stage, at)

        cpu = ["--partition=cpu", "--cpus-per-task=1"]
        gpu = [
            "--partition=gpu",
            "--gres=gpu:h100:1",
            "--cpus-per-task=8",
            "--mem=64G",
            "--time=06:30:00",
            "--signal=B:TERM@300",
        ]
        if index == 0:
            prep = queue("prepare", 0, [*cpu, "--mem=16G", "--time=01:00:00"])
            gpu.append(f"--dependency=afterok:{prep}")
        proposer = queue("propose", index, gpu)
        fits = queue(
            "fit",
            index,
            [
                *cpu,
                f"--dependency=afterok:{proposer}",
                f"--array=0-{len(plan['tasks']) - 1}%8",
                "--mem=16G",
                "--time=00:40:00",
                "--signal=B:TERM@120",
            ],
        )
        finish = queue(
            "finish",
            index,
            [*cpu, f"--dependency=afterany:{fits}", "--mem=4G", "--time=00:15:00"],
        )
        if index + 1 < rounds:
            queue(
                "dispatch",
                index + 1,
                [*cpu, f"--dependency=afterok:{finish}", "--mem=4G", "--time=00:15:00"],
            )
        else:
            prune = queue(
                "prune",
                index,
                [
                    *cpu,
                    f"--dependency=afterok:{finish}",
                    f"--array=0-{len(plan['tasks']) - 1}%8",
                    "--mem=16G",
                    "--time=01:15:00",
                ],
            )
            queue(
                "report",
                index,
                [*cpu, f"--dependency=afterany:{prune}", "--mem=4G", "--time=00:15:00"],
            )
        if set(adopt) - seen:
            raise ValueError("adoption key not in this submission round")
        value = {
            "protocol": plan["protocol"],
            "commit": commit,
            "plan_sha256": plan["artifact_sha256"],
            "round": index,
            "planned_rounds": rounds,
            "jobs": {
                p.stem: p.read_text().strip()
                for p in sorted((root / "submission-intent").glob("round-*/*.id"))
            },
            "all_rounds_submitted": index + 1 == rounds,
            "one_h100_per_proposer_job": True,
            "scientific_critic": "off",
            "automatic_test_access": False,
        }
        public._write(manifest, value)
        public._write(root / "submission_manifest.json", value)
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument(
        "--adopt", action="append", default=[], metavar="STAGE-ROUND=JOB_ID"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            submit(args.root, args.round, dict(x.split("=", 1) for x in args.adopt)),
            indent=2,
        )
    )
