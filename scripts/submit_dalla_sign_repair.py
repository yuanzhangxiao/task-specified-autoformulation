#!/usr/bin/env python3
"""Submit public-context sign review and paired rescue fits with durable receipts."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import dalla_sign_repair as io
from scripts.recover_review_continuation_submission import scheduler_record
from scripts.submit_shared_process_pilot import source_commit, submit_job


def delta_config(config: Path, root: Path, image: Path) -> Path:
    """Freeze the Delta placement and actual image without changing review budgets."""
    value = io.Config.model_validate(public._read(config)).model_dump(mode="json")
    with image.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    value.update(platform="delta-a40x1", serving_image_sha256=digest)
    target = root / "delta-config.json"
    with public._lock(root / "platform-setup"):
        if target.exists():
            if public._read(target) != value:
                raise ValueError("Delta configuration changed; use a fresh output root")
        else:
            public._write(target, value)
    return target


def submit(
    source: Path,
    root: Path,
    adopt: dict | None = None,
    config: Path | None = None,
    *,
    delta: bool = False,
) -> dict:
    """Prepare without fitting, then submit missing jobs without duplicating any."""
    source, root = source.resolve(), root.resolve()
    commit = source_commit(io.REPO)
    if os.environ.get("AF_COMMIT", commit) != commit:
        raise ValueError("pinned commit differs")
    if not Path(os.environ["AF_PYTHON"]).is_file():
        raise ValueError("AF_PYTHON is missing")
    config = config or io.REPO / "configs/dalla_sign_repair_v1.json"
    settings = io.Config.model_validate(public._read(config))
    delta = delta or settings.platform == "delta-a40x1"
    group = Path(
        "/work/hdd/bibo/yxiao2/phase_b"
        if delta
        else "/scratch/group/p.nairr260351.000/u.yx126462"
    )
    os.environ.setdefault(
        "AF_VLLM_IMAGE",
        "/projects/bibo/yxiao2/containers/vllm-openai-v0.27.1.sif"
        if delta
        else "/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif",
    )
    if delta:
        os.environ.setdefault("AF_HF_HOME", "/projects/bibo/yxiao2/huggingface-cache")
        Path(os.environ["AF_HF_HOME"]).mkdir(parents=True, exist_ok=True)
        config = delta_config(config, root, Path(os.environ["AF_VLLM_IMAGE"]))
        settings = io.Config.model_validate(public._read(config))
    revision = settings.model_settings.model_revision
    if not os.environ.get("AF_HF_HOME"):
        candidates = [
            group / "huggingface-cache",
            Path("/scratch/user/u.yx126462/huggingface-cache"),
        ]
        found = next(
            (
                p
                for p in candidates
                if (p / "hub/models--openai--gpt-oss-20b/snapshots" / revision).is_dir()
            ),
            None,
        )
        if found is None:
            raise ValueError(
                "set AF_HF_HOME to the cache containing the pinned model revision"
            )
        os.environ["AF_HF_HOME"] = str(found)
    os.environ.setdefault("AF_COMPUTE_CACHE_ROOT", str(group / "sign-repair-cache"))
    os.environ.setdefault("AF_IPC_TMP_ROOT", "/tmp/af-sign-" + str(os.getuid()))
    if (
        not Path(os.environ["AF_VLLM_IMAGE"]).is_file()
        or not Path(os.environ["AF_HF_HOME"]).is_dir()
    ):
        raise ValueError("serving image or Hugging Face cache is missing")
    plan = io.freeze(source, config, root)
    identity = {"commit": commit, "plan_sha256": plan["artifact_sha256"]}
    adopt = adopt or {}
    if set(adopt) - {"prepare", "review", "fit", "report"}:
        raise ValueError("unknown adoption stage")
    os.environ.update(
        AF_REPO_ROOT=str(io.REPO),
        AF_OUTPUT_ROOT=str(root),
        AF_INPUTS=str(source),
        AF_COMMIT=commit,
    )
    with public._lock(root / "scheduler"):
        manifest = root / "submission_manifest.json"
        if manifest.exists():
            result = public._read(manifest)
            if result["identity"] != identity:
                raise ValueError("submission identity differs")
            return result
        directory = root / "submission-intent"
        directory.mkdir(exist_ok=True)
        frozen = directory / "identity.json"
        if frozen.exists() and public._read(frozen) != identity:
            raise ValueError("submission intent differs")
        public._write(frozen, identity)
        (root / "logs").mkdir(exist_ok=True)
        worker = io.REPO / "scripts/hpc/run_dalla_sign_repair_aces.sh"

        def queue(stage, options):
            account = (
                (
                    os.environ.get("AF_GPU_ACCOUNT", "bibo-delta-gpu")
                    if stage == "review"
                    else os.environ.get("AF_CPU_ACCOUNT", "bibo-delta-cpu")
                )
                if delta
                else os.environ.get("AF_ACCOUNT", "156264627414")
            )
            opts = [
                "--kill-on-invalid-dep=yes",
                "--export=ALL",
                "--account=" + account,
                "--nodes=1",
                "--ntasks=1",
                f"--job-name=dalla-sign-{stage}",
                f"--output={root}/logs/{stage}-%A_%a.out",
                f"--error={root}/logs/{stage}-%A_%a.err",
                *(["--constraint=projects&work"] if delta else []),
                *options,
            ]
            argv = ["sbatch", "--parsable", *opts, str(worker), stage, "0"]
            intent, ident = (
                directory / f"{stage}.intent.json",
                directory / f"{stage}.id",
            )
            if intent.exists():
                if public._read(intent) != {"argv": argv}:
                    raise ValueError("saved scheduler command differs")
                if ident.exists():
                    job = ident.read_text().strip()
                    if not job.isdecimal() or (stage in adopt and adopt[stage] != job):
                        raise ValueError("confirmed job differs")
                    return job
                if stage not in adopt:
                    raise ValueError(
                        f"Unconfirmed {stage}; inspect receipts and scheduler. "
                        f"Use --adopt {stage}=JOB_ID for a confirmed matching job."
                    )
                job = adopt[stage]
                record = scheduler_record(job, argv, since=intent.stat().st_mtime)
                public._write(directory / f"{stage}.adoption.json", record)
                ident.write_text(job + "\n")
                return job
            if stage in adopt:
                raise ValueError("cannot adopt without saved intent")
            return submit_job(directory, stage, opts, worker, stage, 0)

        prepare = queue(
            "prepare",
            [
                "--partition=cpu",
                "--cpus-per-task=1",
                "--mem=8G",
                "--time=01:00:00" if delta else "--time=00:30:00",
            ],
        )
        review = queue(
            "review",
            [
                f"--dependency=afterok:{prepare}",
                "--partition=gpuA40x4" if delta else "--partition=gpu",
                "--gpus-per-node=1" if delta else "--gres=gpu:h100:1",
                "--cpus-per-task=8",
                "--mem=64G",
                "--time=01:15:00",
                "--signal=B:TERM@300",
            ],
        )
        fit = queue(
            "fit",
            [
                f"--dependency=afterok:{review}",
                "--partition=cpu",
                "--cpus-per-task=1",
                f"--array=0-{len(plan['rows']) - 1}%3",
                "--mem=16G",
                "--time=04:30:00",
            ],
        )
        report = queue(
            "report",
            [
                f"--dependency=afterany:{prepare}:{review}:{fit}",
                "--partition=cpu",
                "--cpus-per-task=1",
                "--mem=4G",
                "--time=00:15:00",
            ],
        )
        result = {
            "protocol": settings.protocol,
            "platform": settings.platform,
            "identity": identity,
            "jobs": {
                "prepare": prepare,
                "review": review,
                "fit": fit,
                "report": report,
            },
            "array_tasks": len(plan["rows"]),
            "review_gpus": 1,
            "fit_gpus": 0,
            "maximum_llm_calls": sum(
                bool(r["review_context"]["eligible_slots"]) for r in plan["rows"]
            )
            * min(
                settings.model_settings.maximum_requests,
                settings.model_settings.attempts_per_step
                * (2 if settings.protocol == "dalla-sign-repair-2" else 1),
            ),
            "test_data_opened": False,
        }
        public._write(manifest, result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Use one Delta A40 and pin the existing local SIF in a fresh plan",
    )
    parser.add_argument("--adopt", action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            submit(
                args.inputs,
                args.root,
                dict(x.split("=", 1) for x in args.adopt),
                args.config,
                delta=args.delta,
            ),
            indent=2,
        )
    )
