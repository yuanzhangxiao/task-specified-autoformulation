#!/usr/bin/env python3
"""Submit the eight frozen construction models to a bounded ACES CPU array."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "construction-fit-aces-execution-1"
ENTRYPOINTS = (
    "scripts/submit_construction_fit_aces.py",
    "scripts/qualify_construction_fitting.py",
    "scripts/hpc/run_construction_fit_aces.sh",
    "scripts/hpc/start_construction_fit_aces.sh",
)


def file_hash(path: Path) -> str:
    """Hash a runtime or entrypoint without loading large files into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def campaign_plan(root: Path) -> dict:
    """Verify immutable scientific inputs before any scheduler action."""
    from autoformalism.research import construction_fit_qualification as campaign

    plan, _ = campaign.verify(root)
    if len(plan.get("tasks", [])) != 8:
        raise ValueError("construction fitting qualification requires eight tasks")
    return plan


def runtime_identity() -> dict:
    """Pin the actual scientific Python and any explicitly selected CPU image."""
    mode = os.environ.get("AF_FIT_RUNTIME", "native")
    if mode not in {"native", "container"}:
        raise ValueError("AF_FIT_RUNTIME must be native or container")
    runtime = {
        "mode": mode,
        "python": sys.executable,
        "python_sha256": file_hash(Path(sys.executable)),
        "python_version": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "sympy", "pydantic")
        },
    }
    if mode == "container":
        image = Path(os.environ["AF_CPU_IMAGE"]).resolve()
        runtime.update(image=str(image), image_sha256=file_hash(image))
    else:
        python = os.environ.get("AF_PYTHON")
        if not python or Path(python).resolve() != Path(sys.executable).resolve():
            raise ValueError("submit using the selected AF_PYTHON runtime")
        runtime["selected_python"] = python
    return runtime


def execution_identity(root: Path) -> dict:
    """Bind plan, runtime and fixed CPU resource requests independently of Git."""
    plan = campaign_plan(root)
    return {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "root": str(root),
        "runtime": runtime_identity(),
        "entrypoint_sha256": {name: file_hash(REPO / name) for name in ENTRYPOINTS},
        "account": os.environ.get("AF_ACCOUNT", "156264627414"),
        "partition": "cpu",
        "array": "0-7%4",
        "cpus_per_task": 1,
        "memory": "8G",
        "wall_time": "01:00:00",
        "exclude": "ac042",
        "gpus": 0,
        "llm_calls": 0,
        "test_data_opened": False,
        "automatic_followup": False,
    }


def verify_execution(root: Path) -> dict:
    """A compute worker may not silently switch code, inputs or Python runtime."""
    root = root.resolve()
    expected = execution_identity(root)
    saved = sealed_read(root / "launcher_execution.json")
    if saved != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("construction fitting launcher identity differs")
    return saved


def confirmed_job(directory: Path, key: str, argv: list[str]) -> str | None:
    """Reuse a confirmed receipt; never retry an uncertain scheduler submission."""
    identifier = directory / f"{key}.id"
    intent = directory / f"{key}.intent.json"
    receipt = directory / f"{key}.reply.json"
    if identifier.exists():
        job = identifier.read_text().strip()
        reply = json.loads(receipt.read_text()) if receipt.exists() else {}
        if (
            not intent.exists()
            or json.loads(intent.read_text()).get("argv") != argv
            or not job.isdecimal()
            or reply.get("returncode") != 0
            or reply.get("stdout", "").strip().split(";")[0] != job
        ):
            raise ValueError(f"unconfirmed {key} receipt; inspect {directory}")
        return job
    if intent.exists() or receipt.exists():
        raise ValueError(f"uncertain {key} submission; inspect {directory}")
    return None


def submit(root: Path) -> dict:
    """One eight-task array and one afterany report, with durable submission intent."""
    root = root.resolve()
    execution = execution_identity(root)
    os.environ.update(AF_REPO_ROOT=str(REPO), AF_OUTPUT_ROOT=str(root))
    directory = root / "submissions" / "cpu-fit-1"
    with public._lock(directory):
        saved_execution = sealed_write(root / "launcher_execution.json", execution)
        identity = {"execution_sha256": saved_execution["artifact_sha256"]}
        sealed_write(directory / "identity.json", identity)
        manifest = directory / "manifest.json"
        if manifest.exists():
            saved = sealed_read(manifest)
            if saved.get("execution_sha256") != identity["execution_sha256"]:
                raise ValueError("construction fitting submission identity differs")
            return saved
        (root / "logs").mkdir(exist_ok=True)
        base = [
            "--kill-on-invalid-dep=yes",
            "--nodes=1",
            "--ntasks=1",
            "--export=ALL",
            f"--account={execution['account']}",
            "--partition=cpu",
            "--cpus-per-task=1",
            "--mem=8G",
            "--exclude=ac042",
        ]
        worker = REPO / "scripts/hpc/run_construction_fit_aces.sh"
        jobs = {}
        for key, extra, stage in (
            (
                "fit",
                ["--array=0-7%4", "--time=01:00:00"],
                "run",
            ),
            ("report", ["--time=00:15:00"], "report"),
        ):
            options = [
                *base,
                f"--job-name=construction-fit-{key}",
                f"--output={root}/logs/{key}-%A_%a.out",
                f"--error={root}/logs/{key}-%A_%a.err",
                *extra,
                *([f"--dependency=afterany:{jobs['fit']}"] if key == "report" else []),
            ]
            argv = ["sbatch", "--parsable", *options, str(worker), stage, "0"]
            job = confirmed_job(directory, key, argv)
            if job is None:
                job = submit_job(directory, key, options, worker, stage, 0)
            jobs[key] = job
        return sealed_write(
            manifest,
            {
                **identity,
                "jobs": jobs,
                "tasks": 8,
                "maximum_parallel_tasks": 4,
                "gpus": 0,
                "llm_calls": 0,
                "test_data_opened": False,
                "automatic_followup": False,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", choices=("submit", "verify"), default="submit"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = (submit if args.command == "submit" else verify_execution)(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
