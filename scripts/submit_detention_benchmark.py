#!/usr/bin/env python3
"""Submit one sealed, eight-arm CPU benchmark qualification on ACES."""

import argparse
import importlib.util
import os
from pathlib import Path

from autoformalism.benchmarks.detention import DetentionConfig
from autoformalism.fitting import public_fitting as public

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "detention_submission_support", ROOT / "scripts/submit_shared_process_pilot.py"
)
_support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_support)


def submit(output: Path) -> dict:
    """Never repeat uncertain scheduler submissions or silently change a source."""
    output = output.resolve()
    commit = _support.source_commit(ROOT)
    if os.environ.get("AF_COMMIT", commit) != commit:
        raise ValueError("pinned commit differs")
    python = Path(os.environ["AF_PYTHON"])
    if not python.is_file():
        raise ValueError("AF_PYTHON does not exist")
    config_path = ROOT / "configs/detention_benchmark_v1.json"
    config = DetentionConfig.model_validate_json(config_path.read_text())
    identity = {
        "commit": commit,
        "config_sha256": public.content_sha256(config),
        "repo_root": str(ROOT),
        "output_root": str(output),
        "python": str(python),
    }
    os.environ.update(
        AF_REPO_ROOT=str(ROOT),
        AF_OUTPUT_ROOT=str(output),
        AF_COMMIT=commit,
        PYTHONDONTWRITEBYTECODE="1",
    )
    with public._lock(output / "scheduler"):
        manifest = output / "submission_manifest.json"
        if manifest.exists():
            value = public._read(manifest)
            if value["identity"] != identity:
                raise ValueError("submission identity changed")
            return value
        intent = output / "submission-intent"
        try:
            intent.mkdir()
        except FileExistsError:
            raise ValueError(
                "partial/uncertain submission; inspect receipts and jobs"
            ) from None
        public._write(intent / "identity.json", identity)
        (output / "logs").mkdir(exist_ok=True)
        jobs = {}

        def queue(stage, extra):
            jobs[stage] = _support.submit_job(
                intent,
                stage,
                [
                    "--parsable",
                    "--partition=cpu",
                    "--nodes=1",
                    "--ntasks=1",
                    "--cpus-per-task=1",
                    "--export=ALL",
                    "--kill-on-invalid-dep=yes",
                    "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
                    f"--job-name=basins-{stage}",
                    f"--output={output}/logs/{stage}-%A_%a.out",
                    f"--error={output}/logs/{stage}-%A_%a.err",
                    *extra,
                ],
                ROOT / "scripts/hpc/run_detention_benchmark_aces.sh",
                stage,
                0,
            )
            return jobs[stage]

        gate = queue("prepare", ["--mem=8G", "--time=00:30:00"])
        fit = queue(
            "fit",
            [
                f"--dependency=afterok:{gate}",
                "--array=0-7%4",
                "--mem=16G",
                "--time=00:40:00",
            ],
        )
        queue("report", [f"--dependency=afterany:{fit}", "--mem=4G", "--time=00:10:00"])
        value = {
            "identity": identity,
            "jobs": jobs,
            "tasks": 8,
            "gpus": 0,
            "cpus_per_task": 1,
            "submission_complete": True,
            "automatic_followup": False,
            "test_data_access": False,
        }
        public._write(manifest, value)
        return value


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(submit(args.output), indent=2))
