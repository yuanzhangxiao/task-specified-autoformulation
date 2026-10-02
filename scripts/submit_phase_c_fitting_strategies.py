#!/usr/bin/env python3
"""Submit the fitting-strategy comparison to Delta CPUs."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import transcription_campaign as q

REPO = Path(__file__).resolve().parents[1]


def submit(
    root: Path,
    config: Path,
    *,
    account: str,
    concurrency: int,
    inputs: Path | None = None,
    matched_source: Path | None = None,
) -> dict:
    """Persist each intent/receipt; uncertain submissions cannot be repeated."""
    root = root.resolve()
    q.prepare(
        root,
        q.CampaignConfig.model_validate_json(config.read_text()),
        inputs,
        matched_source,
    )
    plan, _ = q.verify(root)
    python = os.environ["AF_PYTHON"]
    if not Path(python).is_file() or not 1 <= concurrency <= 8:
        raise ValueError("require an existing Python and concurrency 1..8")
    worker = REPO / "scripts/hpc/run_phase_c_fitting_strategies_delta.sh"
    commit = (
        (REPO / "SOURCE_COMMIT").read_text().strip()
        if (REPO / "SOURCE_COMMIT").exists()
        else subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    )
    identity = {
        "plan_sha256": public.content_sha256(plan),
        "commit": commit,
        "account": account,
        "concurrency": concurrency,
        "python": python,
        "repo": str(REPO),
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
    }
    os.environ.update(
        AF_REPO_ROOT=str(REPO), AF_OUTPUT_ROOT=str(root), AF_COMMIT=commit
    )
    (root / "logs").mkdir(exist_ok=True)
    receipts = root / "submission"
    with public._lock(receipts):
        seal(receipts / "identity.json", identity)
        jobs = {}
        for stage in ("prepare", "fit", "report"):
            options = [
                "--parsable",
                f"--account={account}",
                "--partition=cpu",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=1",
                "--mem=8G",
                "--time="
                + {"prepare": "00:45:00", "fit": "00:20:00", "report": "00:05:00"}[
                    stage
                ],
                "--export=ALL",
                "--kill-on-invalid-dep=yes",
                f"--job-name=fitting-strategy-{stage}",
                f"--output={root}/logs/{stage}-%A_%a.out",
                f"--error={root}/logs/{stage}-%A_%a.err",
            ]
            if stage == "fit":
                options += [
                    f"--dependency=afterok:{jobs['prepare']}",
                    f"--array=0-{len(plan['tasks']) - 1}%{concurrency}",
                ]
            elif stage == "report":
                options += [f"--dependency=afterany:{jobs['prepare']}:{jobs['fit']}"]
            argv = ["sbatch", *options, str(worker), stage]
            intent, receipt, job_file = (
                receipts / f"{stage}.{suffix}"
                for suffix in ("intent.json", "reply.json", "id")
            )
            if job_file.exists():
                if (
                    public._read(intent) != {"argv": argv}
                    or not job_file.read_text().strip().isdecimal()
                ):
                    raise ValueError("saved scheduler identity differs")
                jobs[stage] = job_file.read_text().strip()
                continue
            if intent.exists():
                raise ValueError(
                    f"Unconfirmed {stage} submission; inspect {receipts} "
                    "and sacct before recovery"
                )
            public._write(intent, {"argv": argv})
            try:
                reply = subprocess.run(argv, capture_output=True, text=True, timeout=45)
            except subprocess.TimeoutExpired:
                public._write(receipt, {"status": "uncertain_timeout"})
                raise ValueError(
                    f"Scheduler outcome unknown; inspect {receipts}"
                ) from None
            public._write(
                receipt,
                {
                    "returncode": reply.returncode,
                    "stdout": reply.stdout,
                    "stderr": reply.stderr,
                },
            )
            if reply.returncode or not re.fullmatch(
                r"[0-9]+(?:;[A-Za-z0-9_-]+)?", reply.stdout.strip()
            ):
                raise ValueError(f"Scheduler reply unconfirmed; inspect {receipt}")
            jobs[stage] = reply.stdout.strip().split(";")[0]
            job_file.write_text(jobs[stage] + "\n")
        result = {
            "identity": identity,
            "jobs": jobs,
            "array_tasks": len(plan["tasks"]),
            "gpus": 0,
            "live_llm_calls": 0,
            "test_data_opened": False,
        }
        public._write(root / "submission_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=REPO / "fitting-inputs.json")
    parser.add_argument("--matched-source", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/phase_c_fitting_strategies_v1.json",
    )
    parser.add_argument("--account", default="bibo-delta-cpu")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            submit(
                args.root,
                args.config,
                account=args.account,
                concurrency=args.concurrency,
                inputs=args.inputs,
                matched_source=args.matched_source,
            ),
            indent=2,
        )
    )
