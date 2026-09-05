#!/usr/bin/env python3
"""Prepare, supervise, and summarize the frozen CPU-only fitter diagnostic."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from time import monotonic

# Set before importing numerical libraries, including in direct CLI invocations.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

from autoformalism.rebuttal.fitter_diagnostic import (  # noqa: E402
    ARMS,
    DiagnosticPlan,
    execute_task,
    prepare_diagnostic,
    read_json,
    sha256,
    summarize_diagnostic,
    verify_freeze,
    write_json,
)


@contextmanager
def _lock(path: Path):
    """Serialize duplicate preparation/tasks; crashes release the OS lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def run_supervised(root: Path, index: int) -> dict:
    """Bound the entire fit and replay, including unbounded library finalization."""
    with _lock(root / "locks" / f"{index:03d}.lock"):
        manifest = verify_freeze(root)
        if not 0 <= index < len(manifest["tasks"]):
            raise ValueError("task index outside frozen matrix")
        task = manifest["tasks"][index]
        identity = {**task, "freeze_sha256": sha256(root / "freeze.json")}
        destination = root / "results" / f"{index:03d}.json"
        if destination.exists():
            result = read_json(destination)
            if any(result.get(key) != value for key, value in identity.items()):
                raise ValueError("existing result belongs to a different task/freeze")
            return result
        plan = DiagnosticPlan.model_validate(manifest["plan"])
        limit = plan.replay_seconds + plan.supervisor_grace_seconds
        if task["arm"] != "agent_replay":
            limit += plan.fit_seconds
        destination.parent.mkdir(parents=True, exist_ok=True)
        log_path = root / "results" / f"{index:03d}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--output-root",
            str(root),
            "--task-index",
            str(index),
        ]
        started = monotonic()
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=log, start_new_session=True
            )
            try:
                return_code = process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                result = {
                    **identity,
                    "status": "timeout",
                    "error": f"worker exceeded {limit:g} seconds",
                }
            except BaseException:
                # Interactive cancellation must not leave an orphan CPU worker.
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            else:
                if return_code == 0 and destination.exists():
                    result = read_json(destination)
                    if any(result.get(key) != value for key, value in identity.items()):
                        raise ValueError("worker result identity differs")
                else:
                    result = {
                        **identity,
                        "status": "worker_failed",
                        "error": f"worker exited {return_code}; see {log_path.name}",
                    }
        result.update(
            task_seconds=monotonic() - started, test_data_opened=False, llm_calls=0
        )
        write_json(destination, result)
        return result


def _write_summary(root: Path) -> dict:
    result = summarize_diagnostic(root)
    write_json(root / "summary.json", result)
    lines = [
        "# Frozen fitter diagnostic",
        "",
        "Development results only. Ratios compare the same frozen model.",
        "All prespecified repetitions and failed tasks are retained.",
        "",
        "| Benchmark / repetition | Arm | Status | Train NMSE | "
        "Validation NMSE | Ratio to agent |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        for arm in ARMS:
            value = row["arms"][arm]
            numbers = [
                value.get(key)
                for key in (
                    "train_nmse",
                    "validation_nmse",
                    "validation_ratio_to_agent",
                )
            ]
            rendered = [
                "—" if number is None else f"{number:.6g}" for number in numbers
            ]
            lines.append(
                f"| {row['benchmark_id']} / {row['repetition']} | {arm} | "
                f"{value['status']} | {' | '.join(rendered)} |"
            )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Run one explicit stage without requiring notebooks or provider credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "worker", "summarize"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase_b_fitter_diagnostic_v1.json")
    )
    parser.add_argument("--public-data-root", type=Path)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--refresh-root", type=Path)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    if args.stage == "prepare":
        if any(
            value is None
            for value in (
                args.public_data_root,
                args.historical_root,
                args.refresh_root,
            )
        ):
            parser.error(
                "prepare requires public-data-root, historical-root and refresh-root"
            )
        plan = DiagnosticPlan.model_validate(read_json(args.config))
        with _lock(root / "locks" / "prepare.lock"):
            result = prepare_diagnostic(
                plan,
                public_root=args.public_data_root,
                historical_root=args.historical_root,
                refresh_root=args.refresh_root,
                output_root=root,
            )
        print(
            f"Frozen {len(result['sources'])} sources and {len(result['tasks'])} tasks"
        )
        print(
            {
                status: sum(source["status"] == status for source in result["sources"])
                for status in ("ready", "source_missing", "source_invalid")
            }
        )
    elif args.stage in {"run", "worker"}:
        if args.task_index is None:
            parser.error("run/worker requires task-index")
        if args.stage == "run":
            result = run_supervised(root, args.task_index)
        else:
            result = execute_task(root, args.task_index)
            write_json(root / "results" / f"{args.task_index:03d}.json", result)
        print(f"Task {args.task_index}: {result['status']}")
    else:
        result = _write_summary(root)
        print(
            f"Expected models: {result['expected_models']}; "
            f"complete: {result['complete_by_arm']}"
        )
        print(root / "summary.md")


if __name__ == "__main__":
    main()
