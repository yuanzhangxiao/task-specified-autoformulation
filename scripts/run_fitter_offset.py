#!/usr/bin/env python3
"""Freeze, supervise and summarize the user-run paired signed-offset experiment."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
from pathlib import Path
from time import monotonic

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json  # noqa: E402
from autoformalism.rebuttal.fitter_offset import (  # noqa: E402
    OffsetPlan,
    checkpoint,
    execute_offset,
    prepare_offset,
    summarize_offset,
    verify_offset,
)
from autoformalism.staged_topology import content_hash  # noqa: E402


def run_supervised(output: Path, index: int) -> dict:
    """Keep partial checkpoints and terminate the process group at its hard cap."""
    frozen = verify_offset(output)
    if not 0 <= index < len(frozen["tasks"]):
        raise ValueError("unknown task index")
    task = frozen["tasks"][index]
    identity = content_hash([frozen["freeze_sha256"], task])
    root = output / "results" / task["name"]
    root.mkdir(parents=True, exist_ok=True)
    result = checkpoint(root / "result.json", identity)
    if result is not None:
        return result
    plan = OffsetPlan.model_validate(frozen["plan"])
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--output",
        str(output),
        "--task-index",
        str(index),
    ]
    started = monotonic()
    with (root / "worker.log").open("a") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=log, start_new_session=True
        )
        try:
            code = process.wait(timeout=plan.worker_seconds(index))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            result = {
                "status": "timeout",
                "error": "supervisor wall-clock limit reached",
            }
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        else:
            result = checkpoint(root / "result.json", identity) if code == 0 else None
            if result is None:
                result = {
                    "status": "worker_failed",
                    "error": f"worker exited {code}; see worker.log",
                }
    result.update(
        identity=identity,
        task=task,
        task_seconds=monotonic() - started,
        test_data_opened=False,
        private_reference_opened=False,
        llm_calls=0,
    )
    write_json(root / "result.json", result)
    return result


def write_summary(output: Path) -> dict:
    """Display both baselines, every solver score and explicit replay failures."""
    result = summarize_offset(output)
    write_json(output / "summary.json", result)
    baselines = result["baselines"]
    lines = [
        "# Paired signed-offset experiment",
        "",
        "Only the offset role/domain changes; equations and initials are identical.",
        "Two shared starts; Radau fits on training only. No test data or selection.",
        "Completion means numerical verification, not optimizer convergence "
        "or scientific adequacy.",
        "",
        "| Baseline | Train NMSE | Validation NMSE |",
        "| --- | ---: | ---: |",
    ]
    for label, key in (
        ("Always zero", "zero_nmse"),
        ("Constant training mean", "training_mean_nmse"),
    ):
        lines.append(
            f"| {label} | {baselines['train'][key]:.9g} | "
            f"{baselines['validation'][key]:.9g} |"
        )
    lines += [
        "",
        f"Training mean: {baselines['training_mean']:.12g}",
        f"Shared starts: `{result['starts']}`",
        "",
        "## All tasks",
        "",
        "| Task | Status | Optimizer success | Calls | Fit seconds |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in result["rows"]:
        fit = (row["result"] or {}).get("fit", {})
        lines.append(
            f"| {row['task']['name']} | {row['status']} | "
            f"{fit.get('optimizer_success', '—')} | "
            f"{fit.get('actual_residual_calls', '—')} | {fit.get('fit_seconds', '—')} |"
        )
    for row in result["rows"]:
        record = row["result"] or {}
        lines += [
            "",
            "## " + row["task"]["name"],
            "",
            f"Status: {row['status']}; error: {record.get('error')}",
        ]
        if row["task"]["kind"] == "guard":
            lines.extend(
                f"- {name}: {case['status']}; baseline check: "
                f"{case.get('analytic_baseline_pass', 'n/a')}"
                for name, case in record.get("cases", {}).items()
            )
            for name, case in record.get("cases", {}).items():
                lines.append(f"- {name} pointwise checks: `{case.get('checks')}`")
                for method, replay in case.get("replays", {}).items():
                    errors = sorted({f["message"] for f in replay.get("failures", [])})
                    if errors:
                        lines.append(f"- {name}/{method} failures: {errors}")
            continue
        fit = record.get("fit", {})
        lines += [
            f"Stop: {fit.get('message')}; selection: {fit.get('selection')}",
            f"Parameters: `{record.get('parameters')}`",
            "",
            "| Replay | Status | NMSE | Prediction mean | Prediction RMS | "
            "Prediction SD | Failures |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]

        def number(value):
            return "—" if value is None else f"{value:.9g}"

        for name, replay in record.get("replays", {}).items():
            errors = sorted({f["message"] for f in replay.get("failures", [])})
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        replay["status"],
                        *[
                            number(replay.get(k))
                            for k in (
                                "normalized_mse",
                                "prediction_mean",
                                "prediction_rms",
                                "prediction_std",
                            )
                        ],
                        str(errors),
                    ]
                )
                + " |"
            )
        lines += ["", f"Pointwise solver/refinement checks: `{record.get('checks')}`"]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    """Keep preparation, execution and reporting independently runnable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "worker", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if args.stage in {"run", "worker"} and args.task_index is None:
        parser.error("run/worker requires task-index")
    if args.stage == "worker":
        print(execute_offset(output, args.task_index)["status"])
        return
    lock_name = f"task-{args.task_index}" if args.stage == "run" else args.stage
    locks = output / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{lock_name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.stage == "prepare":
            if args.source is None or args.config is None:
                parser.error("prepare requires source and config")
            result = prepare_offset(
                OffsetPlan.model_validate(read_json(args.config)), args.source, output
            )
            print(result["freeze_sha256"], "tasks", len(result["tasks"]))
        elif args.stage == "run":
            print(run_supervised(output, args.task_index)["status"])
        else:
            write_summary(output)
            print(output / "summary.md")


if __name__ == "__main__":
    main()
