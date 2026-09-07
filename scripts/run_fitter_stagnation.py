#!/usr/bin/env python3
"""Prepare, supervise and summarize the user-run fitter stagnation diagnosis."""

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
from autoformalism.rebuttal.fitter_stagnation import (  # noqa: E402
    StagnationPlan,
    checkpoint,
    execute_diagnosis,
    prepare_diagnosis,
    summarize_diagnosis,
    verify_diagnosis,
)
from autoformalism.staged_topology import content_hash  # noqa: E402


def run_supervised(output: Path, index: int) -> dict:
    """Enforce a hard per-task cap while keeping all completed checkpoints."""
    frozen = verify_diagnosis(output)
    if not 0 <= index < len(frozen["tasks"]):
        raise ValueError("unknown task index")
    task = frozen["tasks"][index]
    identity = content_hash([frozen["freeze_sha256"], task])
    root = output / "results" / task["name"]
    root.mkdir(parents=True, exist_ok=True)
    result = checkpoint(root / "result.json", identity)
    if result is not None:
        return result
    plan = StagnationPlan.model_validate(frozen["plan"])
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
    """Write compact user-readable outcomes; keep detailed traces in each task."""
    result = summarize_diagnosis(output)
    write_json(output / "summary.json", result)
    lines = [
        "# Fitter stagnation diagnosis",
        "",
        "Same model and all-one start. All final scores use the same "
        "tighter ODE tolerance.",
        "No model selection or test data. A complete task need not have "
        "optimizer convergence.",
        "",
        "| Arm | Status | Optimizer success | Train NMSE | Validation NMSE | "
        "nfev | Actual residual calls | Optimality |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def format_value(value):
        return (
            "—"
            if value is None
            else f"{value:.6g}"
            if isinstance(value, float)
            else str(value)
        )

    for row in result["rows"]:
        values = [
            row[k]
            for k in (
                "name",
                "status",
                "optimizer_success",
                "train_nmse",
                "validation_nmse",
                "nfev",
                "actual_residual_calls",
                "optimality",
            )
        ]
        lines.append("| " + " | ".join(format_value(v) for v in values) + " |")
        if row["error"]:
            lines.extend(["", f"{row['name']}: {row['error']}", ""])
    lines.extend(["", "## Optimizer stops and parameter values", ""])
    for row in result["rows"][1:]:
        lines.append(
            f"- {row['name']}: {row['message']}; selection={row['selection']}; "
            f"parameters={row['parameters']}"
        )
    profile = output / "results/step_profile/result.json"
    if profile.exists():
        record = read_json(profile)
        lines.extend(
            [
                "",
                "## Training-subset derivative sensitivity",
                "",
                f"Frozen trajectory IDs: {record.get('trajectory_ids', [])}",
                "",
                "| Anchor | ODE tolerance | Step factor | "
                "Central gradient infinity norm | Jacobian condition | "
                "Forward/central disagreement |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for report in record.get("reports", []):
            for estimate in report["steps"]:
                values = [
                    report["anchor"],
                    report["ode_tolerance"],
                    estimate["step_factor"],
                    estimate["central"]["gradient_inf_norm"],
                    estimate["central"]["condition_number"],
                    estimate["forward_central_relative_difference"],
                ]
                lines.append("| " + " | ".join(format_value(v) for v in values) + " |")
        lines.extend(
            [
                "",
                "Condition numbers use raw physical parameter coordinates; "
                "they are descriptive, not an identifiability proof.",
                "",
                "Repeat and integration-tolerance checks:",
            ]
        )
        for report in record.get("reports", []):
            difference = format_value(report["exact_repeat_residual_difference"])
            lines.append(
                f"- {report['anchor']} / {report['ode_tolerance']}: "
                f"exact-repeat residual difference norm={difference}"
            )
            for comparison in report["adjacent_steps"]:
                difference = format_value(comparison["central_relative_difference"])
                lines.append(
                    f"  - Central Jacobian relative difference from "
                    f"{comparison['left_step']:.6g} to "
                    f"{comparison['right_step']:.6g}: {difference}"
                )
        for comparison in record.get("tolerance_comparison", []):
            difference = format_value(comparison["residual_difference_norm"])
            lines.append(
                f"- {comparison['anchor']}: current/tight residual "
                f"difference norm={difference}"
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    """Require explicit preparation and task execution, independently of notebooks."""
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
        print(execute_diagnosis(output, args.task_index)["status"])
        return
    lock_name = f"task-{args.task_index}" if args.stage == "run" else args.stage
    locks = output / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{lock_name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.stage == "prepare":
            if args.config is None or args.source is None:
                parser.error("prepare requires config and source")
            result = prepare_diagnosis(
                StagnationPlan.model_validate(read_json(args.config)),
                args.source,
                output,
            )
            print(result["freeze_sha256"], "tasks", len(result["tasks"]))
        elif args.stage == "run":
            print(run_supervised(output, args.task_index)["status"])
        else:
            write_summary(output)
            print(output / "summary.md")


if __name__ == "__main__":
    main()
