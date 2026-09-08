#!/usr/bin/env python3
"""Prepare and supervise the user-run fitter runtime and accuracy campaign."""

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
from autoformalism.rebuttal.fitter_runtime import (  # noqa: E402
    RuntimePlan,
    checkpoint,
    execute_runtime,
    prepare_runtime,
    summarize_runtime,
    verify_runtime,
)
from autoformalism.staged_topology import content_hash  # noqa: E402


def run_supervised(output: Path, index: int) -> dict:
    """Enforce a hard per-task cap while keeping all completed checkpoints."""
    frozen = verify_runtime(output)
    if not 0 <= index < len(frozen["tasks"]):
        raise ValueError("unknown task index")
    task = frozen["tasks"][index]
    identity = content_hash([frozen["freeze_sha256"], task])
    root = output / "results" / task["name"]
    root.mkdir(parents=True, exist_ok=True)
    result = checkpoint(root / "result.json", identity)
    if result is not None:
        return result
    plan = RuntimePlan.model_validate(frozen["plan"])
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
    """Expose every numerical gate and fit outcome in the user's review report."""
    result = summarize_runtime(output)
    write_json(output / "summary.json", result)
    lines = [
        "# Fitter runtime and accuracy — corrected input integration",
        "",
        "Frozen model; all fits start at all ones. No test data or LLM calls.",
        "A complete task can still have optimizer timeout or non-convergence.",
        "",
    ]

    def fmt(value):
        if value is None:
            return "—"
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    def table_row(values):
        lines.append("| " + " | ".join(fmt(v) for v in values) + " |")

    lines += [
        "## Fixed-vector profiles",
        "",
        f"Times cover {len(result['profile_trajectory_ids'])} fixed training "
        "trajectories; timing fields below are nested. "
        "Compute excludes array checkpoint I/O.",
        "",
        "| Anchor | Method | Tolerance | Accuracy pass | Median compute s | "
        "Median array checkpoint s | "
        "Residual RMS difference | Max difference |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"][:3]:
        record = row["result"] or {}
        for entry in record.get("comparisons", []):
            table_row(
                [
                    row["task"]["anchor"],
                    entry["method"],
                    entry["tolerance"],
                    entry["accuracy_pass"],
                    entry.get("median_compute_seconds"),
                    entry.get("median_array_checkpoint_seconds"),
                    entry.get("rms"),
                    entry.get("maximum_absolute"),
                ]
            )
    for row in result["rows"][:3]:
        record = row["result"] or {}
        lines += [
            "",
            "### " + row["task"]["anchor"],
            "",
            f"Status: {row['status']}; reference: {record.get('reference')}",
            f"Reference refinement checks: {record.get('reference_checks')}",
            "Derivative points verified across solvers: "
            f"{record.get('derivative_points_verified')}",
            "",
            "| Case | Status | Total simulation s | Integration s | "
            "RHS s | RHS calls | "
            "Solver nfev | Observation s | Forcing setup s | "
            "Observation expression s |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, entry in record.get("cases", {}).items():
            totals = {
                key: sum(t.get(key, 0) for t in entry.get("trajectories", []))
                for key in (
                    "total_seconds",
                    "integration_seconds",
                    "rhs_seconds",
                    "rhs_calls",
                    "solver_nfev",
                    "observation_seconds",
                    "observation_forcing_seconds",
                    "observation_expression_seconds",
                )
            }
            table_row([name, entry["status"], *totals.values()])
        for name in ("Radau_reference", "DOP853_reference"):
            for trajectory in (
                record.get("cases", {}).get(name, {}).get("trajectories", [])
            ):
                lines.append(
                    f"- {name} / {trajectory['trajectory_id']}: "
                    f"segments={trajectory.get('integration_segments')}; "
                    f"state ranges={trajectory.get('state_ranges')}"
                )
        lines += [
            "",
            "Derivative consistency (reference solver at rtol=1e-9):",
            "",
            "| Policy | Status | Gradient infinity norm | "
            "Difference from smaller scaled step | "
            "Cross-solver Jacobian difference | Actual steps |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
        for entry in record.get("derivatives", []):
            table_row(
                [
                    entry["name"],
                    entry["status"],
                    entry.get("matrix", {}).get("gradient_inf_norm"),
                    entry.get("relative_difference_from_smaller_scaled"),
                    entry.get("jacobian_reference_relative_difference"),
                    entry.get("actual_steps"),
                ]
            )
            if entry.get("error"):
                lines.append(f"Derivative check {entry['name']}: {entry['error']}")
        lines += [
            "",
            "Python profile: separate intrusive measurement, "
            "excluded from timing comparisons.",
            "Profiled trajectory: "
            f"{record.get('interpreter', {}).get('trajectory_id')}",
            "",
        ]
        for entry in record.get("interpreter", {}).get("functions", [])[:10]:
            lines.append(
                f"- {entry['file']}:{entry['line']} {entry['function']}: "
                f"self={entry['self_seconds']:.5g}s, "
                f"cumulative={entry['cumulative_seconds']:.5g}s, "
                f"calls={entry['calls']}"
            )
    lines += [
        "",
        "## Matched fits",
        "",
        f"Each fit gets {result['plan']['fit_seconds']:g} seconds. "
        "Final scores use DOP853 at 1e-9/1e-11; Radau independently checks "
        "the same values on all train and validation trajectories.",
        "",
        "| Arm | Status | Optimizer success | Train NMSE | Validation NMSE | "
        "Actual calls | nfev | Fit s | Train solver score difference | "
        "Validation solver score difference |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"][3:9]:
        record = row["result"] or {}
        fit = record.get("fit", {})
        replay = record.get("replays", {}).get("DOP853") or {}
        differences = record.get("replay_score_difference") or {}
        table_row(
            [
                row["task"]["name"],
                row["status"],
                fit.get("optimizer_success"),
                replay.get("train", {}).get("normalized_mse"),
                replay.get("validation", {}).get("normalized_mse"),
                fit.get("actual_residual_calls"),
                fit.get("nfev"),
                fit.get("fit_seconds"),
                differences.get("train"),
                differences.get("validation"),
            ]
        )
    for row in result["rows"][3:9]:
        record = row["result"] or {}
        fit = record.get("fit", {})
        lines += [
            "",
            f"### {row['task']['name']}",
            "",
            f"Accuracy guard: {record.get('guard')}",
            f"Stop: {fit.get('message')}; selection: {fit.get('selection')}",
            f"Parameters: {fit.get('parameters')}",
            f"Iteration callbacks: {len(fit.get('iterations', []))}; "
            f"last callback: {fit.get('iterations', [])[-1:]}",
        ]
        if record.get("error"):
            lines.append("Error: " + record["error"])
    lines += [
        "",
        "## Replay of previous fitted vectors",
        "",
        "Exact previous parameters; no new optimization. "
        "Corrected scores use DOP853, checked by Radau.",
        "",
        "| Previous arm | Status | Old train NMSE | Corrected train NMSE | "
        "Old validation NMSE | Corrected validation NMSE | "
        "Train solver difference | Validation solver difference |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"][9:]:
        record = row["result"] or {}
        old = record.get("previous_scores") or {}
        replay = record.get("replays", {}).get("DOP853") or {}
        differences = record.get("replay_score_difference") or {}
        table_row(
            [
                row["task"]["arm"],
                row["status"],
                old.get("train", {}).get("normalized_mse"),
                replay.get("train", {}).get("normalized_mse"),
                old.get("validation", {}).get("normalized_mse"),
                replay.get("validation", {}).get("normalized_mse"),
                differences.get("train"),
                differences.get("validation"),
            ]
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
        print(execute_runtime(output, args.task_index)["status"])
        return
    lock_name = f"task-{args.task_index}" if args.stage == "run" else args.stage
    locks = output / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{lock_name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.stage == "prepare":
            if args.config is None or args.source is None:
                parser.error("prepare requires config and source")
            result = prepare_runtime(
                RuntimePlan.model_validate(read_json(args.config)),
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
