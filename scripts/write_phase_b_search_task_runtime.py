#!/usr/bin/env python3
"""Write one failure-safe Phase-B search task resource record."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _read_process_time(path: Path | None) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "search_process_elapsed_seconds": None,
        "search_process_user_cpu_seconds": None,
        "search_process_system_cpu_seconds": None,
        "search_process_max_rss_kib": None,
    }
    if path is None or not path.is_file():
        return result
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    result.update(
        {
            "search_process_elapsed_seconds": _optional_float(
                values.get("elapsed_seconds")
            ),
            "search_process_user_cpu_seconds": _optional_float(
                values.get("user_cpu_seconds")
            ),
            "search_process_system_cpu_seconds": _optional_float(
                values.get("system_cpu_seconds")
            ),
            "search_process_max_rss_kib": (
                None
                if not values.get("max_rss_kib")
                else int(values["max_rss_kib"])
            ),
        }
    )
    return result


def write_runtime_record(
    *,
    output: Path,
    task_index: int,
    arm_id: str,
    benchmark_id: str,
    tier: str,
    repetition: int,
    started_epoch_seconds: float,
    finished_epoch_seconds: float,
    exit_code: int,
    allocated_cpus: int,
    allocated_gpus: int,
    gpu_inventory_path: Path | None,
    process_time_path: Path | None,
) -> dict[str, object]:
    """Create one typed-enough JSON record without reading benchmark data."""
    if finished_epoch_seconds < started_epoch_seconds:
        raise ValueError("task finish time precedes start time")
    elapsed = finished_epoch_seconds - started_epoch_seconds
    gpu_inventory = (
        None
        if gpu_inventory_path is None or not gpu_inventory_path.is_file()
        else gpu_inventory_path.read_text(encoding="utf-8").strip()
    )
    record: dict[str, object] = {
        "schema_version": "phase-b-search-task-runtime-1",
        "task_index": task_index,
        "arm_id": arm_id,
        "benchmark_id": benchmark_id,
        "tier": tier,
        "repetition": repetition,
        "started_epoch_seconds": started_epoch_seconds,
        "finished_epoch_seconds": finished_epoch_seconds,
        "task_elapsed_wall_seconds": elapsed,
        "exit_code": exit_code,
        "allocated_cpus": allocated_cpus,
        "allocated_gpus": allocated_gpus,
        "allocated_cpu_core_hours": elapsed * allocated_cpus / 3600.0,
        "allocated_gpu_hours": elapsed * allocated_gpus / 3600.0,
        "gpu_inventory": gpu_inventory,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "queue_time_included": False,
        "monetary_cost_usd": None,
        "monetary_cost_status": "not_priced_local_open_weight_model",
        **_read_process_time(process_time_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(output)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--started-epoch-seconds", type=float, required=True)
    parser.add_argument("--finished-epoch-seconds", type=float)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--allocated-cpus", type=int, required=True)
    parser.add_argument("--allocated-gpus", type=int, required=True)
    parser.add_argument("--gpu-inventory", type=Path)
    parser.add_argument("--process-time", type=Path)
    args = parser.parse_args()
    record = write_runtime_record(
        output=args.output,
        task_index=args.task_index,
        arm_id=args.arm_id,
        benchmark_id=args.benchmark_id,
        tier=args.tier,
        repetition=args.repetition,
        started_epoch_seconds=args.started_epoch_seconds,
        finished_epoch_seconds=(
            time.time()
            if args.finished_epoch_seconds is None
            else args.finished_epoch_seconds
        ),
        exit_code=args.exit_code,
        allocated_cpus=args.allocated_cpus,
        allocated_gpus=args.allocated_gpus,
        gpu_inventory_path=args.gpu_inventory,
        process_time_path=args.process_time,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
