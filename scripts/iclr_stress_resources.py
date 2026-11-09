"""Associate Slurm accounting records with paired discovery blocks."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read


def cpu_seconds(value: str) -> float | None:
    """Parse the day/hour/minute or minute/second format emitted by sacct."""
    if not value or value in {"Unknown", "N/A"}:
        return None
    days, _, clock = value.rpartition("-")
    fields = [float(x) for x in clock.split(":")]
    if len(fields) > 3 or any(x < 0 for x in fields):
        raise ValueError("invalid Slurm CPU duration")
    return int(days or "0") * 86400 + sum(
        x * 60**i for i, x in enumerate(reversed(fields))
    )


def allocation_gpus(tres: str) -> int | None:
    """Prefer Slurm's aggregate GPU TRES over GPU-type subtotals."""
    if not tres or tres in {"Unknown", "N/A"}:
        return None
    counts = {
        k: int(v)
        for atom in tres.split(",")
        for k, sep, v in [atom.partition("=")]
        if sep and (k == "gres/gpu" or k.startswith("gres/gpu:"))
    }
    return counts.get("gres/gpu", sum(counts.values()))


def collect(plan_path: Path, sources: list[Path], dest: Path) -> list[dict]:
    """Preserve retries and record whole-block scheduler allocation once per job."""
    plan = sealed_read(plan_path)
    if plan.get("protocol") != "iclr-controlled-search-1":
        raise ValueError("expected a sealed controlled search plan")
    blocks = {b["block"]: b for b in plan["blocks"]}
    ledger = {}
    for source in sources:
        with source.open() as stream:
            reader = csv.DictReader(stream, delimiter="|")
            required = {
                "JobID",
                "ElapsedRaw",
                "TotalCPU",
                "AllocCPUS",
                "AllocTRES",
                "State",
            }
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"required sacct fields missing from {source}")
            for row in reader:
                job = row["JobID"]
                # Only the requested array-task allocation. Child steps and the
                # array parent can contain the same resource accounting.
                match = re.fullmatch(r"\d+_(\d+)", job)
                if not match:
                    continue
                i = int(match.group(1))
                if i not in blocks:
                    raise ValueError(f"array index {i} outside campaign plan")
                if job in ledger and ledger[job] != row:
                    raise ValueError(f"conflicting accounting records for {job}")
                ledger[job] = row
    records = []
    for job, row in ledger.items():
        block = blocks[int(job.rsplit("_", 1)[1])]
        seconds = float(row["ElapsedRaw"]) if row["ElapsedRaw"] else None
        gpus = allocation_gpus(row["AllocTRES"])
        records.append(
            {
                "plan_sha256": plan["artifact_sha256"],
                "job": job,
                "block": block["block"],
                "benchmark_id": block["cell"],
                "repetition": block["seed"],
                "treatments": "+".join(t["treatment"] for t in block["tasks"]),
                "scheduler_state": row["State"],
                "wall_seconds": seconds,
                "allocated_cpus": row["AllocCPUS"],
                "measured_cpu_seconds": cpu_seconds(row["TotalCPU"]),
                "allocated_gpus": gpus,
                "allocated_gpu_hours": gpus * seconds / 3600
                if gpus is not None and seconds is not None
                else None,
            }
        )
    if not records:
        raise ValueError("no array task allocations found")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.plan, args.sacct, args.output)
