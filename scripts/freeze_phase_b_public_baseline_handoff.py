#!/usr/bin/env python3
"""Freeze public-only baseline selections for transfer to sealed evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from autoformalism.rebuttal.baseline_pilot import BaselinePilotTask


def main() -> None:
    """Copy only typed selection/status artifacts and record every digest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    tasks = tuple(
        BaselinePilotTask.model_validate_json(line)
        for line in args.task_plan.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for task in tasks:
        run = (
            args.runs_root.expanduser().resolve()
            / task.method
            / f"{task.benchmark_id}_{task.tier}_seed{task.repetition}"
        )
        result = run / "result.json"
        status = run / "run_status.json"
        source = result if result.is_file() else status
        if not source.is_file():
            raise ValueError(
                f"baseline task has no terminal artifact: {task.task_index}"
            )
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"baseline artifact is not an object: {source}")
        if source == result:
            if payload.get("test_data_opened") is not False:
                raise ValueError(f"baseline result is not public-only: {source}")
            if (
                payload.get("method"),
                payload.get("benchmark_id"),
                payload.get("tier"),
                payload.get("seed"),
            ) != (
                task.method,
                task.benchmark_id,
                task.tier,
                task.repetition,
            ):
                raise ValueError(f"baseline result identity differs: {source}")
        destination = output / "tasks" / f"task_{task.task_index:03d}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if destination.read_bytes() != source.read_bytes():
                raise ValueError(f"frozen handoff artifact differs: {destination}")
        else:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix="handoff-", delete=False
            ) as handle:
                temporary = Path(handle.name)
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        records.append(
            {
                "task_index": task.task_index,
                "method": task.method,
                "benchmark_id": task.benchmark_id,
                "tier": task.tier,
                "repetition": task.repetition,
                "artifact_kind": "selection" if source == result else "failure",
                "path": str(destination.relative_to(output)),
                "sha256": _sha256(destination),
            }
        )
    ledger = output / "handoff_sources.jsonl"
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    _write_once(ledger, text)
    manifest = {
        "schema_version": "phase-b-public-baseline-handoff-1",
        "status": "frozen_for_delta_transfer",
        "task_count": len(tasks),
        "selection_count": sum(
            record["artifact_kind"] == "selection" for record in records
        ),
        "failure_count": sum(
            record["artifact_kind"] == "failure" for record in records
        ),
        "task_plan_sha256": _sha256(args.task_plan.expanduser().resolve()),
        "handoff_sources_sha256": _sha256(ledger),
        "test_data_included": False,
        "private_reference_included": False,
        "llm_raw_responses_included": False,
    }
    manifest_path = output / "handoff_manifest.json"
    _write_once(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"frozen handoff differs: {path}")
        return
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
