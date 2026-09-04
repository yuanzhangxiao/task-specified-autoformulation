#!/usr/bin/env python3
"""Freeze public splits and shared candidates for the CasADi pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.casadi_initializer_pilot import (
    canonical_casadi_initializer_plan_sha256,
    casadi_initializer_task_count,
    casadi_initializer_task_identity,
    load_casadi_initializer_pilot_plan,
)
from autoformalism.schemas import CandidateModel


def prepare_pilot(
    config_path: Path,
    source_replay_root: Path,
    public_data_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Validate all immutable inputs and create a write-once pilot freeze."""
    config_path = config_path.expanduser().resolve()
    source_replay_root = source_replay_root.expanduser().resolve()
    public_data_root = public_data_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    plan = load_casadi_initializer_pilot_plan(config_path)
    plan_sha256 = canonical_casadi_initializer_plan_sha256(plan)

    source_manifest_path = source_replay_root / "proposer_repair_replay.json"
    source_ledger_path = source_replay_root / "artifact_ledger.jsonl"
    source_manifest = _read_object(source_manifest_path)
    expected_source = plan.source_replay
    for key, expected in {
        "schema_version": expected_source.manifest_schema_version,
        "status": expected_source.required_status,
        "source_plan_sha256": expected_source.source_plan_sha256,
        "replay_plan_sha256": expected_source.replay_plan_sha256,
        "artifact_ledger_sha256": expected_source.artifact_ledger_sha256,
        "replay_result_count": expected_source.replay_result_count,
    }.items():
        if source_manifest.get(key) != expected:
            raise ValueError(f"source replay differs at {key}")
    if _sha256(source_ledger_path) != expected_source.artifact_ledger_sha256:
        raise ValueError("source replay artifact ledger differs")
    source_ledger = _load_ledger(source_ledger_path)

    frozen = output_root / "frozen"
    _write_once(frozen / "plan.json", config_path.read_bytes())
    candidate_rows: list[dict[str, object]] = []
    condition = plan.source_candidate_condition
    for candidate_index, (cell, repetition) in enumerate(
        (cell, repetition)
        for cell in plan.cells
        for repetition in plan.repetitions
    ):
        relative = (
            Path("finalists")
            / condition.directory_name
            / f"task_{candidate_index:03d}.json"
        )
        source_candidate = source_replay_root / relative
        _verify_ledger(relative, source_candidate, source_ledger)
        candidate = CandidateModel.model_validate_json(
            source_candidate.read_text(encoding="utf-8")
        )
        frozen_candidate = (
            frozen / "candidates" / f"candidate_{candidate_index:03d}.json"
        )
        _write_once(
            frozen_candidate,
            candidate.model_dump_json(indent=2).encode() + b"\n",
        )
        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "benchmark_id": cell.benchmark_id,
                "tier": cell.tier,
                "repetition": repetition,
                "source_path": str(relative),
                "source_file_sha256": _sha256(source_candidate),
                "frozen_candidate_sha256": _sha256(frozen_candidate),
            }
        )

    public_rows: list[dict[str, object]] = []
    for cell in plan.cells:
        public_root = public_data_root / "phase_b_v1" / cell.benchmark_id
        prompt_path = public_root / "proposer_prompt.txt"
        if _sha256(prompt_path) != cell.public_prompt_sha256:
            raise ValueError(f"public proposer prompt differs: {cell.benchmark_id}")
        public_rows.append(
            {
                "benchmark_id": cell.benchmark_id,
                "public_prompt_sha256": _sha256(prompt_path),
                "train_sha256": _sha256(public_root / "train.csv"),
                "validation_sha256": _sha256(public_root / "validation.csv"),
            }
        )

    _write_once_jsonl(frozen / "candidate_manifest.jsonl", candidate_rows)
    _write_once_jsonl(frozen / "public_input_manifest.jsonl", public_rows)
    task_rows: list[dict[str, object]] = []
    for task_index in range(casadi_initializer_task_count(plan)):
        fit_condition, cell, repetition, candidate_index = (
            casadi_initializer_task_identity(plan, task_index)
        )
        task_rows.append(
            {
                "task_index": task_index,
                "condition_id": fit_condition.condition_id,
                "benchmark_id": cell.benchmark_id,
                "tier": cell.tier,
                "repetition": repetition,
                "candidate_index": candidate_index,
            }
        )
    _write_once_jsonl(frozen / "task_plan.jsonl", task_rows)
    freeze = {
        "schema_version": "phase-b-casadi-initializer-pilot-freeze-1",
        "status": "frozen_before_fitting",
        "development_only": True,
        "plan_sha256": plan_sha256,
        "source_replay_manifest_sha256": _sha256(source_manifest_path),
        "source_replay_artifact_ledger_sha256": _sha256(source_ledger_path),
        "candidate_manifest_sha256": _sha256(
            frozen / "candidate_manifest.jsonl"
        ),
        "public_input_manifest_sha256": _sha256(
            frozen / "public_input_manifest.jsonl"
        ),
        "task_plan_sha256": _sha256(frozen / "task_plan.jsonl"),
        "candidate_count": len(candidate_rows),
        "task_count": len(task_rows),
        "same_candidate_across_conditions": True,
        "equal_total_wall_time_budget": True,
        "observed_derivatives_supplied": False,
        "latent_values_supplied": False,
        "latent_derivatives_supplied": False,
        "private_reference_available_to_fitter": False,
        "test_data_opened": False,
        "weighted_overall_score_defined": False,
    }
    _write_once(
        frozen / "freeze_manifest.json",
        (json.dumps(freeze, indent=2, sort_keys=True) + "\n").encode(),
    )
    return freeze


def _load_ledger(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source ledger path: {relative}")
        if str(relative) in result:
            raise ValueError(f"duplicate source ledger path: {relative}")
        result[str(relative)] = item
    return result


def _verify_ledger(
    relative: Path, path: Path, ledger: dict[str, dict[str, Any]]
) -> None:
    item = ledger.get(str(relative))
    if item is None:
        raise ValueError(f"source candidate is absent from ledger: {relative}")
    if _sha256(path) != item.get("sha256") or path.stat().st_size != item.get(
        "size_bytes"
    ):
        raise ValueError(f"source candidate differs from ledger: {relative}")


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_once_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    _write_once(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode(),
    )


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"frozen pilot artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    parser.add_argument("--source-replay-root", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    result = prepare_pilot(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
