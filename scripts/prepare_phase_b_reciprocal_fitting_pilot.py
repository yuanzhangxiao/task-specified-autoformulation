#!/usr/bin/env python3
"""Freeze shared candidates and exact derivatives before the fitting pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.reciprocal_fitting_pilot import (
    canonical_reciprocal_fitting_plan_sha256,
    load_reciprocal_fitting_pilot_plan,
    reciprocal_fitting_task_count,
    reciprocal_fitting_task_identity,
)
from autoformalism.schemas import CandidateModel


def prepare_pilot(
    config_path: Path,
    source_replay_root: Path,
    derivative_overlay_root: Path,
    public_data_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Validate immutable inputs and create a portable write-once freeze."""
    paths = [
        config_path,
        source_replay_root,
        derivative_overlay_root,
        public_data_root,
        output_root,
    ]
    (
        config_path,
        source_replay_root,
        derivative_overlay_root,
        public_data_root,
        output_root,
    ) = (item.expanduser().resolve() for item in paths)
    plan = load_reciprocal_fitting_pilot_plan(config_path)
    plan_sha256 = canonical_reciprocal_fitting_plan_sha256(plan)
    range_ownership = (
        plan.schema_version == "phase-b-parameter-range-ownership-pilot-1"
    )
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

    overlay_manifest_path = derivative_overlay_root / "manifest.json"
    overlay_manifest = _read_object(overlay_manifest_path)
    if (
        overlay_manifest.get("schema_version")
        != "phase-b-exact-observed-derivative-overlay-1"
        or overlay_manifest.get("status") != "complete"
        or overlay_manifest.get("test_data_opened") is not False
        or overlay_manifest.get("config_sha256") != _sha256(config_path)
    ):
        raise ValueError("exact derivative overlay manifest differs")
    overlay_cells = {
        str(item["benchmark_id"]): item
        for item in overlay_manifest.get("cells", [])
    }

    frozen = output_root / "frozen"
    _write_once(frozen / "plan.json", config_path.read_bytes())
    condition = plan.source_candidate_condition
    candidate_rows: list[dict[str, object]] = []
    derivative_rows: list[dict[str, object]] = []
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
        canonical = candidate.model_dump_json(indent=2).encode() + b"\n"
        _write_once(frozen_candidate, canonical)
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

    for cell in plan.cells:
        public_root = public_data_root / "phase_b_v1" / cell.benchmark_id
        if _sha256(public_root / "proposer_prompt.txt") != cell.public_prompt_sha256:
            raise ValueError(f"public proposer prompt differs: {cell.benchmark_id}")
        overlay_cell = overlay_cells.get(cell.benchmark_id)
        if overlay_cell is None:
            raise ValueError(
                "exact derivative overlay cell is absent: "
                f"{cell.benchmark_id}"
            )
        for split_name in ("train",):
            source_derivative = (
                derivative_overlay_root / cell.benchmark_id / f"{split_name}.csv"
            )
            expected = overlay_cell["splits"][split_name]
            if _sha256(source_derivative) != expected["derivative_overlay_sha256"]:
                raise ValueError("exact derivative overlay file differs")
            public_split = public_root / f"{split_name}.csv"
            if _sha256(public_split) != expected["public_split_sha256"]:
                raise ValueError("public split differs from derivative overlay source")
            frozen_derivative = (
                frozen / "derivatives" / cell.benchmark_id / f"{split_name}.csv"
            )
            _copy_once(source_derivative, frozen_derivative)
            derivative_rows.append(
                {
                    "benchmark_id": cell.benchmark_id,
                    "split": split_name,
                    "public_split_sha256": _sha256(public_split),
                    "derivative_overlay_sha256": _sha256(frozen_derivative),
                    "derivative_channels": expected["derivative_channels"],
                }
            )

    _write_once_jsonl(frozen / "candidate_manifest.jsonl", candidate_rows)
    _write_once_jsonl(frozen / "derivative_manifest.jsonl", derivative_rows)
    task_rows = []
    for task_index in range(reciprocal_fitting_task_count(plan)):
        condition, cell, repetition, candidate_index = (
            reciprocal_fitting_task_identity(plan, task_index)
        )
        task_rows.append(
            {
                "task_index": task_index,
                "condition_id": condition.condition_id,
                "benchmark_id": cell.benchmark_id,
                "tier": cell.tier,
                "repetition": repetition,
                "candidate_index": candidate_index,
            }
        )
    _write_once_jsonl(frozen / "task_plan.jsonl", task_rows)
    freeze = {
        "schema_version": (
            "phase-b-parameter-range-ownership-pilot-freeze-1"
            if range_ownership
            else "phase-b-reciprocal-fitting-pilot-freeze-1"
        ),
        "status": "frozen_before_fitting",
        "development_only": True,
        "plan_sha256": plan_sha256,
        "source_replay_manifest_sha256": _sha256(source_manifest_path),
        "source_replay_artifact_ledger_sha256": _sha256(source_ledger_path),
        "derivative_overlay_manifest_sha256": _sha256(overlay_manifest_path),
        "candidate_manifest_sha256": _sha256(frozen / "candidate_manifest.jsonl"),
        "derivative_manifest_sha256": _sha256(frozen / "derivative_manifest.jsonl"),
        "task_plan_sha256": _sha256(frozen / "task_plan.jsonl"),
        "candidate_count": len(candidate_rows),
        "task_count": len(task_rows),
        "exact_training_observed_derivatives_supplied": True,
        "validation_derivatives_supplied": False,
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


def _copy_once(source: Path, destination: Path) -> None:
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise ValueError(f"frozen pilot copy differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-replay-root", type=Path, required=True)
    parser.add_argument("--derivative-overlay-root", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = prepare_pilot(
        config_path=arguments.config,
        source_replay_root=arguments.source_replay_root,
        derivative_overlay_root=arguments.derivative_overlay_root,
        public_data_root=arguments.public_data_root,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
