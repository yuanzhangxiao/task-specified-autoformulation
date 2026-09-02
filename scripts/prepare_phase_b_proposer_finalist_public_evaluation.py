#!/usr/bin/env python3
"""Freeze the repaired-finalist public evaluation before numerical fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal.proposer_finalist_evaluation import (
    canonical_plan_sha256,
    finalist_task_count,
    load_proposer_finalist_evaluation_plan,
)


def main() -> None:
    """Validate declared inputs and create a write-once experiment freeze."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-replay-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-contract-root", type=Path, required=True)
    parser.add_argument("--mechanism-spec-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    plan = load_proposer_finalist_evaluation_plan(args.config)
    source_manifest = args.source_replay_root / "proposer_repair_replay.json"
    source_ledger = args.source_replay_root / "artifact_ledger.jsonl"
    source_payload = _read_object(source_manifest)
    expected_source = plan.source_replay
    for key, expected in {
        "schema_version": expected_source.manifest_schema_version,
        "status": expected_source.required_status,
        "source_plan_sha256": expected_source.source_plan_sha256,
        "replay_plan_sha256": expected_source.replay_plan_sha256,
        "artifact_ledger_sha256": expected_source.artifact_ledger_sha256,
        "replay_result_count": expected_source.replay_result_count,
    }.items():
        if source_payload.get(key) != expected:
            raise ValueError(f"source replay differs at {key}")
    if _sha256(source_ledger) != expected_source.artifact_ledger_sha256:
        raise ValueError("source replay artifact ledger SHA-256 differs")
    ledger_paths: set[str] = set()
    for line in source_ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source replay ledger path: {relative}")
        if str(relative) in ledger_paths:
            raise ValueError(f"duplicate source replay ledger path: {relative}")
        ledger_paths.add(str(relative))
        artifact = args.source_replay_root / relative
        if (
            _sha256(artifact) != item.get("sha256")
            or artifact.stat().st_size != item.get("size_bytes")
        ):
            raise ValueError(f"source replay artifact differs: {relative}")
    for directory in (
        args.data_root,
        args.target_contract_root / "specs",
        args.mechanism_spec_root / "specs",
    ):
        if not directory.is_dir():
            raise ValueError(f"required input directory is missing: {directory}")
    for cell in plan.cells:
        prompt = (
            args.data_root
            / "phase_b_v1"
            / cell.benchmark_id
            / "proposer_prompt.txt"
        )
        target = (
            args.target_contract_root / "specs" / f"{cell.benchmark_id}.json"
        )
        mechanism = (
            args.mechanism_spec_root / "specs" / f"{cell.benchmark_id}.json"
        )
        for path, expected, label in (
            (prompt, cell.public_prompt_sha256, "public proposer prompt"),
            (
                target,
                cell.public_target_contract_sha256,
                "public target contract",
            ),
            (
                mechanism,
                cell.public_mechanism_spec_sha256,
                "public mechanism specification",
            ),
        ):
            if _sha256(path) != expected:
                raise ValueError(
                    f"{label} SHA-256 differs: benchmark={cell.benchmark_id}"
                )

    frozen_plan = args.output_root / "frozen" / "plan.json"
    _write_once(frozen_plan, args.config.read_bytes())
    manifest = {
        "schema_version": "phase-b-proposer-finalist-public-evaluation-freeze-1",
        "status": "frozen_before_public_fits",
        "plan_sha256": canonical_plan_sha256(plan),
        "plan_file_sha256": _sha256(frozen_plan),
        "source_replay_manifest_sha256": _sha256(source_manifest),
        "source_replay_artifact_ledger_sha256": _sha256(source_ledger),
        "task_count": finalist_task_count(plan),
        "condition_count": len(plan.conditions),
        "matched_pair_count": len(plan.cells) * len(plan.repetitions),
        "development_only": True,
        "new_llm_calls_permitted": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "weighted_overall_score_defined": False,
        "automatic_operating_point_selection": False,
    }
    manifest_path = args.output_root / "frozen" / "freeze_manifest.json"
    _write_once(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _read_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"required source artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"frozen evaluation artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
