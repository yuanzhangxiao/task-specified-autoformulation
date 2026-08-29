#!/usr/bin/env python3
"""Merge private hidden-subspace shards in original subject order."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject
from autoformalism.rebuttal.phase_b_hidden_subspace import (
    PhaseBHiddenSubspaceOutcome,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    raw = args.subjects.read_bytes()
    subjects_sha256 = hashlib.sha256(raw).hexdigest()
    original = tuple(
        FrozenEvaluationSubject.model_validate_json(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    )
    if not original:
        raise ValueError("original hidden-evaluation subjects are empty")
    original_ids = [item.subject_id for item in original]
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("original hidden-evaluation identifiers must be unique")

    updated_by_id: dict[str, FrozenEvaluationSubject] = {}
    outcome_by_id: dict[str, PhaseBHiddenSubspaceOutcome] = {}
    manifests: list[dict[str, object]] = []
    shard_roots = sorted((args.input_root / "shards").glob("shard_*"))
    if not shard_roots:
        raise ValueError("no hidden-subspace shard directories found")
    for shard_root in shard_roots:
        manifest = json.loads(
            (shard_root / "hidden_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("subjects_sha256") != subjects_sha256:
            raise ValueError(f"hidden shard input hash differs: {shard_root}")
        subjects_path = shard_root / "hidden_subjects.shard.jsonl"
        outcomes_path = shard_root / "hidden_outcomes.shard.jsonl"
        if manifest.get("hidden_subjects_sha256") != _sha256(subjects_path):
            raise ValueError(f"hidden shard subject hash differs: {shard_root}")
        if manifest.get("hidden_outcomes_sha256") != _sha256(outcomes_path):
            raise ValueError(f"hidden shard outcome hash differs: {shard_root}")
        subjects = tuple(
            FrozenEvaluationSubject.model_validate_json(line)
            for line in subjects_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        outcomes = tuple(
            PhaseBHiddenSubspaceOutcome.model_validate_json(line)
            for line in outcomes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if len(subjects) != int(manifest.get("assigned_count", -1)):
            raise ValueError(f"hidden shard assigned count differs: {shard_root}")
        if {item.subject_id for item in subjects} != {
            item.subject_id for item in outcomes
        }:
            raise ValueError(f"hidden shard subjects/outcomes differ: {shard_root}")
        for subject in subjects:
            if subject.subject_id in updated_by_id:
                raise ValueError(f"duplicate hidden subject: {subject.subject_id}")
            updated_by_id[subject.subject_id] = subject
        for outcome in outcomes:
            outcome_by_id[outcome.subject_id] = outcome
        manifests.append(manifest)

    shard_counts = {int(item["shard_count"]) for item in manifests}
    if len(shard_counts) != 1:
        raise ValueError("hidden shards disagree on shard count")
    shard_count = shard_counts.pop()
    shard_indices = {int(item["shard_index"]) for item in manifests}
    if shard_indices != set(range(shard_count)):
        raise ValueError(
            f"hidden shards are incomplete: expected={list(range(shard_count))}, "
            f"actual={sorted(shard_indices)}"
        )
    contracts = {str(item["contracts_sha256"]) for item in manifests}
    wall_times = {
        float(item["maximum_trajectory_wall_time_seconds"])
        for item in manifests
    }
    if len(contracts) != 1 or len(wall_times) != 1:
        raise ValueError("hidden shards disagree on frozen evaluation configuration")
    if set(updated_by_id) != set(original_ids):
        raise ValueError(
            "merged hidden subjects differ from frozen input; "
            f"missing={sorted(set(original_ids) - set(updated_by_id))}, "
            f"extra={sorted(set(updated_by_id) - set(original_ids))}"
        )

    original_by_id = {item.subject_id: item for item in original}
    for identifier, updated in updated_by_id.items():
        frozen = original_by_id[identifier]
        if _immutable_subject_payload(updated) != _immutable_subject_payload(frozen):
            raise ValueError(f"hidden evaluation changed frozen fields: {identifier}")
        if updated.hidden_mechanisms[: len(frozen.hidden_mechanisms)] != (
            frozen.hidden_mechanisms
        ):
            raise ValueError(
                f"hidden evaluation changed existing endpoints: {identifier}"
            )
        if (
            outcome_by_id[identifier].candidate_sha256
            != frozen.source_provenance.candidate_sha256
        ):
            raise ValueError(f"hidden outcome candidate differs: {identifier}")

    updated = tuple(updated_by_id[identifier] for identifier in original_ids)
    outcomes = tuple(outcome_by_id[identifier] for identifier in original_ids)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    subjects_path = output_root / "hidden_subjects.jsonl"
    outcomes_path = output_root / "hidden_outcomes.jsonl"
    _write_jsonl(subjects_path, updated)
    _write_jsonl(outcomes_path, outcomes)
    manifest = {
        "schema_version": "phase-b-hidden-subspace-merge-1",
        "status": "complete",
        "subjects_sha256": subjects_sha256,
        "contracts_sha256": contracts.pop(),
        "subject_count": len(original),
        "shard_count": shard_count,
        "maximum_trajectory_wall_time_seconds": wall_times.pop(),
        "available_count": sum(item.status == "available" for item in outcomes),
        "unrecovered_count": sum(item.status == "unrecovered" for item in outcomes),
        "not_applicable_count": sum(
            item.status == "not_applicable" for item in outcomes
        ),
        "failed_count": sum(item.status == "failed" for item in outcomes),
        "parameter_refit_applied": False,
        "alignment_fitted_on_training_only": True,
        "hidden_subjects_sha256": _sha256(subjects_path),
        "hidden_outcomes_sha256": _sha256(outcomes_path),
    }
    _write_text_atomic(
        output_root / "hidden_merge_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"merged {len(updated)} hidden subjects; "
        f"available={manifest['available_count']} "
        f"unrecovered={manifest['unrecovered_count']} "
        f"not_applicable={manifest['not_applicable_count']} "
        f"failed={manifest['failed_count']}"
    )


def _immutable_subject_payload(
    subject: FrozenEvaluationSubject,
) -> dict[str, object]:
    payload = subject.model_dump(mode="json")
    payload.pop("private_metrics_opened_after_freeze")
    payload.pop("hidden_mechanisms")
    return payload


def _write_jsonl(path: Path, values: tuple[object, ...]) -> None:
    _write_text_atomic(
        path,
        "".join(value.model_dump_json() + "\n" for value in values),  # type: ignore[attr-defined]
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
