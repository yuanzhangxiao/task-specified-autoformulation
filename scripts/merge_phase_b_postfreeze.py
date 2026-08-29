#!/usr/bin/env python3
"""Merge deterministic post-freeze shards in original subject order."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject
from autoformalism.rebuttal.postfreeze_evaluation import PostFreezeEvaluationOutcome


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
        raise ValueError("original frozen subject manifest is empty")
    original_ids = [item.subject_id for item in original]
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("original frozen subject identifiers must be unique")
    updated_by_id: dict[str, FrozenEvaluationSubject] = {}
    outcome_by_id: dict[str, PostFreezeEvaluationOutcome] = {}
    manifests = []
    shard_roots = sorted((args.input_root / "shards").glob("shard_*"))
    if not shard_roots:
        raise ValueError("no post-freeze shard directories found")
    for shard_root in shard_roots:
        manifest = json.loads(
            (shard_root / "postfreeze_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("subjects_sha256") != subjects_sha256:
            raise ValueError(f"shard input hash differs: {shard_root}")
        subjects_path = shard_root / "postfreeze_subjects.shard.jsonl"
        outcomes_path = shard_root / "postfreeze_outcomes.shard.jsonl"
        if manifest.get("postfreeze_subjects_sha256") != _sha256(subjects_path):
            raise ValueError(f"shard subject hash differs: {shard_root}")
        if manifest.get("postfreeze_outcomes_sha256") != _sha256(outcomes_path):
            raise ValueError(f"shard outcome hash differs: {shard_root}")
        manifests.append(manifest)
        subjects = tuple(
            FrozenEvaluationSubject.model_validate_json(line)
            for line in subjects_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        outcomes = tuple(
            PostFreezeEvaluationOutcome.model_validate_json(line)
            for line in outcomes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if len(subjects) != int(manifest.get("assigned_count", -1)):
            raise ValueError(f"shard assigned count differs: {shard_root}")
        if {item.subject_id for item in subjects} != {
            item.subject_id for item in outcomes
        }:
            raise ValueError(f"shard subjects/outcomes differ: {shard_root}")
        for subject in subjects:
            if subject.subject_id in updated_by_id:
                raise ValueError(f"duplicate post-freeze subject: {subject.subject_id}")
            updated_by_id[subject.subject_id] = subject
        for outcome in outcomes:
            outcome_by_id[outcome.subject_id] = outcome
    if len({int(item["shard_count"]) for item in manifests}) != 1:
        raise ValueError("post-freeze shards disagree on shard count")
    wall_times = {
        float(item["maximum_trajectory_wall_time_seconds"])
        for item in manifests
    }
    if len(wall_times) != 1:
        raise ValueError("post-freeze shards disagree on trajectory wall time")
    shard_count = int(manifests[0]["shard_count"])
    shard_indices = {int(item["shard_index"]) for item in manifests}
    if shard_indices != set(range(shard_count)):
        raise ValueError(
            f"post-freeze shards are incomplete: expected={list(range(shard_count))}, "
            f"actual={sorted(shard_indices)}"
        )
    if set(updated_by_id) != set(original_ids):
        raise ValueError(
            "merged post-freeze subjects differ from frozen input; "
            f"missing={sorted(set(original_ids) - set(updated_by_id))}, "
            f"extra={sorted(set(updated_by_id) - set(original_ids))}"
        )
    original_by_id = {item.subject_id: item for item in original}
    for identifier, updated_subject in updated_by_id.items():
        frozen_subject = original_by_id[identifier]
        if _immutable_subject_payload(updated_subject) != _immutable_subject_payload(
            frozen_subject
        ):
            raise ValueError(
                f"post-freeze evaluation changed frozen subject fields: {identifier}"
            )
        original_interventions = frozen_subject.interventions
        if updated_subject.interventions[: len(original_interventions)] != (
            original_interventions
        ):
            raise ValueError(
                f"post-freeze evaluation changed existing interventions: {identifier}"
            )
        outcome = outcome_by_id[identifier]
        if (
            outcome.candidate_sha256
            != frozen_subject.source_provenance.candidate_sha256
        ):
            raise ValueError(f"post-freeze outcome candidate differs: {identifier}")
    updated = tuple(updated_by_id[identifier] for identifier in original_ids)
    outcomes = tuple(outcome_by_id[identifier] for identifier in original_ids)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    subjects_path = output_root / "postfreeze_subjects.jsonl"
    outcomes_path = output_root / "postfreeze_outcomes.jsonl"
    _write_jsonl(subjects_path, updated)
    _write_jsonl(outcomes_path, outcomes)
    manifest = {
        "schema_version": "phase-b-postfreeze-merge-1",
        "status": "complete",
        "subjects_sha256": subjects_sha256,
        "subject_count": len(original),
        "shard_count": shard_count,
        "target_available_count": sum(
            item.target_status == "available" for item in outcomes
        ),
        "target_failed_count": sum(item.target_status == "failed" for item in outcomes),
        "test_data_opened_after_freeze": True,
        "parameter_refit_applied": False,
        "trajectory_initial_refit_applied": False,
        "evaluation_protocol": "unseen_condition_free_rollout",
        "maximum_trajectory_wall_time_seconds": wall_times.pop(),
        "postfreeze_subjects_sha256": _sha256(subjects_path),
        "postfreeze_outcomes_sha256": _sha256(outcomes_path),
    }
    _write_text_atomic(
        output_root / "postfreeze_merge_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"merged {len(updated)} post-freeze subjects; "
        f"available={manifest['target_available_count']} "
        f"failed={manifest['target_failed_count']}"
    )


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


def _immutable_subject_payload(
    subject: FrozenEvaluationSubject,
) -> dict[str, object]:
    payload = subject.model_dump(mode="json")
    for field in (
        "private_metrics_opened_after_freeze",
        "target_prediction",
        "interventions",
    ):
        payload.pop(field)
    return payload


if __name__ == "__main__":
    main()
