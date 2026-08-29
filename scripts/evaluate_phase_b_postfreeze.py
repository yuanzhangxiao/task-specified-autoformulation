#!/usr/bin/env python3
"""Replay common frozen subjects on sealed Phase-B test trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from autoformalism.baselines.raw_data_agent import raw_agent_validation_context
from autoformalism.config import DataConfig
from autoformalism.data import (
    BenchmarkLoader,
    BenchmarkRegistry,
    DatasetSplit,
    DevelopmentDataset,
    FrozenTestAccess,
)
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import FitConfig
from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject
from autoformalism.rebuttal.postfreeze_evaluation import (
    PostFreezeEvaluationOutcome,
    evaluate_subject_on_test,
    outcome_for_subject,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--maximum-trajectory-wall-time-seconds",
        type=float,
        default=300.0,
    )
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must lie in [0, shard count)")
    if args.maximum_trajectory_wall_time_seconds <= 0.0:
        parser.error("maximum trajectory wall time must be positive")

    raw_subjects = args.subjects.read_bytes()
    subjects_sha256 = hashlib.sha256(raw_subjects).hexdigest()
    subjects = _read_subjects(raw_subjects)
    assigned = subjects[args.shard_index :: args.shard_count]
    fit_config = FitConfig(
        integration_backend="solve_ivp",
        allow_derivative_regression=False,
        relative_tolerance=1e-7,
        absolute_tolerance=1e-9,
        maximum_wall_time_seconds=args.maximum_trajectory_wall_time_seconds,
    )

    registry = BenchmarkRegistry()
    loader = BenchmarkLoader(registry)
    development_cache: dict[
        tuple[str, str], tuple[DevelopmentDataset, ValidationContext]
    ] = {}
    for subject in subjects:
        development, context = _development_for_subject(
            subject,
            public_data_root=args.public_data_root,
            registry=registry,
            loader=loader,
            cache=development_cache,
        )
        if context != subject.validation_context:
            raise ValueError(
                "serialized validation context differs from public data for "
                f"{subject.subject_id}"
            )
        if development.benchmark_id != subject.benchmark_id:
            raise ValueError(f"public benchmark identity differs: {subject.subject_id}")
    output_root = args.output_root.expanduser().resolve()
    shard_root = (
        output_root
        if args.shard_count == 1
        else output_root / "shards" / f"shard_{args.shard_index}"
    )
    shard_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output_root / "postfreeze_freeze_receipt.json"
    _write_or_validate_receipt(
        receipt_path,
        subjects_sha256=subjects_sha256,
        subject_count=len(subjects),
        maximum_trajectory_wall_time_seconds=(
            args.maximum_trajectory_wall_time_seconds
        ),
    )
    test_cache: dict[tuple[str, str], DatasetSplit] = {}
    updated_subjects: list[FrozenEvaluationSubject] = []
    outcomes: list[PostFreezeEvaluationOutcome] = []
    checkpoint_root = shard_root / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    for index, subject in enumerate(assigned, start=1):
        checkpoint = checkpoint_root / f"{subject.subject_id}.json"
        if checkpoint.is_file():
            updated, outcome = _load_checkpoint(checkpoint, subject)
        else:
            error: Exception | None = None
            try:
                development, context = development_cache[
                    (subject.benchmark_id, subject.tier)
                ]
                test = _test_for_subject(
                    subject,
                    public_data_root=args.public_data_root,
                    loader=loader,
                    selection_hash=subjects_sha256,
                    cache=test_cache,
                )
                updated = evaluate_subject_on_test(
                    subject,
                    training_split=development.train,
                    test_split=test,
                    fit_config=fit_config,
                )
            except Exception as exc:  # preserve a per-subject test failure
                error = exc
                updated = _failure_without_test_split(subject, exc)
            outcome = outcome_for_subject(updated, error=error)
            _write_checkpoint(checkpoint, updated, outcome)
        updated_subjects.append(updated)
        outcomes.append(outcome)
        print(
            f"postfreeze {index}/{len(assigned)} {subject.subject_id} "
            f"status={outcome.status} target={outcome.target_status}",
            flush=True,
        )

    subjects_name = (
        "postfreeze_subjects.jsonl"
        if args.shard_count == 1
        else "postfreeze_subjects.shard.jsonl"
    )
    outcomes_name = (
        "postfreeze_outcomes.jsonl"
        if args.shard_count == 1
        else "postfreeze_outcomes.shard.jsonl"
    )
    subjects_path = shard_root / subjects_name
    outcomes_path = shard_root / outcomes_name
    _write_jsonl(subjects_path, tuple(updated_subjects))
    _write_jsonl(outcomes_path, tuple(outcomes))
    manifest = {
        "schema_version": "phase-b-postfreeze-run-1",
        "status": "complete",
        "subjects_sha256": subjects_sha256,
        "subject_count": len(subjects),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_count": len(assigned),
        "target_available_count": sum(
            item.target_status == "available" for item in outcomes
        ),
        "target_failed_count": sum(item.target_status == "failed" for item in outcomes),
        "test_data_opened_after_freeze": True,
        "parameter_refit_applied": False,
        "trajectory_initial_refit_applied": False,
        "evaluation_protocol": "unseen_condition_free_rollout",
        "maximum_trajectory_wall_time_seconds": (
            args.maximum_trajectory_wall_time_seconds
        ),
        "postfreeze_subjects_sha256": _sha256(subjects_path),
        "postfreeze_outcomes_sha256": _sha256(outcomes_path),
    }
    _write_text_atomic(
        shard_root / "postfreeze_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )


def _read_subjects(raw: bytes) -> tuple[FrozenEvaluationSubject, ...]:
    subjects = tuple(
        FrozenEvaluationSubject.model_validate_json(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    )
    identifiers = [item.subject_id for item in subjects]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("frozen subject identifiers must be unique")
    if any(not item.selection_frozen for item in subjects):
        raise ValueError("every post-freeze subject must be frozen")
    return subjects


def _development_for_subject(
    subject: FrozenEvaluationSubject,
    *,
    public_data_root: Path,
    registry: BenchmarkRegistry,
    loader: BenchmarkLoader,
    cache: dict[
        tuple[str, str], tuple[DevelopmentDataset, ValidationContext]
    ],
) -> tuple[DevelopmentDataset, ValidationContext]:
    key = (subject.benchmark_id, subject.tier)
    if key not in cache:
        config = DataConfig(
            root=public_data_root.expanduser().resolve(),
            benchmark_id=subject.benchmark_id,
            tier=subject.tier,
        )
        development = loader.load_development(config)
        context = raw_agent_validation_context(
            development,
            registry.get(subject.benchmark_id),
        )
        cache[key] = (development, context)
    return cache[key]


def _test_for_subject(
    subject: FrozenEvaluationSubject,
    *,
    public_data_root: Path,
    loader: BenchmarkLoader,
    selection_hash: str,
    cache: dict[tuple[str, str], DatasetSplit],
) -> DatasetSplit:
    key = (subject.benchmark_id, subject.tier)
    if key not in cache:
        config = DataConfig(
            root=public_data_root.expanduser().resolve(),
            benchmark_id=subject.benchmark_id,
            tier=subject.tier,
        )
        cache[key] = loader.load_test(
            config,
            access=FrozenTestAccess(
                benchmark_id=subject.benchmark_id,
                tier=subject.tier,
                selection_hash=selection_hash,
            ),
        )
    return cache[key]


def _failure_without_test_split(
    subject: FrozenEvaluationSubject,
    error: Exception,
) -> FrozenEvaluationSubject:
    payload = subject.model_dump(mode="json")
    payload.update(
        {
            "private_metrics_opened_after_freeze": True,
            "target_prediction": {
                "status": "failed",
                "message": f"{type(error).__name__}: {error}",
            },
        }
    )
    return FrozenEvaluationSubject.model_validate(payload)


def _write_or_validate_receipt(
    path: Path,
    *,
    subjects_sha256: str,
    subject_count: int,
    maximum_trajectory_wall_time_seconds: float,
) -> None:
    receipt = {
        "schema_version": "phase-b-postfreeze-freeze-receipt-1",
        "subjects_sha256": subjects_sha256,
        "subject_count": subject_count,
        "all_subjects_validated_before_test_access": True,
        "all_public_contexts_validated_before_test_access": True,
        "maximum_trajectory_wall_time_seconds": (
            maximum_trajectory_wall_time_seconds
        ),
    }
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != receipt:
            raise ValueError("existing post-freeze receipt differs from inputs")
        return
    _write_text_atomic(path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _load_checkpoint(
    path: Path,
    original: FrozenEvaluationSubject,
) -> tuple[FrozenEvaluationSubject, PostFreezeEvaluationOutcome]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    updated = FrozenEvaluationSubject.model_validate(payload["subject"])
    outcome = PostFreezeEvaluationOutcome.model_validate(payload["outcome"])
    if (
        updated.subject_id != original.subject_id
        or outcome.subject_id != original.subject_id
        or updated.source_provenance.candidate_sha256
        != original.source_provenance.candidate_sha256
        or outcome.candidate_sha256 != original.source_provenance.candidate_sha256
    ):
        raise ValueError(f"checkpoint does not match subject {original.subject_id}")
    return updated, outcome


def _write_checkpoint(
    path: Path,
    subject: FrozenEvaluationSubject,
    outcome: PostFreezeEvaluationOutcome,
) -> None:
    _write_text_atomic(
        path,
        json.dumps(
            {
                "subject": subject.model_dump(mode="json"),
                "outcome": outcome.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
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


if __name__ == "__main__":
    main()
