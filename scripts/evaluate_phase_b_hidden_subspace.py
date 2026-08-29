#!/usr/bin/env python3
"""Score frozen Phase-B mechanism-response subspaces against private truth."""

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
from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.rebuttal.phase_b_hidden_subspace import (
    PhaseBHiddenSubspaceContract,
    PhaseBHiddenSubspaceOutcome,
    evaluate_phase_b_hidden_subspace,
    failed_hidden_subspace_result,
    phase_b_hidden_subspace_contract,
    phase_b_reference_directions,
)

ENDPOINT_ID = "claimed_mechanism_response_subspace"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--private-data-root", type=Path, required=True)
    parser.add_argument("--mechanism-config-root", type=Path, required=True)
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
    registry = BenchmarkRegistry()
    loader = BenchmarkLoader(registry)
    development_cache: dict[
        tuple[str, str], tuple[DevelopmentDataset, ValidationContext]
    ] = {}
    mechanism_specs: dict[str, MechanismEvaluationSpec] = {}
    for subject in subjects:
        development, context = _development_for_subject(
            subject,
            public_data_root=args.public_data_root,
            registry=registry,
            loader=loader,
            cache=development_cache,
        )
        if development.benchmark_id != subject.benchmark_id:
            raise ValueError(f"public benchmark identity differs: {subject.subject_id}")
        if context != subject.validation_context:
            raise ValueError(
                "serialized validation context differs from public data for "
                f"{subject.subject_id}"
            )
        mechanism_specs[subject.benchmark_id] = _mechanism_spec(
            args.mechanism_config_root,
            subject,
        )

    output_root = args.output_root.expanduser().resolve()
    shard_root = (
        output_root
        if args.shard_count == 1
        else output_root / "shards" / f"shard_{args.shard_index}"
    )
    shard_root.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(
        output_root / "hidden_freeze_receipt.json",
        {
            "schema_version": "phase-b-hidden-freeze-receipt-1",
            "subjects_sha256": subjects_sha256,
            "subject_count": len(subjects),
            "all_public_contexts_validated_before_private_access": True,
            "maximum_trajectory_wall_time_seconds": (
                args.maximum_trajectory_wall_time_seconds
            ),
        },
    )

    contracts = {
        benchmark_id: phase_b_hidden_subspace_contract(
            benchmark_id,
            data_root=args.private_data_root.expanduser().resolve(),
        )
        for benchmark_id in sorted({item.benchmark_id for item in subjects})
    }
    for benchmark_id, contract in contracts.items():
        if (
            contract.public_prompt_sha256
            != mechanism_specs[benchmark_id].public_prompt_sha256
        ):
            raise ValueError(
                f"hidden/public prompt commitments differ: {benchmark_id}"
            )
    contracts_path = output_root / "hidden_subspace_contracts.jsonl"
    contracts_text = "".join(
        contracts[key].model_dump_json() + "\n" for key in sorted(contracts)
    )
    _write_or_validate_text(contracts_path, contracts_text)
    contracts_sha256 = _sha256(contracts_path)

    test_cache: dict[tuple[str, str], DatasetSplit] = {}
    reference_cache: dict[tuple[str, tuple[tuple[str, float], ...]], object] = {}
    updated_subjects: list[FrozenEvaluationSubject] = []
    outcomes: list[PhaseBHiddenSubspaceOutcome] = []
    checkpoint_root = shard_root / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    for index, subject in enumerate(assigned, start=1):
        checkpoint = checkpoint_root / f"{subject.subject_id}.json"
        contract = contracts[subject.benchmark_id]
        if checkpoint.is_file():
            updated, outcome = _load_checkpoint(checkpoint, subject, contract)
        else:
            try:
                development, _ = development_cache[
                    (subject.benchmark_id, subject.tier)
                ]
                test = _test_for_subject(
                    subject,
                    public_data_root=args.public_data_root,
                    loader=loader,
                    selection_hash=subjects_sha256,
                    cache=test_cache,
                )
                reference = None
                if contract.mode != "not_applicable":
                    reference_key = (
                        contract.sha256,
                        tuple(sorted(subject.target_prediction.normalization_scales.items())),
                    )
                    if reference_key not in reference_cache:
                        reference_cache[reference_key] = phase_b_reference_directions(
                            training_split=development.train,
                            test_split=test,
                            contract=contract,
                            normalization_scales=(
                                subject.target_prediction.normalization_scales
                            ),
                            private_data_root=args.private_data_root,
                        )
                    reference = reference_cache[reference_key]
                updated, outcome = evaluate_phase_b_hidden_subspace(
                    subject,
                    training_split=development.train,
                    test_split=test,
                    public_mechanism_spec=mechanism_specs[subject.benchmark_id],
                    contract=contract,
                    private_data_root=args.private_data_root,
                    maximum_trajectory_wall_time_seconds=(
                        args.maximum_trajectory_wall_time_seconds
                    ),
                    reference_directions=reference,  # type: ignore[arg-type]
                )
            except Exception as exc:
                updated, outcome = failed_hidden_subspace_result(
                    subject,
                    contract,
                    exc,
                )
            _write_checkpoint(checkpoint, updated, outcome)
        updated_subjects.append(updated)
        outcomes.append(outcome)
        print(
            f"hidden {index}/{len(assigned)} {subject.subject_id} "
            f"status={outcome.status}",
            flush=True,
        )

    subjects_name = (
        "hidden_subjects.jsonl"
        if args.shard_count == 1
        else "hidden_subjects.shard.jsonl"
    )
    outcomes_name = (
        "hidden_outcomes.jsonl"
        if args.shard_count == 1
        else "hidden_outcomes.shard.jsonl"
    )
    subjects_path = shard_root / subjects_name
    outcomes_path = shard_root / outcomes_name
    _write_jsonl(subjects_path, tuple(updated_subjects))
    _write_jsonl(outcomes_path, tuple(outcomes))
    manifest = {
        "schema_version": "phase-b-hidden-subspace-run-1",
        "status": "complete",
        "subjects_sha256": subjects_sha256,
        "contracts_sha256": contracts_sha256,
        "subject_count": len(subjects),
        "assigned_count": len(assigned),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "maximum_trajectory_wall_time_seconds": (
            args.maximum_trajectory_wall_time_seconds
        ),
        "available_count": sum(item.status == "available" for item in outcomes),
        "unrecovered_count": sum(item.status == "unrecovered" for item in outcomes),
        "not_applicable_count": sum(
            item.status == "not_applicable" for item in outcomes
        ),
        "failed_count": sum(item.status == "failed" for item in outcomes),
        "parameter_refit_applied": False,
        "alignment_fitted_on_training_only": True,
        "test_data_opened_after_freeze": True,
        "hidden_subjects_sha256": _sha256(subjects_path),
        "hidden_outcomes_sha256": _sha256(outcomes_path),
    }
    _write_text_atomic(
        shard_root / "hidden_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )


def _read_subjects(raw: bytes) -> tuple[FrozenEvaluationSubject, ...]:
    subjects = tuple(
        FrozenEvaluationSubject.model_validate_json(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    )
    if not subjects:
        raise ValueError("hidden evaluation subject manifest is empty")
    identifiers = [item.subject_id for item in subjects]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("hidden evaluation subject identifiers must be unique")
    if any(not item.selection_frozen for item in subjects):
        raise ValueError("every hidden evaluation subject must be frozen")
    if any(
        any(unit.mechanism_id == ENDPOINT_ID for unit in item.hidden_mechanisms)
        for item in subjects
    ):
        raise ValueError("hidden subspace endpoint is already populated")
    return subjects


def _development_for_subject(
    subject: FrozenEvaluationSubject,
    *,
    public_data_root: Path,
    registry: BenchmarkRegistry,
    loader: BenchmarkLoader,
    cache: dict[tuple[str, str], tuple[DevelopmentDataset, ValidationContext]],
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


def _mechanism_spec(
    root: Path,
    subject: FrozenEvaluationSubject,
) -> MechanismEvaluationSpec:
    path = root.expanduser().resolve() / f"{subject.benchmark_id}.json"
    spec = MechanismEvaluationSpec.model_validate_json(path.read_text(encoding="utf-8"))
    if (spec.benchmark_id, spec.tier) != (subject.benchmark_id, subject.tier):
        raise ValueError(f"public mechanism contract differs: {subject.subject_id}")
    return spec


def _load_checkpoint(
    path: Path,
    original: FrozenEvaluationSubject,
    contract: PhaseBHiddenSubspaceContract,
) -> tuple[FrozenEvaluationSubject, PhaseBHiddenSubspaceOutcome]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    updated = FrozenEvaluationSubject.model_validate(payload["subject"])
    outcome = PhaseBHiddenSubspaceOutcome.model_validate(payload["outcome"])
    if (
        updated.subject_id != original.subject_id
        or outcome.subject_id != original.subject_id
        or outcome.candidate_sha256 != original.source_provenance.candidate_sha256
        or outcome.contract_sha256 != contract.sha256
    ):
        raise ValueError(f"hidden checkpoint differs: {original.subject_id}")
    return updated, outcome


def _write_checkpoint(
    path: Path,
    subject: FrozenEvaluationSubject,
    outcome: PhaseBHiddenSubspaceOutcome,
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


def _write_or_validate_json(path: Path, payload: dict[str, object]) -> None:
    _write_or_validate_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_or_validate_text(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"existing frozen artifact differs: {path}")
        return
    _write_text_atomic(path, text)


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
