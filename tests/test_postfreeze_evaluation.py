"""Tests for method-neutral frozen-parameter held-out replay."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from autoformalism.baselines.raw_data_agent import raw_agent_validation_context
from autoformalism.benchmarks import (
    phase_b_protocols,
    phase_b_public_spec,
    simulate_phase_b,
    write_public_production_bundle,
)
from autoformalism.config import DataConfig
from autoformalism.data import (
    BenchmarkLoader,
    BenchmarkRegistry,
    DatasetSplit,
    SplitName,
    Trajectory,
)
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.final_evaluation import (
    FrozenEvaluationSubject,
    FrozenParameterization,
    SourceArtifactProvenance,
    TargetPredictionEndpoint,
)
from autoformalism.rebuttal.postfreeze_evaluation import (
    evaluate_subject_on_test,
    outcome_for_subject,
)
from autoformalism.schemas import CandidateModel


def _trajectory(identifier: str, values: tuple[float, ...]) -> Trajectory:
    array = np.asarray(values, dtype=float)
    return Trajectory(
        trajectory_id=identifier,
        time=np.arange(len(array), dtype=float),
        targets={"x": array},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={},
    )


def _splits() -> tuple[DatasetSplit, DatasetSplit]:
    training = DatasetSplit(
        SplitName.TRAIN,
        (
            _trajectory("train_0", (1.0, 1.0, 1.0)),
            _trajectory("train_1", (2.0, 2.0, 2.0)),
        ),
        "train-fingerprint",
    )
    test = DatasetSplit(
        SplitName.TEST,
        (
            _trajectory("test_0", (3.0, 3.0, 3.0)),
            _trajectory("test_1", (4.0, 4.0, 4.0)),
        ),
        "test-fingerprint",
    )
    return training, test


def _candidate(*, with_parameter: bool = False) -> CandidateModel:
    payload: dict[str, object] = {
        "candidate_id": "constant_state",
        "parent_candidate_id": None,
        "states": [{"name": "x", "kind": "observed"}],
        "state_equations": [
            {"state": "x", "rhs": "0.0" if not with_parameter else "-k * x"}
        ],
        "observation_mappings": [{"channel": "x", "expression": "x"}],
        "initial_conditions": [{"state": "x", "scope": "global", "expression": "x"}],
    }
    if with_parameter:
        payload["parameters"] = [
            {
                "name": "k",
                "scope": "global",
                "bounds": {"lower": 0.0, "upper": 2.0},
                "initialization_range": {"lower": 0.1, "upper": 1.0},
            }
        ]
    return CandidateModel.model_validate(payload)


def _subject(*, missing_parameter: bool = False) -> FrozenEvaluationSubject:
    candidate = _candidate(with_parameter=missing_parameter)
    candidate_hash = hashlib.sha256(
        candidate.model_dump_json().encode("utf-8")
    ).hexdigest()
    return FrozenEvaluationSubject(
        subject_id="subject",
        method="method",
        benchmark_id="benchmark",
        tier="easy",
        repetition=0,
        private_metrics_opened_after_freeze=False,
        source_provenance=SourceArtifactProvenance(
            adapter="direct_candidate",
            request_id="subject",
            source_path="candidate.json",
            source_sha256="a" * 64,
            candidate_sha256=candidate_hash,
        ),
        candidate=candidate,
        parameterization=FrozenParameterization(
            status="missing" if missing_parameter else "not_required"
        ),
        validation_context=ValidationContext(targets=("x",)),
        target_prediction=TargetPredictionEndpoint(status="missing"),
    )


def test_exact_frozen_free_rollout_populates_target_and_interventions() -> None:
    training, test = _splits()

    updated = evaluate_subject_on_test(
        _subject(), training_split=training, test_split=test
    )

    assert updated.private_metrics_opened_after_freeze is True
    assert updated.target_prediction.status == "available"
    assert updated.target_prediction.evaluation_protocol == (
        "unseen_condition_free_rollout"
    )
    assert updated.target_prediction.normalized_mse == 0.0
    assert updated.target_prediction.per_target_normalized_mse == {"x": 0.0}
    assert updated.target_prediction.normalization_scales == {"x": 0.5}
    assert updated.target_prediction.trajectory_count == 2
    assert updated.target_prediction.successful_trajectory_count == 2
    assert len(updated.interventions) == 2
    assert all(item.status == "available" for item in updated.interventions)
    assert all(item.target_nmse == 0.0 for item in updated.interventions)
    outcome = outcome_for_subject(updated)
    assert outcome.status == "complete"
    assert outcome.error is None


def test_incomplete_parameterization_fails_without_refitting() -> None:
    training, test = _splits()

    updated = evaluate_subject_on_test(
        _subject(missing_parameter=True),
        training_split=training,
        test_split=test,
    )

    assert updated.target_prediction.status == "failed"
    assert updated.target_prediction.normalized_mse is None
    assert updated.target_prediction.failed_trajectories == ("test_0", "test_1")
    assert all(item.status == "failed" for item in updated.interventions)
    outcome = outcome_for_subject(updated)
    assert outcome.status == "failed"
    assert outcome.error_type == "TargetPredictionFailure"
    assert "not replay-complete" in (outcome.error or "")


def test_free_rollout_rejects_non_test_split() -> None:
    training, _ = _splits()

    try:
        evaluate_subject_on_test(
            _subject(), training_split=training, test_split=training
        )
    except ValueError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("non-test split was accepted")


def test_postfreeze_cli_opens_test_only_after_frozen_manifest(
    tmp_path: Path,
) -> None:
    private_root = Path(__file__).parents[1] / "data_raw"
    spec = phase_b_public_spec(
        "dalla_man",
        "easy",
        "named",
        task="T1",
        data_root=private_root,
    )
    protocols = phase_b_protocols("dalla_man", task="T1")
    selected_protocols = tuple(
        next(item for item in protocols if item.split == split)
        for split in ("train", "validation", "test")
    )
    trajectories = tuple(
        simulate_phase_b(item, data_root=private_root) for item in selected_protocols
    )
    public_root = tmp_path / "public"
    write_public_production_bundle(
        public_root / "phase_b_v1" / spec.benchmark_id,
        spec,
        trajectories,
    )
    registry = BenchmarkRegistry()
    development = BenchmarkLoader(registry).load_development(
        DataConfig(
            root=public_root,
            benchmark_id=spec.benchmark_id,
            tier="easy",
        )
    )
    context = raw_agent_validation_context(development, registry.get(spec.benchmark_id))
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "constant_glucose",
            "parent_candidate_id": None,
            "states": [{"name": "Gp", "kind": "observed"}],
            "state_equations": [{"state": "Gp", "rhs": "0.0"}],
            "observation_mappings": [{"channel": "Gp", "expression": "Gp"}],
            "initial_conditions": [
                {"state": "Gp", "scope": "global", "expression": "Gp"}
            ],
        }
    )
    candidate_hash = hashlib.sha256(
        candidate.model_dump_json().encode("utf-8")
    ).hexdigest()
    subject = FrozenEvaluationSubject(
        subject_id="phase-b-subject",
        method="test",
        benchmark_id=spec.benchmark_id,
        tier="easy",
        repetition=0,
        private_metrics_opened_after_freeze=False,
        source_provenance=SourceArtifactProvenance(
            adapter="direct_candidate",
            request_id="phase-b-subject",
            source_path="candidate.json",
            source_sha256="a" * 64,
            candidate_sha256=candidate_hash,
        ),
        candidate=candidate,
        parameterization=FrozenParameterization(status="not_required"),
        validation_context=context,
        target_prediction=TargetPredictionEndpoint(status="missing"),
    )
    second_payload = subject.model_dump(mode="json")
    second_payload["subject_id"] = "phase-b-subject-2"
    second_payload["repetition"] = 1
    second_payload["source_provenance"]["request_id"] = "phase-b-subject-2"
    second = FrozenEvaluationSubject.model_validate(second_payload)
    subjects = tmp_path / "subjects.jsonl"
    subjects.write_text(
        subject.model_dump_json() + "\n" + second.model_dump_json() + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "postfreeze"
    repository = Path(__file__).parents[1]

    for shard_index in range(2):
        subprocess.run(
            [
                sys.executable,
                str(repository / "scripts/evaluate_phase_b_postfreeze.py"),
                "--subjects",
                str(subjects),
                "--public-data-root",
                str(public_root),
                "--output-root",
                str(output),
                "--shard-index",
                str(shard_index),
                "--shard-count",
                "2",
            ],
            cwd=repository,
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/merge_phase_b_postfreeze.py"),
            "--subjects",
            str(subjects),
            "--input-root",
            str(output),
            "--output-root",
            str(output),
        ],
        cwd=repository,
        check=True,
    )

    updated = tuple(
        FrozenEvaluationSubject.model_validate_json(line)
        for line in (output / "postfreeze_subjects.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    manifest = json.loads(
        (output / "postfreeze_merge_manifest.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (output / "postfreeze_freeze_receipt.json").read_text(encoding="utf-8")
    )
    assert [item.subject_id for item in updated] == [
        "phase-b-subject",
        "phase-b-subject-2",
    ]
    assert all(item.target_prediction.status == "available" for item in updated)
    assert all(item.target_prediction.trajectory_count == 1 for item in updated)
    assert manifest["subject_count"] == 2
    assert manifest["target_available_count"] == 2
    assert manifest["parameter_refit_applied"] is False
    assert manifest["trajectory_initial_refit_applied"] is False
    assert receipt["all_subjects_validated_before_test_access"] is True
    assert receipt["all_public_contexts_validated_before_test_access"] is True

    shard_subjects = (
        output / "shards" / "shard_0" / "postfreeze_subjects.shard.jsonl"
    )
    shard_subjects.write_text(
        shard_subjects.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    tampered = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/merge_phase_b_postfreeze.py"),
            "--subjects",
            str(subjects),
            "--input-root",
            str(output),
            "--output-root",
            str(tmp_path / "tampered-merge"),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tampered.returncode != 0
    assert "shard subject hash differs" in tampered.stderr
