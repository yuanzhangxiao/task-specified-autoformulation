"""Tests for the non-scalar frozen Phase-B evaluation contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.final_evaluation import (
    FrozenEvaluationSubject,
    FrozenParameterization,
    HiddenMechanismEndpoint,
    InterventionEndpoint,
    QualitativeLLMEndpoint,
    SourceArtifactProvenance,
    TargetPredictionEndpoint,
    certify_runtime_validity,
    evaluate_frozen_subject,
    evaluation_summary,
)
from autoformalism.rebuttal.final_evaluation_adapters import SourceAdapterOutcome
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.schemas import CandidateModel


def _candidate(*, dynamic_memory: bool = True) -> CandidateModel:
    states = [
        {
            "name": "memory",
            "kind": "latent" if dynamic_memory else "observed",
            "mechanisms": ["input_memory"],
        },
        {"name": "target", "kind": "observed"},
    ]
    return CandidateModel.model_validate(
        {
            "candidate_id": "candidate",
            "parent_candidate_id": None,
            "states": states,
            "processes": [],
            "state_equations": [
                {"state": "memory", "rhs": "input_u - memory"},
                {"state": "target", "rhs": "memory - target"},
            ],
            "observation_mappings": [{"channel": "target", "expression": "target"}],
            "initial_conditions": [
                {"state": "memory", "scope": "global", "fixed_value": 0.0},
                {"state": "target", "scope": "global", "fixed_value": 0.0},
            ],
        }
    )


def _spec(*, signed: bool = False) -> MechanismEvaluationSpec:
    return MechanismEvaluationSpec.model_validate(
        {
            "benchmark_id": "synthetic",
            "tier": "hard",
            "required_mechanisms": [
                {
                    "id": "input_memory",
                    "required_drivers": ["input_u"],
                    "required_targets": ["target"],
                    "requires_dynamic_memory": True,
                    "required_sign": "positive" if signed else "unspecified",
                }
            ],
        }
    )


def _subject(*, private_opened: bool = True) -> FrozenEvaluationSubject:
    candidate = _candidate()
    candidate_sha256 = hashlib.sha256(
        candidate.model_dump_json().encode("utf-8")
    ).hexdigest()
    return FrozenEvaluationSubject(
        subject_id="subject",
        method="method",
        benchmark_id="synthetic",
        tier="hard",
        repetition=0,
        private_metrics_opened_after_freeze=private_opened,
        source_provenance=SourceArtifactProvenance(
            adapter="direct_candidate",
            request_id="subject",
            source_path="candidate.json",
            source_sha256=candidate_sha256,
            candidate_sha256=candidate_sha256,
        ),
        candidate=candidate,
        parameterization=FrozenParameterization(status="not_required"),
        validation_context=ValidationContext(
            targets=("target",),
            external_inputs=("input_u",),
        ),
        target_prediction=TargetPredictionEndpoint(
            status="available",
            normalized_mse=0.2,
            per_target_normalized_mse={"target": 0.2},
        ),
        hidden_mechanisms=(
            HiddenMechanismEndpoint(
                mechanism_id="input_memory",
                status="available",
                recovered=True,
                aligned_test_nmse=0.1,
            ),
        ),
        interventions=(
            InterventionEndpoint(
                case_id="shift",
                status="available",
                target_nmse=0.3,
                response_direction_correct=True,
            ),
        ),
        qualitative_llm=QualitativeLLMEndpoint(
            status="available",
            protocol="paired-question-consensus",
            requested_calls=2,
            successful_calls=2,
            pass_count=3,
        ),
    )


def test_mechanism_compliance_is_conjunctive_per_requirement() -> None:
    result = evaluate_mechanisms(_candidate(dynamic_memory=False), _spec())

    assert result.mechanism_coverage == 1.0
    assert result.structural_validity == pytest.approx(3 / 4)
    assert result.graph_mechanism_compliance == 0.0
    assert result.mechanism_annotation_compliance == 0.0
    assert result.mechanism_compliance == 0.0
    assert result.mechanism_compliance_complete is True
    assert result.mechanism_results[0].status == "failed"


def test_graph_compliance_does_not_require_proposer_annotation() -> None:
    payload = _candidate().model_dump(mode="json")
    payload["states"][0]["mechanisms"] = []

    result = evaluate_mechanisms(CandidateModel.model_validate(payload), _spec())

    assert result.graph_mechanism_compliance == 1.0
    assert result.mechanism_compliance == 1.0
    assert result.mechanism_annotation_compliance == 0.0
    assert result.graph_compliant_mechanisms == ("input_memory",)
    assert result.annotation_compliant_mechanisms == ()
    assert result.annotation_repairs[0].status == "unambiguous"
    assert result.annotation_repairs[0].suggested_components == ("memory",)


def test_public_mechanism_endpoint_retains_specification_provenance() -> None:
    payload = _spec().model_dump(mode="json")
    payload["source"] = "public_prompt"
    payload["public_prompt_sha256"] = "a" * 64
    payload["required_mechanisms"][0]["public_requirement"] = "input memory"

    record = evaluate_frozen_subject(
        _subject(), MechanismEvaluationSpec.model_validate(payload)
    )

    assert record.public_mechanism.specification_source == "public_prompt"
    assert record.public_mechanism.public_prompt_sha256 == "a" * 64


def test_uncertified_required_sign_is_not_counted_as_compliant() -> None:
    result = evaluate_mechanisms(_candidate(), _spec(signed=True))

    assert result.structural_validity == 1.0
    assert result.mechanism_compliance == 0.0
    assert result.mechanism_compliance_complete is False
    assert result.mechanism_results[0].status == "ambiguous"


def test_compliance_averages_small_conjunctive_groups() -> None:
    payload = _spec().model_dump(mode="json")
    payload["required_mechanisms"].append(
        {
            "id": "missing_mechanism",
            "required_drivers": ["other_input"],
            "required_targets": ["target"],
            "requires_dynamic_memory": False,
        }
    )

    result = evaluate_mechanisms(
        _candidate(), MechanismEvaluationSpec.model_validate(payload)
    )

    assert result.mechanism_compliance == 0.5
    assert result.compliant_mechanisms == ("input_memory",)


def test_compliance_requires_a_path_to_every_requested_target() -> None:
    payload = _spec().model_dump(mode="json")
    payload["required_mechanisms"][0]["required_targets"] = [
        "target",
        "other_target",
    ]

    result = evaluate_mechanisms(
        _candidate(), MechanismEvaluationSpec.model_validate(payload)
    )

    assert result.mechanism_coverage == 1.0
    assert result.mechanism_compliance == 0.0
    group = result.mechanism_results[0]
    assert group.status == "failed"
    assert any(
        item.predicate == "target_path:other_target" and item.status == "failed"
        for item in group.predicates
    )


def test_private_metrics_require_frozen_postselection_access() -> None:
    with pytest.raises(ValidationError, match="post-freeze"):
        _subject(private_opened=False)


def test_runtime_validity_is_computed_from_candidate_and_public_context() -> None:
    valid = certify_runtime_validity(
        _candidate(),
        ValidationContext(targets=("target",), external_inputs=("input_u",)),
    )
    invalid = certify_runtime_validity(
        _candidate(),
        ValidationContext(targets=("other",), external_inputs=("input_u",)),
    )

    assert valid.valid is True
    assert valid.failures == ()
    assert invalid.valid is False
    assert any("MISSING_OBSERVATION_MAPPING" in item for item in invalid.failures)


def test_available_target_metrics_must_cover_exact_public_targets() -> None:
    payload = _subject().model_dump(mode="json")
    payload["target_prediction"]["per_target_normalized_mse"] = {}

    with pytest.raises(ValidationError, match="differ from public targets"):
        FrozenEvaluationSubject.model_validate(payload)


def test_runtime_invalid_candidate_is_rejected_before_mechanism_scoring() -> None:
    payload = _subject().model_dump(mode="json")
    payload["validation_context"]["targets"] = ["other"]
    payload["target_prediction"] = {"status": "failed"}
    payload["hidden_mechanisms"] = []
    payload["interventions"] = []
    subject = FrozenEvaluationSubject.model_validate(payload)

    record = evaluate_frozen_subject(subject, _spec())

    assert record.runtime.valid is False
    assert record.public_mechanism.status == "invalid_runtime"
    assert record.public_mechanism.evaluation is None


def test_runtime_invalid_candidate_cannot_carry_private_scores() -> None:
    payload = _subject().model_dump(mode="json")
    payload["validation_context"]["targets"] = ["other"]
    payload["target_prediction"]["per_target_normalized_mse"] = {"other": 0.2}
    subject = FrozenEvaluationSubject.model_validate(payload)

    with pytest.raises(ValueError, match="cannot carry available private metrics"):
        evaluate_frozen_subject(subject, _spec())


def test_final_record_keeps_metrics_separate_without_overall_score() -> None:
    record = evaluate_frozen_subject(_subject(), _spec())
    summary = evaluation_summary((record,))

    assert record.public_mechanism.evaluation is not None
    assert record.public_mechanism.evaluation.mechanism_compliance == 1.0
    assert record.target_prediction.normalized_mse == 0.2
    assert record.complexity.additive_term_count == 4
    assert summary["mean_target_test_nmse"] == 0.2
    assert summary["mean_public_mechanism_compliance"] == 1.0
    assert summary["hidden_mechanism_recovery_rate"] == 1.0
    assert summary["mean_hidden_nmse_conditional_on_recovery"] == 0.1
    assert summary["replay_complete_rate"] == 1.0
    assert summary["parameterization_status_counts"]["not_required"] == 1
    assert summary["qualitative_llm_requested_calls"] == 2
    assert "overall_score" not in summary


def test_missing_public_contract_is_explicit() -> None:
    record = evaluate_frozen_subject(_subject(), None)

    assert record.public_mechanism.status == "missing"
    assert record.public_mechanism.evaluation is None


def test_final_evaluation_cli_writes_separate_endpoint_artifacts(
    tmp_path: Path,
) -> None:
    subjects = tmp_path / "subjects.jsonl"
    subjects.write_text(_subject().model_dump_json() + "\n", encoding="utf-8")
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "synthetic_hard.json").write_text(
        _spec().model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    outcomes = tmp_path / "outcomes.jsonl"
    outcome_rows = (
        SourceAdapterOutcome(
            request_id="subject",
            source_kind="autoformalism",
            source_path="summary.json",
            status="adapted",
            subject_id="subject",
        ),
        SourceAdapterOutcome(
            request_id="failed",
            source_kind="raw_data_agent",
            source_path="failed-run",
            status="failed",
            error_type="ValueError",
            error="no candidate",
        ),
    )
    outcomes.write_text(
        "".join(item.model_dump_json() + "\n" for item in outcome_rows),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    repository = Path(__file__).parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/assemble_phase_b_final_evaluation.py"),
            "--subjects",
            str(subjects),
            "--mechanism-config-root",
            str(specs),
            "--source-outcomes",
            str(outcomes),
            "--output-root",
            str(output),
        ],
        cwd=repository,
        check=True,
    )

    summary = json.loads(
        (output / "final_evaluation_summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output / "final_evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert summary["mean_public_mechanism_compliance"] == 1.0
    assert summary["source_completion_rate"] == 0.5
    assert "overall_score" not in summary
    assert manifest["weighted_overall_score_defined"] is False
    assert (output / "final_evaluation_metrics.csv").is_file()
    assert (output / "final_evaluation_records.jsonl").is_file()


def test_structure_only_cli_computes_runtime_instead_of_accepting_a_flag(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(_candidate().model_dump_json(), encoding="utf-8")
    context = tmp_path / "context.json"
    context.write_text(
        ValidationContext(
            targets=("other",), external_inputs=("input_u",)
        ).model_dump_json(),
        encoding="utf-8",
    )
    output = tmp_path / "record.json"
    repository = Path(__file__).parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/evaluate_candidate_deterministically.py"),
            "--candidate",
            str(candidate),
            "--validation-context",
            str(context),
            "--subject-id",
            "subject",
            "--method",
            "method",
            "--benchmark-id",
            "synthetic",
            "--tier",
            "hard",
            "--repetition",
            "0",
            "--output",
            str(output),
        ],
        cwd=repository,
        check=True,
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["runtime"]["valid"] is False
    assert record["public_mechanism"]["status"] == "invalid_runtime"
    assert record["target_prediction"]["status"] == "missing"
