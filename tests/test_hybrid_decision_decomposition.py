"""Tests for frozen hybrid-decision component analysis."""

from __future__ import annotations

import csv
import json
import sys

import pytest

from autoformalism.judging import HybridScoringConfig, score_hybrid_pair
from autoformalism.rebuttal.hybrid_labels import HybridCalibrationLabels
from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    CandidateAbsoluteAssessment,
    HybridJudgeResult,
    PairedAbsoluteAssessment,
    RelativeAssessment,
    RelativeCriterion,
    RelativeVerdict,
    RequirementRegistry,
)
from scripts.analyze_hybrid_decision_decomposition import (
    decompose_row,
    main,
)


def _paired(
    criterion: AbsoluteCriterion,
    left: AbsoluteVerdict,
    right: AbsoluteVerdict,
) -> PairedAbsoluteAssessment:
    return PairedAbsoluteAssessment(
        criterion=criterion,
        subject_id="candidate",
        candidate_a=CandidateAbsoluteAssessment(verdict=left, evidence="Left."),
        candidate_b=CandidateAbsoluteAssessment(verdict=right, evidence="Right."),
    )


def _fixture() -> tuple[
    dict[str, str],
    dict[str, HybridCalibrationLabels],
    HybridScoringConfig,
]:
    config = HybridScoringConfig()
    requirements = RequirementRegistry()
    deterministic = (
        _paired(
            AbsoluteCriterion.TASK_INPUTS_REACH_TARGETS,
            AbsoluteVerdict.PASS,
            AbsoluteVerdict.PASS,
        ),
    )
    result = HybridJudgeResult(
        absolute_assessments=(
            _paired(
                AbsoluteCriterion.SOURCE_ROLES_CONSISTENT,
                AbsoluteVerdict.PASS,
                AbsoluteVerdict.FAIL,
            ),
            _paired(
                AbsoluteCriterion.SINK_ROLES_CONSISTENT,
                AbsoluteVerdict.PASS,
                AbsoluteVerdict.PASS,
            ),
            _paired(
                AbsoluteCriterion.SEMANTIC_FLUXES_NOT_DUPLICATED,
                AbsoluteVerdict.PASS,
                AbsoluteVerdict.PASS,
            ),
        ),
        comparative_assessments=tuple(
            RelativeAssessment(
                criterion=criterion,
                verdict=RelativeVerdict.CANDIDATE_A,
                evidence="Candidate A is preferable.",
            )
            for criterion in RelativeCriterion
        ),
    )
    score = score_hybrid_pair(result, deterministic, requirements, config)
    row = {
        "pair_id": "pair_1",
        "mutation_type": "wrong_meal_sink",
        "judge_model": "vllm:test",
        "repetition": "0",
        "order": "baseline_a",
        "baseline_position": "A",
        "baseline_preference": "baseline",
        "baseline_decision_value": str(score.decision_value),
        "requirements": requirements.model_dump_json(),
        "deterministic_assessments": json.dumps(
            [item.model_dump(mode="json") for item in deterministic]
        ),
        "absolute_assessments": json.dumps(
            [item.model_dump(mode="json") for item in result.absolute_assessments]
        ),
        "comparative_assessments": json.dumps(
            [
                item.model_dump(mode="json")
                for item in result.comparative_assessments
            ]
        ),
    }
    labels = {
        "pair_1": HybridCalibrationLabels(
            pair_id="pair_1",
            overall_preference="baseline",
            absolute_labels=(),
            comparative_labels=(),
        )
    }
    return row, labels, config


def test_decomposition_reconstructs_additive_decision() -> None:
    row, labels, config = _fixture()

    result = decompose_row(row, labels=labels, config=config)

    assert result["absolute_delta_for_baseline"] == pytest.approx(
        0.4841269841269842
    )
    assert result["atomic_sensitive_marginal_for_baseline"] == pytest.approx(
        result["absolute_delta_for_baseline"]
    )
    assert result["other_absolute_delta_for_baseline"] == pytest.approx(0.0)
    assert result["baseline_comparative_preference"] == 1.0
    assert result["weighted_comparative_contribution_for_baseline"] == 0.25
    assert result["final_decision_for_baseline"] == pytest.approx(
        result["absolute_delta_for_baseline"] + 0.25
    )
    balance = next(
        item for item in result["groups"] if item["kind"] == "balance_semantics"
    )
    assert balance["complete_delta"] == 1.0
    assert balance["partial_delta"] == pytest.approx(1.0 / 3.0)


def test_cli_reports_failures_and_representative_example(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, labels, _config = _fixture()
    scores = tmp_path / "scores.csv"
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        labels["pair_1"].model_dump_json() + "\n", encoding="utf-8"
    )
    failures = tmp_path / "failures.jsonl"
    failures.write_text(
        json.dumps(
            {
                "judge_model": "vllm:test",
                "failure_stage": "atomic_evidence",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_hybrid_decision_decomposition.py",
            "--scores",
            str(scores),
            "--failures",
            str(failures),
            "--labels",
            str(labels_path),
            "--output",
            str(output),
        ],
    )

    main()

    metrics = json.loads(output.read_text(encoding="utf-8"))
    model = metrics["models"]["vllm:test"]
    assert model["response_success_rate"] == 0.5
    assert model["failure_stages"] == {"atomic_evidence": 1}
    assert model["final_accuracy"] == 1.0
    assert model["representative_examples"][0]["predicted_preference"] == "baseline"
    assert output.with_suffix(".md").is_file()
