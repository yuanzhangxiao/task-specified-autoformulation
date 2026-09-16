"""Replay selected baseline equations without a train-plus-validation refit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.final_evaluation_adapters import (
    SourceAdapterRequest,
    adapt_source,
)


def _payload(method: str) -> dict:
    """Represent a completed, training-fitted development selection."""
    return BaselineDevelopmentResult(
        method=method,
        benchmark_id="fixture",
        tier="easy",
        seed=0,
        equations={"y": "u - 0.5*y"},
        selected_hyperparameters={"threshold": 0.1},
        training_normalized_mse=0.2,
        validation_normalized_mse=0.3,
    ).model_dump(mode="json")


def _adapt(tmp_path: Path, payload: dict, method: str):
    """Load the same exact input file used for source hashing."""
    path = tmp_path / "development.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    request = SourceAdapterRequest(
        request_id=method, source_kind=method, source_path=path
    )
    return adapt_source(
        request, ValidationContext(targets=("y",), external_inputs=("u",))
    )


@pytest.mark.parametrize("method", ["sindy", "pysr"])
def test_preserve_development_equations_and_source_without_final_refit(
    tmp_path, method
):
    subject = _adapt(tmp_path, _payload(method), method)

    assert subject.candidate.state_equations[0].rhs == "u - 0.5*y"
    assert subject.candidate.initial_conditions[0].expression == "y"
    assert subject.parameterization.status == "not_required"
    assert subject.target_prediction.status == "missing"
    assert not subject.private_metrics_opened_after_freeze
    assert (
        subject.source_provenance.source_sha256
        == hashlib.sha256((tmp_path / "development.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"test_data_opened": True},
        {"test_normalized_mse": 0.001},
        {"status": "failed"},
    ],
)
def test_reject_test_contamination_and_incomplete_development(tmp_path, mutation):
    with pytest.raises(ValidationError):
        _adapt(tmp_path, {**_payload("sindy"), **mutation}, "sindy")


def test_reject_wrong_baseline_identity(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        _adapt(tmp_path, _payload("pysr"), "sindy")


def test_reject_wrong_target_set(tmp_path):
    with pytest.raises(ValueError, match="differ from public targets"):
        _adapt(tmp_path, {**_payload("sindy"), "equations": {"z": "-z"}}, "sindy")
