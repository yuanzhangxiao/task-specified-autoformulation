"""Tests for method adapters into the frozen final-evaluation contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.final_evaluation import (
    FrozenEvaluationSubject,
    FrozenParameterization,
    SourceArtifactProvenance,
    TargetPredictionEndpoint,
)
from autoformalism.rebuttal.final_evaluation_adapters import (
    SourceAdapterRequest,
    adapt_source,
    source_identity,
)
from autoformalism.schemas import CandidateModel
from scripts.export_phase_b_frozen_subjects import _read_requests


def _context() -> ValidationContext:
    return ValidationContext(targets=("x",), external_inputs=("u",))


def _candidate(*, ranged_initial: bool = False) -> CandidateModel:
    initial = (
        {
            "state": "x",
            "scope": "global",
            "initialization_range": {"lower": -1.0, "upper": 1.0},
        }
        if ranged_initial
        else {"state": "x", "scope": "global", "fixed_value": 0.0}
    )
    return CandidateModel.model_validate(
        {
            "candidate_id": "candidate",
            "parent_candidate_id": None,
            "states": [{"name": "x", "kind": "observed"}],
            "state_equations": [{"state": "x", "rhs": "u - k * x"}],
            "observation_mappings": [{"channel": "x", "expression": "x"}],
            "parameters": [
                {
                    "name": "k",
                    "scope": "global",
                    "bounds": {"lower": 0.0, "upper": 2.0},
                    "initialization_range": {"lower": 0.1, "upper": 1.0},
                }
            ],
            "initial_conditions": [initial],
        }
    )


def _baseline_payload(method: str) -> dict[str, object]:
    return {
        "method": method,
        "benchmark_id": "benchmark",
        "tier": "easy",
        "seed": 3,
        "equations": {"x": "u - 0.5 * x"},
        "selected_hyperparameters": {"threshold": 0.1},
        "training_normalized_mse": 0.1,
        "validation_normalized_mse": 0.2,
        "test_normalized_mse": 0.3,
        "test_per_target_normalized_mse": {"x": 0.3},
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_autoformalism_adapter_freezes_parameters_and_ignores_test(
    tmp_path: Path,
) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "status": "complete",
        "benchmark_id": "benchmark",
        "tier": "easy",
        "seed": 2,
        "selected_candidate": _candidate().model_dump(mode="json"),
        "final_global_parameters": {"k": 0.4},
        "final_global_initial_conditions": {},
        "test_normalized_mse": 0.01,
        "test_per_target_normalized_mse": {"x": 0.01},
    }
    _write_json(source, payload)

    subject = adapt_source(
        SourceAdapterRequest(
            request_id="auto-0", source_kind="autoformalism", source_path=source
        ),
        _context(),
    )

    assert subject.parameterization.status == "available"
    assert subject.parameterization.global_parameters == {"k": 0.4}
    assert subject.target_prediction.status == "missing"
    assert subject.private_metrics_opened_after_freeze is False
    assert subject.source_provenance.adapter == "autoformalism_summary"


def test_autoformalism_adapter_marks_legacy_missing_initial_as_partial(
    tmp_path: Path,
) -> None:
    source = tmp_path / "summary.json"
    _write_json(
        source,
        {
            "status": "complete",
            "benchmark_id": "benchmark",
            "tier": "easy",
            "seed": 0,
            "selected_candidate": _candidate(ranged_initial=True).model_dump(
                mode="json"
            ),
            "final_global_parameters": {"k": 0.4},
        },
    )

    subject = adapt_source(
        SourceAdapterRequest(
            request_id="auto-legacy",
            source_kind="autoformalism",
            source_path=source,
        ),
        _context(),
    )

    assert subject.parameterization.status == "partial"
    assert subject.parameterization.global_initial_conditions == {}


def test_raw_agent_adapter_uses_returned_fitted_parameters(tmp_path: Path) -> None:
    run = tmp_path / "raw-run"
    _write_json(
        run / "run_config.json",
        {
            "provider": "openai",
            "model": "gpt-test",
            "benchmark_id": "benchmark",
            "tier": "easy",
            "repetition": 4,
        },
    )
    _write_json(run / "candidate.json", _candidate().model_dump(mode="json"))
    _write_json(
        run / "evaluation.json",
        {
            "schema_version": "raw-data-agent-fitted-evaluation-1",
            "fitted_parameter_values": {"k": 0.75},
            "test_normalized_mse": 0.001,
        },
    )

    request = SourceAdapterRequest(
        request_id="raw-4", source_kind="raw_data_agent", source_path=run
    )
    assert source_identity(request) == ("benchmark", "easy", 4)
    subject = adapt_source(request, _context())

    assert subject.method == "raw_data_agent:openai:gpt-test"
    assert subject.parameterization.global_parameters == {"k": 0.75}
    assert subject.parameterization.status == "available"
    assert subject.target_prediction.status == "missing"
    assert set(subject.source_provenance.auxiliary_sha256) == {
        "candidate.json",
        "evaluation.json",
        "run_config.json",
    }


@pytest.mark.parametrize("method", ["sindy", "pysr"])
def test_symbolic_adapter_preserves_equation_but_not_embedded_test_metrics(
    method: str,
    tmp_path: Path,
) -> None:
    source = tmp_path / f"{method}.json"
    _write_json(source, _baseline_payload(method))

    subject = adapt_source(
        SourceAdapterRequest(request_id=method, source_kind=method, source_path=source),
        _context(),
    )

    assert subject.candidate.state_equations[0].rhs == "u - 0.5 * x"
    assert subject.parameterization.status == "not_required"
    assert subject.target_prediction.status == "missing"


def test_d3_adapter_uses_selected_checkpoint_generation(tmp_path: Path) -> None:
    source = tmp_path / "result.json"
    payload = _baseline_payload("d3_native_no_tools")
    payload["selected_hyperparameters"] = {"selected_generation": 7}
    _write_json(source, payload)
    _write_json(
        tmp_path / "d3_checkpoint.json",
        {
            "records": [
                {
                    "generation": 7,
                    "candidate": _candidate().model_dump(mode="json"),
                    "parameters": {"k": 0.6},
                }
            ]
        },
    )

    subject = adapt_source(
        SourceAdapterRequest(request_id="d3", source_kind="d3", source_path=source),
        _context(),
    )

    assert subject.parameterization.global_parameters == {"k": 0.6}
    assert subject.parameterization.status == "available"
    assert "d3_checkpoint.json" in subject.source_provenance.auxiliary_sha256


def test_frozen_subject_rejects_out_of_bounds_fitted_value() -> None:
    candidate = _candidate()
    provenance = SourceArtifactProvenance(
        adapter="direct_candidate",
        request_id="bad",
        source_path="candidate.json",
        source_sha256="a" * 64,
        candidate_sha256="b" * 64,
    )

    with pytest.raises(ValidationError, match="outside bounds"):
        FrozenEvaluationSubject(
            subject_id="bad",
            method="method",
            benchmark_id="benchmark",
            tier="easy",
            repetition=0,
            private_metrics_opened_after_freeze=False,
            source_provenance=provenance,
            candidate=candidate,
            parameterization=FrozenParameterization(
                status="available", global_parameters={"k": 3.0}
            ),
            validation_context=_context(),
            target_prediction=TargetPredictionEndpoint(status="missing"),
        )


def test_request_manifest_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    line = SourceAdapterRequest(
        request_id="same", source_kind="sindy", source_path=tmp_path / "result.json"
    ).model_dump_json()
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identifiers must be unique"):
        _read_requests(path)
