"""Native D3 scoring in the frozen test pipeline.

D3 learns `x_next = x + f` with no dt multiplier. Handing that expression to an
ODE solver silently reinterprets it, so the frozen stage routes on declared
execution semantics and labels the result with its own protocol.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.rebuttal.final_evaluation import (
    FrozenEvaluationSubject,
)
from autoformalism.rebuttal.final_evaluation_adapters import (
    SourceAdapterRequest,
    adapt_source,
)


def _development_result(payload: dict) -> dict:
    return {
        "schema_version": "phase-b-baseline-development-result-1",
        "method": "d3_native_no_tools",
        "benchmark_id": "phase_b_cell",
        "tier": "easy",
        "seed": 0,
        "equations": {"y": "y + 1.0"},
        "selected_hyperparameters": {"selected_generation": 2},
        "selection_payload": payload,
        "training_normalized_mse": 0.1,
        "validation_normalized_mse": 0.2,
        "status": "development_complete",
        "test_data_opened": False,
    }


def _candidate() -> dict:
    return {
        "candidate_id": "d3_candidate",
        "parent_candidate_id": None,
        "states": [{"name": "y", "kind": "observed"}],
        "state_equations": [{"state": "y", "rhs": "a * y"}],
        "observation_mappings": [{"channel": "y", "expression": "y"}],
        "parameters": [
            {
                "name": "a",
                "scope": "global",
                "bounds": {"lower": -2.0, "upper": 2.0},
                "initialization_range": {"lower": -1.0, "upper": 1.0},
            }
        ],
        "initial_conditions": [{"state": "y", "scope": "global", "fixed_value": 0.0}],
    }


def _campaign_run(tmp_path: Path, *, plan: str = "a" * 64) -> Path:
    """Mirror the campaign: result.json is a wrapper, the selection is sealed."""
    run = tmp_path / "results" / "7"
    run.mkdir(parents=True)
    (run / "result.json").write_text(
        json.dumps(
            {
                "benchmark_id": "phase_b_cell",
                "tier": "easy",
                "repetition": 0,
                "status": "complete",
                "plan_sha256": plan,
                "protocol": "phase-b-d3-native-validation-1",
            }
        ),
        encoding="utf-8",
    )
    (run / "native-selection.json").write_text(
        json.dumps(
            {
                "plan_sha256": "a" * 64,
                "selection": _development_result(
                    {
                        "candidate": _candidate(),
                        "parameters": {"a": -0.5},
                        "target_scales": {"y": 1.0},
                        "selected_generation": 2,
                    }
                ),
            }
        ),
        encoding="utf-8",
    )
    return run / "result.json"


def _context():
    from autoformalism.expressions import ValidationContext

    return ValidationContext(
        targets=("y",),
        auxiliaries=(),
        external_inputs=(),
        fixed_covariates=(),
    )


def test_campaign_selection_is_read_from_its_own_seal(tmp_path: Path) -> None:
    path = _campaign_run(tmp_path)
    subject = adapt_source(
        SourceAdapterRequest(
            request_id="d3-0", source_kind="d3", source_path=path
        ),
        _context(),
    )
    assert subject.method == "d3_native_no_tools"
    assert subject.execution_semantics == "discrete_increment_recursive_rollout"
    assert subject.parameterization.global_parameters == {"a": -0.5}
    # the sealed selection is hashed alongside the wrapper
    assert "native-selection.json" in subject.source_provenance.auxiliary_sha256


def test_a_selection_from_another_plan_is_refused(tmp_path: Path) -> None:
    path = _campaign_run(tmp_path, plan="b" * 64)
    with pytest.raises(ValueError, match="different plans"):
        adapt_source(
            SourceAdapterRequest(
                request_id="d3-0", source_kind="d3", source_path=path
            ),
            _context(),
        )


def test_discrete_subject_cannot_be_labelled_free_rollout() -> None:
    """The schema refuses a discrete model scored as a continuous one."""
    from autoformalism.schemas import CandidateModel

    common = {
        "subject_id": "s",
        "method": "d3_native_no_tools",
        "benchmark_id": "phase_b_cell",
        "tier": "easy",
        "repetition": 0,
        "private_metrics_opened_after_freeze": True,
        "source_provenance": {
            "adapter": "d3_result",
            "request_id": "r",
            "source_path": "/tmp/result.json",
            "source_sha256": "0" * 64,
            "candidate_sha256": "1" * 64,
        },
        "candidate": CandidateModel.model_validate(_candidate()),
        "parameterization": {"status": "available", "global_parameters": {"a": -0.5}},
        "validation_context": _context(),
        "execution_semantics": "discrete_increment_recursive_rollout",
    }
    scored = {
        "status": "available",
        "normalized_mse": 0.5,
        "per_target_normalized_mse": {"y": 0.5},
        "normalization_scales": {"y": 1.0},
        "trajectory_count": 1,
        "successful_trajectory_count": 1,
    }
    with pytest.raises(ValueError, match="must be scored by"):
        FrozenEvaluationSubject.model_validate(
            {
                **common,
                "target_prediction": {
                    **scored,
                    "evaluation_protocol": "unseen_condition_free_rollout",
                },
            }
        )
    subject = FrozenEvaluationSubject.model_validate(
        {
            **common,
            "target_prediction": {
                **scored,
                "evaluation_protocol": (
                    "unseen_condition_recursive_discrete_rollout"
                ),
            },
        }
    )
    assert subject.target_prediction.normalized_mse == 0.5


def test_hidden_subspace_reports_unsupported_rather_than_raising() -> None:
    """A discrete model has no derivative to perturb; that is not a failure."""
    text = Path(
        "src/autoformalism/rebuttal/phase_b_hidden_subspace.py"
    ).read_text(encoding="utf-8")
    assert "hidden evaluation requires the common free-rollout protocol" not in text
    assert "hidden response subspace requires the common free-rollout" in text
    assert 'status="not_applicable"' in text


def test_failed_discrete_replay_keeps_its_own_protocol() -> None:
    text = Path(
        "src/autoformalism/rebuttal/postfreeze_evaluation.py"
    ).read_text(encoding="utf-8")
    assert "protocol: str = \"unseen_condition_free_rollout\"" in text
    assert "unseen_condition_recursive_discrete_rollout" in text
    # routed by declared semantics, never by a method name
    assert (
        'subject.execution_semantics == "discrete_increment_recursive_rollout"'
        in text
    )
    assert "d3_native" not in text


def test_sealed_test_evaluator_refuses_development_splits() -> None:
    pytest.importorskip("torch")
    from autoformalism.baselines.d3_rollout import evaluate_sealed_test

    assert callable(evaluate_sealed_test)
    source = Path("src/autoformalism/baselines/d3_rollout.py").read_text(
        encoding="utf-8"
    )
    assert "sealed replay requires the test split" in source
    # the development entry point keeps refusing test data
    assert "require TRAIN and VALIDATION splits, never TEST" in source


# --- vendored external campaigns -----------------------------------------


@pytest.mark.parametrize("kind", ["llm_sr", "llm_ode"])
def test_a_vendored_campaign_selection_is_adapted_without_reading_upstream(
    tmp_path: Path, kind: str
) -> None:
    """Only the sealed selection is read, so upstream internals never matter."""
    run = tmp_path / kind
    run.mkdir()
    (run / "result.json").write_text(
        json.dumps(
            {
                "benchmark_id": "phase_b_cell",
                "tier": "easy",
                "repetition": 0,
                "status": "complete",
                "plan_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    selection = _development_result(
        {
            "candidate": _candidate(),
            "parameters": {"a": -0.5},
            "selected_generation": 3,
        }
    )
    selection["method"] = f"{kind}_vendored"
    (run / "native-selection.json").write_text(
        json.dumps({"plan_sha256": "c" * 64, "selection": selection}),
        encoding="utf-8",
    )
    subject = adapt_source(
        SourceAdapterRequest(
            request_id=f"{kind}-0",
            source_kind=kind,
            source_path=run / "result.json",
        ),
        _context(),
    )
    assert subject.method == f"{kind}_vendored"
    # both discover a continuous right-hand side, so the ODE replay applies
    assert subject.execution_semantics == "continuous_ode_free_rollout"
    assert subject.parameterization.global_parameters == {"a": -0.5}
    assert "native-selection.json" in subject.source_provenance.auxiliary_sha256


def test_a_vendored_selection_from_another_plan_is_refused(tmp_path: Path) -> None:
    run = tmp_path / "llm_sr"
    run.mkdir()
    (run / "result.json").write_text(
        json.dumps({"plan_sha256": "d" * 64}), encoding="utf-8"
    )
    selection = _development_result({"candidate": _candidate(), "parameters": {}})
    selection["method"] = "llm_sr_vendored"
    (run / "native-selection.json").write_text(
        json.dumps({"plan_sha256": "e" * 64, "selection": selection}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different plans"):
        adapt_source(
            SourceAdapterRequest(
                request_id="llm_sr-0",
                source_kind="llm_sr",
                source_path=run / "result.json",
            ),
            _context(),
        )
