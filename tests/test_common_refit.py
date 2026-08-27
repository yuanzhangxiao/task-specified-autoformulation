"""Tests for the common two-stage candidate evaluator."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from autoformalism.baselines.common_refit import (
    CommonRefitConfig,
    evaluate_common_refit,
)
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import FitConfig
from autoformalism.schemas import ProposerCandidateV2, enrich_proposal_v2
from scripts.refit_raw_data_agent_candidate import _load_frozen_source


def _config() -> CommonRefitConfig:
    return CommonRefitConfig(
        screening_fit=FitConfig(
            integration_backend="fixed_rk4",
            allow_derivative_regression=False,
            random_seed=3,
        ),
        final_fit=FitConfig(
            integration_backend="solve_ivp",
            allow_derivative_regression=False,
            random_seed=3,
        ),
    )


def _candidate():
    proposal = ProposerCandidateV2.model_validate(
        {
            "schema_version": "2",
            "candidate_id": "candidate",
            "states": [
                {
                    "name": "x",
                    "kind": "observed",
                    "observed_channel": "x",
                    "rhs": "-k * x",
                }
            ],
            "algebraics": [],
            "parameters": [
                {"name": "k", "bounds": {"lower": 0.0, "upper": 2.0}}
            ],
        }
    )
    return enrich_proposal_v2(proposal, ("x",))


def test_common_refit_warm_starts_final_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_fit(
        model, training, validation, config, *, initial_global_parameters=None
    ):
        del model, training, validation
        calls.append((config, initial_global_parameters))
        return SimpleNamespace(success=True, global_parameters={"k": 0.25})

    monkeypatch.setattr(
        "autoformalism.baselines.common_refit.fit_candidate", fake_fit
    )
    dataset = SimpleNamespace(train=object(), validation=object())

    result = evaluate_common_refit(
        _candidate(), dataset, ValidationContext(targets=("x",)), _config()
    )

    assert result.final_fit is not None
    assert len(calls) == 2
    assert calls[0][1] is None
    assert calls[1][1] == {"k": 0.25}


def test_common_refit_stops_after_failed_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_fit(model, training, validation, config, **kwargs):
        del model, training, validation, kwargs
        calls.append(config)
        return SimpleNamespace(success=False, global_parameters={})

    monkeypatch.setattr(
        "autoformalism.baselines.common_refit.fit_candidate", fake_fit
    )
    dataset = SimpleNamespace(train=object(), validation=object())

    result = evaluate_common_refit(
        _candidate(), dataset, ValidationContext(targets=("x",)), _config()
    )

    assert result.final_fit is None
    assert len(calls) == 1


def test_common_refit_rejects_adaptive_screening() -> None:
    with pytest.raises(ValueError, match="screening fit must use fixed_rk4"):
        CommonRefitConfig(
            screening_fit=FitConfig(
                integration_backend="solve_ivp",
                allow_derivative_regression=False,
            ),
            final_fit=FitConfig(
                integration_backend="solve_ivp",
                allow_derivative_regression=False,
            ),
        )


def test_common_refit_loads_autoformalism_summary(tmp_path) -> None:
    summary = tmp_path / "run" / "summary.json"
    summary.parent.mkdir()
    summary.write_text(
        json.dumps(
            {
                "benchmark_id": "cell",
                "tier": "easy",
                "seed": 7,
                "selected_candidate": _candidate().model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    source = _load_frozen_source(source_run=None, source_summary=summary)

    assert source.source_kind == "autoformalism_summary"
    assert source.provider == "autoformalism"
    assert source.repetition == 7
    assert source.candidate.candidate_id == "candidate"
