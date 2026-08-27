"""Common two-stage numerical evaluation for frozen candidate structures."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import model_validator

from autoformalism.data import DevelopmentDataset
from autoformalism.expressions import (
    ValidationContext,
    compile_candidate,
    repair_protected_declarations,
)
from autoformalism.fitting import FitConfig, FitResult, fit_candidate
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema


class CommonRefitConfig(StrictSchema):
    """Frozen screening and final-fit settings shared across methods."""

    schema_version: str = "common-candidate-refit-config-1"
    screening_fit: FitConfig
    final_fit: FitConfig

    @model_validator(mode="after")
    def enforce_two_stage_protocol(self) -> CommonRefitConfig:
        """Keep screening fast and final evaluation adaptive."""
        if self.screening_fit.integration_backend != "fixed_rk4":
            raise ValueError("screening fit must use fixed_rk4")
        if self.final_fit.integration_backend != "solve_ivp":
            raise ValueError("final fit must use solve_ivp")
        if (
            self.screening_fit.allow_derivative_regression
            or self.final_fit.allow_derivative_regression
        ):
            raise ValueError("common refit must use rollout objectives")
        if self.screening_fit.random_seed != self.final_fit.random_seed:
            raise ValueError("screening and final random seeds must match")
        return self


@dataclass(frozen=True)
class CommonRefitResult:
    """Compiled candidate plus screening and optional warm-started final fits."""

    candidate: CandidateModel
    repairs: tuple[str, ...]
    warnings: tuple[dict[str, str], ...]
    screening_fit: FitResult
    final_fit: FitResult | None


def evaluate_common_refit(
    candidate: CandidateModel,
    dataset: DevelopmentDataset,
    context: ValidationContext,
    config: CommonRefitConfig,
) -> CommonRefitResult:
    """Evaluate one frozen structure without pruning, judging, or test access."""
    repaired, repairs = repair_protected_declarations(candidate, context)
    compiled = compile_candidate(repaired, context)
    screening = fit_candidate(
        compiled,
        dataset.train,
        dataset.validation,
        config.screening_fit,
    )
    final = fit_candidate(
        compiled,
        dataset.train,
        dataset.validation,
        config.final_fit,
        initial_global_parameters=(
            screening.global_parameters if screening.success else None
        ),
    )
    warnings = tuple(
        {
            "code": item.code,
            "location": item.location,
            "message": item.message,
        }
        for item in compiled.validated.warnings
    )
    return CommonRefitResult(
        candidate=repaired,
        repairs=repairs,
        warnings=warnings,
        screening_fit=screening,
        final_fit=final,
    )
