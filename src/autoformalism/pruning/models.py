"""Typed configuration and diagnostics for post-fit term pruning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field

from autoformalism.fitting import FitResult
from autoformalism.schemas import CandidateModel


class PruningConfig(BaseModel):
    """Deterministic support generation and validation-selection controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    validation_mse_tolerance: float = Field(default=0.01, ge=0.0)
    contribution_epsilon: float = Field(default=1e-12, gt=0.0)
    threshold_epsilon: float = Field(default=1e-12, gt=0.0)
    maximum_normalized_contribution: float = Field(default=0.05, gt=0.0)
    preserve_external_input_terms: bool = True
    require_target_dynamics: bool = True


@dataclass(frozen=True)
class TermContribution:
    """Normalized RMS contribution of one complete additive term."""

    term_id: str
    location: str
    expression: str
    normalized_rms: float
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class PruningCandidateResult:
    """One generated support and its deterministic/numerical outcome."""

    threshold: float
    retained_term_ids: tuple[str, ...]
    removed_term_ids: tuple[str, ...]
    removed_parameters: tuple[str, ...]
    candidate: CandidateModel | None
    fit_result: FitResult | None
    accepted: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class PruningResult:
    """Baseline, evaluated supports, and selected simplest valid model."""

    unpruned_candidate: CandidateModel
    unpruned_fit: FitResult
    contributions: tuple[TermContribution, ...]
    thresholds: tuple[float, ...]
    candidates: tuple[PruningCandidateResult, ...]
    selected_candidate: CandidateModel
    selected_fit: FitResult
    selected_threshold: float
    selected_removed_terms: tuple[str, ...]
    selected_removed_parameters: tuple[str, ...]
    contribution_by_term: Mapping[str, float]
    persistence_training_mse: float
    persistence_validation_mse: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contribution_by_term",
            MappingProxyType(dict(self.contribution_by_term)),
        )
