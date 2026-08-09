"""Typed contract for redesigned benchmark suites.

This module describes a future suite without registering it as runnable data.
That separation prevents an incomplete redesign from silently replacing the
historical production benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TierDesign(BaseModel):
    """Family-specific meaning of one difficulty tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["easy", "hard"]
    difficulty_basis: Literal["mechanism_and_observability", "description"]
    description_mode: Literal["inherit_semantic_variant", "functional", "opaque"]
    mechanism_burden: Literal["focused", "coupled", "matched"]
    trajectory_regime: Literal["shared_excitation", "matched"]
    description: str = Field(min_length=1)


class BenchmarkFamilyDefinition(BaseModel):
    """A benchmark family and its independent experimental axes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["dalla_man", "cstr", "alien_device"]
    tasks: tuple[str, ...] = Field(min_length=1)
    dynamics_conditions: tuple[
        Literal["canonical", "perturbed", "not_applicable"], ...
    ] = Field(min_length=1)
    semantic_variants: tuple[
        Literal["named", "obfuscated", "functional", "opaque", "not_applicable"],
        ...,
    ] = Field(min_length=1)
    tiers: tuple[TierDesign, ...]
    paired_numeric_data_across_semantic_variants: bool
    paired_numeric_data_across_tiers: bool
    paired_input_protocols_across_tiers: bool
    identifiability_requirement: Literal[
        "data_identifiable", "task_identifiable", "equivalence_class"
    ]
    primary_claim: str = Field(min_length=1)
    hidden_evaluation: Literal["coordinate", "subspace", "equivalence", "none"]
    full_factorial_definition: bool = True

    @model_validator(mode="after")
    def tiers_are_complete(self) -> BenchmarkFamilyDefinition:
        """Require exactly the frozen easy/hard pair."""
        if tuple(tier.name for tier in self.tiers) != ("easy", "hard"):
            raise ValueError("every Phase-B family must define easy and hard")
        return self

    @property
    def number_of_cells(self) -> int:
        """Return full-factorial cells before methods and seeds."""
        return (
            len(self.tasks)
            * len(self.dynamics_conditions)
            * len(self.semantic_variants)
            * len(self.tiers)
        )


class EvaluationContract(BaseModel):
    """Evaluation endpoints frozen before generating benchmark data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_metrics: tuple[str, ...] = Field(min_length=1)
    diagnostic_metrics: tuple[str, ...] = Field(min_length=1)
    dalla_rollout_horizons_minutes: tuple[int, ...] = Field(min_length=1)
    generic_rollout_horizon_fractions: tuple[float, ...] = Field(min_length=1)
    selection_uses_validation_only: bool
    private_references_evaluation_only: bool

    @model_validator(mode="after")
    def safeguards_are_enabled(self) -> EvaluationContract:
        """Protect held-out evaluation and require increasing horizons."""
        if not self.selection_uses_validation_only:
            raise ValueError("selection must use validation data only")
        if not self.private_references_evaluation_only:
            raise ValueError("private references must be evaluation-only")
        horizons = self.dalla_rollout_horizons_minutes
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("Dalla rollout horizons must be unique and increasing")
        fractions = self.generic_rollout_horizon_fractions
        if tuple(sorted(set(fractions))) != fractions or not all(
            0.0 < value <= 1.0 for value in fractions
        ):
            raise ValueError(
                "generic rollout fractions must be unique, increasing, and in (0, 1]"
            )
        return self


class BenchmarkSuiteSpec(BaseModel):
    """Frozen scientific design for a benchmark-suite version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suite_version: Literal["phase_b_v1"]
    status: Literal["design_frozen_data_pending"]
    families: tuple[BenchmarkFamilyDefinition, ...]
    evaluation: EvaluationContract

    @model_validator(mode="after")
    def suite_axes_are_valid(self) -> BenchmarkSuiteSpec:
        """Validate that difficulty and semantic controls are not conflated."""
        by_name = {family.family: family for family in self.families}
        if set(by_name) != {"dalla_man", "cstr", "alien_device"}:
            raise ValueError("suite must define Dalla Man, CSTR, and alien device")
        if len(by_name) != len(self.families):
            raise ValueError("benchmark families must be unique")

        dalla = by_name["dalla_man"]
        if dalla.tasks != ("T1", "T2", "T3", "T4"):
            raise ValueError("Dalla Man must cover T1-T4")
        if dalla.dynamics_conditions != ("canonical", "perturbed"):
            raise ValueError("Dalla Man must compare canonical and perturbed dynamics")

        for name in ("dalla_man", "cstr"):
            family = by_name[name]
            if family.semantic_variants != ("named", "obfuscated"):
                raise ValueError(
                    f"{name} must keep named/obfuscated as a separate axis"
                )
            if not family.paired_numeric_data_across_semantic_variants:
                raise ValueError(
                    f"{name} semantic controls must use paired numeric data"
                )
            if family.paired_numeric_data_across_tiers:
                raise ValueError(f"{name} tiers must encode task/data difficulty")
            if any(
                tier.difficulty_basis != "mechanism_and_observability"
                for tier in family.tiers
            ):
                raise ValueError(f"{name} tiers must be mechanism/observability tiers")
            if not family.paired_input_protocols_across_tiers:
                raise ValueError(f"{name} tiers must share input protocols")

        alien = by_name["alien_device"]
        if alien.semantic_variants != ("functional", "opaque"):
            raise ValueError("alien must keep functional/opaque as a semantic axis")
        if not alien.paired_numeric_data_across_semantic_variants:
            raise ValueError("alien semantic controls must use paired numeric data")
        if alien.paired_numeric_data_across_tiers:
            raise ValueError("alien tiers must encode task/data difficulty")
        if any(
            tier.difficulty_basis != "mechanism_and_observability"
            for tier in alien.tiers
        ):
            raise ValueError("alien tiers must be mechanism/observability tiers")
        if not alien.paired_input_protocols_across_tiers:
            raise ValueError("alien tiers must share input protocols")
        return self

    @property
    def number_of_cells(self) -> int:
        """Return full-factorial cells before methods and seeds."""
        return sum(family.number_of_cells for family in self.families)


def load_suite_spec(path: Path) -> BenchmarkSuiteSpec:
    """Load and strictly validate a suite specification from JSON."""
    return BenchmarkSuiteSpec.model_validate(json.loads(path.read_text()))
