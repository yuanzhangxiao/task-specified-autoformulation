"""Typed configuration and results for benchmark baselines."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BaselineConfig(BaseModel):
    """One reproducible baseline run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal[
        "persistence",
        "sindy",
        "pysr",
        "llm_feature_sindy",
        "d3_native_no_tools",
    ]
    seed: int = 0
    llm_model: str | None = None
    sindy_thresholds: tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
    )
    pysr_iterations: int = Field(default=40, ge=1)
    maximum_expression_size: int = Field(default=30, ge=3)
    d3_generations: int = Field(default=20, ge=1)
    d3_patience: int = Field(default=20, ge=1)
    wall_timeout_seconds: float = Field(default=1_800.0, gt=0.0)


class BaselineResult(BaseModel):
    """Serializable metrics and discovered equations from one baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    benchmark_id: str
    tier: str
    seed: int
    equations: dict[str, str]
    selected_hyperparameters: dict[str, float | int | str]
    training_normalized_mse: float
    validation_normalized_mse: float
    test_normalized_mse: float
    test_per_target_normalized_mse: dict[str, float]
    elapsed_wall_seconds: float | None = Field(default=None, ge=0.0)
    wall_timeout_seconds: float | None = Field(default=None, gt=0.0)
    status: Literal["complete"] = "complete"


class BaselineRunStatus(BaseModel):
    """Always-written process-level completion, failure, or timeout status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["complete", "failed", "timed_out"]
    elapsed_wall_seconds: float = Field(ge=0.0)
    wall_timeout_seconds: float = Field(gt=0.0)
    error: str | None = None
