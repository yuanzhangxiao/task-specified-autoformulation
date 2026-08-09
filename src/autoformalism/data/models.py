"""Typed data contracts for supported benchmark layouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.data.exceptions import ChannelRoleError


class SplitName(str, Enum):
    """Dataset split names used on disk."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class ChannelRole(str, Enum):
    """Permitted uses of public columns."""

    TARGET = "target"
    AUXILIARY = "auxiliary"
    EXTERNAL_INPUT = "external_input"
    FIXED_COVARIATE = "fixed_covariate"


class TierRoles(BaseModel):
    """Channel roles for one observability tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    targets: tuple[str, ...]
    auxiliaries: tuple[str, ...] = ()

    @model_validator(mode="after")
    def roles_are_disjoint(self) -> TierRoles:
        """Ensure generated targets can never be supplied auxiliaries."""
        if not self.targets:
            raise ValueError("at least one target channel is required")
        overlap = set(self.targets) & set(self.auxiliaries)
        if overlap:
            raise ValueError(f"target/auxiliary overlap: {sorted(overlap)}")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("duplicate target channel")
        if len(set(self.auxiliaries)) != len(self.auxiliaries):
            raise ValueError("duplicate auxiliary channel")
        return self


class BenchmarkSpec(BaseModel):
    """Normalized registry specification for a public benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str
    relative_root: Path
    manifest_relative_path: Path
    tier_roles: Mapping[str, TierRoles]
    time_column: str
    trajectory_id_column: str | None = None
    external_inputs: tuple[str, ...]
    fixed_covariates: tuple[str, ...] = ()
    input_filename_template: str
    tier_directory_template: str = "{tier}"
    data_layout: Literal["legacy_split_files", "tidy_split_file"] = (
        "legacy_split_files"
    )
    split_filename_template: str | None = None
    sampling_interval: float = Field(gt=0.0)
    clean_observations_available: bool = True
    one_step_target_history: bool = True

    @model_validator(mode="after")
    def all_roles_are_disjoint(self) -> BenchmarkSpec:
        """Reject ambiguous roles at registry construction time."""
        side_roles = set(self.external_inputs) | set(self.fixed_covariates)
        if set(self.external_inputs) & set(self.fixed_covariates):
            raise ValueError("external inputs and fixed covariates overlap")
        for tier, roles in self.tier_roles.items():
            overlap = (set(roles.targets) | set(roles.auxiliaries)) & side_roles
            if overlap:
                raise ValueError(f"{tier} observed/input role overlap: {overlap}")
        if self.data_layout == "tidy_split_file" and not self.split_filename_template:
            raise ValueError("tidy split layout requires split_filename_template")
        if self.data_layout == "legacy_split_files" and self.split_filename_template:
            raise ValueError("legacy split layout cannot set split_filename_template")
        return self


@dataclass(frozen=True)
class SplitPaths:
    """Resolved files used to load one split."""

    observations: Path
    derivatives: Path
    inputs: Path


@dataclass(frozen=True)
class Trajectory:
    """One complete trajectory with immutable numeric arrays."""

    trajectory_id: str
    time: NDArray[np.float64]
    targets: Mapping[str, NDArray[np.float64]]
    auxiliaries: Mapping[str, NDArray[np.float64]]
    external_inputs: Mapping[str, NDArray[np.float64]]
    fixed_covariates: Mapping[str, Any]
    derivatives: Mapping[str, NDArray[np.float64]]

    def __post_init__(self) -> None:
        """Freeze mappings and arrays to prevent accidental split mutation."""
        for values in (
            self.time,
            *self.targets.values(),
            *self.auxiliaries.values(),
            *self.external_inputs.values(),
            *self.derivatives.values(),
        ):
            values.setflags(write=False)
        object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))
        object.__setattr__(
            self, "auxiliaries", MappingProxyType(dict(self.auxiliaries))
        )
        object.__setattr__(
            self, "external_inputs", MappingProxyType(dict(self.external_inputs))
        )
        object.__setattr__(
            self, "fixed_covariates", MappingProxyType(dict(self.fixed_covariates))
        )
        object.__setattr__(
            self, "derivatives", MappingProxyType(dict(self.derivatives))
        )

    @property
    def number_of_rows(self) -> int:
        """Return the number of samples in this trajectory."""
        return len(self.time)


@dataclass(frozen=True)
class DatasetSplit:
    """A named, fingerprinted collection of trajectories."""

    name: SplitName
    trajectories: tuple[Trajectory, ...]
    fingerprint: str


@dataclass(frozen=True)
class BenchmarkDataset:
    """Strictly separated train, validation, and test splits."""

    benchmark_id: str
    tier: str
    roles: TierRoles
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit

    def __post_init__(self) -> None:
        """Validate split identities and content separation."""
        expected = (
            (self.train, SplitName.TRAIN),
            (self.validation, SplitName.VALIDATION),
            (self.test, SplitName.TEST),
        )
        for split, name in expected:
            if split.name is not name:
                raise ChannelRoleError(f"expected {name.value}, got {split.name.value}")
        fingerprints = {split.fingerprint for split, _ in expected}
        if len(fingerprints) != 3:
            raise ChannelRoleError("dataset splits have identical fingerprints")


@dataclass(frozen=True)
class DevelopmentDataset:
    """Train and validation data available during structural selection."""

    benchmark_id: str
    tier: str
    roles: TierRoles
    train: DatasetSplit
    validation: DatasetSplit

    def __post_init__(self) -> None:
        if self.train.name is not SplitName.TRAIN:
            raise ChannelRoleError("development train split has the wrong name")
        if self.validation.name is not SplitName.VALIDATION:
            raise ChannelRoleError("development validation split has the wrong name")
        if self.train.fingerprint == self.validation.fingerprint:
            raise ChannelRoleError("development splits have identical fingerprints")


@dataclass(frozen=True)
class FrozenTestAccess:
    """Explicit authorization to open one frozen benchmark test split."""

    benchmark_id: str
    tier: str
    selection_hash: str

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.tier or not self.selection_hash:
            raise ValueError("test access grant fields must be nonempty")
