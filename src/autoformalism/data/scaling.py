"""Training-only standardization for numeric data channels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from autoformalism.data.exceptions import ScalingError
from autoformalism.data.models import DatasetSplit, SplitName


@dataclass(frozen=True)
class Scale:
    """Mean and nonzero standard deviation fitted on training values."""

    mean: float
    standard_deviation: float


@dataclass(frozen=True)
class ScaledTrajectory:
    """Scaled numeric channels for one trajectory."""

    trajectory_id: str
    targets: Mapping[str, NDArray[np.float64]]
    auxiliaries: Mapping[str, NDArray[np.float64]]
    external_inputs: Mapping[str, NDArray[np.float64]]


class TrainingScaler:
    """Fit standardization statistics on training data exactly once."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._epsilon = epsilon
        self._scales: Mapping[str, Scale] | None = None

    @property
    def scales(self) -> Mapping[str, Scale]:
        """Return fitted immutable scaling parameters."""
        if self._scales is None:
            raise ScalingError("scaler has not been fitted")
        return self._scales

    def fit(self, split: DatasetSplit) -> TrainingScaler:
        """Fit all numeric target, auxiliary, and input channels on train."""
        if split.name is not SplitName.TRAIN:
            raise ScalingError("scaling parameters may be fitted on training only")
        if self._scales is not None:
            raise ScalingError("scaler is already fitted")
        values_by_name: dict[str, list[np.ndarray]] = {}
        for trajectory in split.trajectories:
            for namespace, values in (
                ("target", trajectory.targets),
                ("auxiliary", trajectory.auxiliaries),
                ("input", trajectory.external_inputs),
            ):
                for name, value in values.items():
                    if np.issubdtype(value.dtype, np.number):
                        values_by_name.setdefault(f"{namespace}:{name}", []).append(
                            value.astype(float, copy=False)
                        )
        if not values_by_name:
            raise ScalingError("training split has no numeric channels")
        scales: dict[str, Scale] = {}
        for name, arrays in values_by_name.items():
            combined = np.concatenate(arrays)
            if not np.isfinite(combined).all():
                raise ScalingError(f"cannot scale nonfinite channel {name}")
            standard_deviation = max(float(np.std(combined)), self._epsilon)
            scales[name] = Scale(float(np.mean(combined)), standard_deviation)
        self._scales = MappingProxyType(scales)
        return self

    def transform(self, split: DatasetSplit) -> tuple[ScaledTrajectory, ...]:
        """Apply training-fitted scales without mutating the source split."""
        scales = self.scales
        return tuple(
            ScaledTrajectory(
                trajectory_id=trajectory.trajectory_id,
                targets=self._transform_mapping("target", trajectory.targets, scales),
                auxiliaries=self._transform_mapping(
                    "auxiliary", trajectory.auxiliaries, scales
                ),
                external_inputs=self._transform_mapping(
                    "input", trajectory.external_inputs, scales
                ),
            )
            for trajectory in split.trajectories
        )

    @staticmethod
    def _transform_mapping(
        namespace: str,
        values: Mapping[str, np.ndarray],
        scales: Mapping[str, Scale],
    ) -> Mapping[str, np.ndarray]:
        transformed: dict[str, np.ndarray] = {}
        for name, value in values.items():
            if not np.issubdtype(value.dtype, np.number):
                continue
            key = f"{namespace}:{name}"
            if key not in scales:
                raise ScalingError(
                    f"channel {key} was not present when fitting on training"
                )
            scale = scales[key]
            result = (value.astype(float) - scale.mean) / scale.standard_deviation
            result.setflags(write=False)
            transformed[name] = result
        return MappingProxyType(transformed)
