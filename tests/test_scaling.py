"""Training-only scaling tests."""

from pathlib import Path

import numpy as np
import pytest

from autoformalism.config import DataConfig
from autoformalism.data.exceptions import ScalingError
from autoformalism.data.loader import BenchmarkLoader
from autoformalism.data.registry import BenchmarkRegistry
from autoformalism.data.scaling import TrainingScaler


def _dataset(root: Path, registry: BenchmarkRegistry):
    return BenchmarkLoader(registry).load(
        DataConfig(root=root, benchmark_id="synthetic", tier="easy")
    )


def test_scaler_fits_train_and_applies_same_scale_to_validation(
    synthetic_root: Path,
    synthetic_registry: BenchmarkRegistry,
) -> None:
    dataset = _dataset(synthetic_root, synthetic_registry)
    scaler = TrainingScaler().fit(dataset.train)

    scaled_train = scaler.transform(dataset.train)
    scaled_validation = scaler.transform(dataset.validation)

    combined_train = np.concatenate([item.targets["y"] for item in scaled_train])
    assert np.mean(combined_train) == pytest.approx(0.0)
    assert np.std(combined_train) == pytest.approx(1.0)
    expected = (
        dataset.validation.trajectories[0].targets["y"][0]
        - scaler.scales["target:y"].mean
    ) / scaler.scales["target:y"].standard_deviation
    assert scaled_validation[0].targets["y"][0] == pytest.approx(expected)


def test_scaler_rejects_validation_fit_and_refit(
    synthetic_root: Path,
    synthetic_registry: BenchmarkRegistry,
) -> None:
    dataset = _dataset(synthetic_root, synthetic_registry)
    scaler = TrainingScaler()

    with pytest.raises(ScalingError, match="training only"):
        scaler.fit(dataset.validation)

    scaler.fit(dataset.train)
    with pytest.raises(ScalingError, match="already fitted"):
        scaler.fit(dataset.train)

