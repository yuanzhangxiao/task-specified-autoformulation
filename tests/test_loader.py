"""Strict split loading and role-validation tests."""

from pathlib import Path

import pandas as pd
import pytest

from autoformalism.config import DataConfig
from autoformalism.data.exceptions import (
    ChannelRoleError,
    DataAlignmentError,
    DataFileNotFoundError,
    MissingColumnError,
)
from autoformalism.data.loader import BenchmarkLoader
from autoformalism.data.models import SplitName
from autoformalism.data.registry import BenchmarkRegistry


def _load(
    synthetic_root: Path,
    synthetic_registry: BenchmarkRegistry,
):
    config = DataConfig(
        root=synthetic_root,
        benchmark_id="synthetic",
        tier="easy",
    )
    return BenchmarkLoader(synthetic_registry).load(config)


def test_groups_trajectories_and_separates_splits(
    synthetic_root: Path,
    synthetic_registry: BenchmarkRegistry,
) -> None:
    dataset = _load(synthetic_root, synthetic_registry)

    assert dataset.train.name is SplitName.TRAIN
    assert dataset.validation.name is SplitName.VALIDATION
    assert dataset.test.name is SplitName.TEST
    assert [item.trajectory_id for item in dataset.train.trajectories] == [
        "train_0",
        "train_1",
    ]
    first = dataset.train.trajectories[0]
    assert first.time.tolist() == [0.0, 1.0]
    assert set(first.targets) == {"y"}
    assert set(first.auxiliaries) == {"a"}
    assert set(first.external_inputs) == {"u"}
    assert first.fixed_covariates == {"c": 1.0}
    assert first.targets["y"].flags.writeable is False
    assert len(
        {
            dataset.train.fingerprint,
            dataset.validation.fingerprint,
            dataset.test.fingerprint,
        }
    ) == 3


def test_development_loading_does_not_open_test(
    synthetic_root: Path,
    synthetic_registry: BenchmarkRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = BenchmarkLoader(synthetic_registry)
    opened: list[SplitName] = []
    original = loader._load_split

    def recording_load(spec, roles, split, paths):
        opened.append(split)
        return original(spec, roles, split, paths)

    monkeypatch.setattr(loader, "_load_split", recording_load)
    config = DataConfig(
        root=synthetic_root,
        benchmark_id="synthetic",
        tier="easy",
    )

    development = loader.load_development(config)

    assert development.train.name is SplitName.TRAIN
    assert development.validation.name is SplitName.VALIDATION
    assert opened == [SplitName.TRAIN, SplitName.VALIDATION]

    test = loader.load_test(config)
    assert test.name is SplitName.TEST
    assert opened[-1] is SplitName.TEST


def test_missing_required_file_is_clear(
    synthetic_root: Path,
    synthetic_registry: BenchmarkRegistry,
) -> None:
    (synthetic_root / "public/synthetic/easy/Y_val.csv").unlink()

    with pytest.raises(DataFileNotFoundError, match=r"Y_val\.csv"):
        _load(synthetic_root, synthetic_registry)


def test_missing_required_column_is_clear(
    synthetic_root: Path,
    synthetic_registry: BenchmarkRegistry,
) -> None:
    path = synthetic_root / "public/synthetic/input_train.csv"
    frame = pd.read_csv(path).drop(columns="u")
    frame.to_csv(path, index=False)

    with pytest.raises(MissingColumnError, match="u"):
        _load(synthetic_root, synthetic_registry)


def test_unassigned_observation_column_is_rejected(
    synthetic_root: Path,
    synthetic_registry: BenchmarkRegistry,
) -> None:
    path = synthetic_root / "public/synthetic/easy/X_train.csv"
    frame = pd.read_csv(path)
    frame["leak"] = 1.0
    frame.to_csv(path, index=False)

    with pytest.raises(ChannelRoleError, match="without target/auxiliary roles"):
        _load(synthetic_root, synthetic_registry)


def test_bad_row_alignment_and_sampling_are_rejected(
    synthetic_root: Path,
    synthetic_registry: BenchmarkRegistry,
) -> None:
    path = synthetic_root / "public/synthetic/input_train.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "t"] = 2.0
    frame.to_csv(path, index=False)

    with pytest.raises(DataAlignmentError, match="sampling interval"):
        _load(synthetic_root, synthetic_registry)


def test_noncontiguous_trajectory_is_rejected(
    synthetic_root: Path,
    synthetic_registry: BenchmarkRegistry,
) -> None:
    path = synthetic_root / "public/synthetic/input_train.csv"
    frame = pd.read_csv(path)
    frame["trajectory_id"] = ["a", "b", "a", "b"]
    frame.to_csv(path, index=False)

    with pytest.raises(DataAlignmentError, match="not contiguous"):
        _load(synthetic_root, synthetic_registry)
