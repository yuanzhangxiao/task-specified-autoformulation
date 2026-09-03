from pathlib import Path

import numpy as np
import pytest

from autoformalism.data import (
    DatasetSplit,
    DerivativeProvenance,
    SplitName,
    Trajectory,
    attach_exact_derivative_overlay,
)


def _split() -> DatasetSplit:
    return DatasetSplit(
        SplitName.TRAIN,
        (
            Trajectory(
                "train_000",
                np.asarray([0.0, 1.0]),
                {"y": np.asarray([1.0, 2.0])},
                {},
                {"u": np.asarray([0.0, 1.0])},
                {},
                {"y": np.asarray([99.0, 99.0])},
                DerivativeProvenance.ESTIMATED,
            ),
        ),
        "public-observation-fingerprint",
    )


def test_exact_derivative_overlay_preserves_observations(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text(
        "trajectory_id,t,d__y\ntrain_000,0,0.25\ntrain_000,1,0.5\n",
        encoding="utf-8",
    )

    result = attach_exact_derivative_overlay(_split(), path)

    assert result.fingerprint != "public-observation-fingerprint"
    assert result.trajectories[0].derivative_provenance is DerivativeProvenance.EXACT
    assert result.trajectories[0].targets["y"] == pytest.approx([1.0, 2.0])
    assert result.trajectories[0].derivatives["y"] == pytest.approx([0.25, 0.5])


def test_exact_derivative_overlay_rejects_time_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text(
        "trajectory_id,t,d__y\ntrain_000,0,0.25\ntrain_000,2,0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="times differ"):
        attach_exact_derivative_overlay(_split(), path)
