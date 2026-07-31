"""Configuration tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.config import DataConfig, load_config


def test_loads_dotenv(synthetic_root: Path, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                f"AUTOFORMALISM_DATA_ROOT={synthetic_root}",
                "AUTOFORMALISM_BENCHMARK_ID=synthetic",
                "AUTOFORMALISM_TIER=medium",
                "AUTOFORMALISM_USE_CLEAN_OBSERVATIONS=true",
            )
        ),
        encoding="utf-8",
    )

    settings = load_config(env_file)

    assert settings.data_root == synthetic_root
    assert settings.benchmark_id == "synthetic"
    assert settings.tier == "medium"
    assert settings.use_clean_observations is True


def test_rejects_missing_data_root(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="data root is not a directory"):
        DataConfig(
            root=tmp_path / "missing",
            benchmark_id="synthetic",
            tier="easy",
        )

