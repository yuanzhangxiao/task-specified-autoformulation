"""Typed application configuration with ``.env`` support."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    """Configuration for selecting and loading a benchmark tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path
    benchmark_id: str
    tier: Literal["easy", "medium", "hard"]
    use_clean_observations: bool = False
    scaling_epsilon: float = Field(default=1e-8, gt=0.0)

    @field_validator("root")
    @classmethod
    def root_must_exist(cls, value: Path) -> Path:
        """Resolve and validate the configured data root."""
        resolved = value.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"data root is not a directory: {resolved}")
        return resolved


class AppConfig(BaseSettings):
    """Environment-backed process configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AUTOFORMALISM_",
        extra="ignore",
        frozen=True,
    )

    data_root: Path
    benchmark_id: str = "original_b1"
    tier: Literal["easy", "medium", "hard"] = "easy"
    use_clean_observations: bool = False
    scaling_epsilon: float = Field(default=1e-8, gt=0.0)

    def data_config(self) -> DataConfig:
        """Return the validated data-loading subset."""
        return DataConfig(
            root=self.data_root,
            benchmark_id=self.benchmark_id,
            tier=self.tier,
            use_clean_observations=self.use_clean_observations,
            scaling_epsilon=self.scaling_epsilon,
        )


def load_config(env_file: Path | None = None) -> AppConfig:
    """Load settings from the environment and an optional dotenv file."""
    if env_file is None:
        return AppConfig()
    return AppConfig(_env_file=env_file)

