"""Benchmark registry, loading, and scaling."""

from autoformalism.data.derivative_overlay import attach_exact_derivative_overlay
from autoformalism.data.loader import BenchmarkLoader
from autoformalism.data.models import (
    BenchmarkDataset,
    BenchmarkSpec,
    ChannelRole,
    DatasetSplit,
    DerivativeProvenance,
    DevelopmentDataset,
    FrozenTestAccess,
    SplitName,
    Trajectory,
)
from autoformalism.data.registry import BenchmarkRegistry
from autoformalism.data.scaling import TrainingScaler

__all__ = [
    "BenchmarkDataset",
    "BenchmarkLoader",
    "BenchmarkRegistry",
    "BenchmarkSpec",
    "ChannelRole",
    "DatasetSplit",
    "DerivativeProvenance",
    "DevelopmentDataset",
    "FrozenTestAccess",
    "SplitName",
    "TrainingScaler",
    "Trajectory",
    "attach_exact_derivative_overlay",
]
