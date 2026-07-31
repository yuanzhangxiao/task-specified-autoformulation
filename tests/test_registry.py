"""Benchmark registry tests."""

import pytest

from autoformalism.data.exceptions import BenchmarkNotFoundError
from autoformalism.data.registry import BenchmarkRegistry


def test_default_registry_is_narrow_and_deterministic() -> None:
    registry = BenchmarkRegistry()

    assert registry.identifiers() == (
        "benchmark5",
        "benchmark6",
        "obfuscated_original_case01",
        "obfuscated_perturbed_case01",
        "original_b1",
        "perturbed_b1",
    )


def test_unknown_identifier_lists_supported_values() -> None:
    with pytest.raises(BenchmarkNotFoundError, match="supported"):
        BenchmarkRegistry().get("not_registered")

