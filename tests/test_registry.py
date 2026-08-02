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


def test_original_b1_selects_one_numeric_meal_encoding() -> None:
    spec = BenchmarkRegistry().get("original_b1")

    assert spec.external_inputs == ("meal_event_g",)
    assert spec.fixed_covariates == ("body_weight_kg",)
    assert spec.one_step_target_history


def test_all_registered_benchmarks_use_one_step_target_history() -> None:
    registry = BenchmarkRegistry()

    assert all(
        registry.get(identifier).one_step_target_history
        for identifier in registry.identifiers()
    )
