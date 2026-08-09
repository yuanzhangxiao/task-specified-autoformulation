"""Benchmark registry tests."""

import pytest

from autoformalism.benchmarks import phase_b_public_spec
from autoformalism.data.exceptions import BenchmarkNotFoundError
from autoformalism.data.registry import BenchmarkRegistry


def test_default_registry_contains_historical_and_phase_b_specs() -> None:
    registry = BenchmarkRegistry()

    historical = (
        "benchmark5",
        "benchmark6",
        "obfuscated_original_case01",
        "obfuscated_perturbed_case01",
        "original_b1",
        "perturbed_b1",
    )
    assert registry.identifiers()[:6] == historical
    assert len(registry.identifiers()) == 46
    assert len(set(registry.identifiers())) == 46


def test_unknown_identifier_lists_supported_values() -> None:
    with pytest.raises(BenchmarkNotFoundError, match="supported"):
        BenchmarkRegistry().get("not_registered")


def test_original_b1_selects_one_numeric_meal_encoding() -> None:
    spec = BenchmarkRegistry().get("original_b1")

    assert spec.external_inputs == ("meal_event_g",)
    assert spec.fixed_covariates == ("body_weight_kg",)
    assert spec.one_step_target_history


def test_phase_b_registry_matches_public_ids_and_disables_target_reset() -> None:
    registry = BenchmarkRegistry()
    identifiers: set[str] = set()
    for task in ("T1", "T2", "T3", "T4"):
        for dynamics in ("canonical", "perturbed"):
            for variant in ("named", "obfuscated"):
                for tier in ("easy", "hard"):
                    identifiers.add(
                        phase_b_public_spec(
                            "dalla_man",
                            tier,
                            variant,
                            task=task,
                            dynamics=dynamics,
                        ).benchmark_id
                    )
    for family, variants in (
        ("cstr", ("named", "obfuscated")),
        ("alien_device", ("functional", "opaque")),
    ):
        for variant in variants:
            for tier in ("easy", "hard"):
                identifiers.add(
                    phase_b_public_spec(family, tier, variant).benchmark_id
                )

    assert len(identifiers) == 40
    assert identifiers <= set(registry.identifiers())
    assert all(
        not registry.get(identifier).one_step_target_history
        for identifier in identifiers
    )
    assert all(
        registry.get(identifier).data_layout == "tidy_split_file"
        for identifier in identifiers
    )
