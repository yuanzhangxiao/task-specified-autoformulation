"""Tests for the frozen Phase-B benchmark-suite design."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.benchmarks import BenchmarkSuiteSpec, load_suite_spec

SPEC = Path("configs/benchmarks/phase_b_suite_v1.json")


def _family(suite: BenchmarkSuiteSpec, name: str):
    return next(family for family in suite.families if family.family == name)


def test_dalla_and_cstr_separate_difficulty_from_obfuscation() -> None:
    suite = load_suite_spec(SPEC)

    for name in ("dalla_man", "cstr"):
        family = _family(suite, name)
        assert family.semantic_variants == ("named", "obfuscated")
        assert family.paired_numeric_data_across_semantic_variants
        assert not family.paired_numeric_data_across_tiers
        assert family.paired_input_protocols_across_tiers
        assert all(
            tier.difficulty_basis == "mechanism_and_observability"
            for tier in family.tiers
        )


def test_alien_separates_difficulty_from_functional_description() -> None:
    suite = load_suite_spec(SPEC)
    alien = _family(suite, "alien_device")

    assert alien.semantic_variants == ("functional", "opaque")
    assert alien.paired_numeric_data_across_semantic_variants
    assert not alien.paired_numeric_data_across_tiers
    assert alien.paired_input_protocols_across_tiers
    assert all(
        tier.difficulty_basis == "mechanism_and_observability"
        for tier in alien.tiers
    )


def test_phase_b_suite_covers_all_dalla_axes() -> None:
    suite = load_suite_spec(SPEC)
    dalla = _family(suite, "dalla_man")

    assert dalla.tasks == ("T1", "T2", "T3", "T4")
    assert dalla.dynamics_conditions == ("canonical", "perturbed")
    assert dalla.number_of_cells == 32
    assert suite.number_of_cells == 40


def test_phase_b_suite_rejects_medium_tier() -> None:
    payload = json.loads(SPEC.read_text())
    payload["families"][0]["tiers"][1]["name"] = "medium"

    with pytest.raises(ValidationError, match=r"easy.*hard"):
        BenchmarkSuiteSpec.model_validate(payload)


def test_phase_b_suite_rejects_semantic_dalla_tiers() -> None:
    payload = json.loads(SPEC.read_text())
    payload["families"][0]["tiers"][0]["difficulty_basis"] = "description"

    with pytest.raises(ValidationError, match="mechanism/observability"):
        BenchmarkSuiteSpec.model_validate(payload)
