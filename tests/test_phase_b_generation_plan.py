"""Tests for the Phase-B candidate-generation preflight."""

from __future__ import annotations

import pytest

from scripts.plan_phase_b_candidate_generation import _legacy_source


@pytest.mark.parametrize(
    ("benchmark_id", "expected"),
    (
        ("phase_b_dalla_man_t1_canonical_named_easy", "original_b1"),
        ("phase_b_dalla_man_t4_perturbed_named_hard", "perturbed_b1"),
        (
            "phase_b_anonymous_system_t2_canonical_obfuscated_easy",
            "obfuscated_original_case01",
        ),
        (
            "phase_b_anonymous_system_t3_perturbed_obfuscated_hard",
            "obfuscated_perturbed_case01",
        ),
        (
            "phase_b_cstr_controlled_reactor_mechanism_canonical_named_easy",
            "benchmark5",
        ),
        (
            "phase_b_anonymous_system_task_canonical_obfuscated_hard",
            "benchmark5",
        ),
        (
            "phase_b_alien_device_unknown_device_mechanism_canonical_functional_easy",
            "benchmark6",
        ),
        ("phase_b_anonymous_system_task_canonical_opaque_hard", "benchmark6"),
    ),
)
def test_legacy_source_mapping(benchmark_id: str, expected: str) -> None:
    assert _legacy_source(benchmark_id) == expected


def test_legacy_source_mapping_fails_closed() -> None:
    with pytest.raises(ValueError, match="cannot map"):
        _legacy_source("phase_b_unknown_task_canonical_named_easy")
