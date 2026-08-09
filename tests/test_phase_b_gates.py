"""Tests for task-specific Phase-B release gates."""

from pathlib import Path

import pytest

from autoformalism.benchmarks import (
    alien_sensitivity_selected_auxiliaries,
    audit_task_gates,
    mechanism_gate_definition,
    phase_b_protocols,
    simulate_phase_b,
)

DATA_ROOT = Path("data_raw")


def test_mechanism_definitions_match_frozen_tier_claims() -> None:
    easy = mechanism_gate_definition("dalla_man", "easy", task="T1")
    hard = mechanism_gate_definition("dalla_man", "hard", task="T1")

    assert easy.mechanisms == ("gastric_memory", "meal_appearance")
    assert hard.mechanisms == (
        "gastric_memory",
        "meal_appearance",
        "glucose_exchange",
    )
    assert easy.claimed_dimension == 2
    assert hard.claimed_dimension == 3

    cstr_hard = mechanism_gate_definition("cstr", "hard", data_root=DATA_ROOT)
    alien_hard = mechanism_gate_definition(
        "alien_device", "hard", data_root=DATA_ROOT
    )
    assert cstr_hard.claimed_dimension == 2
    assert alien_hard.claimed_dimension == 3


def test_private_mechanism_scale_changes_reference_but_not_protocol() -> None:
    protocol = phase_b_protocols("cstr")[1]
    nominal = simulate_phase_b(protocol, data_root=DATA_ROOT)
    shifted = simulate_phase_b(
        protocol,
        data_root=DATA_ROOT,
        private_mechanism_scales={"reaction_heat": 1.01},
    )

    assert nominal.protocol_id == shifted.protocol_id
    assert nominal.model_dump()["inputs"].shape == shifted.model_dump()["inputs"].shape
    assert pytest.approx(nominal.states) != shifted.states


def test_unknown_private_mechanism_is_rejected() -> None:
    protocol = phase_b_protocols("alien_device")[0]

    with pytest.raises(ValueError, match="unknown alien mechanism"):
        simulate_phase_b(
            protocol,
            data_root=DATA_ROOT,
            private_mechanism_scales={"invented": 1.1},
        )


def test_private_initial_offset_is_auditable_and_not_in_protocol() -> None:
    protocol = phase_b_protocols("alien_device")[0]
    shifted = simulate_phase_b(
        protocol,
        data_root=DATA_ROOT,
        private_initial_offsets={"z1": 0.1},
    )

    assert "private_initial_offsets" not in protocol.specification
    assert shifted.states[0, shifted.state_names.index("z1")] == pytest.approx(0.1)


def test_alien_auxiliary_selection_is_deterministic() -> None:
    first = alien_sensitivity_selected_auxiliaries(DATA_ROOT)
    second = alien_sensitivity_selected_auxiliaries(DATA_ROOT)

    assert first == second
    assert len(set(first)) == 2
    assert all(name.startswith("z") for name in first)


def test_complete_cstr_easy_gate_report_is_auditable() -> None:
    definition = mechanism_gate_definition("cstr", "easy", data_root=DATA_ROOT)
    report = audit_task_gates(definition, data_root=DATA_ROOT)

    assert len(report.singular_values) == len(definition.mechanisms)
    assert len(report.ablations) == len(definition.mechanisms)
    assert report.rank_at_1e3 <= len(definition.mechanisms)
    assert report.release_ready == (
        report.rank_pass
        and report.condition_pass
        and report.stable_rank_pass
        and report.ablation_pass
        and report.basic.finite_rollouts_pass
        and report.basic.input_design_pass
        and report.basic.persistence_pass
    )
