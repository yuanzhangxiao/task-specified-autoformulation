"""Tests for private Phase-B protocol execution and release safeguards."""

from pathlib import Path

import numpy as np
import pytest

from autoformalism.benchmarks import (
    audit_basic_gates,
    phase_b_protocols,
    simulate_phase_b,
    write_private_bundle,
)
from autoformalism.rebuttal.dalla_man import (
    STATE_INDEX,
    DallaManExternalForcing,
    simulate_dalla_man,
)

DATA_ROOT = Path("data_raw")


@pytest.mark.parametrize("family", ["dalla_man", "cstr", "alien_device"])
def test_phase_b_protocols_have_frozen_split_counts(family: str) -> None:
    protocols = phase_b_protocols(family, task="T4" if family == "dalla_man" else None)

    assert len(protocols) == 26
    assert sum(item.split == "train" for item in protocols) == 16
    assert sum(item.split == "validation" for item in protocols) == 4
    assert sum(item.split == "test" for item in protocols) == 6
    assert len({item.protocol_id for item in protocols}) == 26


def test_dalla_external_forcing_is_explicit_and_changes_trajectory() -> None:
    baseline = simulate_dalla_man(meals=(), duration=60, dt=1, variant="original")
    forced = simulate_dalla_man(
        meals=(),
        duration=60,
        dt=1,
        variant="original",
        external_forcing=DallaManExternalForcing(
            glucose_mg_per_kg_min=((10, 30, 0.8),),
            insulin_pmol_per_kg_min=((10, 30, 0.3),),
        ),
    )

    assert forced.derived["glucose_forcing"][15] == pytest.approx(0.8)
    assert forced.derived["insulin_forcing"][15] == pytest.approx(0.3)
    assert not np.allclose(baseline.states, forced.states)


def test_dalla_glucose_initial_shift_conserves_gp_plus_gt() -> None:
    protocols = phase_b_protocols("dalla_man", task="T1")
    basal_protocol = next(
        item for item in protocols if item.protocol_id == "train_basal"
    )
    shifted_protocol = next(
        item for item in protocols if item.protocol_id == "train_initial_gp_plus10"
    )

    basal = simulate_phase_b(basal_protocol, data_root=DATA_ROOT)
    shifted = simulate_phase_b(shifted_protocol, data_root=DATA_ROOT)
    indices = [STATE_INDEX["Gp"], STATE_INDEX["Gt"]]

    assert shifted.states[0, STATE_INDEX["Gp"]] > basal.states[0, STATE_INDEX["Gp"]]
    assert shifted.states[0, indices].sum() == pytest.approx(
        basal.states[0, indices].sum()
    )


@pytest.mark.parametrize(
    ("family", "target", "expected_shape"),
    [
        ("dalla_man", "Gp", (301, 12)),
        ("cstr", "T", (301, 3)),
        ("alien_device", "y", (601, 6)),
    ],
)
def test_private_simulators_are_finite_and_basic_gates_are_not_release_gates(
    family: str, target: str, expected_shape: tuple[int, int]
) -> None:
    protocols = phase_b_protocols(family, task="T4" if family == "dalla_man" else None)
    selected = tuple(item for item in protocols if item.split == "train")
    trajectories = tuple(
        simulate_phase_b(item, data_root=DATA_ROOT) for item in selected
    )
    report = audit_basic_gates(family, trajectories, target_name=target)

    assert trajectories[0].states.shape == expected_shape
    assert report.finite_rollout_fraction == 1.0
    assert report.finite_rollouts_pass
    assert not report.standalone_release_ready
    assert "task_scaled_sensitivity_rank" in report.pending_gates


def test_private_bundle_refuses_protocol_trajectory_mismatch(tmp_path: Path) -> None:
    protocols = phase_b_protocols("alien_device")[:2]
    trajectory = simulate_phase_b(protocols[0], data_root=DATA_ROOT)

    with pytest.raises(ValueError, match="order must match"):
        write_private_bundle(tmp_path, protocols, (trajectory,))


def test_private_bundle_marks_reference_unavailable_to_methods(tmp_path: Path) -> None:
    protocol = phase_b_protocols("cstr")[0]
    trajectory = simulate_phase_b(protocol, data_root=DATA_ROOT)

    write_private_bundle(tmp_path, (protocol,), (trajectory,))

    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert '"private_reference": true' in manifest
    assert '"available_to_discovery_methods": false' in manifest
