"""Perturbed CSTR dynamics: a bounded rate multiplier that keeps the preparation."""

from pathlib import Path

import numpy as np
import pytest

from autoformalism.benchmarks import phase_b_protocols, simulate_phase_b
from autoformalism.benchmarks.phase_b_generation import (
    CSTR_PERTURBATION,
    cstr_rate_multiplier,
)
from autoformalism.benchmarks.phase_c_protocols import phase_c_protocols

DATA_ROOT = Path("data_raw")
SPEC = (
    DATA_ROOT
    / "benchmark5_anonymous_nonlinear_process/private/system_specification.json"
)
needs_private = pytest.mark.skipif(
    not SPEC.exists(), reason="private CSTR specification absent"
)


def test_multiplier_is_one_at_equilibrium_and_bounded() -> None:
    a = CSTR_PERTURBATION["amplitude"]
    assert cstr_rate_multiplier(365.0, 365.0, "perturbed") == 1.0
    assert cstr_rate_multiplier(300.0, 365.0, "canonical") == 1.0
    hot = cstr_rate_multiplier(500.0, 365.0, "perturbed")
    cold = cstr_rate_multiplier(250.0, 365.0, "perturbed")
    assert hot == pytest.approx(1 - a) and cold == pytest.approx(1 + a)
    assert (
        cstr_rate_multiplier(370.0, 365.0, "perturbed")
        < 1.0
        < cstr_rate_multiplier(360.0, 365.0, "perturbed")
    )


def test_unknown_cstr_dynamics_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown CSTR dynamics"):
        cstr_rate_multiplier(365.0, 365.0, "other")  # type: ignore[arg-type]


@needs_private
def test_perturbed_cstr_keeps_the_basal_preparation() -> None:
    basal = next(p for p in phase_c_protocols("cstr") if p.protocol_id == "train_zero")
    canonical = simulate_phase_b(basal, data_root=DATA_ROOT)
    perturbed = simulate_phase_b(basal, dynamics="perturbed", data_root=DATA_ROOT)
    assert np.allclose(perturbed.states, canonical.states, atol=1e-9)
    assert np.allclose(perturbed.states[0], canonical.states[0])


@needs_private
def test_perturbed_cstr_changes_only_the_driven_response() -> None:
    driven = [p for p in phase_c_protocols("cstr") if p.split != "test"]
    gaps = []
    for protocol in driven:
        canonical = simulate_phase_b(protocol, data_root=DATA_ROOT)
        perturbed = simulate_phase_b(
            protocol, dynamics="perturbed", data_root=DATA_ROOT
        )
        assert np.array_equal(perturbed.time, canonical.time)
        assert np.array_equal(perturbed.inputs, canonical.inputs)
        assert np.array_equal(perturbed.states[0], canonical.states[0])
        gaps.append(
            float(np.max(np.abs(perturbed.states[:, 1] - canonical.states[:, 1])))
        )
    assert max(gaps) > 1.0


@needs_private
def test_only_dalla_and_cstr_define_perturbed_dynamics() -> None:
    alien = phase_b_protocols("alien_device")[0]
    with pytest.raises(ValueError, match="perturbed dynamics condition"):
        simulate_phase_b(alien, dynamics="perturbed", data_root=DATA_ROOT)
