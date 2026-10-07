"""Larger controls retain full feedback, anchored coordinates and training isolation."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import larger_coupled_inputs as controls
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.identifiable_campaign import SETTINGS
from autoformalism.fitting.profiled_coupled import ProfiledCoupled
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


@pytest.fixture(scope="module")
def larger_inputs(tmp_path_factory):
    path = tmp_path_factory.mktemp("larger-inputs") / "inputs.json"
    controls.export_inputs(path)
    return path


def test_generic_starts_and_separation(larger_inputs):
    data = read_seal(larger_inputs)
    before = controls.bases(data)
    assert len(before) == 12
    assert controls.export_inputs(larger_inputs)["identity"] == public.content_sha256(
        data
    )
    for n in (3, 6):
        for seed in range(3):
            a, b = (
                before[f"linear{n}{suffix}_s{seed}"] for suffix in ("", "_fast_slow")
            )
            assert a["start"] == b["start"]
            assert len(a["start"]) == 2 * n
    for case in data["cases"].values():
        assert case["identifiability"]["passed"]
        case["validation"] = {"poison": True}
        case["reference_parameters"] = {"poison": True}
        case["identifiability"] = {"poison": True}
    assert controls.bases(data) == before
    for base in before.values():
        assert set(base) == {"request", "start", "nodes", "training", "coordinates"}


@pytest.mark.parametrize("name", controls.CASES)
def test_matrix_exponential_basis_and_sensitivity(larger_inputs, name):
    data = read_seal(larger_inputs)
    case = data["cases"][name]
    base = controls.bases(data)[f"{name}_s0"]
    model, _, _ = public._lower(PublicFitRequest.model_validate(base["request"]))
    system = SymbolicODE(model)
    profile = ProfiledCoupled(system)
    n = case["states"]
    assert len(profile.outer_names) == len(profile.gain_names) == n
    assert profile.audit["augmented_states"] == n * (n + 1) ** 2
    assert set(profile.gain_names) == {"gain"} | {
        f"init_x{i}_value" for i in range(1, n)
    }
    row = public.unpack_split(
        PublicSplit.model_validate(case["training"])
    ).trajectories[0]
    for parameters in (case["reference_parameters"], base["start"]):
        matrix = controls.matrix(n, parameters)
        assert matrix + matrix.T == pytest.approx(
            -2 * np.diag([parameters[f"a{i}"] for i in range(n)])
        )
        assert np.max(np.linalg.eigvals(matrix).real) < 0
        vector = np.array([parameters[name] for name in system.names])
        design, offset, da, db, count = profile.trajectory(
            row, vector, SETTINGS, monotonic() + 30
        )
        predicted, jac, _, _ = symbolic_rollout(
            system, row, vector, SETTINGS, monotonic() + 30, sensitivities=True
        )
        truth = controls.reference(case["training"]["rows"][0], parameters, n)[:, 0]
        assert predicted[:, 0] == pytest.approx(truth, abs=2e-7)
        assert offset + design @ vector[profile.gain_indices] == pytest.approx(
            truth, abs=2e-7
        )
        assert design == pytest.approx(jac[:, 0, profile.gain_indices], abs=2e-6)
        outer = db + np.einsum("nkp,k->np", da, vector[profile.gain_indices])
        assert outer == pytest.approx(jac[:, 0, profile.outer_indices], abs=2e-6)
        assert count["segments"] == 24


def test_incomplete_or_modified_roster_rejected(larger_inputs):
    data = read_seal(larger_inputs)
    bad = deepcopy(data)
    bad["commons"].pop("linear6_s2")
    with pytest.raises(ValueError, match="all twelve"):
        controls.bases(bad)
    bad = deepcopy(data)
    bad["commons"]["linear6_s2"]["request"]["parameter_guesses"]["gain"] = 3.0
    with pytest.raises(ValueError, match="request/start"):
        controls.bases(bad)
    bad = deepcopy(data)
    bad["cases"]["linear3"]["training"]["name"] = "val"
    with pytest.raises(ValueError):
        controls.bases(bad)


def test_deficient_rank_blocks_case(monkeypatch):
    original = controls.reference

    def unidentifiable(row, parameters, n):
        return original(row, parameters | {"gain": 1.0}, n)

    monkeypatch.setattr(controls, "reference", unidentifiable)
    with pytest.raises(ValueError, match="unqualified control"):
        controls.make_case("linear3")
