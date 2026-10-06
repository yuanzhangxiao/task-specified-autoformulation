"""Structural and numerical checks for bounded terminal-output variable projection."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.identifiable_campaign import SETTINGS
from autoformalism.fitting.profiled_output import ProfiledOutput, project
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from tests.profiled_fixture import TRUTH, make_base


@pytest.fixture(scope="module")
def base():
    return make_base()[0]


def system(base):
    return SymbolicODE(
        public._lower(PublicFitRequest.model_validate(base["request"]))[0]
    )


def test_symbolic_certificate_and_exact_filtered_reconstruction(base):
    s = system(base)
    p = ProfiledOutput(s)
    assert set(p.gain_names) == {"c", "d"}
    assert p.audit["outer_unknowns"] == 3
    train = public.unpack_split(PublicSplit.model_validate(base["training"]))
    for vector in (base["start"], TRUTH):
        theta = np.array([vector[n] for n in s.names])
        for row in train.trajectories:
            a, b, da, db, counts = p.trajectory(row, theta, SETTINGS, monotonic() + 10)
            predicted, jac, _, _ = symbolic_rollout(
                s, row, theta, SETTINGS, monotonic() + 10, sensitivities=True
            )
            assert b + a @ theta[p.gain_indices] == pytest.approx(
                predicted[:, 0], abs=1e-7
            )
            derivative = db + np.einsum("nkp,k->np", da, theta[p.gain_indices])
            assert derivative == pytest.approx(jac[:, 0, p.outer_indices], abs=2e-6)
            aa, bb, _, _, _ = p.trajectory(
                row, theta, SETTINGS, monotonic() + 10, sensitivities=False
            )
            assert aa == pytest.approx(a, abs=1e-7)
            assert bb == pytest.approx(b, abs=1e-7)
            assert counts["segments"] >= 8  # retain the short public pulse


@pytest.mark.parametrize(
    "y_rhs,z_rhs,gains,match",
    [
        ("-b*y+c*tanh(z)-d*u", "-a*z+u+y", None, "feeds back"),
        ("-b*y*y+c*tanh(z)-d*u", "-a*z+u", None, "not linear"),
        ("-b*y+c*c*tanh(z)-d*u", "-a*z+u", ("c", "d"), "not conditionally"),
        ("-b*y+c*d*tanh(z)", "-a*z+u", ("c", "d"), "jointly affine"),
        ("-b*y+c*tanh(z)-d*u", "-a*z+c*u", ("c", "d"), "not conditionally"),
        ("-b*y+c*tanh(z)-d*u", "-a*z+u", ("init_z_value",), "not conditionally"),
    ],
)
def test_unsafe_structure_is_rejected(base, y_rhs, z_rhs, gains, match):
    b = deepcopy(base)
    b["request"]["base_candidate"]["state_equations"] = [
        {"state": "y", "rhs": y_rhs},
        {"state": "z", "rhs": z_rhs},
    ]
    with pytest.raises(ValueError, match=match):
        ProfiledOutput(system(b), gains)


def test_observation_mapping_and_expired_deadline(base):
    b = deepcopy(base)
    b["request"]["base_candidate"]["observation_mappings"][0]["expression"] = "y+z"
    b["request"]["initialization_plan"]["rules"]["y"] = {
        "initial": {"mode": "value", "guess": 0.2}
    }
    with pytest.raises(ValueError, match="identity-observed"):
        ProfiledOutput(system(b))
    s = system(base)
    row = public.unpack_split(
        PublicSplit.model_validate(base["training"])
    ).trajectories[0]
    with pytest.raises(TimeoutError):
        ProfiledOutput(s).trajectory(
            row, np.array([TRUTH[n] for n in s.names]), SETTINGS, 0
        )


@pytest.mark.parametrize("active", [0, -1, 1])
def test_profiled_residual_derivative_including_residual_correction(active):
    rng = np.random.default_rng(481)
    a0, da = rng.normal(size=(30, 3)), rng.normal(size=(30, 3, 2))
    db = rng.normal(size=(30, 2))
    truth = np.array([0.8, {-1: -0.7, 0: 1.1, 1: 10.7}[active], 1.6])
    b0 = -a0 @ truth + rng.normal(size=30) * 0.1
    lo, hi = np.zeros(3), np.ones(3) * 10

    def at(q):
        return project(a0 + np.einsum("nkp,p->nk", da, q), b0 + db @ q, da, db, lo, hi)

    center = at(np.zeros(2))
    assert center["audit"]["active_mask"][1] == active
    for j in range(2):
        step = np.eye(2)[j] * 1e-6
        fd = (at(step)["residual"] - at(-step)["residual"]) / 2e-6
        assert center["jacobian"][:, j] == pytest.approx(fd, abs=1e-7)
    assert (
        np.sum(center["residual"] ** 2)
        <= np.sum((a0 @ np.clip(truth, lo, hi) + b0) ** 2) + 1e-10
    )


def test_rank_deficiency_and_invalid_projection_not_given_zero_jacobian():
    a = np.ones((6, 2))
    with pytest.raises(ValueError, match="rank-deficient"):
        project(a, np.ones(6), np.zeros((6, 2, 1)), np.zeros((6, 1)), [0, 0], [10, 10])
    a[0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid"):
        project(a, np.ones(6), np.zeros((6, 2, 1)), np.zeros((6, 1)), [0, 0], [10, 10])


def test_complete_profiled_rollout_jacobian_by_outer_finite_differences(base):
    s = system(base)
    profiled = ProfiledOutput(s)
    train = public.unpack_split(PublicSplit.model_validate(base["training"]))
    theta = np.array([base["start"][n] for n in s.names])

    def at(vector):
        rows = [
            profiled.trajectory(row, vector, SETTINGS, monotonic() + 10)
            for row in train.trajectories
        ]
        return project(
            np.concatenate([r[0] for r in rows]),
            np.concatenate(
                [
                    r[1] - t.targets["y"]
                    for r, t in zip(rows, train.trajectories, strict=True)
                ]
            ),
            np.concatenate([r[2] for r in rows]),
            np.concatenate([r[3] for r in rows]),
            [0.001, 0.001],
            [100, 100],
        )

    center = at(theta)
    for k, index in enumerate(profiled.outer_indices):
        step = np.eye(len(theta))[index] * 1e-4
        fd = (at(theta + step)["residual"] - at(theta - step)["residual"]) / 2e-4
        assert center["jacobian"][:, k] == pytest.approx(fd, abs=2e-5)
