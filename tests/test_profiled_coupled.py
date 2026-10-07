"""Coupled basis reconstruction, latent initial profiling, and rejection boundaries."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import coupled_campaign as campaign
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.identifiable_campaign import SETTINGS
from autoformalism.fitting.profiled_coupled import ProfiledCoupled
from autoformalism.fitting.profiled_output import ProfiledOutput, project
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    path = tmp_path_factory.mktemp("coupled") / "inputs.json"
    campaign.export_inputs(path)
    return read_seal(path)


def system(base):
    return SymbolicODE(
        public._lower(PublicFitRequest.model_validate(base["request"]))[0]
    )


@pytest.mark.parametrize("case", campaign.CASES)
def test_coupled_reconstruction_and_derivatives(data, case):
    base = campaign.bases(data)[f"{case}_s0"]
    s = system(base)
    profiled = ProfiledCoupled(s)
    assert set(profiled.gain_names) == {"c", "init_z_value"}
    assert set(profiled.outer_names) == {"a", "b"}
    assert profiled.audit["profiled_initial_parameters"] == ["init_z_value"]
    with pytest.raises(ValueError, match="feeds back"):
        ProfiledOutput(s)
    row = public.unpack_split(
        PublicSplit.model_validate(base["training"])
    ).trajectories[1]
    for parameters in (base["start"], data["cases"][case]["reference_parameters"]):
        theta = np.array([parameters[n] for n in s.names])
        a, b, da, db, counts = profiled.trajectory(
            row, theta, SETTINGS, monotonic() + 10
        )
        predicted, jac, _, _ = symbolic_rollout(
            s, row, theta, SETTINGS, monotonic() + 10, sensitivities=True
        )
        assert b + a @ theta[profiled.gain_indices] == pytest.approx(
            predicted[:, 0], abs=2e-7
        )
        derivative = db + np.einsum("nkp,k->np", da, theta[profiled.gain_indices])
        assert derivative == pytest.approx(jac[:, 0, profiled.outer_indices], abs=2e-6)
        assert a == pytest.approx(jac[:, 0, profiled.gain_indices], abs=2e-6)
        aa, bb, _, _, _ = profiled.trajectory(
            row, theta, SETTINGS, monotonic() + 10, sensitivities=False
        )
        assert aa == pytest.approx(a, abs=2e-7)
        assert bb == pytest.approx(b, abs=2e-7)
        assert counts["segments"] >= 12  # all public input corners retained


@pytest.mark.parametrize(
    "rhs,gains,match",
    [
        ("-a*z-b*y+c*u+y*z", None, "jointly affine in states"),
        ("-a*z-b*y+c*u", ("a",), "conditionally affine"),
        ("-a*z-b*y+c*c*u", ("c",), "conditionally affine"),
        ("-a*z-b*y+c*init_z_value*u", ("c", "init_z_value"), "jointly affine"),
    ],
)
def test_unsafe_blocks_rejected(data, rhs, gains, match):
    base = deepcopy(campaign.bases(data)["coupled_linear_s0"])
    # init_z_value is introduced by lowering; use an already lowered model for
    # cross-products with initializer parameters.
    model = public._lower(PublicFitRequest.model_validate(base["request"]))[0]
    raw = model.validated.candidate.model_dump(mode="json")
    raw["state_equations"][1]["rhs"] = rhs
    candidate = type(model.validated.candidate).model_validate(raw)
    from autoformalism.expressions import compile_candidate

    with pytest.raises(ValueError, match=match):
        ProfiledCoupled(
            SymbolicODE(compile_candidate(candidate, model.validated.context)), gains
        )


def test_profiled_jacobian_with_latent_initial_at_interior_and_bound(data):
    base = campaign.bases(data)["coupled_linear_s0"]
    s = system(base)
    p = ProfiledCoupled(s)
    rows = public.unpack_split(
        PublicSplit.model_validate(base["training"])
    ).trajectories
    theta = np.array([base["start"][n] for n in s.names])
    for initial_upper in (10.0, -2.0):
        lower = np.array([0.001 if n == "c" else -10 for n in p.gain_names])
        upper = np.array([100.0 if n == "c" else initial_upper for n in p.gain_names])

        def at(vector, lower=lower, upper=upper):
            values = [p.trajectory(r, vector, SETTINGS, monotonic() + 15) for r in rows]
            return project(
                np.concatenate([v[0] for v in values]),
                np.concatenate(
                    [v[1] - r.targets["y"] for v, r in zip(values, rows, strict=True)]
                ),
                np.concatenate([v[2] for v in values]),
                np.concatenate([v[3] for v in values]),
                lower,
                upper,
            )

        center = at(theta)
        if initial_upper < 0:
            assert (
                center["audit"]["active_mask"][p.gain_names.index("init_z_value")] == 1
            )
        for j, index in enumerate(p.outer_indices):
            delta = np.eye(len(theta))[index] * 1e-4
            fd = (at(theta + delta)["residual"] - at(theta - delta)["residual"]) / 2e-4
            assert center["jacobian"][:, j] == pytest.approx(fd, abs=3e-5)


def test_nonidentity_output_and_timeout(data):
    base = deepcopy(campaign.bases(data)["coupled_linear_s0"])
    base["request"]["base_candidate"]["observation_mappings"][0]["expression"] = "y+z"
    base["request"]["initialization_plan"]["rules"]["y"] = {
        "initial": {"mode": "value", "guess": 0.2}
    }
    with pytest.raises(ValueError, match="identity-observed"):
        ProfiledCoupled(system(base))
    base = campaign.bases(data)["coupled_linear_s0"]
    s = system(base)
    row = public.unpack_split(
        PublicSplit.model_validate(base["training"])
    ).trajectories[0]
    with pytest.raises(TimeoutError):
        ProfiledCoupled(s).trajectory(
            row, np.array([base["start"][n] for n in s.names]), SETTINGS, 0
        )
