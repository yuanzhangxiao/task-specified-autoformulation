"""Channel-name invariance of the opt-in scalar collocation adapter."""

from __future__ import annotations

import json
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.sensitivity_probe import SymbolicODE, SymbolicOracle
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from scripts.smoke_public_fitting import control


def renamed_control():
    """Change only a public channel spelling, including its boundary map."""
    request, train, val = control("collocation-single-target-v2")
    return (
        PublicFitRequest.model_validate_json(
            request.model_dump_json().replace("v01", "glucose")
        ),
        PublicSplit.model_validate_json(
            train.model_dump_json().replace("v01", "glucose")
        ),
        PublicSplit.model_validate_json(
            val.model_dump_json().replace("v01", "glucose")
        ),
    )


def test_named_profile_keeps_legacy_settings_and_capability(tmp_path):
    request, train, val = renamed_control()
    legacy, _, _ = control("collocation-feasible-v1")
    assert public.profile_settings(request) == public.profile_settings(legacy)
    public.prepare_fit(request, train, val, tmp_path / "named")
    assert public.inspect_fit(tmp_path / "named")["capability_supported"]
    old = request.model_copy(update={"profile": "collocation-feasible-v1"})
    public.prepare_fit(old, train, val, tmp_path / "old")
    assert not public.inspect_fit(tmp_path / "old")["capability_supported"]


def test_named_oracle_residuals_and_sensitivities_are_unchanged(tmp_path):
    outputs = []
    for i, (request, train, _) in enumerate(
        (control("collocation-single-target-v2"), renamed_control())
    ):
        model, guesses, _ = public._lower(request)
        oracle = SymbolicOracle(
            SymbolicODE(model),
            public.unpack_split(train),
            0.5,
            FitConfig(integration_method="Radau"),
            tmp_path / str(i),
            monotonic() + 30,
            sensitivities=True,
        )
        vector = oracle.vector(guesses)
        outputs.append((oracle(vector), oracle.last_jac))
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    np.testing.assert_allclose(outputs[0][1], outputs[1][1], rtol=0, atol=0)


def test_named_collocation_real_fit_and_resume(tmp_path):
    request, train, val = renamed_control()
    public.prepare_fit(request, train, val, tmp_path)
    result = public.execute_fit(tmp_path)
    assert result.status == "complete", result.message
    assert result.training.normalized_mse < 1e-8
    assert result.validation.normalized_mse < 1e-8
    assert set(result.training.per_target_normalized_mse) == {"glucose"}
    backend = json.loads((tmp_path / "backend_result.json").read_text())
    assert backend["initializer"]["success"]
    assert public.execute_fit(tmp_path) == result


@pytest.mark.parametrize(
    "profile", ["collocation-feasible-v1", "collocation-single-target-v2"]
)
def test_multi_target_remains_unsupported(tmp_path, profile):
    request, train, val = control(profile, multiple_targets=True)
    public.prepare_fit(request, train, val, tmp_path)
    assert not public.inspect_fit(tmp_path)["capability_supported"]
    assert public.execute_fit(tmp_path).status == "capability_unsupported"
    assert not (tmp_path / "started.json").exists()
