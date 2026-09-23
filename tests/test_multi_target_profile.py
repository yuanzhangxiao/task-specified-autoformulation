"""Joint output fitting, consistent Jacobians, failure dimensions and sealed replay."""

from dataclasses import replace
from time import monotonic

import numpy as np
import pytest

from autoformalism.data import TrainingScaler
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sensitivity_probe as sensitivity
from autoformalism.fitting.feasibility import EvaluationBudget, GuardedOracle
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.fitting.stagnation import RolloutOracle
from scripts.smoke_multi_target_fitting import control, run
from scripts.smoke_public_fitting import control as scalar_control


def oracle(tmp_path, *, cls=sensitivity.SymbolicOracle, **kwargs):
    request, train, _, _ = control()
    model, _, _ = public._lower(request)
    training = public.unpack_split(train)
    fitted = TrainingScaler().fit(training)
    scales = {
        c: fitted.scales[f"target:{c}"].standard_deviation
        for c in request.context.targets
    }
    settings = FitConfig(
        integration_method="Radau", relative_tolerance=1e-9, absolute_tolerance=1e-11
    )
    result = cls(
        sensitivity.SymbolicODE(model),
        training,
        scales,
        settings,
        tmp_path / "symbolic",
        monotonic() + 60,
        sensitivities=True,
        **kwargs,
    )
    production = RolloutOracle(
        model, training, scales, settings, tmp_path / "production", None
    )
    return result, production, result.vector(dict(request.parameter_guesses))


def test_all_channel_residuals_match_production_and_jacobian_matches_differences(
    tmp_path,
):
    symbolic, production, x = oracle(tmp_path)
    residual = symbolic(x)
    jacobian = symbolic.jacobian(x).copy()
    assert residual.shape == (2 * 3 * 21,)
    assert jacobian.shape == (len(residual), len(x))
    np.testing.assert_allclose(residual, production(x), rtol=2e-6, atol=2e-7)
    for column in range(len(x)):
        step = np.eye(len(x))[column] * 1e-5
        numerical = (production(x + step) - production(x - step)) / (2e-5)
        np.testing.assert_allclose(jacobian[:, column], numerical, rtol=2e-4, atol=2e-5)
    c = symbolic.names.index("c")
    assert np.all(jacobian[:42, c] == 0)  # Only the algebraic output depends on c.
    assert np.linalg.norm(jacobian[42:63, c]) > 0


@pytest.mark.parametrize(
    "scales",
    [0.5, {}, {"Gp": 1}, {"Gp": 1, "I": 1, "U": 0}, {"Gp": 1, "I": np.nan, "U": 1}],
)
def test_missing_invalid_or_shared_scalar_scales_rejected(scales):
    with pytest.raises(ValueError):
        sensitivity.observation_scales(("Gp", "I", "U"), scales)


@pytest.mark.parametrize("timeout", [False, True])
def test_failed_or_timed_out_rollout_keeps_full_residual_dimension(
    tmp_path, monkeypatch, timeout
):
    if timeout:
        obj, _, x = oracle(
            tmp_path,
            cls=GuardedOracle,
            budget=EvaluationBudget(monotonic() + 60, 10),
            point_seconds=5,
        )
        obj.with_sensitivities = False
    else:
        obj, _, x = oracle(tmp_path)

    def fail(*args, **kwargs):
        raise TimeoutError("point limit") if timeout else ValueError("bad output")

    monkeypatch.setattr(sensitivity, "symbolic_rollout", fail)
    result = obj(x)
    assert result.shape == (126,)
    assert np.all(result == obj.settings.failure_penalty)
    if not timeout:
        assert obj.last_jac.shape == (126, len(x))


def test_scalar_profile_settings_and_predictions_are_identical(tmp_path):
    old, train, val = scalar_control("collocation-single-target-v2")
    new = old.model_copy(update={"profile": "collocation-multi-target-v1"})
    assert public.profile_settings(old) == public.profile_settings(new)
    public.prepare_fit(new, train, val, tmp_path)
    assert public.inspect_fit(tmp_path)["capability_supported"]
    model, guesses, _ = public._lower(old)
    outputs = []
    for index, scale in enumerate((0.5, {"v01": 0.5})):
        obj = sensitivity.SymbolicOracle(
            sensitivity.SymbolicODE(model),
            public.unpack_split(train),
            scale,
            FitConfig(integration_method="Radau"),
            tmp_path / str(index),
            monotonic() + 30,
            sensitivities=True,
        )
        outputs.append((obj(obj.vector(guesses)), obj.last_jac))
    for column in range(2):
        np.testing.assert_array_equal(outputs[0][column], outputs[1][column])


def test_two_and_three_outputs_real_fit_child_fit_and_exact_resume(tmp_path):
    result = run(tmp_path)
    assert result["multi_target_child_fit"] == "pass"


def test_missing_target_cannot_be_silently_scored_as_complete():
    metric = {
        "normalized_mse": 0,
        "per_target_normalized_mse": {"Gp": 0, "I": 0},
        "failed_trajectories": [],
    }
    assert not public._metrics(metric, ("Gp", "I", "U")).available


def test_future_validation_labels_never_drive_any_output():
    request, _, val, truth = control()
    model, _, _ = public._lower(request)
    row = public.unpack_split(val).trajectories[0]
    changed = replace(
        row,
        targets={
            channel: np.concatenate((values[:1], values[1:] * 100 + 400))
            for channel, values in row.targets.items()
        },
    )
    settings = FitConfig(integration_method="Radau")
    first = simulate_trajectory(model, row, truth, {}, settings)
    second = simulate_trajectory(model, changed, truth, {}, settings)
    assert first.success and second.success
    for channel in request.context.targets:
        np.testing.assert_array_equal(
            first.predictions[channel], second.predictions[channel]
        )
