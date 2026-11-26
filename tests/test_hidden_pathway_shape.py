"""Check observable weak elimination under signed, finite-level input schedules."""

import numpy as np
import pytest

from autoformalism.research.hidden_pathway_shape import (
    LINEAR,
    PREVIOUS,
    fit_hidden_pathway,
    forecast_hidden_pathway,
)


def row(label, time, source, output):
    return {
        "trajectory_id": label,
        "time": np.asarray(time).tolist(),
        "inputs": {"a": np.asarray(source).tolist()},
        "targets": {"v": np.asarray(output).tolist()},
        "auxiliaries": {},
        "covariates": {},
    }


def clean(levels=(0, 1, 2, 3), rates=(0.4, 1.3)):
    a, b = rates
    t = np.linspace(0, 8, 321)
    rows = []
    for level in levels:
        for j in (0, 1):
            # Separate free initial transients; the fitted algorithm sees v alone.
            A, B = 0.7 + level * 0.13 + j * 0.6, 0.3 - 0.14 * level - j * 0.5
            out = (1 + level**2) / (a * b) + A * np.exp(-a * t) + B * np.exp(-b * t)
            rows.append(row(f"train_{level}_{j}", t, np.full(len(t), level), out))
    return rows


def fit(rows, interpolation=PREVIOUS, **kwargs):
    return fit_hidden_pathway(
        rows, target="v", source="a", interpolation=interpolation, **kwargs
    )


def test_unknown_curve_and_pole_pair_recovered_with_changing_initials():
    # Bound follows from the linear-interpolation remainder dt² max |v''|/8.
    result = fit(clean(), interpolation=LINEAR, output_error=0.0002)
    assert result["status"] == "estimated" and result["rank"] == 6
    np.testing.assert_allclose(result["q"], [1, 2, 5, 10], rtol=1e-5)
    np.testing.assert_allclose(
        [result["rate_sum"], result["rate_product"]], [1.7, 0.52], rtol=1e-5
    )
    families = result["family_screen"]
    assert families["quadratic"]["status"] == families["cubic"]["status"] == "estimated"
    assert (
        families["affine"]["status"] == families["fixed_tanh"]["status"] == "excluded"
    )
    assert fit(clean())["family_screen"]["affine"]["reason"] == "no_error_bound"


def test_two_input_levels_leave_shapes_equivalent():
    f = fit(clean(range(2)), output_error=0.0002)
    assert f["status"] == "estimated"
    screen = f["family_screen"]
    assert screen["affine"]["status"] == screen["fixed_tanh"]["status"] == "estimated"
    assert screen["quadratic"]["reason"] == "rank_or_condition"
    np.testing.assert_allclose(screen["affine"]["q"], screen["fixed_tanh"]["q"])


def test_no_excitation_continuous_ramps_and_nonphysical_poles():
    const = clean()
    for r in const:
        r["targets"]["v"] = [r["inputs"]["a"][0]] * len(r["time"])
    assert fit(const)["reason"] == "rank_or_condition"
    assert fit(clean(), max_condition=2)["reason"] == "rank_or_condition"
    for r in const:
        r["inputs"]["a"] = r["time"]
    assert fit(const, LINEAR)["reason"] == "insufficient_level_windows"
    assert fit(const, "unknown")["reason"] == "unknown_interpolation"
    assert fit(clean(rates=(-0.2, 0.6)))["reason"] == "nominal_poles"


def test_switched_inputs_forecast_preserves_continuity_and_unknown_levels_fail():
    rng = np.random.default_rng(120)
    data = []
    rates = {"rate_sum": 1.7, "rate_product": 0.52, "levels": [0, 1], "q": [0, 2]}
    for j in range(4):
        times = np.r_[0, np.cumsum(rng.uniform(0.03, 0.05, 400))]
        u = np.repeat(rng.integers(0, 2, size=21), 20)[: len(times)]
        z = forecast_hidden_pathway(
            times, u, y0=j + 1, dy0=0.1 - j, interpolation=PREVIOUS, **rates
        )
        data.append(row(f"train_{j}", times, u, z))
    f = fit(data, output_error=0.003)
    assert f["status"] == "estimated" and f["rank"] == 4
    np.testing.assert_allclose(f["q"], [0, 2], atol=0.002)
    with pytest.raises(ValueError, match="ramp"):
        forecast_hidden_pathway(
            [0, 1], [0, 1], y0=1, dy0=0, interpolation=LINEAR, **rates
        )
    with pytest.raises(ValueError, match="unestimated"):
        forecast_hidden_pathway(
            [0, 1], [3, 4], y0=1, dy0=0, interpolation=PREVIOUS, **rates
        )
    data[0]["trajectory_id"] = "test_0"
    with pytest.raises(ValueError, match="train"):
        fit(data)
