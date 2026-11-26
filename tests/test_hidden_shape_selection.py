"""Check event-level decisions, latent equivalence and observation-prefix forecasts."""

from fractions import Fraction

import numpy as np
import pytest

from autoformalism.research.hidden_shape_selection import (
    compare_event_errors,
    equivalent_power_readouts,
)
from scripts.check_hidden_pathway_controls import (
    PREFIX_END,
    TIME,
    event,
    prefix_forecast,
)


def test_compares_same_events_with_a_material_margin():
    errors = {"a": {"1": 0.0, "2": 0.0, "3": 0.1}, "b": {"1": 1.0, "2": 1.0, "3": 0.1}}
    assert compare_event_errors(errors, 0.2)["winner"] == "a"
    errors["a"]["3"] = 0.5
    assert compare_event_errors(errors, 0.2)["winner"] is None
    errors["b"].pop("1")
    with pytest.raises(ValueError, match="same events"):
        compare_event_errors(errors, 0.2)


def test_invalid_or_indistinguishable_forecasts_never_decide():
    assert compare_event_errors({"a": {"1": 1.0}, "b": {"1": 1.0}}, 0)["winner"] is None
    errors = {"a": {"1": 0.0, "2": 1.0}, "b": {"1": 0.0, "2": 1.0}}
    assert compare_event_errors(errors, 0)["status"] == "unresolved"
    errors["b"]["2"] = float("nan")
    assert compare_event_errors(errors, 0)["reason"] == "incomplete forecasts"


def test_coordinate_change_preserves_the_output_exactly():
    assert equivalent_power_readouts(2, 1, Fraction(2, 5), Fraction(4, 5))
    assert not equivalent_power_readouts(2, 1, Fraction(2, 5), Fraction(6, 5))
    with pytest.raises(ValueError, match="exact positive rational"):
        equivalent_power_readouts(2, 1, 0.4, 0.8)


def test_forecast_from_only_the_permitted_prefix():
    row = event("offset_plus_square", 1.5, 2, 1, 2, noise=0.0)
    y = np.array(row["targets"]["y"])
    prefix = y[TIME <= PREFIX_END]
    actual = prefix_forecast(TIME, prefix, 0.4 + 1.3, 0.4 * 1.3, 1 + 1.5**2)
    np.testing.assert_allclose(actual, y, atol=2e-14)
    with pytest.raises(ValueError, match="only the declared prefix"):
        prefix_forecast(TIME, y, 1.7, 0.52, 3.25)


def test_repeated_pole_forecast():
    y = (1 + TIME) * np.exp(-TIME)
    result = prefix_forecast(TIME, y[TIME <= PREFIX_END], 2.0, 1.0, 0.0)
    np.testing.assert_allclose(result, y, atol=1e-14)
