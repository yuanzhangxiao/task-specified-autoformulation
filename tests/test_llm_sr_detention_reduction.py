"""Exact signed/zero quotient lifts preserve the frozen original model family."""

import json
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.fitting import llm_sr_detention_reduction as reduction
from autoformalism.fitting import llm_sr_detention_refit as refit
from autoformalism.fitting.llm_sr_instantaneous import compile_current_value
from tests.test_llm_sr_detention_refit import example


def domain():
    item, context, training = example(15)
    return item, context, refit.domain_policy(item, context, training)["bounds"]


def test_random_feasible_original_points_lift_without_a_frozen_gauge():
    _, _, bounds = domain()
    rng = np.random.default_rng(33)
    for _ in range(250):
        source = {name: rng.uniform(*pair) for name, pair in bounds.items()}
        reduced = reduction.reduce_parameters(source)
        lifted = reduction.lift_parameters(reduced, bounds)
        assert reduction.reduce_parameters(lifted) == reduced
        for name, value in lifted.items():
            assert Fraction(bounds[name][0]) <= value <= Fraction(bounds[name][1])
        low, high = reduction.feasible_g_interval(reduced, bounds)
        assert low <= Fraction(float(source["k_gw"])) <= high
        preferred = reduction.lift_parameters(
            reduced, bounds, preferred_g=source["k_gw"]
        )
        assert preferred == {name: Fraction(float(x)) for name, x in source.items()}


@pytest.mark.parametrize("g,kappa", [(0, 0), (0, 2), (0.5, 0), (-0.5, 0)])
def test_zero_product_has_no_division_or_artificial_sign_assumption(g, kappa):
    item, _, bounds = domain()
    source = item.parameters | {"k_gw": g, "kappa": kappa}
    reduced = reduction.reduce_parameters(source)
    lifted = reduction.lift_parameters(
        reduced, bounds, preferred_g=g, preferred_kappa=kappa
    )
    assert lifted["k_gw"] == g
    assert lifted["kappa"] == kappa
    assert reduction.reduce_parameters(lifted) == reduced


@pytest.mark.parametrize("g", [-0.7, 0, 0.7])
def test_positive_kappa_lower_bound_constrains_both_product_signs(g):
    item, _, bounds = domain()
    bounds = bounds | {"kappa": [0.5, 2.0]}
    source = item.parameters | {"k_gw": g, "kappa": 1}
    reduced = reduction.reduce_parameters(source)
    low, high = reduction.feasible_g_interval(reduced, bounds)
    if g == 0:
        assert low == high == 0
    elif g > 0:
        assert low >= Fraction(g) / 2 and high <= Fraction(g) / Fraction(0.5)
    else:
        assert low >= Fraction(g) / Fraction(0.5) and high <= Fraction(g) / 2
    assert (
        reduction.reduce_parameters(reduction.lift_parameters(reduced, bounds))
        == reduced
    )


def test_fixed_zero_kappa_permits_all_feasible_signed_g_values():
    item, _, bounds = domain()
    bounds = bounds | {"kappa": [0, 0]}
    reduced = reduction.reduce_parameters(item.parameters | {"kappa": 0})
    low, high = reduction.feasible_g_interval(reduced, bounds)
    assert low < high
    for g in (low, high):
        lifted = reduction.lift_parameters(reduced, bounds, preferred_g=g)
        assert lifted["kappa"] == 0
    with pytest.raises(ValueError, match="zero kappa"):
        reduction.lift_parameters(reduced | {"d": 1}, bounds)


def test_exact_boundary_roundtrips_and_correlated_domain_rejection():
    item, _, bounds = domain()
    for side in (0, 1):
        source = {name: pair[side] for name, pair in bounds.items()}
        reduced = reduction.reduce_parameters(source)
        lifted = reduction.lift_parameters(reduced, bounds)
        assert reduction.reduce_parameters(lifted) == reduced
    reduced = reduction.reduce_parameters(item.parameters)
    # Each coefficient is finite, but no original g can realize this combination.
    with pytest.raises(ValueError, match="no feasible original lift"):
        reduction.lift_parameters(reduced | {"M": -1e6}, bounds)
    with pytest.raises(ValueError, match="no feasible original lift"):
        reduction.lift_parameters(reduced | {"d": 1e6}, bounds)
    with pytest.raises(ValueError, match="unchanged coordinate"):
        reduction.lift_parameters(reduced | {"dt": 0}, bounds)


@pytest.mark.parametrize("g,kappa", [(0.5, 2), (-0.5, 2), (0, 2), (0.5, 0)])
def test_original_piecewise_rhs_equivalence_including_dry_and_capacity_guards(g, kappa):
    item, context, bounds = domain()
    model = compile_current_value(item, context)
    source = item.parameters | {
        "k_gw": g,
        "kappa": kappa,
        "k_loss": 0.2,
        "h_max": 0.25,
        "dt": 2,
    }
    reduced = reduction.reduce_parameters(source)
    lifted = {
        n: float(v) for n, v in reduction.lift_parameters(reduced, bounds).items()
    }
    for h in (-0.1, 0, 0.1, 0.2, 0.3, 1, 10):
        for rate in (0, 0.1, 2, 10):
            values = {
                "inflow_down": rate * 1200,
                "area_down": 1200,
                "crest_down": 0.2,
                "warning_depth": 0.6,
            }
            forcing = SimpleNamespace(value=lambda n, _t, v=values: v[n])
            before = model.rhs(0, [h], source, forcing)
            after = model.rhs(0, [h], lifted, forcing)
            np.testing.assert_allclose(after, before, atol=2e-14, rtol=2e-14)


def test_certificate_reports_exact_identity_and_float_representation_limits():
    item, _, bounds = domain()
    certificate = reduction.reduction_certificate(item.parameters, bounds)
    assert certificate["rank_ceiling"] == 8
    assert certificate["declared_coordinates"] == 9
    assert certificate["original_bounds_satisfied"]
    assert certificate["exact_combinations_preserved"]
    assert not certificate["reduced_domain_is_cartesian_box"]
    assert not certificate["optimizer_coordinates_changed"]
    assert not certificate["unique_original_coefficients_certified"]
    assert not certificate["scientific_validity_certified"]
    assert certificate["floating_representative_requires_independent_rollout"]
    assert certificate["representative_parameters"] != item.parameters
    assert max(certificate["representative_roundoff_absolute"].values()) < 1e-13
    assert json.loads(json.dumps(certificate)) == certificate


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.1"])
def test_invalid_coordinate_values_fail_closed(value):
    item, _, bounds = domain()
    with pytest.raises(ValueError, match="finite real"):
        reduction.reduction_certificate(item.parameters | {"C": value}, bounds)


@pytest.mark.parametrize(
    "name,pair", [("kappa", [-1, 1]), ("dt", [0, 1]), ("C", [1, 0])]
)
def test_invalid_bound_domains_do_not_claim_exact_reduction(name, pair):
    item, _, bounds = domain()
    with pytest.raises(ValueError):
        reduction.reduction_certificate(item.parameters, bounds | {name: pair})


def test_roster_and_outside_original_domain_fail_closed():
    item, _, bounds = domain()
    with pytest.raises(ValueError, match="roster"):
        reduction.reduce_parameters(item.parameters | {"unused": 1})
    with pytest.raises(ValueError, match="outside"):
        reduction.reduction_certificate(item.parameters | {"h_max": -1}, bounds)
