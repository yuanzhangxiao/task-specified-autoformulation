"""The concentration control solves a convex QP; negative predictions aren't clipped."""

from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import concentration_profile as c
from tests.test_insulin_revision_fit import specimen  # noqa: F401


def test_active_inequality_changes_coefficients_and_certifies_kkt():
    a, b = np.eye(2), np.array([-1.0, -2.0])
    x, residual, audit = c.constrained_least_squares(
        a, b, [-np.inf] * 2, [np.inf] * 2, [[-1.0, -1.0]], [1.0], monotonic() + 10
    )
    assert x == pytest.approx([0.0, 1.0], abs=1e-7)
    assert residual == pytest.approx(a @ x + b)
    assert audit["normalized_kkt_residual"] <= audit["kkt_tolerance"]
    assert audit["minimum_scaled_concentration"] >= -c.PRIMAL_TOLERANCE
    assert not audit["continuous_positivity_certified"]


def test_unconstrained_feasible_solution_is_unchanged():
    x, _, audit = c.constrained_least_squares(
        np.eye(2),
        [-1.0, -2.0],
        [0.0, 0.0],
        [10.0, 10.0],
        [[1.0, 1.0]],
        [0.0],
        monotonic() + 10,
    )
    assert x == pytest.approx([1.0, 2.0])
    assert audit["cutting_plane_solves"] == 0


def test_infeasible_domain_and_expired_budget_do_not_return_a_fit():
    with pytest.raises(c.projection.ProfileUnavailable):
        c.constrained_least_squares(
            np.eye(1), [0.0], [0.0], [1.0], [[1.0]], [-2.0], monotonic() + 5
        )
    with pytest.raises(TimeoutError):
        c.constrained_least_squares(
            np.eye(1), [0.0], [0.0], [1.0], [[1.0]], [0.0], monotonic() - 1
        )


def test_refined_constraint_grid_preserves_training_objective(specimen):  # noqa: F811
    from autoformalism.fitting.insulin_revision_fit import InsulinRevisionFit

    fit = InsulinRevisionFit(*specimen)
    plain = fit.profiler_factory(fit.oracle)
    positive = c.InsulinConcentrationProfile(fit.oracle)
    a = plain.design(fit.start, monotonic() + 10)
    b = positive.design(fit.start, monotonic() + 10)
    np.testing.assert_allclose(a["design"], b["design"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(a["offset"], b["offset"], rtol=1e-10, atol=1e-10)
    assert b["constraint_design"].shape[0] == 2 * 13
    p = positive.profile(fit.start, monotonic() + 10)
    assert p["joint_solve"]["status"] == "conditional_optimum"
    assert p["maximum_pairing_discrepancy"] < 1e-7
