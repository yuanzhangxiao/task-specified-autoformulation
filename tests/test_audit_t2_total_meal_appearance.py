from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from scripts import audit_t2_total_meal_appearance as audit


def test_piecewise_linear_input_integral_verifies_against_integrated_filter():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 20.0])
    q = np.array([0.0, 6.0, 12.0, 6.0, 0.0, 0.0, 0.0])
    gain, tau = 2.0, 3.0
    d, area = 0.0, 0.0
    expected = [0.0]
    for a, b in pairwise(t):

        def rhs(time, states):
            return [(np.interp(time, t, q) - states[0]) / tau, gain * states[0]]

        sol = solve_ivp(rhs, [a, b], [d, area], rtol=1e-12, atol=1e-12)
        d, area = sol.y[:, -1]
        expected.append(area)
    actual = audit.filter_source(t, q, gain, tau)
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    assert audit._window(t, {"area": actual}, 2, 8)["area"] == pytest.approx(
        expected[-2] - expected[2]
    )
    with pytest.raises(ValueError, match="measured"):
        audit._window(t, {"area": actual}, 1, 9)
    with pytest.raises(ValueError, match="positive time constant"):
        audit.filter_source(t, q, 1, -2)
