import casadi as ca
import numpy as np
import pytest

from scripts.audit_t2_disposal_balance import balance, event, integrated_disposal


def toy(text="x-2*U"):
    return {
        "formula": {"equations": {"Gp": text}, "parameters": {}},
        "scales": {"Gp": 2, "U": 1},
    }


def test_irregular_grid_consistent_affine_balance():
    t = [0, 1, 4]
    u = [1, 3, 2]
    area = integrated_disposal(np.array(t)) @ u
    np.testing.assert_allclose(area, [0, 2, 9.5])
    g = 60 + 4 * np.array(t) - 2 * area
    row = {
        "trajectory_id": "train_000",
        "time": t,
        "inputs": {"x": [4] * 3},
        "auxiliaries": {},
        "targets": {"Gp": g, "U": u},
    }
    r = event(row, toy())
    assert r["measured_U_glucose_scaled_sse"] < 1e-24
    row["targets"]["Gp"] = g + np.array([0, 0, 10])
    inconsistent = event(row, toy())
    free = (
        inconsistent["free_U_glucose_scaled_sse"]
        + inconsistent["free_U_disposal_scaled_sse"]
    )
    assert 0 < free < inconsistent["measured_U_glucose_scaled_sse"]


def test_only_fixed_linear_disposal_is_supported():
    for expr in ("x*U", "U**2+x", "Gp+U", "x"):
        with pytest.raises(ValueError):
            balance(toy(expr), {"x"})
    names, s, b = balance(toy("tanh(x)-2*U"), {"x"})
    assert names == ["x"] and b == -2
    assert float(s(ca.DM(2))) == pytest.approx(np.tanh(2))
    with pytest.raises(ValueError):
        integrated_disposal(np.array([1, 1]))
