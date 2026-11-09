"""Attribution at measured insulin inputs and an isolated fit-domain restriction."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_t2_insulin_terms as rates
from scripts import run_t2_infusion_control as control


def test_driven_insulin_on_an_irregular_grid_has_the_exact_initial_memory():
    c = {
        "max_secretion": 4.0,
        "sigmoid_slope": 0.4,
        "sigmoid_midpoint": 180.0,
        "clearance": 0.02,
    }
    t = [0.0, 2.0, 6.0, 15.0]
    g = [180.0] * len(t)
    expected = 100 + (5 - 100) * np.exp(-0.3)
    assert rates.forced_insulin(c, t, g, 5) == pytest.approx(expected)
    d = rates.forced_insulin(
        c, t, [170.0, 190.0, 210.0, 150.0], 35
    ) - rates.forced_insulin(c, t, [170.0, 190.0, 210.0, 150.0], 25)
    assert d == pytest.approx(10 * np.exp(-0.3))
    with pytest.raises(ValueError, match="increasing"):
        rates.forced_insulin(c, [0.0, 1.0, 0.0], [180.0] * 3, 5)


def test_balanced_steady_trajectory_has_zero_attribution_error():
    t = np.arange(0.0, 300.125, 0.125)
    knots = np.arange(301.0)
    c = {
        "max_secretion": 4.0,
        "sigmoid_midpoint": 180.0,
        "sigmoid_slope": 0.1,
        "clearance": 0.1,
        "appearance_rate": 0.02,
        "basal_disposal_drift": 0.0,
        "insulin_to_X": 0.0,
        "X_decay": 0.2,
    }
    # Zero absorption and disposal keep glucose constant in this example.
    row = SimpleNamespace(
        time=knots,
        targets={
            "Gp": np.full(301, 180.0),
            "I": np.full(301, 20.0),
            "U": np.zeros(301),
        },
        external_inputs={rates.MEAL: np.zeros(301), rates.INFUSION: np.zeros(301)},
    )
    z = np.zeros(len(t))
    s = {
        "Gp": z + 180.0,
        "I": z + 20.0,
        "U": z.copy(),
        "A": z.copy(),
        "appearance": z.copy(),
        "X": z.copy(),
    }
    terms = {"net_self": z - 2.0, "glucose_secretion": z + 2.0}
    result = rates.recovery_record(c, terms, {"Gp": z, "I": z}, s, row, t)
    assert result["I300_error"] == 0
    np.testing.assert_allclose(list(result["error_components"].values()), 0, atol=1e-11)
    for v in result["closed_loop_flux_checks"].values():
        assert v["I300"] == pytest.approx(20.0)
        assert v["Gp300"] == pytest.approx(180.0)
    row.external_inputs[rates.INFUSION][290] = 1.0
    with pytest.raises(ValueError, match="zero ingestion"):
        rates.recovery_record(c, terms, {"Gp": z, "I": z}, s, row, t)


def test_late_glucose_substitution_is_not_an_endpoint_only_rate():
    c = {
        "max_secretion": 20.0,
        "sigmoid_midpoint": 180.0,
        "sigmoid_slope": 0.1,
        "clearance": 0.03,
    }
    t = np.arange(241.0, 301.0)
    first = np.linspace(100, 220, len(t))
    other = np.full_like(first, 220)
    assert rates.secretion(c, first[-1]) == rates.secretion(c, other[-1])
    assert rates.forced_insulin(c, t, first, 30) < rates.forced_insulin(c, t, other, 30)


def test_infusion_control_preserves_inputs_and_changes_only_one_parameter_domain():
    params = [
        {"name": control.GAIN, "role": "coefficient", "domain": "real"},
        {
            "name": control.OUTER,
            "role": "nonnegative_coefficient",
            "domain": "nonnegative",
        },
    ]
    data = {
        "models": {
            b: {
                "problem": {"base_candidate": {"parameters": deepcopy(params)}},
                "model_sha256": "original",
            }
            for b in ("c15-3", "c15-2")
        }
    }
    original = deepcopy(data)
    fixed = control.bound_infusion(data)
    assert data == original
    for out in fixed["models"].values():
        a, b = out["problem"]["base_candidate"]["parameters"]
        assert a == params[0] | {
            "role": "nonnegative_coefficient",
            "domain": "nonnegative",
        }
        assert b == params[1]
    data["models"]["c15-2"]["problem"]["base_candidate"]["parameters"][1]["role"] = (
        "coefficient"
    )
    with pytest.raises(ValueError, match="domains"):
        control.bound_infusion(data)


def test_unknown_handoff_is_rejected_before_use(tmp_path):
    path = tmp_path / "pulse.tar.gz"
    path.write_bytes(b"different package")
    with pytest.raises(ValueError, match="different pulse"):
        rates.attachment(path)


def test_endpoint_replay_resumes_and_binds_the_scientific_plan(tmp_path, monkeypatch):
    calls = []

    def replay(model, entry, expected, *, interventions):
        calls.append(interventions)
        return {"record": "checked"}

    monkeypatch.setattr(rates, "run_one", replay)
    point = {"parameter_sha256": "coefficients"}
    for _ in range(2):
        assert rates.saved_replay(tmp_path, "original", None, point, None, True) == {
            "record": "checked"
        }
    assert calls == [True]
    with pytest.raises(ValueError, match="checkpoint differs"):
        rates.saved_replay(tmp_path, "different", None, point, None, True)
