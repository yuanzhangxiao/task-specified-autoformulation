"""Equation facts do not certify mechanisms or infer laws from descriptions."""

from copy import deepcopy

from autoformalism.rebuttal.process_equation_diagnostics import diagnose


def fixture():
    """Two identical outlets with an initial gauge in the ongoing balance."""
    return {
        "context": {"external_inputs": ["inflow_up"], "time_symbol": "t"},
        "candidate": {
            "states": [{"name": n} for n in ("h_up", "h_down")],
            "processes": [
                {"name": "q", "expression": "h_up"},
                {"name": "r", "expression": "1*h_up"},
            ],
            "state_equations": [
                {"state": "h_up", "rhs": "inflow_up-a*q-b*r+g*initial_up"},
                {"state": "h_down", "rhs": "c*q-d*h_down"},
            ],
            "initial_conditions": [{"state": "h_up", "expression": "initial_up"}],
        },
    }


def test_duplicate_laws_and_boundary_forcing_are_advisory():
    bundle = fixture()
    before = deepcopy(bundle)
    result = diagnose(bundle, None, {"training": {"rows": []}})
    assert bundle == before
    duplicate = result["duplicate_process_laws"][0]
    assert duplicate["processes"] == ["q", "r"]
    assert duplicate["common_consumers"] == [{"state": "h_up", "processes": ["q", "r"]}]
    assert not duplicate["parameter_nonidentifiability_proved"]
    assert not result["hard_rejection"]
    assert not result["scientific_compliance_certified"]
    assert any(
        f["code"] == "INITIAL_COVARIATE_IN_ONGOING_RHS" for f in result["findings"]
    )
    assert any(f["code"] == "AFFINE_DEPTH_LAW" for f in result["findings"])


def test_threshold_structure_is_not_threshold_certification():
    bundle = fixture()
    bundle["candidate"]["processes"][1]["expression"] = "sqrt(max(h_up-crest_up,0))**3"
    bundle["candidate"]["state_equations"][0]["rhs"] = "inflow_up-a*q-b*r"
    result = diagnose(bundle, None, {"training": {"rows": []}})
    assert not result["duplicate_process_laws"]
    assert not any(
        f["code"] == "INITIAL_COVARIATE_IN_ONGOING_RHS" for f in result["findings"]
    )
    upstream = next(t for t in result["expression_traits"] if t["name"] == "h_up")
    assert upstream["threshold_operators"] == ["max"]
    assert not upstream["affine_in_dynamic_channels"]
    assert not upstream["threshold_behavior_certified"]
    bundle["candidate"]["description"] = "Constant, nonlinear, threshold, impossible."
    assert diagnose(bundle, None, {"training": {"rows": []}}) == result


def test_algebraic_cycle_is_reported_not_expanded_forever():
    bundle = fixture()
    bundle["candidate"]["processes"][0]["expression"] = "r"
    bundle["candidate"]["processes"][1]["expression"] = "q"
    result = diagnose(bundle, None, {"training": {"rows": []}})
    assert all(t["status"] == "unavailable" for t in result["expression_traits"])
    assert all("cycle" in t["error"] for t in result["expression_traits"])
