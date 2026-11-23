"""Conservative transfer witnesses at the function-to-topology boundary."""

from copy import deepcopy

import pytest

from autoformalism.research.trajectory_basin_balance import check_transfer_balance


@pytest.fixture
def case():
    return {
        "provenance": {"case": "phase_c_detention_coupled_noise0_v1"},
        "evidence": {
            "trajectories": [
                {
                    "trajectory_id": "train_000",
                    "covariates": {
                        "area_up": 800,
                        "area_down": 1200,
                        "crest_up": 0.35,
                        "initial_up": 0.55,
                    },
                }
            ]
        },
    }


GRAPH = {
    "variable_types": {"h_up": "differential", "h_down": "differential"},
    "initializers": {
        "h_up": {"mode": "observed_initial", "channel": "initial_up"},
        "h_down": {"mode": "observed_initial", "channel": "h_down"},
    },
}


def formula(enter, leave="k*max(h_up - crest_up,0)"):
    return {
        "equations": {
            "h_up": f"(inflow_up - ({leave}))/area_up",
            "h_down": f"(inflow_down+({enter})-b*max(0,h_down-crest_down))/area_down",
        },
        "parameters": {"k": 0.0, "b": 0.1},
    }


def test_proof_passes_even_at_zero_guess_and_with_reordered_head(case):
    result = check_transfer_balance(case, GRAPH, formula("k*max(0,h_up-crest_up)"))
    assert result["status"] == "pass"
    assert "upstream_volume_terms" in result["witness"]


def test_extra_area_is_incompatible_with_every_active_transfer(case):
    result = check_transfer_balance(
        case, GRAPH, formula("area_up*k*max(0,h_up-crest_up)")
    )
    assert result["status"] == "fail"
    assert result["witness"]["net_volume_multiplier"] == "799"
    assert result["witness"]["trajectory"] == "train_000"
    # An unrelated nonzero parameterization can make two independent gains agree.
    free = formula("area_up*j*max(0,h_up-crest_up)")
    free["parameters"]["j"] = 0.7
    assert check_transfer_balance(case, GRAPH, free)["status"] == "unverified"


@pytest.mark.parametrize(
    "covariates",
    [{"area_up": 1}, {"initial_up": 0.35}, {"area_down": 0}, {"area_up": float("nan")}],
)
def test_requires_a_positive_known_geometry_and_active_head_witness(case, covariates):
    case["evidence"]["trajectories"][0]["covariates"].update(covariates)
    result = check_transfer_balance(
        case, GRAPH, formula("area_up*k*max(0,h_up-crest_up)")
    )
    assert result["status"] == "unverified"


def test_vacuous_and_unrecognized_forms_are_inconclusive(case):
    for candidate in (
        formula("k", "0"),
        formula("area_up*k*tanh(h_up)", "k*tanh(h_up)"),
    ):
        assert check_transfer_balance(case, GRAPH, candidate)["status"] == "unverified"


def test_scope_is_bound_to_benchmark_and_named_depths(case):
    f = formula("k*max(h_up-crest_up,0)")
    graph = deepcopy(GRAPH)
    graph["initializers"]["h_up"]["mode"] = "constant"
    assert check_transfer_balance(case, graph, f)["status"] == "unverified"
    case["provenance"]["case"] = "another_case"
    assert check_transfer_balance(case, GRAPH, f)["status"] == "not_applicable"
