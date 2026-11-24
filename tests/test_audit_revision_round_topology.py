import pytest

from scripts.audit_revision_round_topology import indicators, summarize


def graph(**sources):
    types = dict.fromkeys(sources, "differential")
    types["U"] = "algebraic"
    return {"variable_types": types, "sources": sources}


def test_shared_memory_without_disposal_path():
    model = graph(I=["I"], A=["I"], U=["A"], Gp=["meal_rate_g_per_min", "A"])
    result = indicators(model, {"A"})
    assert result["I_through_latent_memory_to_U"]
    assert result["shared_insulin_driven_latent_direct_to_U_and_Gp"]
    assert result["U_path_or_shared_latent"]
    assert not result["U_any_path_to_Gp"]
    assert not result["U_direct_to_Gp"]
    assert result["meal_to_Gp"]


def test_indirect_disposal_and_dangling_cycle():
    model = graph(I=["A"], A=["I"], U=["I"], M=["U"], Gp=["M"])
    result = indicators(model, {"A"})
    assert result["U_any_path_to_Gp"]
    assert not result["U_direct_to_Gp"]
    assert not result["I_through_latent_memory_to_U"]
    assert not result["shared_insulin_driven_latent_direct_to_U_and_Gp"]


def test_direct_disposal_and_indirect_insulin_via_glucose():
    model = graph(I=[], A=["Gp"], U=["A"], Gp=["I", "A", "U"])
    result = indicators(model, {"A"})
    assert result["U_direct_to_Gp"]
    assert result["U_any_path_to_Gp"]
    assert not result["shared_insulin_driven_latent_direct_to_U_and_Gp"]
    with pytest.raises(ValueError, match="latent differential"):
        indicators(model, {"U"})


def test_multiplicity_and_pairing():
    present = {"direct": True}
    absent = {"direct": False}
    result = summarize(
        [
            {
                "20B": {"A": present, "B": present},
                "uniform": {"A": present, "C": absent},
            },
            {
                "20B": {"A": present, "C": absent},
                "uniform": {"B": present, "C": absent},
            },
            {
                "20B": {"A": present, "C": absent},
                "uniform": {"A": present, "B": present},
            },
        ],
        ["direct"],
    )
    assert result["direct"]["picks_with_feature"] == {"20B": 4, "uniform": 4}
    assert result["direct"]["draws_with_more"] == {"20B": 1, "equal": 1, "uniform": 1}
