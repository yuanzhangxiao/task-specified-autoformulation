"""A zero-gain nested control must agree on all surviving outputs and initials."""

from copy import deepcopy

import pytest

from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from scripts.iclr_t2_memory_ablation import common_equations
from tests.test_prefix_initialization import problem


def build(*, memory):
    row = problem().model_dump(mode="json")
    candidate = row["base_candidate"]
    for state in candidate["states"]:
        if state["name"] == "x":
            state["name"] = "I"
    candidate["initial_conditions"][0]["state"] = "I"
    candidate["observation_mappings"][0]["expression"] = "I"
    rhs = "-theta*I+z+r"
    if memory:
        rhs += "+gain_M*M"
        candidate["states"].append({"name": "M", "kind": "latent"})
        candidate["state_equations"].append({"state": "M", "rhs": "k_M_rise*r-M/tau_M"})
        candidate["initial_conditions"].append(
            {"state": "M", "scope": "global", "fixed_value": 0}
        )
        row["graph"]["rules"]["M"] = {"initial": {"mode": "map", "expression": "0"}}
        names = {
            "k_M_rise": "coefficient",
            "tau_M": "positive_shape",
            "gain_M": "coefficient",
        }
    else:
        rhs += "+basal_floor*(1-tanh(I/eps))"
        names = {"eps": "positive_shape", "basal_floor": "coefficient"}
    candidate["state_equations"][0] = {"state": "I", "rhs": rhs}
    for name, role in names.items():
        candidate["parameters"].append({"name": name, "scope": "global", "role": role})
        row["parameter_guesses"][name] = 1.0
    return row


def test_nested_equations_include_dynamic_readouts_and_initials():
    child, base = map(
        PrefixFitProblem.model_validate, [build(memory=True), build(memory=False)]
    )
    proof = common_equations(child, base)
    assert proof["common_expressions"] == 4
    assert "gain_M" not in proof["insulin_at_zero_gain"]
    assert proof["extra_initial"] == "0"
    assert "tau_M" in proof["extra_state_rhs"]


@pytest.mark.parametrize("edit", ["output", "initial", "persistent_input"])
def test_other_surviving_effect_cannot_be_attributed_to_removed_coupling(edit):
    child = build(memory=True)
    other = deepcopy(child)
    if edit == "output":
        other["base_candidate"]["observation_mappings"][0]["expression"] = "I+M"
    elif edit == "initial":
        other["graph"]["rules"]["M"] = {
            "initial": {"mode": "value", "scope": "trajectory", "guess": 0.2}
        }
    else:
        other["base_candidate"]["state_equations"][1]["rhs"] += "+M"
    with pytest.raises(ValueError, match="nested memory/control"):
        common_equations(
            PrefixFitProblem.model_validate(other),
            PrefixFitProblem.model_validate(build(memory=False)),
        )
