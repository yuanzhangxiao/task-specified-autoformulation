"""Deletion consistency, initial boundaries and training-only open-loop ranking."""

import copy

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.pruning import process_aware as pruning
from autoformalism.schemas.public_fitting import PublicFitRequest
from scripts.smoke_public_fitting import control


def example():
    parent, train, val = control("collocation-single-target-v2")
    raw = parent.model_dump(mode="json")
    raw["base_candidate"]["processes"] = [{"name": "q", "expression": "b*z"}]
    raw["base_candidate"]["state_equations"] = [
        {"state": "x", "rhs": "-a*x + q"},
        {"state": "z", "rhs": "-q"},
    ]
    return PublicFitRequest.model_validate(raw), train, val


def test_shared_process_is_one_atomic_removal_and_cleans_latent_boundary():
    parent, train, _ = example()
    units, blocked = pruning.removal_units(parent)
    assert not blocked
    assert not any(u["slots"] == ["x:1"] for u in units)
    unit = next(u for u in units if u["id"] == "process:q")
    assert set(unit["slots"]) == {"x:1", "z:0"}
    child, audit = pruning.remove(parent, unit)
    assert [s.name for s in child.base_candidate.states] == ["x"]
    assert not child.base_candidate.processes
    assert not child.initialization_plan.rules
    seed = sibling_fit.compatible_seed(
        parent,
        child,
        {"a": 0.7, "b": 1.2, "init_z_scale": 0.5},
        train,
        allow_initialization_changes=True,
    )
    assert seed["parameters"] == {"a": 0.7}
    assert set(seed["removed_parameters"]) == {"b", "init_z_scale"}
    assert audit["removed"]["initializers"] == ["z"]
    assert pruning.smaller(audit["before"], audit["after"])
    assert (
        child.base_candidate.observation_mappings
        == parent.base_candidate.observation_mappings
    )


def test_pruning_law_subterm_propagates_without_duplicate_functions():
    parent, _, _ = example()
    raw = parent.model_dump(mode="json")
    raw["base_candidate"]["processes"][0]["expression"] = "b*z + a*z"
    parent = PublicFitRequest.model_validate(raw)
    units, _ = pruning.removal_units(parent)
    child, _ = pruning.remove(parent, next(u for u in units if u["id"] == "term:q:1"))
    assert len(child.base_candidate.processes) == 1
    assert child.base_candidate.state_equations == parent.base_candidate.state_equations
    assert {p.name for p in child.base_candidate.parameters} == {"a", "b"}
    assert child.initialization_plan == parent.initialization_plan


@pytest.mark.parametrize("expression", ["tanh(q)", "1/q", "q*q", "q**2"])
def test_nonlinear_process_use_is_not_automatically_deleted(expression):
    parent, _, _ = example()
    raw = parent.model_dump(mode="json")
    raw["base_candidate"]["state_equations"][0]["rhs"] = f"-a*x + {expression}"
    units, blocked = pruning.removal_units(PublicFitRequest.model_validate(raw))
    assert not any(u["id"] == "process:q" for u in units)
    assert blocked[0]["id"] == "process:q"


def test_overlapping_shared_groups_cannot_be_partially_deleted():
    parent, _, _ = example()
    raw = parent.model_dump(mode="json")
    raw["base_candidate"]["processes"].append({"name": "r", "expression": "z"})
    raw["base_candidate"]["state_equations"][0]["rhs"] = "-a*x + q*r + r"
    raw["base_candidate"]["state_equations"][1]["rhs"] = "-q-r"
    units, blocked = pruning.removal_units(PublicFitRequest.model_validate(raw))
    assert {u["id"] for u in blocked} == {"process:q", "process:r"}
    assert not any(u["kind"] == "process" for u in units)


def test_real_open_loop_training_contributions_do_not_use_later_targets():
    parent, train, val = example()
    parameters = {"a": 0.7, "b": 1.2, "init_z_scale": 0.5}
    before = pruning.contributions(parent, parameters, train)
    raw = train.model_dump(mode="json")
    for row in raw["rows"]:
        row["targets"]["v01"][1:] = [1000] * (len(row["time"]) - 1)
    changed = train.model_validate(raw)
    assert pruning.contributions(parent, parameters, changed) == before
    assert all(v >= 0 for v in before.values())
    with pytest.raises(ValueError, match="training data"):
        pruning.contributions(parent, parameters, val)
    with pytest.raises(TimeoutError):
        pruning.contributions(parent, parameters, train, seconds=0)


def test_complete_vector_required_and_parent_unchanged():
    parent, train, _ = example()
    before = copy.deepcopy(parent)
    with pytest.raises(ValueError, match="complete fitted"):
        pruning.contributions(parent, {"a": 1}, train)
    for unit in pruning.removal_units(parent)[0]:
        pruning.remove(parent, unit)
    assert parent == before
    public._lower(parent)


def test_orphan_cleanup_cannot_silently_remove_one_transfer_consumer():
    parent, _, _ = example()
    raw = parent.model_dump(mode="json")
    raw["base_candidate"]["processes"][0]["expression"] = "b*x"
    # z is disconnected from output but consumes the same transfer as x.
    parent = PublicFitRequest.model_validate(raw)
    unit = next(u for u in pruning.removal_units(parent)[0] if u["id"] == "term:x:0")
    with pytest.raises(ValueError, match="partially remove a shared process"):
        pruning.remove(parent, unit)


def test_incomplete_transfer_removal_cannot_bypass_unit_builder():
    parent, _, _ = example()
    with pytest.raises(ValueError, match="permitted atomic removal"):
        pruning.remove(parent, {"id": "process:q", "kind": "process", "slots": ["x:1"]})
