"""Complete compatible starts, immutable scientific boundaries and consumed budgets."""

import copy

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit as sibling
from autoformalism.schemas.public_fitting import PublicFitRequest
from scripts.smoke_public_fitting import control


def example():
    parent, train, val = control("collocation-feasible-v1")
    raw = parent.model_dump(mode="json")
    raw["base_candidate"]["state_equations"][1]["rhs"] = "-beta*z"
    raw["base_candidate"]["parameters"][1]["name"] = "beta"
    raw["parameter_guesses"] = {}
    return parent, PublicFitRequest.model_validate(raw), train, val


def test_complete_start_carries_initializers_and_only_identical_declarations():
    parent, child, train, _ = example()
    parameters = {"a": 0.9, "b": 1.3, "init_z_scale": 2.4}
    seed = sibling.compatible_seed(parent, child, parameters, train)
    assert seed["parameters"]["init_z_scale"] == 2.4
    assert seed["parameters"]["a"] == 0.9
    assert seed["retained_initializer_parameters"] == ["init_z_scale"]
    assert seed["fresh_parameters"] == ["beta"]
    assert seed["removed_parameters"] == ["b"]
    assert parent.initialization_plan == child.initialization_plan
    raw = child.model_dump(mode="json")
    raw["base_candidate"]["parameters"][0]["role"] = "coefficient"
    raw["base_candidate"]["parameters"][0]["domain"] = "real"
    changed = sibling.compatible_seed(
        parent, PublicFitRequest.model_validate(raw), parameters, train
    )
    assert "a" in changed["changed_declarations"]
    assert "a" not in changed["retained_parameters"]


@pytest.mark.parametrize(
    "parameters",
    [
        {"a": 1},
        {"a": 1, "b": 1, "init_z_scale": float("nan")},
        {"a": -1, "b": 1, "init_z_scale": 1},
    ],
)
def test_bad_parent_vector_rejected(parameters):
    parent, child, train, _ = example()
    with pytest.raises(ValueError):
        sibling.compatible_seed(parent, child, parameters, train)


def test_initializer_plan_cannot_be_used_to_smuggle_learned_values():
    parent, child, train, _ = example()
    raw = child.model_dump(mode="json")
    raw["initialization_plan"]["rules"]["z"]["initial"]["parameters"][0]["guess"] = 2.4
    with pytest.raises(ValueError, match="scientific initialization plan"):
        sibling.compatible_seed(
            parent,
            PublicFitRequest.model_validate(raw),
            {"a": 1, "b": 1, "init_z_scale": 2.4},
            train,
        )


def prepared(tmp_path):
    parent, child, train, val = example()
    return sibling.prepare_child_fit(
        parent,
        child,
        {"a": 0.9, "b": 1.3, "init_z_scale": 2.4},
        train,
        val,
        tmp_path,
        lineage={"transaction": "synthetic"},
    )


def test_exact_resume_and_existing_profile(tmp_path, monkeypatch):
    frozen = prepared(tmp_path)
    calls = []

    def backend(request, model, train, val, guesses, settings, directory):
        calls.append(copy.deepcopy(guesses))
        assert guesses == frozen["seed"]["parameters"]
        assert settings == public.profile_settings(request)
        metric = {
            "normalized_mse": 0.1,
            "per_target_normalized_mse": {"v01": 0.1},
            "failed_trajectories": [],
        }
        return {
            "parameters": guesses,
            "training": metric,
            "validation": metric,
            "refinement": {"budget_exhausted": True, "actual_residual_calls": 3},
        }

    monkeypatch.setattr(public, "_run_backend", backend)
    result = sibling.execute_child_fit(tmp_path)
    assert result.status == "complete" and result.budget_exhausted
    assert result.identity == frozen["identity"]
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    assert sibling.execute_child_fit(tmp_path) == result
    assert prepared(tmp_path) == frozen
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    assert len(calls) == 1


def test_interruption_cannot_grant_second_allocation(tmp_path, monkeypatch):
    frozen = prepared(tmp_path)
    public._write(tmp_path / "started.json", {"identity": frozen["identity"]})
    monkeypatch.setattr(public, "_run_backend", lambda *args: pytest.fail("new budget"))
    assert sibling.execute_child_fit(tmp_path).status == "interrupted"
    assert sibling.execute_child_fit(tmp_path).status == "interrupted"


@pytest.mark.parametrize("field", ["seed", "parent_parameters", "lineage"])
def test_freeze_tamper_rejected(tmp_path, field):
    frozen = prepared(tmp_path)
    frozen[field] = {}
    public._write(tmp_path / "freeze.json", frozen)
    with pytest.raises((ValueError, KeyError)):
        sibling.inspect_child_fit(tmp_path)
