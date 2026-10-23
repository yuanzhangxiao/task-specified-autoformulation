"""Construction bookkeeping preserves all executable uses and scoped semantics."""

import json
from copy import deepcopy

import pytest

from autoformalism.research import construction_parameter_audit as audit
from autoformalism.schemas.prefix_initialization import PrefixFitProblem


def problem(*, rhs="(1+g)*u-(1/a+b/tau)*x", **candidate_changes):
    """Create a small observed/latent system without fitting or numerical backends."""
    payload = {
        "base_candidate": {
            "candidate_id": "audit_fixture",
            "parent_candidate_id": None,
            "states": [{"name": s, "kind": "latent"} for s in ("x", "z")],
            "state_equations": [
                {"state": "x", "rhs": rhs},
                {"state": "z", "rhs": "-z"},
            ],
            "processes": [],
            "observation_mappings": [{"channel": "y", "expression": "x"}],
            "parameters": [
                {"name": "a", "scope": "global", "role": "time_constant"},
                {"name": "b", "scope": "global", "role": "nonnegative_coefficient"},
                {"name": "g", "scope": "global", "role": "nonnegative_coefficient"},
                {"name": "tau", "scope": "global", "role": "time_constant"},
            ],
            "initial_conditions": [
                {"state": s, "scope": "global", "fixed_value": 0} for s in ("x", "z")
            ],
            **candidate_changes,
        },
        "context": {"targets": ["y"], "external_inputs": ["u"]},
        "graph": {
            "rules": {
                "z": {"initial": {"mode": "value", "scope": "trajectory", "guess": 1}}
            }
        },
    }
    # Keep the common parameter roster syntactically declared while permitting
    # an audit of coefficients used only in an unconsumed process definition.
    payload["base_candidate"]["processes"].append(
        {"name": "unconsumed_parameters", "expression": "a+b+g+tau"}
    )
    return PrefixFitProblem.model_validate(payload)


def names(report):
    return set(report["inactive_parameter_certificate"]["parameters"])


def candidates(report, argument):
    return [
        item
        for item in report["simplification_candidates"]
        if item["location"] == "rhs:x" and item["argument"] == argument
    ]


def test_affine_collection_is_read_only_and_domains_are_preserved():
    source = problem()
    before = source.model_dump(mode="json")
    result = audit.assess(source)
    assert result["status"] == "assessed"
    assert result == audit.assess(source)
    assert source.model_dump(mode="json") == before
    rows = {row["location"]: row for row in result["equations"]}
    assert rows["rhs:x"]["coefficients"] == {"u": "1 + g", "x": "-(1 / a + b / tau)"}
    assert rows["rhs:x"]["offset"] == "0"
    shifted = candidates(result, "u")[0]
    assert shifted["kind"] == "single_parameter_reexpression"
    assert shifted["potential_parameter_count_reduction"] == 0
    decay = candidates(result, "x")[0]
    assert decay["parameters"] == ["a", "b", "tau"]
    assert decay["potential_parameter_count_reduction"] == 2
    assert not decay["domain_image_derived"]
    assert not decay["automatic_rewrite_permitted"]
    assert decay["original_domains"]["a"]["domain"] == "positive"
    assert decay["original_domains"]["b"]["domain"] == "nonnegative"
    assert not names(result)
    assert not result["parameter_fitting_performed"]
    assert not result["numerical_rollout_performed"]
    json.dumps(result, allow_nan=False, sort_keys=True)


def test_unused_definition_closure_and_parameter_certificate():
    source = problem(
        rhs="-x",
        processes=[
            {"name": "q", "expression": "a*z"},
            {"name": "r", "expression": "q/tau"},
        ],
    )
    result = audit.assess(source)
    assert names(result) == {"a", "b", "g", "tau"}
    assert {row["name"] for row in result["unused_process_definitions"]} == {
        "q",
        "r",
        "unconsumed_parameters",
    }
    assert not result["inactive_parameter_certificate"][
        "execution_domain_safety_proven"
    ]
    assert "eagerly" in result["inactive_parameter_certificate"]["limitation"]
    assert "init_z_value" not in names(result)


def test_transitive_shared_process_consumers_prevent_private_simplification():
    source = problem(
        processes=[
            {"name": "q", "expression": "(1+g)*u"},
            {"name": "r", "expression": "q/a"},
        ],
        state_equations=[
            {"state": "x", "rhs": "q-x"},
            {"state": "z", "rhs": "r-z"},
        ],
    )
    result = audit.assess(source)
    assert not candidates(result, "u")
    retained = next(
        item
        for item in result["retained_composite_coefficients"]
        if item["location"] == "rhs:x"
    )
    assert "rhs:z" in retained["external_uses"]
    assert [r["name"] for r in result["unused_process_definitions"]] == [
        "unconsumed_parameters"
    ]
    assert {"b", "tau"} <= names(result)


def test_parameter_shared_between_two_coefficients_is_retained():
    result = audit.assess(problem(rhs="(1+g)*x+g*u"))
    assert not candidates(result, "x")
    retained = next(
        item
        for item in result["retained_composite_coefficients"]
        if item["location"] == "rhs:x"
    )
    assert "rhs:x/coefficient:u" in retained["external_uses"]


def test_constraint_subject_and_output_use_block_collapse_and_inactivity():
    source = problem(
        rhs="(1+g)*u-x",
        processes=[{"name": "protected", "expression": "g*z"}],
        constraints=[{"subject": "protected", "kind": "nonnegative"}],
        observation_mappings=[
            {"channel": "y", "expression": "x"},
            {"channel": "w", "expression": "b*z"},
        ],
    )
    payload = source.model_dump(mode="json")
    payload["context"]["targets"] = ["y", "w"]
    result = audit.assess(PrefixFitProblem.model_validate(payload))
    assert result["status"] == "assessed"
    assert not candidates(result, "u")
    assert "g" not in names(result)
    assert "b" not in names(result)
    retained = next(
        item
        for item in result["retained_composite_coefficients"]
        if item["location"] == "rhs:x"
    )
    assert "constraint:0" in retained["external_uses"]


def test_initializer_local_name_is_not_mistaken_for_equation_parameter_sharing():
    source = problem(rhs="(1+g)*u-x")
    payload = source.model_dump(mode="json")
    payload["graph"]["rules"]["z"]["initial"] = {
        "mode": "map",
        "expression": "g*reading",
        "parameters": [{"name": "g"}],
        "public_initials": {"reading": "y"},
    }
    result = audit.assess(PrefixFitProblem.model_validate(payload))
    assert candidates(result, "u")
    assert "init_z_g" not in names(result)
    local = next(row for row in result["parameters"] if row["parameter"] == "init_z_g")
    assert local["active_consumers"]["locations"] == ["initial:z"]
    assert result["initializer_audit"]["bindings"]["z"]["public_initials"] == {
        "reading": "y"
    }


def test_dependent_initializer_roots_have_all_transitive_initial_uses():
    payload = problem(rhs="-x").model_dump(mode="json")
    payload["base_candidate"]["states"].append({"name": "v", "kind": "latent"})
    payload["base_candidate"]["state_equations"].append({"state": "v", "rhs": "-v"})
    payload["base_candidate"]["initial_conditions"].append(
        {"state": "v", "scope": "global", "fixed_value": 0}
    )
    payload["graph"]["rules"]["v"] = {
        "initial": {
            "mode": "map",
            "expression": "p*ancestor",
            "state_initials": {"ancestor": "z"},
            "parameters": [{"name": "p"}],
        }
    }
    result = audit.assess(PrefixFitProblem.model_validate(payload))
    assert result["status"] == "assessed"
    assert "init_z_value" not in names(result)
    assert "init_v_p" not in names(result)
    root = next(
        row for row in result["parameters"] if row["parameter"] == "init_z_value"
    )
    assert root["active_consumers"]["locations"] == ["initial:v", "initial:z"]


def test_declared_parameter_bounds_are_never_replaced_with_an_unrestricted_gain():
    payload = problem().model_dump(mode="json")
    payload["base_candidate"]["parameters"][2]["bounds"] = {"lower": 2, "upper": 3}
    result = audit.assess(PrefixFitProblem.model_validate(payload))
    assert candidates(result, "u")[0]["original_domains"]["g"]["bounds"] == {
        "lower": 2,
        "upper": 3,
    }
    assert not candidates(result, "u")[0]["domain_image_derived"]


def test_nonlinear_equation_defers_coefficient_collection_but_retains_usage():
    result = audit.assess(problem(rhs="a*x*z+b*u"))
    row = next(row for row in result["equations"] if row["location"] == "rhs:x")
    assert row["status"] == "nonlinear_deferred"
    assert "coefficients" not in row
    assert not result["simplification_candidates"]
    assert names(result) == {"g", "tau"}


@pytest.mark.parametrize(
    "rhs", ["__import__('os').system('echo bad')", "x[0]", "missing*x"]
)
def test_invalid_or_unsupported_expression_gives_no_certificate(rhs):
    result = audit.assess(problem(rhs=rhs))
    assert result["status"] == "unavailable"
    assert "inactive_parameter_certificate" not in result


def test_cycle_gives_no_partial_usage_certificate():
    result = audit.assess(
        problem(
            processes=[
                {"name": "q", "expression": "r+a"},
                {"name": "r", "expression": "q+b"},
            ]
        )
    )
    assert result["status"] == "unavailable"
    assert "inactive_parameter_certificate" not in result


def test_expansion_is_bounded_without_truncating_an_equation_into_a_false_model(
    monkeypatch,
):
    monkeypatch.setattr(audit, "MAX_EXPANDED_NODES", 8)
    result = audit.assess(problem())
    row = next(row for row in result["equations"] if row["location"] == "rhs:x")
    assert row["status"] == "expansion_deferred"
    assert "expanded_expression" not in row
    assert not names(result)
    view = audit.prompt_view(result)
    assert view["omitted_equation_count"] >= 1


def test_prompt_projection_has_exact_equations_and_bounded_examples():
    result = audit.assess(problem())
    view = audit.prompt_view(result)
    assert "d(x)/dt = (1 + g) * u - (1 / a + b / tau) * x" in view["expanded_equations"]
    assert "still has one unknown" in view["instruction"]
    assert "parameters" not in view
    assert "initializer_audit" not in view
    copied = deepcopy(result)
    copied["simplification_candidates"] *= 20
    capped = audit.prompt_view(copied)
    assert len(capped["private_coefficient_examples"]) == 12
    assert capped["private_coefficient_count"] == 40
    assert audit.prompt_view({"status": "unavailable", "reason": "invalid"}) == {
        "status": "unavailable",
        "reason": "invalid",
    }


def test_public_quantity_named_constant_does_not_collide_with_affine_offset():
    payload = problem(rhs="constant+a").model_dump(mode="json")
    payload["context"]["external_inputs"] = ["constant"]
    result = audit.assess(PrefixFitProblem.model_validate(payload))
    row = next(row for row in result["equations"] if row["location"] == "rhs:x")
    assert row["coefficients"] == {"constant": "1"}
    assert row["offset"] == "a"


def test_fixed_scaling_is_a_unit_review_not_an_unproved_rejection():
    payload = problem(rhs="1000*u-x").model_dump(mode="json")
    payload["base_candidate"]["states"][0]["unit"] = "mg/kg"
    result = audit.assess(PrefixFitProblem.model_validate(payload))
    item = result["fixed_scaling_review"][0]
    assert item["argument"] == "u"
    assert item["fixed_coefficient"] == "1000"
    assert item["target_declared_unit"] == "mg/kg"
    assert item["argument_declared_unit"] == "unspecified"
    assert item["status"] == "advisory_units_review"
    assert "do not invent" in item["instruction"]
    assert audit.prompt_view(result)["fixed_scaling_count"] == 1


def test_invalid_initializer_aliases_do_not_produce_partial_certificate():
    payload = problem().model_dump(mode="json")
    payload["graph"]["rules"]["z"]["initial"] = {
        "mode": "map",
        "expression": "p",
        "parameters": [{"name": "p"}, {"name": "unused"}],
    }
    result = audit.assess(PrefixFitProblem.model_validate(payload))
    assert result["status"] == "unavailable"
    assert "unused" in result["reason"]
    assert "inactive_parameter_certificate" not in result


def test_unit_factors_are_hidden_but_inverse_time_constant_is_not_called_redundant():
    result = audit.assess(problem(rhs="(1+g*1)*u-x/tau"))
    row = next(row for row in result["equations"] if row["location"] == "rhs:x")
    assert row["coefficients"]["u"] == "1 + g"
    assert not candidates(result, "x")
    assert candidates(result, "u")
