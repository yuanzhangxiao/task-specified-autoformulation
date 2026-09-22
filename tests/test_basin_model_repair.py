"""Explicit scientific choices, global mechanical propagation, no silent physics."""

from copy import deepcopy

import pytest

from autoformalism.expressions import ModelValidationError, compile_candidate
from autoformalism.fitting.initialization import (
    LatentInitializationPlan,
    apply_initialization_plan,
)
from autoformalism.search import basin_model_repair as repair
from tests.test_basin_equation_checks import CONTEXT, SURVEY, candidate


def bundle(independent=False):
    model = candidate(
        independent=independent,
        down="(inflow_down-o)/area_down"
        if independent
        else "(inflow_down+q-o)/area_down",
    )
    plan = LatentInitializationPlan.model_validate(
        {
            "rules": {}
            if independent
            else {
                "h_up": {
                    "initial": {
                        "mode": "map",
                        "expression": "initial_up",
                        "parameters": [],
                    }
                }
            }
        }
    )
    lowered, guesses, audit = apply_initialization_plan(
        compile_candidate(model, CONTEXT), plan
    )
    return {
        "source_task": {"arm": "full"},
        "brief": {
            "scientific_context": "Public basin fixture; no hidden generator.",
            "public_variables": [
                {"name": name, "data_role": role}
                for role, names in (
                    ("target", CONTEXT.targets),
                    ("external_input", CONTEXT.external_inputs),
                    ("covariate", CONTEXT.fixed_covariates),
                )
                for name in names
            ],
            "requirements": [
                {
                    "id": "local",
                    "public_requirement": "Local runoff pathway",
                    "targets": ["h_down"],
                    "drivers": ["inflow_down"],
                },
                *(
                    []
                    if independent
                    else [
                        {
                            "id": "memory",
                            "public_requirement": "Upstream storage memory",
                            "targets": ["h_down"],
                            "drivers": ["inflow_up"],
                            "requires_dynamic_memory": True,
                        }
                    ]
                ),
            ],
            "limits": {
                "generated_variables": 64,
                "total_terms": 512,
                "terms_per_equation": 32,
            },
        },
        "context": CONTEXT.model_dump(mode="json"),
        "candidate": lowered.validated.candidate.model_dump(mode="json"),
        "initialization": {
            "base_candidate": model.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "context": CONTEXT.model_dump(mode="json"),
            "guesses": guesses,
            "audit": audit,
        },
    }


def test_fix_known_and_rewrite_all_uses_without_new_gains():
    parent = bundle()
    original = deepcopy(parent)
    result = repair.apply(
        parent,
        {
            "hypothesis": "Known coefficient.",
            "parameter_bindings": [{"parameter": "k", "value": 1}],
        },
        "coupled",
    )
    assert parent == original
    assert "k" not in {p["name"] for p in result["bundle"]["candidate"]["parameters"]}
    assert result["parameter_replacements"] == {"k": "1.0"}
    assert result["changed"]
    assert all(
        c["status"] != "fail"
        for c in repair.assessment(result["bundle"], "coupled", [SURVEY])["checks"]
    )


def test_shared_identity_and_atomic_signs():
    parent = bundle()
    raw = {
        "hypothesis": "One hydraulic coefficient.",
        "parameter_bindings": [{"parameter": "k", "same_as": "d"}],
        "equations": [
            {"component": "h_up", "expression": "(inflow_up-q)/area_up"},
            {"component": "h_down", "expression": "(inflow_down+q-o)/area_down"},
        ],
    }
    tx = repair.apply(parent, raw, "coupled")
    assert {p["name"] for p in tx["bundle"]["candidate"]["parameters"]} == {"d"}
    assert tx["parameter_replacements"] == {"k": "d"}
    assert not any(
        "af_process_gain" in p["name"] for p in tx["bundle"]["candidate"]["parameters"]
    )


@pytest.mark.parametrize(
    "bindings",
    [
        [{"parameter": "k", "value": -1}],
        [{"parameter": "missing", "value": 1}],
        [{"parameter": "k", "same_as": "d"}, {"parameter": "d", "same_as": "k"}],
    ],
)
def test_invalid_binding_is_atomic(bindings):
    parent = bundle()
    original = deepcopy(parent)
    with pytest.raises(ValueError):
        repair.apply(
            parent, {"hypothesis": "bad", "parameter_bindings": bindings}, "coupled"
        )
    assert parent == original


@pytest.mark.parametrize("expression", ["__import__('os')", "unknown", "q"])
def test_existing_compiler_rejects_unsafe_unknown_or_algebraic_cycle(expression):
    with pytest.raises((ValueError, ModelValidationError)):
        repair.apply(
            bundle(),
            {
                "hypothesis": "bad",
                "equations": [{"component": "q", "expression": expression}],
            },
            "coupled",
        )


def test_add_outlet_via_existing_equation_api_and_preserve_boundaries():
    parent = bundle(independent=True)
    original = deepcopy(parent["initialization"])
    tx = repair.apply(
        parent,
        {
            "hypothesis": "Threshold discharge.",
            "equations": [
                {
                    "component": "new_outlet",
                    "kind": "algebraic",
                    "expression": "c*max(0,h_down-crest_down)",
                },
                {
                    "component": "h_down",
                    "expression": "(inflow_down-new_outlet)/area_down",
                },
            ],
            "remove_processes": ["o"],
            "new_parameters": [{"name": "c", "role": "rate"}],
        },
        "independent",
    )
    assert tx["bundle"]["initialization"]["plan"] == original["plan"]
    assert (
        tx["bundle"]["initialization"]["base_candidate"]["initial_conditions"]
        == original["base_candidate"]["initial_conditions"]
    )
    assert {p["name"] for p in tx["bundle"]["candidate"]["parameters"]} == {"c"}


def test_no_new_hidden_states_or_negative_control_leakage():
    for edit in (
        {"component": "hidden", "kind": "dynamic", "expression": "h_down"},
        {"component": "h_down", "expression": "inflow_up-h_down"},
    ):
        with pytest.raises((ValueError, ModelValidationError)):
            repair.apply(
                bundle(independent=True),
                {"hypothesis": "bad", "equations": [edit]},
                "independent",
            )


def test_payload_has_current_full_model_not_metrics():
    b = bundle()
    a = repair.assessment(b, "coupled", [SURVEY])
    payload = repair.payload(b, a, a, 3, None)
    assert payload["assembled_model"] == b["initialization"]["base_candidate"]
    assert "validation" not in payload and "normalized_mse" not in str(payload)


@pytest.mark.parametrize("value", [True, "1", float("inf"), float("nan")])
def test_nonfinite_or_coerced_fixed_values_are_rejected(value):
    with pytest.raises(ValueError):
        repair.ParameterBinding(parameter="k", value=value)


@pytest.mark.parametrize(
    "expression",
    [
        "(inflow_down-o)/area_down",
        "(inflow_up+inflow_down-o)/area_down",
    ],
)
def test_coordinated_repair_preserves_public_memory_contract(expression):
    with pytest.raises(ValueError, match="typed public pathways"):
        repair.apply(
            bundle(),
            {
                "hypothesis": "Remove or bypass memory",
                "equations": [{"component": "h_down", "expression": expression}],
            },
            "coupled",
        )


def test_binding_rewrites_new_rhs_and_shared_law_consumers_see_one_definition():
    tx = repair.apply(
        bundle(),
        {
            "hypothesis": "One law, two consumers and a fixed coefficient",
            "parameter_bindings": [{"parameter": "k", "value": 2}],
            "equations": [
                {"component": "q", "expression": "k*max(0,h_up-crest_up)**2"}
            ],
        },
        "coupled",
    )
    model = tx["bundle"]["candidate"]
    assert next(p["expression"] for p in model["processes"] if p["name"] == "q") == (
        "2.0 * max(0, h_up - crest_up) ** 2"
    )
    assert all("q" in e["rhs"] for e in model["state_equations"])
    assert tx["rewritten_components"] == ["q"]
    # The downstream trajectory also drives its outlet law.
    assert tx["affected_components"] == ["h_down", "h_up", "o", "q"]
