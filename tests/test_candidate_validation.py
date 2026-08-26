"""Deterministic semantic, leakage, closure, and domain validation tests."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from autoformalism.expressions import (
    CandidateValidator,
    ModelValidationError,
    ValidationContext,
    repair_protected_declarations,
)
from autoformalism.schemas import CandidateModel


def candidate_payload() -> dict[str, Any]:
    """Return a valid model with states, a process, forcing, and constraints."""
    return {
        "schema_version": "1",
        "candidate_id": "safe_candidate",
        "parent_candidate_id": None,
        "change_summary": "Initial safe model.",
        "states": [
            {
                "name": "x",
                "kind": "latent",
                "unit": "relative",
                "description": "Latent driven state.",
            },
            {
                "name": "y",
                "kind": "observed",
                "unit": "relative",
                "description": "Generated target state.",
            },
        ],
        "processes": [
            {
                "name": "flux",
                "expression": "gain * softplus(x) / (offset + abs(aux))",
                "unit": "relative/time",
                "description": "Safe generated flux.",
            }
        ],
        "state_equations": [
            {"state": "x", "rhs": "-decay * x + input_u"},
            {"state": "y", "rhs": "flux - decay * y"},
        ],
        "observation_mappings": [
            {"channel": "target", "expression": "y", "unit": "relative"}
        ],
        "parameters": [
            {
                "name": "gain",
                "scope": "global",
                "bounds": {"lower": 0.1, "upper": 5.0},
                "initialization_range": {"lower": 0.5, "upper": 2.0},
                "unit": "1/time",
                "description": "Flux gain.",
            },
            {
                "name": "offset",
                "scope": "global",
                "bounds": {"lower": 1.0, "upper": 3.0},
                "initialization_range": {"lower": 1.1, "upper": 2.0},
                "unit": "relative",
                "description": "Positive denominator offset.",
            },
            {
                "name": "decay",
                "scope": "global",
                "bounds": {"lower": 0.01, "upper": 1.0},
                "initialization_range": {"lower": 0.1, "upper": 0.5},
                "unit": "1/time",
                "description": "State decay.",
            },
        ],
        "initial_conditions": [
            {
                "state": "x",
                "scope": "trajectory_specific",
                "initialization_range": {"lower": -2.0, "upper": 2.0},
            },
            {
                "state": "y",
                "scope": "global",
                "initialization_range": {"lower": 0.0, "upper": 2.0},
            },
        ],
        "constraints": [
            {
                "subject": "y",
                "kind": "nonnegative",
                "description": "Target state remains nonnegative.",
                "bounds": None,
            }
        ],
    }


@pytest.fixture
def context() -> ValidationContext:
    return ValidationContext(
        targets=("target",),
        auxiliaries=("aux",),
        external_inputs=("input_u",),
        fixed_covariates=("covariate",),
        unavailable_observed_channels=("hidden_aux",),
    )


def _candidate(payload: dict[str, Any] | None = None) -> CandidateModel:
    return CandidateModel.model_validate(payload or candidate_payload())


def _codes(error: ModelValidationError) -> set[str]:
    return {item.code for item in error.diagnostics}


def test_valid_candidate_has_deterministic_process_order(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["processes"].insert(
        0,
        {
            "name": "activation",
            "expression": "sigmoid(x)",
            "unit": "relative",
            "description": "Bounded activation.",
        },
    )
    payload["processes"][1]["expression"] = (
        "gain * activation / (offset + abs(aux))"
    )

    validated = CandidateValidator().validate(_candidate(payload), context)

    assert validated.process_order == ("activation", "flux")
    assert validated.forcing_symbols == frozenset({"aux", "input_u"})


def test_no_latent_ablation_rejects_latent_dynamic_state(
    context: ValidationContext,
) -> None:
    restricted = context.model_copy(update={"forbid_latent_states": True})

    with pytest.raises(ModelValidationError) as raised:
        CandidateValidator().validate(_candidate(), restricted)

    assert "LATENT_STATE_FORBIDDEN" in _codes(raised.value)


def test_no_latent_ablation_rejects_unmapped_observed_state(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["states"][0]["kind"] = "observed"
    payload["initial_conditions"][0]["scope"] = "global"
    restricted = context.model_copy(update={"forbid_latent_states": True})

    with pytest.raises(ModelValidationError) as raised:
        CandidateValidator().validate(_candidate(payload), restricted)

    assert "UNOBSERVED_DYNAMIC_STATE_FORBIDDEN" in _codes(raised.value)


def test_supplied_auxiliary_may_be_promoted_to_modeled_state(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["states"][0]["name"] = "aux"
    payload["state_equations"][0] = {
        "state": "aux",
        "rhs": "-decay * aux + input_u",
    }
    payload["initial_conditions"][0]["state"] = "aux"
    payload["processes"][0]["expression"] = (
        "gain * softplus(aux) / (offset + abs(aux))"
    )
    payload["observation_mappings"].append(
        {"channel": "aux", "expression": "aux(t)", "unit": "relative"}
    )

    validated = CandidateValidator().validate(_candidate(payload), context)

    assert "aux" not in validated.forcing_symbols
    assert validated.forcing_symbols == frozenset({"input_u"})
    assert validated.observation_expressions["aux"].symbols == frozenset({"aux"})


def test_protected_declarations_are_repaired_to_supplied_forcing(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["states"][0]["name"] = "input_u"
    payload["state_equations"][0] = {"state": "input_u", "rhs": "0"}
    payload["initial_conditions"][0]["state"] = "input_u"
    payload["processes"][0]["expression"] = (
        "gain * softplus(input_u) / (offset + abs(covariate))"
    )
    payload["parameters"].append(
        {
            "name": "covariate",
            "scope": "global",
            "bounds": {"lower": 1.0, "upper": 100.0},
            "initialization_range": {"lower": 10.0, "upper": 90.0},
        }
    )

    repaired, diagnostics = repair_protected_declarations(
        _candidate(payload), context
    )
    validated = CandidateValidator().validate(repaired, context)

    assert {state.name for state in repaired.states} == {"y"}
    assert "covariate" not in {item.name for item in repaired.parameters}
    assert validated.forcing_symbols == frozenset({"input_u", "covariate"})
    assert len(diagnostics) == 2


def test_repair_drops_only_unmodeled_auxiliary_observation_mappings(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["observation_mappings"].extend(
        [
            {"channel": "aux", "expression": "aux", "unit": "relative"},
            {"channel": "mystery", "expression": "x", "unit": "relative"},
        ]
    )

    repaired, diagnostics = repair_protected_declarations(
        _candidate(payload), context
    )

    assert {item.channel for item in repaired.observation_mappings} == {
        "target",
        "mystery",
    }
    assert diagnostics == (
        "removed redundant supplied-auxiliary observation mapping: aux",
    )


def test_repair_demotes_auxiliary_state_without_derivative_to_supplied_channel(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["states"].append(
        {"name": "aux", "kind": "observed", "unit": "relative"}
    )
    payload["initial_conditions"].append(
        {"state": "aux", "scope": "global", "fixed_value": 0.0}
    )
    payload["observation_mappings"].append(
        {"channel": "aux", "expression": "aux", "unit": "relative"}
    )

    repaired, diagnostics = repair_protected_declarations(
        _candidate(payload), context
    )

    assert "aux" not in {item.name for item in repaired.states}
    assert diagnostics == (
        "used supplied auxiliary because no derivative was declared: aux",
        "removed redundant supplied-auxiliary observation mapping: aux",
    )


def test_repair_removes_unreferenced_process_but_keeps_dependencies(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["processes"].extend(
        [
            {"name": "nested", "expression": "flux + 1"},
            {"name": "unused_rate", "expression": "d(y) / dt"},
        ]
    )
    payload["state_equations"][1]["rhs"] = "nested - decay * y"

    repaired, diagnostics = repair_protected_declarations(
        _candidate(payload), context
    )

    assert {item.name for item in repaired.processes} == {"flux", "nested"}
    assert diagnostics == (
        "removed unreferenced algebraic process: unused_rate",
    )


def test_repair_removes_unused_parameter_without_changing_expressions(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["parameters"].append(
        {
            "name": "unused_gain",
            "scope": "global",
            "bounds": {"lower": 0.0, "upper": 2.0},
            "initialization_range": {"lower": 0.1, "upper": 1.0},
        }
    )
    original_equations = copy.deepcopy(payload["state_equations"])
    original_processes = copy.deepcopy(payload["processes"])

    repaired, diagnostics = repair_protected_declarations(
        _candidate(payload), context
    )

    assert {item.name for item in repaired.parameters} == {
        "gain",
        "offset",
        "decay",
    }
    assert repaired.model_dump(mode="json")["state_equations"] == original_equations
    assert [item.expression for item in repaired.processes] == [
        item["expression"] for item in original_processes
    ]
    assert diagnostics == (
        "removed unused parameter declaration: unused_gain",
    )


def test_repair_retains_parameter_used_only_by_initial_expression(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["parameters"].append(
        {
            "name": "initial_scale",
            "scope": "global",
            "bounds": {"lower": 0.0, "upper": 2.0},
            "initialization_range": {"lower": 0.1, "upper": 1.0},
        }
    )
    payload["initial_conditions"][0] = {
        "state": "x",
        "scope": "global",
        "expression": "initial_scale * covariate",
    }

    repaired, diagnostics = repair_protected_declarations(
        _candidate(payload), context
    )

    assert "initial_scale" in {item.name for item in repaired.parameters}
    assert "removed unused parameter declaration: initial_scale" not in diagnostics


def test_repair_maps_declared_lagged_target_alias() -> None:
    payload = candidate_payload()
    payload["state_equations"][1]["rhs"] = "target_prev - decay * y"
    lagged_context = ValidationContext(
        targets=("target",),
        auxiliaries=("aux",),
        external_inputs=("input_u",),
        fixed_covariates=("covariate",),
        lagged_targets=("target",),
    )

    repaired, diagnostics = repair_protected_declarations(
        _candidate(payload), lagged_context
    )

    assert {
        item.name: item.expression for item in repaired.processes
    }["target_prev"] == "target"
    assert diagnostics == (
        "mapped causal lag alias target_prev to interval-boundary target",
        "bound identity-observed state initialization to causal channel: "
        "y -> target",
        "removed unreferenced algebraic process: flux",
        "removed unused parameter declaration: gain",
        "removed unused parameter declaration: offset",
    )


def test_repair_removes_numeric_bounds_from_qualitative_state_constraints(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["constraints"][0]["bounds"] = {"lower": 0.0, "upper": 10.0}
    payload["constraints"].append(
        {
            "subject": "decay",
            "kind": "bounded",
            "bounds": {"lower": 0.5, "upper": 2.0},
        }
    )

    repaired, diagnostics = repair_protected_declarations(
        _candidate(payload), context
    )

    assert repaired.constraints[0].bounds is None
    assert repaired.constraints[1].bounds is not None
    assert diagnostics == (
        "removed invented numeric bounds from qualitative constraint: y",
    )


def test_repair_drops_constraint_on_undeclared_concept(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["constraints"].append(
        {
            "subject": "conceptual_mechanism",
            "kind": "nonnegative",
            "description": "Prose concept, not a model variable.",
            "bounds": None,
        }
    )

    repaired, diagnostics = repair_protected_declarations(
        _candidate(payload), context
    )
    validated = CandidateValidator().validate(repaired, context)

    assert {item.subject for item in repaired.constraints} == {"y"}
    assert diagnostics == (
        "removed constraint on undeclared subject: conceptual_mechanism",
    )
    assert validated.candidate == repaired


def test_known_positive_fixed_covariate_proves_safe_division() -> None:
    payload = candidate_payload()
    payload["state_equations"][0]["rhs"] = "-decay * x + 1 / covariate"
    context = ValidationContext(
        targets=("target",),
        auxiliaries=("aux",),
        external_inputs=("input_u",),
        fixed_covariates=("covariate",),
        forcing_bounds={"covariate": (78.0, 78.0)},
    )

    CandidateValidator().validate(_candidate(payload), context)


@pytest.mark.parametrize(
    ("expression", "expected_code"),
    [
        ("target + x", "TARGET_LEAKAGE"),
        ("hidden_aux + x", "UNAVAILABLE_OBSERVED_CHANNEL"),
        ("mystery + x", "UNDEFINED_SYMBOL"),
    ],
)
def test_rejects_leakage_unavailable_and_undefined_symbols(
    context: ValidationContext,
    expression: str,
    expected_code: str,
) -> None:
    payload = candidate_payload()
    payload["state_equations"][0]["rhs"] = expression

    with pytest.raises(ModelValidationError) as caught:
        CandidateValidator().validate(_candidate(payload), context)

    assert expected_code in _codes(caught.value)


def test_constraint_subject_validation_uses_benchmark_context(
    context: ValidationContext,
) -> None:
    supplied = candidate_payload()
    supplied["constraints"].append(
        {"subject": "input_u", "kind": "nonnegative", "bounds": None}
    )
    CandidateValidator().validate(_candidate(supplied), context)

    unknown = candidate_payload()
    unknown["constraints"].append(
        {"subject": "mystery", "kind": "nonnegative", "bounds": None}
    )
    with pytest.raises(ModelValidationError) as caught:
        CandidateValidator().validate(_candidate(unknown), context)

    assert "UNKNOWN_CONSTRAINT_SUBJECT" in _codes(caught.value)


def test_target_named_state_is_generated_not_leakage(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["states"][1]["name"] = "target"
    payload["state_equations"][1]["state"] = "target"
    payload["state_equations"][1]["rhs"] = "flux - decay * target"
    payload["initial_conditions"][1]["state"] = "target"
    payload["constraints"][0]["subject"] = "target"
    payload["observation_mappings"][0]["expression"] = "target"

    validated = CandidateValidator().validate(_candidate(payload), context)

    assert "target" in {
        item.name for item in validated.candidate.states
    }


def test_rejects_algebraic_cycles(context: ValidationContext) -> None:
    payload = candidate_payload()
    payload["processes"] = [
        {
            "name": "first",
            "expression": "second + x",
            "unit": "relative",
            "description": "First cyclic process.",
        },
        {
            "name": "second",
            "expression": "first + x",
            "unit": "relative",
            "description": "Second cyclic process.",
        },
    ]
    payload["state_equations"][1]["rhs"] = "first - decay * y"
    payload["parameters"] = [
        item for item in payload["parameters"] if item["name"] == "decay"
    ]

    with pytest.raises(ModelValidationError) as caught:
        CandidateValidator().validate(_candidate(payload), context)

    assert "ALGEBRAIC_CYCLE" in _codes(caught.value)


def test_rejects_unused_and_channel_shadowing_parameters(
    context: ValidationContext,
) -> None:
    unused = candidate_payload()
    unused["parameters"].append(
        {
            "name": "unused",
            "scope": "global",
            "bounds": {"lower": 0.0, "upper": 1.0},
            "initialization_range": {"lower": 0.1, "upper": 0.9},
            "unit": "relative",
            "description": "Not referenced.",
        }
    )
    with pytest.raises(ModelValidationError) as unused_error:
        CandidateValidator().validate(_candidate(unused), context)
    assert "UNUSED_PARAMETER" in _codes(unused_error.value)

    shadowing = candidate_payload()
    shadowing["parameters"][0]["name"] = "aux"
    shadowing["processes"][0]["expression"] = "aux * softplus(x)"
    with pytest.raises(ModelValidationError) as shadowing_error:
        CandidateValidator().validate(_candidate(shadowing), context)
    assert "CHANNEL_NAME_COLLISION" in _codes(shadowing_error.value)

    duplicate_namespace = candidate_payload()
    duplicate_namespace["parameters"][0]["name"] = "x"
    duplicate_namespace["processes"][0]["expression"] = "x * softplus(x)"
    with pytest.raises(ModelValidationError) as duplicate_error:
        CandidateValidator().validate(_candidate(duplicate_namespace), context)
    assert "DECLARATION_NAME_COLLISION" in _codes(duplicate_error.value)


def test_requires_exact_target_observation_mappings(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["observation_mappings"][0]["channel"] = "other"

    with pytest.raises(ModelValidationError) as caught:
        CandidateValidator().validate(_candidate(payload), context)

    assert _codes(caught.value) >= {
        "MISSING_OBSERVATION_MAPPING",
        "UNEXPECTED_OBSERVATION_MAPPING",
    }


def test_validator_defensively_checks_state_equation_closure(
    context: ValidationContext,
) -> None:
    candidate = _candidate()
    bypassed = candidate.model_copy(
        update={"state_equations": candidate.state_equations[:-1]}
    )

    with pytest.raises(ModelValidationError) as caught:
        CandidateValidator().validate(bypassed, context)

    assert "MISSING_STATE_EQUATION" in _codes(caught.value)


def test_rejects_initialization_constraint_conflict(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["initial_conditions"][1]["initialization_range"] = {
        "lower": -1.0,
        "upper": 2.0,
    }

    with pytest.raises(ModelValidationError) as caught:
        CandidateValidator().validate(_candidate(payload), context)

    assert "INITIAL_CONSTRAINT_CONFLICT" in _codes(caught.value)


@pytest.mark.parametrize(
    ("expression", "expected_code"),
    [
        ("log(x)", "DOMAIN_LOG_NONPOSITIVE"),
        ("sqrt(x)", "DOMAIN_SQRT_NEGATIVE"),
    ],
)
def test_detects_domain_risks(
    context: ValidationContext,
    expression: str,
    expected_code: str,
) -> None:
    payload = candidate_payload()
    payload["state_equations"][0]["rhs"] = f"{expression} + input_u"

    with pytest.raises(ModelValidationError) as caught:
        CandidateValidator().validate(_candidate(payload), context)

    assert expected_code in _codes(caught.value)


@pytest.mark.parametrize("expression", ["1 / x", "x**-1"])
def test_division_risks_are_guarded_warnings(
    context: ValidationContext,
    expression: str,
) -> None:
    payload = candidate_payload()
    payload["state_equations"][0]["rhs"] = f"{expression} + input_u"

    validated = CandidateValidator().validate(_candidate(payload), context)

    assert {item.code for item in validated.warnings} == {
        "DOMAIN_DIVISION_GUARDED"
    }


def test_proves_safe_parameter_domains(context: ValidationContext) -> None:
    payload = candidate_payload()
    payload["state_equations"][0]["rhs"] = (
        "log(gain) + sqrt(abs(x)) + 1 / offset + input_u"
    )

    CandidateValidator().validate(_candidate(payload), context)


def test_diagnostics_are_deterministically_sorted(
    context: ValidationContext,
) -> None:
    payload = candidate_payload()
    payload["state_equations"][0]["rhs"] = "target + hidden_aux + mystery"

    with pytest.raises(ModelValidationError) as first:
        CandidateValidator().validate(_candidate(payload), context)
    with pytest.raises(ModelValidationError) as second:
        CandidateValidator().validate(_candidate(copy.deepcopy(payload)), context)

    assert first.value.diagnostics == second.value.diagnostics
    assert first.value.diagnostics == tuple(sorted(first.value.diagnostics))


def test_one_step_protocol_allows_unmodeled_target_on_rhs() -> None:
    payload = candidate_payload()
    payload["state_equations"][0]["rhs"] = "target - decay * x"
    payload["initial_conditions"][0].pop("initialization_range")
    payload["initial_conditions"][0]["fixed_value"] = 0.0
    context = ValidationContext(
        targets=("target",),
        lagged_targets=("target",),
        auxiliaries=("aux",),
        external_inputs=("input_u",),
        fixed_covariates=("covariate",),
    )

    validated = CandidateValidator().validate(_candidate(payload), context)

    assert "target" in validated.forcing_symbols


def test_one_step_null_observed_initialization_uses_mapped_channel() -> None:
    payload = candidate_payload()
    payload["initial_conditions"][0].pop("initialization_range")
    payload["initial_conditions"][0]["fixed_value"] = 0.0
    payload["initial_conditions"][1]["initialization_range"] = None
    context = ValidationContext(
        targets=("target",),
        lagged_targets=("target",),
        auxiliaries=("aux",),
        external_inputs=("input_u",),
        fixed_covariates=("covariate",),
    )

    validated = CandidateValidator().validate(_candidate(payload), context)

    expression = validated.initial_condition_expressions["y"]
    assert expression.symbols == frozenset({"target"})


def test_one_step_fixed_observed_initialization_is_canonically_repaired() -> None:
    payload = candidate_payload()
    payload["initial_conditions"][1] = {
        "state": "y",
        "scope": "global",
        "fixed_value": 0.0,
    }
    context = ValidationContext(
        targets=("target",),
        lagged_targets=("target",),
        auxiliaries=("aux",),
        external_inputs=("input_u",),
        fixed_covariates=("covariate",),
    )

    repaired, diagnostics = repair_protected_declarations(
        _candidate(payload), context
    )

    initial = next(
        item for item in repaired.initial_conditions if item.state == "y"
    )
    assert initial.fixed_value is None
    assert initial.expression == "target"
    assert any(
        "bound identity-observed state initialization" in item
        for item in diagnostics
    )


def test_one_step_latent_initialization_rejects_range_and_unknown_symbol() -> None:
    payload = candidate_payload()
    context = ValidationContext(
        targets=("target",),
        lagged_targets=("target",),
        auxiliaries=("aux",),
        external_inputs=("input_u",),
        fixed_covariates=("covariate",),
    )

    with pytest.raises(ModelValidationError) as ranged:
        CandidateValidator().validate(_candidate(payload), context)
    assert "LATENT_INITIALIZATION_NOT_CAUSAL" in _codes(ranged.value)

    payload["initial_conditions"][0].pop("initialization_range")
    payload["initial_conditions"][0]["expression"] = "unknown + target"
    with pytest.raises(ModelValidationError) as unknown:
        CandidateValidator().validate(_candidate(payload), context)
    assert "INVALID_INITIALIZATION_SYMBOL" in _codes(unknown.value)


def test_contextual_validation_rejects_latent_initialization_without_mode() -> None:
    payload = candidate_payload()
    payload["initial_conditions"][0]["initialization_range"] = None
    context = ValidationContext(
        targets=("target",),
        lagged_targets=("target",),
        auxiliaries=("aux",),
        external_inputs=("input_u",),
        fixed_covariates=("covariate",),
    )

    with pytest.raises(ModelValidationError) as caught:
        CandidateValidator().validate(_candidate(payload), context)

    assert "MISSING_INITIALIZATION_MODE" in _codes(caught.value)
