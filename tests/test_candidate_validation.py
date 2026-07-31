"""Deterministic semantic, leakage, closure, and domain validation tests."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from autoformalism.expressions import (
    CandidateValidator,
    ModelValidationError,
    ValidationContext,
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
        ("1 / x", "DOMAIN_DIVISION_ZERO"),
        ("log(x)", "DOMAIN_LOG_NONPOSITIVE"),
        ("sqrt(x)", "DOMAIN_SQRT_NEGATIVE"),
        ("x**-1", "DOMAIN_DIVISION_ZERO"),
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
