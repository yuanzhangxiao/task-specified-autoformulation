"""Proposer candidate schema tests."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from pydantic import ValidationError

from autoformalism.schemas import CandidateModel


@pytest.fixture
def candidate_payload() -> dict[str, Any]:
    """Return a complete valid candidate using every major schema feature."""
    return {
        "schema_version": "1",
        "candidate_id": "candidate_001",
        "parent_candidate_id": "candidate_000",
        "change_summary": "Added a latent causal response state.",
        "states": [
            {
                "name": "target_state",
                "kind": "observed",
                "unit": "relative",
                "description": "State mapped to the measured target.",
            },
            {
                "name": "latent_response",
                "kind": "latent",
                "unit": "relative",
                "description": "Unobserved input-response state.",
            },
        ],
        "processes": [
            {
                "name": "generated_flux",
                "expression": "gain * latent_response",
                "unit": "relative/time",
                "description": "Generated causal flux.",
            }
        ],
        "state_equations": [
            {
                "state": "target_state",
                "rhs": "generated_flux - decay * target_state",
            },
            {
                "state": "latent_response",
                "rhs": "input_u - latent_response / tau",
            },
        ],
        "observation_mappings": [
            {
                "channel": "measured_target",
                "expression": "target_state",
                "unit": "relative",
            }
        ],
        "parameters": [
            {
                "name": "gain",
                "scope": "global",
                "bounds": {"lower": 0.0, "upper": 10.0},
                "initialization_range": {"lower": 0.1, "upper": 2.0},
                "unit": "1/time",
                "description": "Shared response gain.",
            },
            {
                "name": "decay",
                "scope": "global",
                "bounds": {"lower": 0.0001, "upper": 5.0},
                "initialization_range": {"lower": 0.01, "upper": 1.0},
                "unit": "1/time",
                "description": "Shared target decay.",
            },
            {
                "name": "tau",
                "scope": "trajectory_specific",
                "bounds": {"lower": 0.1, "upper": 100.0},
                "initialization_range": {"lower": 1.0, "upper": 20.0},
                "unit": "time",
                "description": "Trajectory-specific response time.",
            },
        ],
        "initial_conditions": [
            {
                "state": "target_state",
                "scope": "global",
                "initialization_range": {"lower": 0.0, "upper": 5.0},
            },
            {
                "state": "latent_response",
                "scope": "trajectory_specific",
                "initialization_range": {"lower": 0.0, "upper": 5.0},
            },
        ],
        "constraints": [
            {
                "subject": "target_state",
                "kind": "nonnegative",
                "description": "The observed storage cannot be negative.",
                "bounds": None,
            },
            {
                "subject": "gain",
                "kind": "bounded",
                "description": "Use the declared plausible gain interval.",
                "bounds": {"lower": 0.0, "upper": 10.0},
            },
        ],
    }


def test_candidate_json_round_trip(candidate_payload: dict[str, Any]) -> None:
    candidate = CandidateModel.model_validate(candidate_payload)

    encoded = candidate.model_dump_json()
    restored = CandidateModel.model_validate_json(encoded)

    assert restored == candidate
    assert json.loads(encoded)["parent_candidate_id"] == "candidate_000"
    assert restored.states[1].kind.value == "latent"
    assert restored.parameters[2].scope.value == "trajectory_specific"


def test_explicit_lhs_equations_normalize_derivatives_and_processes(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["processes"] = []
    payload["state_equations"] = []
    payload["equations"] = [
        {
            "lhs": "d(target_state)/dt",
            "rhs": "generated_flux - decay * target_state",
        },
        {
            "lhs": "latent_response_rate",
            "rhs": "input_u - latent_response / tau",
            "derivative_of": "latent_response",
        },
        {"lhs": "generated_flux", "rhs": "gain * latent_response"},
    ]

    candidate = CandidateModel.model_validate(payload)

    assert candidate.equations == ()
    assert [(item.state, item.rhs) for item in candidate.state_equations] == [
        ("target_state", "generated_flux - decay * target_state"),
        ("latent_response", "latent_response_rate"),
    ]
    assert {item.name: item.expression for item in candidate.processes} == {
        "latent_response_rate": "input_u - latent_response / tau",
        "generated_flux": "gain * latent_response",
    }
    assert CandidateModel.model_validate_json(candidate.model_dump_json()) == candidate


def test_plain_state_lhs_is_normalized_as_algebraic_not_derivative(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["state_equations"] = payload["state_equations"][1:]
    payload["equations"] = [{"lhs": "target_state", "rhs": "generated_flux"}]

    candidate = CandidateModel.model_validate(payload)

    assert {item.name for item in candidate.states} == {"latent_response"}
    assert {item.state for item in candidate.state_equations} == {"latent_response"}
    assert {item.state for item in candidate.initial_conditions} == {
        "latent_response"
    }
    assert {item.name for item in candidate.processes} == {
        "generated_flux",
        "target_state",
    }


def test_rate_name_is_algebraic_without_explicit_derivative_link(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["equations"] = [{"lhs": "unknown_rate", "rhs": "1"}]

    candidate = CandidateModel.model_validate(payload)

    assert {item.name for item in candidate.processes} == {
        "generated_flux",
        "unknown_rate",
    }


def test_derivative_link_requires_a_declared_state(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["equations"] = [
        {"lhs": "unknown_rate", "rhs": "1", "derivative_of": "unknown"}
    ]

    with pytest.raises(ValidationError, match="undeclared state"):
        CandidateModel.model_validate(payload)


def test_derivative_link_removes_redundant_rate_state(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["states"].append(
        {"name": "target_rate", "kind": "latent", "unit": "relative/time"}
    )
    payload["initial_conditions"].append(
        {"state": "target_rate", "scope": "global", "fixed_value": 0.0}
    )
    payload["equations"] = [
        {
            "lhs": "target_rate",
            "rhs": "generated_flux - decay * target_state",
            "derivative_of": "target_state",
        }
    ]
    payload["state_equations"] = payload["state_equations"][1:]

    candidate = CandidateModel.model_validate(payload)

    assert "target_rate" not in {item.name for item in candidate.states}
    assert "target_rate" not in {
        item.state for item in candidate.initial_conditions
    }
    assert {item.name for item in candidate.processes} >= {"target_rate"}
    assert candidate.state_equations[-1].state == "target_state"
    assert candidate.state_equations[-1].rhs == "target_rate"


def test_explicit_derivative_rejects_conflicting_derivative_link(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["equations"] = [
        {
            "lhs": "d(target_state)/dt",
            "rhs": "1",
            "derivative_of": "latent_response",
        }
    ]

    with pytest.raises(ValidationError, match="conflicts"):
        CandidateModel.model_validate(payload)


def test_identical_algebraic_equations_are_deduplicated(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["equations"] = [
        {"lhs": "extra_rate", "rhs": "gain * latent_response"},
        {"lhs": "extra_rate", "rhs": "gain * latent_response"},
    ]
    payload["parameters"].append(
        {
            "name": "extra_rate",
            "scope": "global",
            "bounds": {"lower": 0.0, "upper": 2.0},
            "initialization_range": {"lower": 0.0, "upper": 1.0},
        }
    )

    candidate = CandidateModel.model_validate(payload)

    assert [item.name for item in candidate.processes].count("extra_rate") == 1
    assert "extra_rate" not in {item.name for item in candidate.parameters}


def test_conflicting_algebraic_equations_remain_invalid(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["equations"] = [
        {"lhs": "extra_rate", "rhs": "gain * latent_response"},
        {"lhs": "extra_rate", "rhs": "decay * latent_response"},
    ]

    with pytest.raises(ValidationError, match="conflicting algebraic"):
        CandidateModel.model_validate(payload)


def test_candidate_allows_omitted_descriptive_metadata(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload.pop("change_summary")
    for state in payload["states"]:
        state.pop("unit")
        state.pop("description")
    for process in payload["processes"]:
        process.pop("unit")
        process.pop("description")
    for mapping in payload["observation_mappings"]:
        mapping.pop("unit")
    for parameter in payload["parameters"]:
        parameter.pop("unit")
        parameter.pop("description")
    for constraint in payload["constraints"]:
        constraint.pop("description")

    candidate = CandidateModel.model_validate(payload)

    assert candidate.change_summary == "unspecified"
    assert all(state.unit == "unspecified" for state in candidate.states)
    assert all(process.description == "unspecified" for process in candidate.processes)
    assert all(
        mapping.unit == "unspecified" for mapping in candidate.observation_mappings
    )
    assert all(parameter.unit == "unspecified" for parameter in candidate.parameters)
    assert all(
        constraint.description == "unspecified" for constraint in candidate.constraints
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("parameters", 0, "bounds"),
            {"lower": 2.0, "upper": 2.0},
            "nondegenerate",
        ),
        (
            ("parameters", 0, "initialization_range"),
            {"lower": -1.0, "upper": 2.0},
            "within bounds",
        ),
        (("parameters", 0, "scope"), "per_run", "scope"),
        (("states", 0, "name"), "not valid", "string_pattern"),
    ],
)
def test_candidate_rejects_invalid_nested_values(
    candidate_payload: dict[str, Any],
    path: tuple[str | int, ...],
    value: Any,
    message: str,
) -> None:
    payload = copy.deepcopy(candidate_payload)
    target: Any = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ValidationError, match=message):
        CandidateModel.model_validate(payload)


def test_candidate_defers_cross_namespace_collisions(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["processes"][0]["name"] = "target_state"

    assert CandidateModel.model_validate(payload).processes[0].name == "target_state"


def test_candidate_requires_equation_and_initialization_for_every_state(
    candidate_payload: dict[str, Any],
) -> None:
    missing_equation = copy.deepcopy(candidate_payload)
    missing_equation["state_equations"].pop()
    # Equation closure is contextual and therefore deferred until the
    # deterministic validator runs after supplied-channel repair.
    assert len(CandidateModel.model_validate(missing_equation).state_equations) == 1

    missing_initial = copy.deepcopy(candidate_payload)
    missing_initial["initial_conditions"].pop()
    assert len(CandidateModel.model_validate(missing_initial).initial_conditions) == 1


def test_identity_mapped_observed_initialization_uses_measured_channel(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["initial_conditions"][0]["scope"] = "trajectory_specific"

    candidate = CandidateModel.model_validate(payload)

    initial = candidate.initial_conditions[0]
    assert initial.scope.value == "global"
    assert initial.expression == "measured_target"
    assert initial.initialization_range is None


def test_identity_mapping_preserves_fixed_initialization_without_context(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["initial_conditions"][0].update(
        {
            "scope": "global",
            "fixed_value": 0.0,
            "expression": None,
            "initialization_range": None,
        }
    )

    candidate = CandidateModel.model_validate(payload)

    initial = candidate.initial_conditions[0]
    assert initial.scope.value == "global"
    assert initial.fixed_value == 0.0
    assert initial.expression is None
    assert initial.initialization_range is None


def test_identity_mapping_overrides_incorrect_latent_label_for_initialization(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["initial_conditions"][0]["scope"] = "trajectory_specific"
    payload["states"][0]["kind"] = "latent"

    candidate = CandidateModel.model_validate(payload)

    initial = candidate.initial_conditions[0]
    assert initial.scope.value == "global"
    assert initial.expression == "measured_target"


def test_unmapped_observed_trajectory_initialization_remains_invalid(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["observation_mappings"][0]["expression"] = "latent_response"
    payload["initial_conditions"][0]["scope"] = "trajectory_specific"

    with pytest.raises(ValidationError, match="only for latent states"):
        CandidateModel.model_validate(payload)


def test_candidate_defers_constraint_subjects_but_rejects_extra_field(
    candidate_payload: dict[str, Any],
) -> None:
    unknown = copy.deepcopy(candidate_payload)
    unknown["constraints"][0]["subject"] = "undeclared"
    assert CandidateModel.model_validate(unknown).constraints[0].subject == "undeclared"

    extra = copy.deepcopy(candidate_payload)
    extra["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CandidateModel.model_validate(extra)


def test_nonnegative_constraint_may_also_refine_numeric_bounds(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["constraints"][0]["bounds"] = {"lower": 0.0, "upper": 100.0}

    constraint = CandidateModel.model_validate(payload).constraints[0]

    assert constraint.kind.value == "nonnegative"
    assert constraint.bounds is not None
    assert constraint.bounds.upper == 100.0


def test_initialization_range_may_declare_fixed_value(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["initial_conditions"][1]["initialization_range"] = {
        "lower": 0.0,
        "upper": 0.0,
    }

    candidate = CandidateModel.model_validate(payload)

    assert candidate.initial_conditions[1].initialization_range.lower == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [("fixed_value", 1.25), ("expression", "0.5 * target + covariate")],
)
def test_initial_condition_supports_explicit_causal_modes(
    candidate_payload: dict[str, Any], field: str, value: Any
) -> None:
    payload = copy.deepcopy(candidate_payload)
    initial = payload["initial_conditions"][1]
    initial.pop("initialization_range")
    initial[field] = value

    candidate = CandidateModel.model_validate(payload)
    restored = CandidateModel.model_validate_json(candidate.model_dump_json())

    assert getattr(restored.initial_conditions[1], field) == value


def test_null_observed_initialization_uses_identity_observation(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    observed = payload["initial_conditions"][0]
    observed["initialization_range"] = None

    candidate = CandidateModel.model_validate(payload)

    assert candidate.initial_conditions[0].expression == "measured_target"


def test_null_latent_initialization_is_deferred_to_contextual_validation(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    latent = payload["initial_conditions"][1]
    latent["initialization_range"] = None

    candidate = CandidateModel.model_validate(payload)

    assert candidate.initial_conditions[1].fixed_value is None
    assert candidate.initial_conditions[1].expression is None
    assert candidate.initial_conditions[1].initialization_range is None


def test_candidate_rejects_nonfinite_number(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["parameters"][0]["bounds"]["upper"] = float("inf")

    with pytest.raises(ValidationError, match="finite"):
        CandidateModel.model_validate(payload)
