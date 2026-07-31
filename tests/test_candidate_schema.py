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


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("parameters", 0, "bounds"), {"lower": 2.0, "upper": 2.0}, "lower"),
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


def test_candidate_rejects_duplicate_names(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["processes"][0]["name"] = "target_state"

    with pytest.raises(ValidationError, match="reuse declared names"):
        CandidateModel.model_validate(payload)


def test_candidate_requires_equation_and_initialization_for_every_state(
    candidate_payload: dict[str, Any],
) -> None:
    missing_equation = copy.deepcopy(candidate_payload)
    missing_equation["state_equations"].pop()
    with pytest.raises(ValidationError, match="state equations must cover"):
        CandidateModel.model_validate(missing_equation)

    missing_initial = copy.deepcopy(candidate_payload)
    missing_initial["initial_conditions"].pop()
    with pytest.raises(ValidationError, match="initial conditions must cover"):
        CandidateModel.model_validate(missing_initial)


def test_trajectory_specific_initialization_requires_latent_state(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["initial_conditions"][0]["scope"] = "trajectory_specific"

    with pytest.raises(ValidationError, match="only for latent states"):
        CandidateModel.model_validate(payload)


def test_candidate_rejects_unknown_constraint_and_extra_field(
    candidate_payload: dict[str, Any],
) -> None:
    unknown = copy.deepcopy(candidate_payload)
    unknown["constraints"][0]["subject"] = "undeclared"
    with pytest.raises(ValidationError, match="undeclared subjects"):
        CandidateModel.model_validate(unknown)

    extra = copy.deepcopy(candidate_payload)
    extra["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CandidateModel.model_validate(extra)


def test_candidate_rejects_nonfinite_number(
    candidate_payload: dict[str, Any],
) -> None:
    payload = copy.deepcopy(candidate_payload)
    payload["parameters"][0]["bounds"]["upper"] = float("inf")

    with pytest.raises(ValidationError, match="finite"):
        CandidateModel.model_validate(payload)

