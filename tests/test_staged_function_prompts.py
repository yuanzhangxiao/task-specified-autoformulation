"""Focused tests for immutable staged function provider prompts."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from autoformalism.search.staged_function_prompts import (
    render_interaction_function_system_prompt,
    render_interaction_function_user_prompt,
    render_latent_initial_system_prompt,
    render_latent_initial_user_prompt,
)


def _runtime_payload(prompt: str) -> dict[str, object]:
    _, serialized = prompt.split("\n", maxsplit=1)
    payload = json.loads(serialized)
    assert isinstance(payload, dict)
    return payload


def test_interaction_function_prompt_has_one_immutable_responsibility() -> None:
    system = render_interaction_function_system_prompt()

    assert "one runtime-selected interaction" in system
    assert "one scalar right-hand-side contribution" in system
    assert "symbol set must equal the selected grouped source" in system
    assert "Use the exact displayed scientific names" in system
    assert "inner FUNCTION slot" in system
    assert "applies the frozen outer assembly\nsign exactly once" in system
    assert "d(x)/dt = ... - (FUNCTION)" in system
    assert "never return\n``-x / tau``" in system
    assert "scientific functional law, not a transcription" in system
    assert "Preserve required nonlinear\nbehavior" in system
    assert "do not silently set unknown rates or gains to one" in system
    assert "known balance contribution" in system
    assert "binary +, -, *, /, and Python **" in system
    assert "integer literal\nwith absolute value at most 16" in system
    assert "Declare every parameter used" in system
    assert "including a shared\nparameter" in system
    assert "Supported roles are coefficient" in system
    assert "must use a scientifically appropriate nonnegative" in system
    assert "Do not emit an assignment" in system
    assert "Do not repair or\nroute to another topology term" in system


def test_interaction_function_request_keeps_runtime_artifacts_separate() -> None:
    prompt = render_interaction_function_user_prompt(
        public_brief_json=json.dumps(
            {"schema_version": "staged-public-brief-1", "targets": ["U"]}
        ),
        inventory_json=json.dumps(
            [
                {"name": "U", "definition": "algebraic"},
                {"name": "X", "definition": "differential"},
            ]
        ),
        equation_sketch_json=json.dumps(
            [{"name": "U", "definition": "algebraic", "terms": []}]
        ),
        selected_term_json=json.dumps(
            {
                "lhs": "U",
                "definition": "algebraic",
                "sources": ["X"],
                "outer_sign": "add",
                "scientific_role": "delayed insulin-dependent disposal",
            }
        ),
        accepted_functions_json=json.dumps(
            [{"lhs": "U", "sources": ["Uii"], "expression": "Uii"}]
        ),
        parameter_registry_json=json.dumps(
            {"gain": {"role": "nonnegative_coefficient"}}
        ),
        diagnostics_json=json.dumps(
            {"items": [{"code": "MISSING_FUNCTION", "lhs": "U"}]}
        ),
    )

    payload = _runtime_payload(prompt)
    assert payload["schema_version"] == "interaction-function-request-1"
    assert payload["selected_term"] == {
        "definition": "algebraic",
        "lhs": "U",
        "outer_sign": "add",
        "scientific_role": "delayed insulin-dependent disposal",
        "sources": ["X"],
    }
    assert payload["accepted_functions"] == [
        {"expression": "Uii", "lhs": "U", "sources": ["Uii"]}
    ]
    assert payload["parameter_registry"] == {
        "gain": {"role": "nonnegative_coefficient"}
    }
    assert payload["runtime_diagnostics"] == {
        "items": [{"code": "MISSING_FUNCTION", "lhs": "U"}]
    }


def test_latent_initial_prompt_enforces_narrow_causal_boundary() -> None:
    system = render_latent_initial_system_prompt()

    assert "exactly one mode" in system
    assert "finite numeric fixed_value" in system
    assert "displayed\nallowed-symbol list" in system
    assert (
        "supplied\nauxiliaries, external inputs, fixed covariates, and time"
        in system
    )
    assert "ordinary\nor lagged target" in system
    assert "fitted\nparameter" in system
    assert "Directly observed state initializers are runtime-derived" in system
    assert "Do not repeat or rename the selected state" in system


def test_latent_initial_request_has_selected_state_and_allowed_symbols() -> None:
    prompt = render_latent_initial_user_prompt(
        public_brief_json=json.dumps(
            {"schema_version": "staged-public-brief-1", "targets": ["Gp"]}
        ),
        inventory_json=json.dumps(
            [
                {"name": "X", "definition": "differential"},
                {"name": "meal_event_g", "definition": "supplied"},
            ]
        ),
        equation_sketch_json=json.dumps(
            [{"name": "X", "definition": "differential", "terms": []}]
        ),
        selected_state_json=json.dumps(
            {"name": "X", "definition": "differential"}
        ),
        allowed_symbols_json=json.dumps(["meal_event_g", "t"]),
        accepted_functions_json=json.dumps(
            [{"lhs": "X", "sources": ["I"], "expression": "gain * I"}]
        ),
    )

    payload = _runtime_payload(prompt)
    assert payload["schema_version"] == "latent-initial-request-1"
    assert payload["selected_state"] == {
        "definition": "differential",
        "name": "X",
    }
    assert payload["allowed_symbols"] == ["meal_event_g", "t"]
    assert "runtime_diagnostics" not in payload


@pytest.mark.parametrize(
    "renderer,kwargs,label",
    [
        (
            render_interaction_function_user_prompt,
            {
                "public_brief_json": "{}",
                "inventory_json": "[]",
                "equation_sketch_json": "[]",
                "selected_term_json": "[]",
                "accepted_functions_json": "[]",
                "parameter_registry_json": "{}",
            },
            "selected_term_json",
        ),
        (
            render_interaction_function_user_prompt,
            {
                "public_brief_json": "{}",
                "inventory_json": "[]",
                "equation_sketch_json": "[]",
                "selected_term_json": "{}",
                "accepted_functions_json": "{}",
                "parameter_registry_json": "{}",
            },
            "accepted_functions_json",
        ),
        (
            render_latent_initial_user_prompt,
            {
                "public_brief_json": '{"bad": NaN}',
                "inventory_json": "[]",
                "equation_sketch_json": "[]",
                "selected_state_json": "{}",
                "allowed_symbols_json": "[]",
                "accepted_functions_json": "[]",
            },
            "public_brief_json",
        ),
        (
            render_latent_initial_user_prompt,
            {
                "public_brief_json": "{}",
                "inventory_json": "[]",
                "equation_sketch_json": "[]",
                "selected_state_json": "{}",
                "allowed_symbols_json": {},
                "accepted_functions_json": "[]",
            },
            "allowed_symbols_json",
        ),
    ],
)
def test_prompt_renderers_reject_invalid_runtime_json(
    renderer: Callable[..., str],
    kwargs: dict[str, object],
    label: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=label):
        renderer(**kwargs)


def test_request_serialization_is_canonical() -> None:
    common = {
        "inventory_json": "[]",
        "equation_sketch_json": "[]",
        "selected_term_json": '{"sources":["x"],"lhs":"z"}',
        "accepted_functions_json": "[]",
        "parameter_registry_json": "{}",
    }

    first = render_interaction_function_user_prompt(
        public_brief_json='{"b":2,"a":1}',
        **common,
    )
    second = render_interaction_function_user_prompt(
        public_brief_json='{"a":1,"b":2}',
        **common,
    )

    assert first == second
