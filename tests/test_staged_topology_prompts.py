"""Focused tests for variable-first staged provider prompts."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from autoformalism.search.staged_topology_prompts import (
    render_equation_topology_system_prompt,
    render_equation_topology_user_prompt,
    render_variable_identification_system_prompt,
    render_variable_identification_user_prompt,
)


def _runtime_payload(prompt: str) -> dict[str, object]:
    _, serialized = prompt.split("\n", maxsplit=1)
    payload = json.loads(serialized)
    assert isinstance(payload, dict)
    return payload


def test_variable_identification_prompt_has_one_scientific_responsibility() -> None:
    system = render_variable_identification_system_prompt()

    assert "supplied, differential, algebraic, or unused" in system
    assert "multiple displayed mechanisms" in system
    assert "Do not propose equations" in system
    assert "valid empty variable list is allowed" in system
    assert "Public-data roles" in system
    assert "public variable currently marked unused may be activated" in system
    assert "only definition update allowed" in system
    assert "every internal\nvariable must preserve" in system


def test_variable_identification_request_keeps_runtime_artifacts_separate() -> None:
    prompt = render_variable_identification_user_prompt(
        public_brief_json=json.dumps(
            {
                "schema_version": "staged-public-brief-1",
                "public_variables": [
                    {"name": "Gp", "data_role": "target"},
                    {"name": "meal_event_g", "data_role": "external_input"},
                ],
            }
        ),
        agenda_json=json.dumps(
            {
                "requirement_ids": [
                    "meal_pathway",
                    "delayed_insulin_action",
                ],
                "target_names": ["Gp", "U"],
            }
        ),
        inventory_json="[]",
        diagnostics_json=json.dumps(
            {"items": [{"code": "MISSING_TARGET", "target": "U"}]}
        ),
    )

    payload = _runtime_payload(prompt)
    assert payload["schema_version"] == "variable-identification-request-1"
    assert payload["agenda"] == {
        "requirement_ids": ["meal_pathway", "delayed_insulin_action"],
        "target_names": ["Gp", "U"],
    }
    assert payload["current_inventory"] == []
    diagnostics = payload["runtime_diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["items"][0]["code"] == "MISSING_TARGET"


def test_equation_topology_prompt_enforces_frozen_inventory_and_grouped_terms() -> (
    None
):
    system = render_equation_topology_system_prompt()

    assert "frozen variable inventory" in system
    assert "complete joint source set" in system
    assert "flatten a multivariate\ninteraction" in system
    assert "inventory_revision" in system
    assert "Do not repeat the left-hand side" in system
    assert "does not claim" in system


def test_equation_topology_request_includes_selected_lhs_and_allowed_sources() -> (
    None
):
    prompt = render_equation_topology_user_prompt(
        public_brief_json=json.dumps(
            {"schema_version": "staged-public-brief-1"}
        ),
        agenda_json=json.dumps(
            {"requirement_ids": ["meal_pathway"], "target_names": ["Gp"]}
        ),
        inventory_json=json.dumps(
            [
                {"name": "Gp", "definition": "differential"},
                {"name": "meal_memory", "definition": "differential"},
                {"name": "meal_event_g", "definition": "supplied"},
            ]
        ),
        selected_lhs_json=json.dumps(
            {"name": "Gp", "definition": "differential"}
        ),
        equation_sketch_json="[]",
        allowed_sources_json=json.dumps(
            ["Gp", "meal_memory", "meal_event_g"]
        ),
    )

    payload = _runtime_payload(prompt)
    assert payload["schema_version"] == "equation-topology-request-1"
    assert payload["selected_lhs"] == {
        "definition": "differential",
        "name": "Gp",
    }
    assert payload["allowed_sources"] == [
        "Gp",
        "meal_memory",
        "meal_event_g",
    ]
    assert "runtime_diagnostics" not in payload


@pytest.mark.parametrize(
    "renderer,kwargs,label",
    [
        (
            render_variable_identification_user_prompt,
            {
                "public_brief_json": '{"bad": NaN}',
                "agenda_json": "{}",
                "inventory_json": "[]",
            },
            "public_brief_json",
        ),
        (
            render_equation_topology_user_prompt,
            {
                "public_brief_json": "{}",
                "agenda_json": "{}",
                "inventory_json": "[]",
                "selected_lhs_json": "{}",
                "equation_sketch_json": "[]",
                "allowed_sources_json": "{}",
            },
            "allowed_sources_json",
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
