"""Explicit safe mutation recipes for adversarial judge candidates."""

from __future__ import annotations

import ast
from typing import Literal

from pydantic import BaseModel, ConfigDict

from autoformalism.schemas import CandidateModel


class MutationRecipe(BaseModel):
    """One auditable mutation applied to a validated candidate payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mutation_type: Literal[
        "named_disconnected_mechanism",
        "narrative_equation_mismatch",
        "replace_symbol",
        "negate_component",
        "replace_state_with_algebraic",
    ]
    component: str | None = None
    mechanism_id: str | None = None
    old_symbol: str | None = None
    new_symbol: str | None = None
    replacement_expression: str | None = None
    narrative: str | None = None


class AdversarialPair(BaseModel):
    """Known valid/adversarial pair with no score or label in judge payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_id: str
    benchmark_id: str
    tier: str
    mutation_type: str
    valid_candidate: CandidateModel
    adversarial_candidate: CandidateModel


def mutate_candidate(
    candidate: CandidateModel, recipe: MutationRecipe
) -> CandidateModel:
    """Apply a restricted structural mutation without executing generated text."""
    payload = candidate.model_dump(mode="json")
    payload["candidate_id"] = f"{candidate.candidate_id}_adv"
    payload["parent_candidate_id"] = candidate.candidate_id
    if recipe.mutation_type == "named_disconnected_mechanism":
        if not recipe.mechanism_id or not recipe.replacement_expression:
            raise ValueError("disconnected mechanism requires id and expression")
        payload["processes"].append(
            {
                "name": _unique_name(candidate, "claimed_mechanism"),
                "expression": recipe.replacement_expression,
                "mechanisms": [recipe.mechanism_id],
                "description": "Plausible named mechanism disconnected from outputs.",
            }
        )
    elif recipe.mutation_type == "narrative_equation_mismatch":
        if not recipe.narrative:
            raise ValueError("narrative mismatch requires narrative text")
        payload["change_summary"] = recipe.narrative
    elif recipe.mutation_type == "replace_symbol":
        if not all((recipe.component, recipe.old_symbol, recipe.new_symbol)):
            raise ValueError("replace_symbol requires component, old, and new")
        _rewrite_component(
            payload,
            recipe.component,
            lambda source: _replace_symbol(
                source, recipe.old_symbol or "", recipe.new_symbol or ""
            ),
        )
    elif recipe.mutation_type == "negate_component":
        if not recipe.component:
            raise ValueError("negate_component requires component")
        _rewrite_component(payload, recipe.component, lambda source: f"-({source})")
    elif recipe.mutation_type == "replace_state_with_algebraic":
        if not recipe.component or not recipe.replacement_expression:
            raise ValueError("state replacement requires component and expression")
        _replace_state(payload, recipe.component, recipe.replacement_expression)
    return CandidateModel.model_validate(payload)


def _rewrite_component(payload: dict, component: str, rewrite) -> None:
    for equation in payload["state_equations"]:
        if equation["state"] == component:
            equation["rhs"] = rewrite(equation["rhs"])
            return
    for process in payload["processes"]:
        if process["name"] == component:
            process["expression"] = rewrite(process["expression"])
            return
    raise ValueError(f"unknown component: {component}")


def _replace_symbol(source: str, old: str, new: str) -> str:
    class Replace(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == old:
                return ast.copy_location(ast.Name(id=new, ctx=node.ctx), node)
            return node

    parsed = ast.parse(source, mode="eval")
    changed = Replace().visit(parsed)
    return ast.unparse(ast.fix_missing_locations(changed))


def _replace_state(payload: dict, name: str, expression: str) -> None:
    state = next((item for item in payload["states"] if item["name"] == name), None)
    if state is None:
        raise ValueError(f"unknown state: {name}")
    payload["states"] = [item for item in payload["states"] if item["name"] != name]
    payload["state_equations"] = [
        item for item in payload["state_equations"] if item["state"] != name
    ]
    payload["initial_conditions"] = [
        item for item in payload["initial_conditions"] if item["state"] != name
    ]
    payload["processes"].append(
        {
            "name": name,
            "expression": expression,
            "mechanisms": state.get("mechanisms", []),
            "description": "Instantaneous replacement for a dynamic mechanism.",
        }
    )


def _unique_name(candidate: CandidateModel, stem: str) -> str:
    occupied = {
        *(item.name for item in candidate.states),
        *(item.name for item in candidate.processes),
        *(item.name for item in candidate.parameters),
    }
    if stem not in occupied:
        return stem
    index = 2
    while f"{stem}_{index}" in occupied:
        index += 1
    return f"{stem}_{index}"
