"""Slot-addressed delivery using the existing restricted scalar-function contract."""

from __future__ import annotations

import ast
from collections import Counter

from pydantic import Field

from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.staged_functions import InteractionFunctionReply
from autoformalism.search import process_assembly_revision as revision
from autoformalism.search.staged_function_prompts import (
    render_equation_function_batch_system_prompt,
    render_interaction_function_system_prompt,
)

POLICY = "identified-function-delivery-1"


class SlotFunction(InteractionFunctionReply):
    """The sole new delivery field identifies the runtime-owned function slot."""

    interaction_id: str = Field(pattern=r"^term_\d+_\d+$")


class FunctionBatch(StrictSchema):
    functions: tuple[SlotFunction, ...] = Field(max_length=64)


def system_prompt() -> str:
    """Reuse scientific/grammar guidance; replace only positional delivery rules."""
    base = render_equation_function_batch_system_prompt()
    base = base.replace(
        "one ordered functions array. Its\n"
        "length and order must exactly match selected_equation.terms.",
        "a functions array. Each entry adds interaction_id from requested_slots. "
        "Return one entry per requested slot; order is immaterial.",
    ).replace("slot or interaction identifiers,", "")
    return (
        base
        + """
The requested_slots map is the authoritative list of blanks to fill. The whole
model is context only. runtime_supplied_slots are already filled and must NOT be
returned. Do not split one grouped-source function into multiple returned terms.
A shared process law is defined once; its consumer uses are runtime supplied.
Return only its intrinsic scalar RHS, without a name or equals sign. Signs and
consumer conversions shown in assembly_contract are applied by the runtime.
Conversion-owned supplied fixed covariates may be absent from the intrinsic law.
Other dependency changes and conversion decisions use a subsequent focused repair.
"""
    )


def normalize_assignment(reply: dict, selected: dict) -> tuple[dict, list[dict]]:
    """Strip only a matching simple algebraic label; never interpret statements."""
    value = dict(reply)
    expression = value["expression"]
    if not isinstance(expression, str) or len(expression) > 4096:
        raise ValueError(
            "expression must be a scalar string of at most 4096 characters"
        )
    try:
        ast.parse(expression, mode="eval")
        return value, []
    except SyntaxError:
        pass
    try:
        tree = ast.parse(expression, mode="exec")
    except SyntaxError as exc:
        raise ValueError("return a parseable scalar RHS expression") from exc
    if (
        selected["definition"] != "algebraic"
        or len(tree.body) != 1
        or not isinstance(tree.body[0], ast.Assign)
        or len(tree.body[0].targets) != 1
        or not isinstance(tree.body[0].targets[0], ast.Name)
        or tree.body[0].targets[0].id != selected["lhs"]
    ):
        raise ValueError(
            "return only the scalar RHS; assignment label does not match "
            "the selected algebraic definition"
        )
    value["expression"] = ast.unparse(tree.body[0].value)
    return value, [
        {
            "code": "EXACT_ALGEBRAIC_LABEL_REMOVED",
            "before": expression,
            "after": value["expression"],
        }
    ]


def unpack(raw: object, requested: list[str]) -> tuple[dict, list[str]]:
    """Keep only uniquely identified requested entries; never guess array positions."""
    if (
        not isinstance(raw, dict)
        or set(raw) != {"functions"}
        or not isinstance(raw["functions"], list)
    ):
        return {}, ["return an object containing a functions array"]
    items = raw["functions"]
    if len(items) > 64:
        return {}, ["function delivery exceeds the 64-entry envelope"]
    ids = [
        item.get("interaction_id")
        for item in items
        if isinstance(item, dict) and isinstance(item.get("interaction_id"), str)
    ]
    counts = Counter(ids)
    mapped, errors = {}, []
    for item in items:
        try:
            function = SlotFunction.model_validate(item)
            name = function.interaction_id
            if name not in requested or counts[name] != 1:
                raise ValueError(f"unrequested or duplicate interaction_id: {name}")
            mapped[name] = function.model_dump(mode="json", exclude={"interaction_id"})
        except ValueError as exc:
            errors.append(str(exc)[:2000])
    return mapped, errors


def interpret(brief, context, source, known, identifier, raw, *, topology=False):
    """Normalize delivery only, then apply all existing expression/graph checks."""
    normalized, labels = normalize_assignment(
        raw, revision.selected_slot(source, identifier)
    )
    i, _ = revision.dep.slot(source, identifier)
    if labels and len(source["equations"][i]["terms"]) != 1:
        raise ValueError("assignment label is ambiguous for a multi-term definition")
    companions = []
    for law in normalized.get("companion_laws", []):
        law, decisions = normalize_assignment(
            law, revision.selected_slot(source, law["interaction_id"])
        )
        labels.extend({"interaction_id": law["interaction_id"], **d} for d in decisions)
        companions.append(law)
    if "companion_laws" in normalized:
        normalized["companion_laws"] = companions
    transaction = revision.prepare(
        brief,
        context,
        source,
        identifier,
        revision.RevisionReply.model_validate(normalized),
        known_functions=known,
        allow_topology_repair=topology,
    )
    return transaction, labels


def repair_grammar() -> str:
    """Reuse the ordinary function grammar and parameter declarations verbatim."""
    text = render_interaction_function_system_prompt()
    return text[
        text.index("The restricted grammar permits") : text.index(
            "Do not emit an assignment"
        )
    ]
