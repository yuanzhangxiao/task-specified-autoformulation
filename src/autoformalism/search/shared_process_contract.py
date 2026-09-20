"""Opt-in process proposals with one defining law and explicit equation uses.

These are construction commitments, not certificates of conservation or science.
Legacy advisory process reviews remain unchanged.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import Field

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    EquationTerm,
    OuterWeightSign,
    ScientificVariable,
)
from autoformalism.staged_topology import content_hash, freeze_inventory

POLICY = "shared-process-contract-1"
SYSTEM = """Propose scientifically meaningful instantaneous processes after variable
selection. Return the requested JSON. Each process has one function, defined later,
and may be used by one or several equations. It has no independent memory or initial
condition. A shared process reuses that SAME law and its parameters in every use.

Give its name, depends_on, used_in_equations_for, and scientific_meaning. Use only
the displayed eligible names. depends_on lists variables determining the law;
used_in_equations_for lists variables whose modeled equations contain it. Supplied
inflows, geometry, thresholds and initial readings have no modeled equations and
cannot appear in used_in_equations_for. A measured target CAN have a modeled equation.
A local outlet can have just one use. Do not invent a second use or an alias merely
to make a process shared. Reuse an existing algebraic name when it already represents
the process. For a transfer describe its reference direction and physical quantity;
signs and conversion dependencies will be decided together in topology construction.
Different basin areas, volumes or units may require conversions at the uses. Similar
formulas need not be a shared physical process. Do not add coupling between disconnected
systems. No functions, parameters, initial conditions or numeric values at this stage.
An empty processes list is valid. Optional suggestions never erase existing variables.
"""

TOPOLOGY_SYSTEM = """Complete the topology of the displayed process uses together.
The process will have ONE defining function of its declared depends_on variables.
For each declared equation target, give its outer_weight_sign, conversion_sources,
and scientific_role. Include every target exactly once. Do not repeat the process
in conversion_sources. Those are only additional variables needed to convert or
scale the process contribution, not a replacement law. Choose signs scientifically;
the runtime never assumes opposite signs or equal gains from the name alone.
For a transfer, preserve the same transferred physical amount across the balances,
accounting for units, areas or volumes. A shared response can need different gains.
Signed processes are allowed: assembly signs do not assert global positivity or
monotonicity. No functional expressions or parameter declarations here.
"""


class ProcessProposal(StrictSchema):
    """Scientific process identity before signs or functional forms are chosen."""

    name: Identifier
    depends_on: tuple[Identifier, ...] = Field(max_length=64)
    used_in_equations_for: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    scientific_meaning: str = Field(min_length=1, max_length=650)


class ProcessReply(StrictSchema):
    """Provider schema; semantic admission is performed separately for each item."""

    processes: tuple[ProcessProposal, ...] = Field(max_length=64)


class ProcessUse(StrictSchema):
    """One topology-owned assembly operation using an existing process identity."""

    target: Identifier
    outer_weight_sign: OuterWeightSign
    conversion_sources: tuple[Identifier, ...] = Field(max_length=63)
    scientific_role: str = Field(min_length=1, max_length=1000)


class UsesReply(StrictSchema):
    """One coherent topology decision for every use of a process."""

    uses: tuple[ProcessUse, ...] = Field(min_length=1, max_length=64)


def admit(brief, inventory, raw: Any) -> tuple[tuple[ScientificVariable, ...], dict]:
    """Salvage independent valid suggestions without guessing invalid relationships."""
    if not isinstance(raw, dict) or set(raw) != {"processes"}:
        raise ValueError("expected processes list")
    items = raw["processes"]
    if not isinstance(items, list) or len(items) > 64:
        raise ValueError("expected at most 64 process suggestions")
    active = {v.name: v for v in inventory if v.definition != "unused"}
    original = {v.name: v for v in inventory}
    duplicates = Counter(
        p.get("name")
        for p in items
        if isinstance(p, dict) and isinstance(p.get("name"), str)
    )
    accepted, rejected, added = [], [], []
    current = inventory
    for index, item in enumerate(items):
        try:
            p = ProcessProposal.model_validate(item)
            if duplicates[p.name] > 1:
                raise ValueError("conflicting/duplicate process name")
            if len(set(p.depends_on)) != len(p.depends_on) or len(
                set(p.used_in_equations_for)
            ) != len(p.used_in_equations_for):
                raise ValueError("duplicate dependency or equation target")
            missing = (set(p.depends_on) | set(p.used_in_equations_for)) - active.keys()
            if missing:
                raise ValueError(f"unknown or unused variables: {sorted(missing)}")
            invalid = [
                n
                for n in p.used_in_equations_for
                if active[n].definition not in {"differential", "algebraic"}
            ]
            if invalid:
                raise ValueError(
                    f"no modeled equation for supplied variables: {invalid}"
                )
            if p.name in (*p.depends_on, *p.used_in_equations_for):
                raise ValueError("a process cannot depend on or consume itself")
            if p.name in original and original[p.name].definition != "algebraic":
                raise ValueError("cannot repurpose a non-algebraic variable")
            new = (
                ()
                if p.name in original
                else (
                    ScientificVariable(
                        name=p.name,
                        definition="algebraic",
                        scientific_role=p.scientific_meaning,
                    ),
                )
            )
            current = freeze_inventory(brief, (*current, *new))
            accepted.append(p.model_dump(mode="json"))
            added.extend(v.name for v in new)
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append(
                {"index": index, "suggestion": item, "error": str(exc)[:6000]}
            )
    return current, {
        "status": "accepted_partial"
        if accepted and rejected
        else "accepted"
        if accepted
        else "skipped_invalid"
        if rejected
        else "empty",
        "suggestions": accepted,
        "rejected_suggestions": rejected,
        "original_suggestions": items,
        "added_names": added,
    }


def propose(brief, enriched: dict, inventory, client, output: Path):
    """One cached call; malformed or invalid optional content retains the inventory."""
    identity = content_hash(
        [POLICY, enriched, [v.model_dump(mode="json") for v in inventory]]
    )
    path = output / "process_review.json"
    if path.exists():
        saved = json.loads(path.read_text())
        payload = {k: v for k, v in saved.items() if k != "sha256"}
        if saved["identity"] != identity or content_hash(payload) != saved["sha256"]:
            raise ValueError("process proposal checkpoint differs")
        return tuple(
            ScientificVariable.model_validate(v) for v in saved["inventory"]
        ), saved
    result = {
        "protocol": POLICY,
        "identity": identity,
        "status": "skipped_invalid",
        "request_hash": None,
        "suggestions": [],
        "added_names": [],
        "original_suggestions": [],
        "rejected_suggestions": [],
        "error": None,
    }
    accepted = inventory
    try:
        record = client.call(
            system=SYSTEM,
            user=json.dumps(
                {
                    "protocol": POLICY,
                    "public_brief": enriched,
                    "inventory": [v.model_dump(mode="json") for v in inventory],
                    "eligible_dependencies": [
                        v.name for v in inventory if v.definition != "unused"
                    ],
                    "eligible_equation_targets": [
                        v.name
                        for v in inventory
                        if v.definition in {"differential", "algebraic"}
                    ],
                },
                sort_keys=True,
            ),
            response_model=ProcessReply,
            step="optional_process_review",
            attempt=0,
        )
        result["request_hash"] = record["request_hash"]
        accepted, audit = admit(brief, inventory, visible_response(record))
        result.update(audit)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        result["error"] = str(exc)[:6000]
    result["inventory"] = [v.model_dump(mode="json") for v in accepted]
    result["sha256"] = content_hash(result)
    atomic_json(path, result)
    return accepted, result


def validate_uses(proposal: dict, reply: UsesReply, inventory) -> dict:
    """Keep target identities fixed and validate explicit conversion dependencies."""
    p = ProcessProposal.model_validate(proposal)
    names = [u.target for u in reply.uses]
    if Counter(names) != Counter(p.used_in_equations_for):
        raise ValueError("include every declared equation target exactly once")
    active = {v.name for v in inventory if v.definition != "unused"}
    for use in reply.uses:
        if (
            len(set(use.conversion_sources)) != len(use.conversion_sources)
            or set(use.conversion_sources) - active
            or p.name in use.conversion_sources
        ):
            raise ValueError("invalid or duplicate conversion sources")
    return {
        "proposal": proposal,
        "uses": [u.model_dump(mode="json") for u in reply.uses],
    }


def definition(binding: dict) -> EquationDefinition:
    """One process law slot: a signed function, with no independent initial state."""
    p = ProcessProposal.model_validate(binding["proposal"])
    return EquationDefinition(
        name=p.name,
        definition="algebraic",
        terms=(
            EquationTerm(
                sources=p.depends_on,
                outer_weight_sign=OuterWeightSign.UNRESTRICTED,
                scientific_role=p.scientific_meaning,
            ),
        ),
    )


def requirements(bindings: list[dict], target: str) -> list[dict]:
    """Expose already decided uses as immutable terms of the target equation."""
    return [
        {
            "process": b["proposal"]["name"],
            "sources": [b["proposal"]["name"], *u["conversion_sources"]],
            "outer_weight_sign": u["outer_weight_sign"],
            "scientific_role": u["scientific_role"],
        }
        for b in bindings
        for u in b["uses"]
        if u["target"] == target
    ]


def validate_equation_uses(bindings: list[dict], equation: EquationDefinition) -> None:
    """Require each agreed process use exactly once; never accept silent replacement."""
    required = {r["process"]: r for r in requirements(bindings, equation.name)}
    processes = {b["proposal"]["name"] for b in bindings}
    for name in processes:
        terms = [t for t in equation.terms if name in t.sources]
        r = required.get(name)
        if r is None:
            if terms:
                raise ValueError(
                    f"undeclared use of shared process {name}; explicit revision needed"
                )
            continue
        if (
            len(terms) != 1
            or set(terms[0].sources) != set(r["sources"])
            or terms[0].outer_weight_sign.value != r["outer_weight_sign"]
        ):
            raise ValueError(f"preserve the agreed shared-process contribution: {r}")


def function_use(bindings: list[dict], lhs: str, sources: list[str]) -> dict | None:
    """Find the shared identity consumed by one immutable interaction slot."""
    rows = [r for r in requirements(bindings, lhs) if set(r["sources"]) == set(sources)]
    if len(rows) > 1:
        raise ValueError("multiple processes in one conversion are unsupported")
    return rows[0] if rows else None


def decorate_function_term(selected: dict, bindings: list[dict]) -> dict:
    """Use the same contract in provider context, saved slots and reconstruction."""
    if not bindings:
        return selected
    use = function_use(bindings, selected["lhs"], selected["sources"])
    if use is None:
        return selected
    return {
        **selected,
        "shared_process_use": use,
        "process_instruction": (
            "Use the shared process once, multiplied or divided by necessary "
            "conversions/gains. Do not inline its law, add a baseline, or apply "
            "a nonlinear function. Its parameters belong to the process equation."
        ),
    }


def validate_contract(contract: dict | None, equations, inventory) -> list[dict]:
    """Reconstruct the shared-law certificate from immutable scientific declarations."""
    if contract is None:
        return []
    if contract["protocol"] != POLICY:
        raise ValueError("unknown shared-process function contract")
    bindings = contract["bindings"]
    definitions = {}
    for binding in bindings:
        validate_uses(binding["proposal"], UsesReply(uses=binding["uses"]), inventory)
        law = definition(binding)
        if law.name in definitions:
            raise ValueError("duplicate shared process definition")
        definitions[law.name] = law
    for equation in equations:
        if equation.name in definitions and equation != definitions.pop(equation.name):
            raise ValueError("shared process must have exactly one defining law")
        validate_equation_uses(bindings, equation)
    if definitions:
        raise ValueError("missing shared process definitions")
    return bindings


def validate_function(expression: str, use: dict | None) -> None:
    """Certify one linear multiplicative use of P, without evaluating provider text."""
    if use is None:
        return
    RestrictedParser().parse(expression, location="shared_process_use")
    node = ast.parse(expression, mode="eval").body
    name = use["process"]

    def count(n):
        return sum(isinstance(x, ast.Name) and x.id == name for x in ast.walk(n))

    def linear(n):
        if isinstance(n, ast.Name):
            return n.id == name
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            return linear(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            return (count(n.left) == 1 and count(n.right) == 0 and linear(n.left)) or (
                count(n.right) == 1 and count(n.left) == 0 and linear(n.right)
            )
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            return count(n.right) == 0 and linear(n.left)
        return False

    if count(node) != 1 or not linear(node):
        raise ValueError(
            f"reuse {name} once as a multiplicative contribution, e.g. "
            f"{name}/area; do not inline, duplicate or nonlinearly transform its law"
        )
