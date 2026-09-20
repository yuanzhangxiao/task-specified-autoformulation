"""Signed process declarations and mechanical assembly, without prose inference."""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.staged_functions import InteractionFunctionReply
from autoformalism.schemas.staged_topology import EquationTerm, ScientificVariable
from autoformalism.search import shared_process_contract as legacy
from autoformalism.staged_topology import content_hash

POLICY = "shared-process-contract-2"
SYSTEM = """Propose optional instantaneous processes after variable selection.
Return processes, or an empty list. Each process has one name, depends_on list,
scientific_meaning, kind, and uses. Each use gives target, sign (positive/negative),
and conversion. Choose ALL use signs NOW; topology will insert them mechanically.
Include each target once PER process. Do not list unrelated system contributions
under one process. Supplied inputs/covariates have no equation to consume a process.
Use an existing algebraic name when appropriate. A process may depend on a dynamic
state it influences, but cannot depend on an algebraic consumer (instantaneous loop).

kind=transfer declares a pairwise transfer with exactly two opposite signs.
kind=influence permits any signs and one or more consumers (including local outlets).
These are assembly policies, not an exhaustive taxonomy or physical certification.
Give conversion as a positive fixed multiplier expression, e.g. 1/area_up, when
known; use "1" for common amount units, or null when unknown. Only positive numeric
constants and displayed covariates with multiplication/division are supported.
Do NOT put the process, its sign, a fitted gain, or a scientific law in conversion.
This optional conversion declares units; null does NOT invalidate the process.
Declare the process's inputs here; its single shared function is generated later.
The runtime fits nonnegative magnitudes; internal nonlinear shapes may be signed.
Do not force shared processes in disconnected systems or invent a second use for
a local outlet. Scientific meaning is explanatory context, not a machine proof.
"""


class SignedUse(StrictSchema):
    """One equation use; null conversion explicitly preserves uncertainty."""

    target: Identifier
    sign: Literal["positive", "negative"]
    conversion: str | None = Field(default=None, min_length=1, max_length=512)


class SignedProcess(StrictSchema):
    name: Identifier
    depends_on: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    kind: Literal["transfer", "influence"]
    uses: tuple[SignedUse, ...] = Field(min_length=1, max_length=64)
    scientific_meaning: str = Field(min_length=1, max_length=650)

    @model_validator(mode="after")
    def unique_signed_uses(self):
        """The same structured rule applies at proposal and independent replay."""
        if len({u.target for u in self.uses}) != len(self.uses):
            raise ValueError("duplicate equation target within this process")
        if self.kind == "transfer" and (
            len(self.uses) != 2
            or {u.sign for u in self.uses} != {"positive", "negative"}
        ):
            raise ValueError("pairwise transfer needs two opposite assembly signs")
        return self


class SignedReply(StrictSchema):
    processes: tuple[SignedProcess, ...] = Field(max_length=64)


def conversion_value(expression: str, covariates: dict[str, float]) -> float:
    """Interpret only positive constants/covariates and products or quotients."""
    try:
        RestrictedParser().parse(expression, location="process conversion")
    except ModelValidationError as exc:
        raise ValueError(str(exc)) from exc

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = float(node.value)
        elif isinstance(node, ast.Name) and node.id in covariates:
            value = float(covariates[node.id])
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            value = left * right if isinstance(node.op, ast.Mult) else left / right
        else:
            raise ValueError("conversion needs positive fixed factors using * or /")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("conversion factors must be finite and positive")
        return value

    return visit(ast.parse(expression, mode="eval").body)


def binding_for(p: SignedProcess) -> dict:
    """Compile to the existing single-law contract; assembly uses are identities."""
    return {
        "proposal": {
            "name": p.name,
            "depends_on": list(p.depends_on),
            "used_in_equations_for": [u.target for u in p.uses],
            "scientific_meaning": p.scientific_meaning,
        },
        "uses": [
            {
                "target": u.target,
                "outer_weight_sign": u.sign,
                "conversion_sources": [],
                "scientific_role": p.scientific_meaning,
            }
            for u in p.uses
        ],
        "signed_declaration": p.model_dump(mode="json"),
    }


def _has_algebraic_cycle(bindings, inventory):
    algebraic = {v.name for v in inventory if v.definition == "algebraic"}
    edges = {name: set() for name in algebraic}
    for b in bindings:
        p = b["proposal"]
        edges[p["name"]].update(set(p["depends_on"]) & algebraic)
        for use in b["uses"]:
            if use["target"] in algebraic:
                edges[use["target"]].add(p["name"])

    finished = set()

    def visit(n, path):
        if n in path:
            return True
        if n in finished:
            return False
        if any(visit(child, path | {n}) for child in edges[n]):
            return True
        finished.add(n)
        return False

    return any(visit(n, set()) for n in edges)


def admit(brief, inventory, raw):
    """Admit independently; malformed conversions do not erase signed processes."""
    if not isinstance(raw, dict) or set(raw) != {"processes"}:
        raise ValueError("expected processes list")
    items = raw["processes"]
    if not isinstance(items, list) or len(items) > 64:
        raise ValueError("expected at most 64 process suggestions")
    duplicates = Counter(
        p.get("name")
        for p in items
        if isinstance(p, dict) and isinstance(p.get("name"), str)
    )
    current, bindings, rejected, conversions = inventory, [], [], []
    covariates = {
        v.name: 1.0 for v in brief.public_variables if v.data_role == "covariate"
    }
    for index, item in enumerate(items):
        try:
            p = SignedProcess.model_validate(item)
            if duplicates[p.name] > 1:
                raise ValueError("duplicate process name")
            b = binding_for(p)
            candidate, audit = legacy.admit(
                brief, current, {"processes": [b["proposal"]]}
            )
            if not audit["suggestions"]:
                raise ValueError(audit["rejected_suggestions"][0]["error"])
            if _has_algebraic_cycle([*bindings, b], candidate):
                raise ValueError(
                    "declared process introduces an instantaneous algebraic cycle"
                )
            # Invalid conversion syntax is unresolved, not a reason to lose the law.
            clean_uses = []
            for use in p.uses:
                try:
                    if use.conversion is not None:
                        conversion_value(use.conversion, covariates)
                    clean_uses.append(use)
                except (ValueError, ArithmeticError) as exc:
                    conversions.append(
                        {
                            "process": p.name,
                            "target": use.target,
                            "expression": use.conversion,
                            "error": str(exc),
                        }
                    )
                    clean_uses.append(use.model_copy(update={"conversion": None}))
            p = p.model_copy(update={"uses": tuple(clean_uses)})
            bindings.append(binding_for(p))
            current = candidate
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append(
                {"index": index, "suggestion": item, "error": str(exc)[:6000]}
            )
    return current, {
        "status": "accepted_partial"
        if bindings and rejected
        else "accepted"
        if bindings
        else "skipped_invalid"
        if rejected
        else "empty",
        "suggestions": [b["proposal"] for b in bindings],
        "bindings": bindings,
        "original_suggestions": items,
        "rejected_suggestions": rejected,
        "conversion_diagnostics": conversions,
        "added_names": [
            v.name for v in current if v.name not in {v.name for v in inventory}
        ],
    }


def propose(brief, enriched, inventory, client, output):
    """One cached call; empty/invalid proposals leave construction open."""
    identity = content_hash(
        [POLICY, enriched, [v.model_dump(mode="json") for v in inventory]]
    )
    path = output / "process_review.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity or saved["sha256"] != content_hash(
            {k: v for k, v in saved.items() if k != "sha256"}
        ):
            raise ValueError("signed process checkpoint differs")
        return tuple(
            ScientificVariable.model_validate(v) for v in saved["inventory"]
        ), saved
    result = {
        "protocol": POLICY,
        "identity": identity,
        "status": "skipped_invalid",
        "request_hash": None,
        "suggestions": [],
        "bindings": [],
        "added_names": [],
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
                    "eligible_equation_targets": [
                        v.name
                        for v in inventory
                        if v.definition in {"differential", "algebraic"}
                    ],
                },
                sort_keys=True,
            ),
            response_model=SignedReply,
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


def assemble(bindings, target, terms):
    """Insert signed uses; tolerate exact repeats, reject contradictory declarations."""
    required = legacy.requirements(bindings, target)
    processes = {b["proposal"]["name"] for b in bindings}
    retained = []
    for term in terms:
        mentioned = set(term.sources) & processes
        if not mentioned:
            retained.append(term)
        elif not any(
            set(term.sources) == set(r["sources"])
            and term.outer_weight_sign.value == r["outer_weight_sign"]
            for r in required
        ):
            raise ValueError("process uses are fixed; return only other equation terms")
    return tuple(retained) + tuple(
        EquationTerm.model_validate(
            {k: r[k] for k in ("sources", "outer_weight_sign", "scientific_role")}
        )
        for r in required
    )


def automatic_function(selected):
    """Consumer identity slots have no LLM call; fitted gains are compiled later."""
    use = selected.get("shared_process_use", {})
    if use.get("automatic_identity"):
        return InteractionFunctionReply(expression=use["process"], parameters=())
    return None
