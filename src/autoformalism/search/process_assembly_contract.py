"""Versioned assembly ownership; scientific functions remain proposer decisions."""

from __future__ import annotations

import ast
import json
from collections import Counter
from copy import deepcopy

from pydantic import StrictBool

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.schemas.staged_functions import InteractionFunctionReply
from autoformalism.search import scientific_verification
from autoformalism.staged_functions import normalize_topology_owned_sign
from autoformalism.staged_topology import _ancestors

POLICY = "process-assembly-contract-1"


class AssemblyFunctionReply(InteractionFunctionReply):
    """Confirm intrinsic conversion overlap only in a focused atomic repair."""

    revise_dependencies: StrictBool = False
    conversion_factor_is_intrinsic: StrictBool = False


def validate_policy(policy: str) -> None:
    """Never interpret an unknown assembly policy as historical behavior."""
    if policy not in {"legacy", POLICY}:
        raise ValueError("unknown process assembly policy")


def decorate(selected: dict, bindings: list[dict]) -> dict:
    """Expose the actual consumer assembly, without requiring prose inference."""
    result = deepcopy(selected)
    binding = next(
        (
            b
            for b in bindings
            if b["proposal"]["name"] == selected["lhs"] and "signed_declaration" in b
        ),
        None,
    )
    if binding is not None:
        uses = binding["signed_declaration"]["uses"]
        result["process_instruction"] = (
            "Define ONE intrinsic law before the declared consumer signs and "
            "conversions. Runtime normalizes redundant whole-law minus factors, "
            "preserving internal differences/calls/powers. This does not require "
            "nonnegative function values. Do not duplicate consumer conversions."
        )
        result["assembly_contract"] = {
            "policy": POLICY,
            "outer_sign_owner": "process_uses",
            "conversions": uses,
            "consumer_templates": [
                {
                    "target": u["target"],
                    "contribution": f"{'-' if u['sign'] == 'negative' else '+'} gain * "
                    f"({u['conversion'] or 'UNRESOLVED_CONVERSION'}) * "
                    f"{selected['lhs']}",
                }
                for u in uses
            ],
            "conversion_overlap_is_not_physical_error_proof": True,
        }
    else:
        result["assembly_contract"] = {
            "policy": POLICY,
            "outer_sign_owner": "function"
            if selected["outer_weight_sign"] == "unrestricted"
            else "topology",
            "conversions": [],
        }
    return result


def _outer_symbols(node: ast.expr, exponent: int = 1) -> Counter:
    """Count outer multiplicative symbols, never inspect inner scientific laws."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _outer_symbols(node.operand, exponent)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        result = _outer_symbols(node.left, exponent)
        result.update(
            _outer_symbols(
                node.right, exponent * (-1 if isinstance(node.op, ast.Div) else 1)
            )
        )
        return result
    return Counter({node.id: exponent}) if isinstance(node, ast.Name) else Counter()


def conversion_overlap(expression: str, selected: dict) -> list[dict]:
    """Find repeated outer covariate factors, not arbitrary symbol overlap."""
    law = RestrictedParser().parse(expression, location="assembly law")
    powers = _outer_symbols(law.tree.body)
    rows = []
    for use in selected.get("assembly_contract", {}).get("conversions", []):
        conversion = use["conversion"]
        if conversion is None:
            continue
        tree = RestrictedParser().parse(conversion, location="consumer conversion")
        factors = {n: e for n, e in _outer_symbols(tree.tree.body).items() if e}
        if factors and all(
            powers[n] * e > 0 and abs(powers[n]) >= abs(e) for n, e in factors.items()
        ):
            rows.append(
                {
                    "target": use["target"],
                    "conversion": conversion,
                    "repeated_covariate_powers": factors,
                    "assembled_contribution": (
                        f"{'-' if use['sign'] == 'negative' else '+'} gain * "
                        f"({conversion}) * ({expression})"
                    ),
                }
            )
    return rows


def prepare(
    reply: InteractionFunctionReply, selected: dict
) -> tuple[InteractionFunctionReply, dict]:
    """Normalize owned signs and request an explicit decision on conversion overlap."""
    contract = selected["assembly_contract"]
    if contract["policy"] != POLICY:
        raise ValueError("assembly slot policy differs")
    plain = InteractionFunctionReply(
        expression=reply.expression, parameters=reply.parameters
    )
    sign = (
        "positive"
        if contract["outer_sign_owner"] == "process_uses"
        else selected["outer_weight_sign"]
    )
    normalized, signs = normalize_topology_owned_sign(plain, outer_weight_sign=sign)
    overlaps = conversion_overlap(normalized.expression, selected)
    intrinsic = bool(getattr(reply, "conversion_factor_is_intrinsic", False))
    if overlaps and not intrinsic:
        raise ValueError(
            "CONSUMER_CONVERSION_OVERLAP: "
            + json.dumps(
                {
                    "uses": overlaps,
                    "repair": "Return the intrinsic law before the conversion. "
                    "If this outer factor is intentionally part of the physical "
                    "law, keep it and set conversion_factor_is_intrinsic=true. Runtime "
                    "does not remove covariates or certify physical units.",
                },
                sort_keys=True,
            )
        )
    return normalized, {
        "raw_reply": reply.model_dump(mode="json"),
        "normalized_reply": normalized.model_dump(mode="json"),
        "sign_normalizations": [s.model_dump(mode="json") for s in signs],
        "consumer_conversion_overlap": overlaps,
        "intrinsic_factor_confirmed": intrinsic,
        "scientific_validity_certified": False,
    }


def process_paths(brief, source: dict) -> set[tuple[str, str]]:
    """Equation-derived paths from every modeled algebraic process to public targets."""
    graph = {
        e["name"]: {n for t in e["terms"] for n in t["sources"]}
        for e in source["equations"]
    }
    processes = {
        e["name"] for e in source["equations"] if e["definition"] == "algebraic"
    }
    targets = {v.name for v in brief.public_variables if v.data_role == "target"}
    return {(p, t) for t in targets for p in processes if p in _ancestors(t, graph)}


def preserve_paths(brief, before: dict, after: dict) -> None:
    """A local source edit cannot silently sever an existing process-target claim."""
    if not scientific_verification.enabled():
        return
    lost = sorted(process_paths(brief, before) - process_paths(brief, after))
    if lost:
        raise ValueError(
            "PROCESS_TARGET_PATH_LOST: "
            + json.dumps(
                {
                    "lost_paths": [{"process": p, "target": t} for p, t in lost],
                    "repair": "Preserve these process-to-target paths locally. "
                    "To remove a process, use a separate topology revision "
                    "updating its consumers. Do not invent a dummy connection.",
                },
                sort_keys=True,
            )
        )


def prompt(system: str) -> str:
    """Make ownership explicit at every batch and focused repair request."""
    return system + (
        "\nAssembly contract: obey each slot's assembly_contract and its displayed "
        "consumer_templates. For fixed topology/process-use signs the runtime "
        "removes redundant explicit OUTER minus factors once; internal signed "
        "differences/calls/powers survive. Return the law before consumer "
        "conversions. Conversion overlap is a question, not proof of wrong physics: "
        "in atomic repair you may explicitly confirm an intentional intrinsic "
        "outer factor with conversion_factor_is_intrinsic=true. No conversion is "
        "silently removed. Local repairs must preserve existing process-to-target "
        "paths; intentional removal belongs to a separate topology revision."
    )
