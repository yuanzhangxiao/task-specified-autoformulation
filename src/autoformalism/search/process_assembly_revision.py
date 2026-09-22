"""Atomic law/conversion repairs and bounded graph edits, without physics inference.

This opt-in interface is deliberately separate from historical construction
policies. Callers commit the returned source and functions together, or neither.
"""

from __future__ import annotations

from copy import deepcopy

from pydantic import Field, StrictBool

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.construction import FunctionalDraft
from autoformalism.schemas.staged_functions import InteractionFunctionReply
from autoformalism.schemas.staged_topology import EquationDefinition, ScientificVariable
from autoformalism.search import function_dependencies as dep
from autoformalism.search import process_assembly_contract as assembly
from autoformalism.search import shared_process_contract as shared
from autoformalism.search import signed_processes as signed
from autoformalism.search.staged_function_runner import _obligation, _selected_term
from autoformalism.staged_functions import (
    bind_function_reply,
    normalize_topology_owned_sign,
    repair_certified_outer_gain_role,
)
from autoformalism.staged_topology import content_hash, lower_topology
from autoformalism.staging import topology_commitment_sha256

POLICY = "process-assembly-revision-1"


class LawReply(InteractionFunctionReply):
    """An expression defines the actual dependencies, without a redundant list."""

    revise_dependencies: StrictBool = False


class ConversionEdit(StrictSchema):
    """Only a named existing consumer's conversion can change, never its sign."""

    target: Identifier
    conversion: str | None = Field(default=None, min_length=1, max_length=512)


class CompanionLaw(LawReply):
    """An explicitly revised ordinary slot in a bounded topology repair."""

    interaction_id: str = Field(pattern=r"^term_\d+_\d+$")


class RevisionReply(LawReply):
    """One scientific transaction, interpreted only under the displayed scope."""

    consumer_conversions: tuple[ConversionEdit, ...] = Field(default=(), max_length=64)
    retain_intrinsic_for: tuple[Identifier, ...] = Field(default=(), max_length=64)
    companion_laws: tuple[CompanionLaw, ...] = Field(default=(), max_length=64)


def selected_slot(source: dict, identifier: str) -> dict:
    """Render signs, consumers and conversions from the actual current source."""
    i, j = dep.slot(source, identifier)
    equations = tuple(EquationDefinition.model_validate(e) for e in source["equations"])
    inventory = tuple(ScientificVariable.model_validate(v) for v in source["inventory"])
    bindings = shared.validate_contract(
        source.get("shared_process_contract"), equations, inventory
    )
    return assembly.decorate(
        shared.decorate_function_term(
            _selected_term(
                equations[i],
                equations[i].terms[j],
                parameter_identity_policy="interaction_local",
            ),
            bindings,
        ),
        bindings,
    )


def conversion_sources(selected: dict) -> set[str]:
    """Keep dependencies of the outer conversion separate from the intrinsic law."""
    return {
        name
        for use in selected["assembly_contract"]["conversions"]
        if use["conversion"] is not None
        for name in RestrictedParser()
        .parse(use["conversion"], location="conversion")
        .symbols
    }


def _binding(source: dict, name: str) -> dict | None:
    return next(
        (
            b
            for b in (source.get("shared_process_contract") or {}).get("bindings", [])
            if b["proposal"]["name"] == name and "signed_declaration" in b
        ),
        None,
    )


def _edit_conversions(
    brief, context, source: dict, identifier: str, edits
) -> list[dict]:
    """Validate fixed positive syntax; never manufacture or infer a unit conversion."""
    if not edits:
        return []
    i, _ = dep.slot(source, identifier)
    binding = _binding(source, source["equations"][i]["name"])
    if binding is None:
        raise ValueError(
            "conversion edits require the defining slot of a signed process"
        )
    uses = binding["signed_declaration"]["uses"]
    allowed = {u["target"] for u in uses}
    names = [e.target for e in edits]
    if len(set(names)) != len(names) or set(names) - allowed:
        raise ValueError("conversion edits need unique existing consumer targets")
    fixed = dict.fromkeys(dep.fixed_sources(brief, context, source), 1.0)
    changes = []
    for edit in edits:
        if edit.conversion is not None:
            signed.conversion_value(edit.conversion, fixed)
        use = next(u for u in uses if u["target"] == edit.target)
        changes.append(
            {
                "target": edit.target,
                "before": use["conversion"],
                "after": edit.conversion,
            }
        )
        use["conversion"] = edit.conversion
    return changes


def _set_sources(
    brief, context, source, identifier, law, *, original_conversion_sources
):
    """Remove only unused conversion-owned fixed covariates without a second flag."""
    i, j = dep.slot(source, identifier)
    term = source["equations"][i]["terms"][j]
    previous = term["sources"]
    actual = set(
        RestrictedParser().parse(law.expression, location="intrinsic law").symbols
    )
    actual -= {p.name for p in law.parameters}
    expected = set(previous)
    allowed = {v["name"] for v in source["inventory"] if v["definition"] != "unused"}
    if actual - allowed:
        raise ValueError(f"unavailable dependency names: {sorted(actual - allowed)}")
    fixed = dep.fixed_sources(brief, context, source)
    omitted_conversion = (expected - actual) & original_conversion_sources & fixed
    if not law.revise_dependencies and (
        expected - actual - omitted_conversion or actual - expected - fixed
    ):
        raise ValueError(
            "DEPENDENCY_DECISION_REQUIRED: "
            f"missing={sorted(expected - actual - omitted_conversion)}, "
            f"extra={sorted(actual - expected - fixed)}; "
            "set revise_dependencies=true for an intentional "
            "state/input/law dependency edit"
        )
    names = [n for n in previous if n in actual] + sorted(actual - expected)
    term["sources"] = names
    binding = _binding(source, source["equations"][i]["name"])
    if binding is not None:
        declaration = {**binding["signed_declaration"], "depends_on": names}
        replacement = signed.binding_for(
            signed.SignedProcess.model_validate(declaration)
        )
        binding.clear()
        binding.update(replacement)
    return {
        "interaction_id": identifier,
        "before": previous,
        "after": names,
        "conversion_only_omissions": sorted(omitted_conversion),
        "explicit_dependency_decision": law.revise_dependencies,
    }


def bind_known(brief, context, source: dict, functions: dict):
    """Rebind retained functions as well as revisions before allowing a commit."""
    inventory = tuple(ScientificVariable.model_validate(v) for v in source["inventory"])
    equations = tuple(EquationDefinition.model_validate(e) for e in source["equations"])
    shared.validate_contract(
        source.get("shared_process_contract"), equations, inventory
    )
    topology, aliases = lower_topology(brief, inventory, equations, context)
    draft = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    for identifier in sorted(functions, key=lambda n: dep.slot(source, n)):
        selected = selected_slot(source, identifier)
        law = InteractionFunctionReply.model_validate(functions[identifier])
        shared.validate_function(law.expression, selected.get("shared_process_use"))
        law, _ = repair_certified_outer_gain_role(
            law,
            set(selected["sources"]),
            outer_weight_sign=selected["outer_weight_sign"],
        )
        draft, _ = bind_function_reply(
            topology, draft, identifier, law, context, aliases, _obligation(selected)
        )
    return topology, draft


def prepare(
    brief,
    context,
    source: dict,
    identifier: str,
    reply: RevisionReply,
    *,
    known_functions: dict | None = None,
    allow_topology_repair: bool = False,
) -> dict:
    """Validate the complete transaction before returning any committed change."""
    reply = RevisionReply.model_validate(reply)
    known = deepcopy(known_functions or {})
    before_selected = selected_slot(source, identifier)
    # JSON-like persisted objects must not retain historical review aliases.
    updated = deepcopy(source)
    if "shared_process_contract" in updated:
        updated["shared_process_contract"] = deepcopy(
            updated["shared_process_contract"]
        )
    conversions = _edit_conversions(
        brief, context, updated, identifier, reply.consumer_conversions
    )
    companions = [r.interaction_id for r in reply.companion_laws]
    if companions and not allow_topology_repair:
        raise ValueError("companion laws require the bounded topology-repair scope")
    if len(set(companions)) != len(companions) or identifier in companions:
        raise ValueError("transaction repeats an interaction")
    for name in companions:
        slot = selected_slot(source, name)
        if slot.get("shared_process_use") or slot["assembly_contract"]["conversions"]:
            raise ValueError(
                "companion edits must be ordinary slots; shared identities stay fixed"
            )
    laws = [(identifier, reply), *((r.interaction_id, r) for r in reply.companion_laws)]
    changes, signs = [], []
    for name, law in laws:
        selected = selected_slot(source, name)
        sign = (
            "positive"
            if selected["assembly_contract"]["outer_sign_owner"] == "process_uses"
            else selected["outer_weight_sign"]
        )
        normalized, normalized_signs = normalize_topology_owned_sign(
            dep.plain(law), outer_weight_sign=sign
        )
        changes.append(
            _set_sources(
                brief,
                context,
                updated,
                name,
                LawReply(
                    **normalized.model_dump(mode="json"),
                    revise_dependencies=law.revise_dependencies,
                ),
                original_conversion_sources=conversion_sources(selected),
            )
        )
        signs.extend(
            {"interaction_id": name, **s.model_dump(mode="json")}
            for s in normalized_signs
        )
        known[name] = normalized.model_dump(mode="json")
    after_selected = selected_slot(updated, identifier)
    overlaps = assembly.conversion_overlap(
        known[identifier]["expression"], after_selected
    )
    overlap_targets = {u["target"] for u in overlaps}
    retained = reply.retain_intrinsic_for
    if len(set(retained)) != len(retained) or set(retained) - overlap_targets:
        raise ValueError(
            "intrinsic retention must name actual overlapping consumer targets"
        )
    if overlap_targets - set(retained):
        raise ValueError("CONVERSION_REPAIR_REQUIRED: " + str(overlaps))
    equations = tuple(
        EquationDefinition.model_validate(e) for e in updated["equations"]
    )
    failures = [c for c in dep.required_checks(brief, equations) if not c["passed"]]
    if failures:
        raise ValueError("PUBLIC_PATHWAY_REPAIR_REQUIRED: " + str(failures))
    assembly.preserve_paths(brief, source, updated)
    topology, draft = bind_known(brief, context, updated, known)
    updated["topology"] = topology.model_dump(mode="json")
    updated["public_structure_checks"] = dep.required_checks(brief, equations)
    updated["public_structure_checks_passed"] = True
    return {
        "policy": POLICY,
        "interaction_id": identifier,
        "scope": "topology" if allow_topology_repair else "local",
        "reply": reply.model_dump(mode="json"),
        "before_source_sha256": content_hash(source),
        "before_functions_sha256": content_hash(known_functions or {}),
        "effective_source": updated,
        "functions": known,
        "draft": draft.model_dump(mode="json"),
        "dependency_changes": changes,
        "conversion_changes": conversions,
        "sign_normalizations": signs,
        "before_conversion_overlap": assembly.conversion_overlap(
            reply.expression, before_selected
        ),
        "remaining_conversion_overlap": overlaps,
        "conversion_status": "retained_intrinsic_unverified"
        if overlaps
        else "no_detected_overlap",
        "scientific_validity_certified": False,
    }


def replay(brief, context, source: dict, known: dict, result: dict) -> dict:
    """Independently reproduce every field; never trust a precomputed transaction."""
    expected = prepare(
        brief,
        context,
        source,
        result["interaction_id"],
        RevisionReply.model_validate(result["reply"]),
        known_functions=known,
        allow_topology_repair=result["scope"] == "topology",
    )
    if expected != result:
        raise ValueError(
            "assembly revision transaction differs from independent replay"
        )
    return expected
