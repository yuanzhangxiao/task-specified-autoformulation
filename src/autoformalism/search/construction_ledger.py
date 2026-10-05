"""A provisional scientific draft shared by fixed and adaptive construction."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    EquationTerm,
    PublicScientificBrief,
    ScientificVariable,
    VariableReply,
)
from autoformalism.search import shared_process_contract, signed_processes
from autoformalism.search.variable_bindings import MechanismBinding
from autoformalism.staged_topology import (
    _ancestors,
    audit_explicit_equation_polarity,
    compile_equation_polarity_policy,
    lower_topology,
    merge_variable_reply,
    public_structure_checks,
)


class GeneratedVariable(StrictSchema):
    """Only generated LHS choices; public sources need no activation decision."""

    name: Identifier
    definition: Literal["differential", "algebraic"]
    scientific_role: str = Field(min_length=1, max_length=1000)


class EquationUpdate(StrictSchema):
    """Replace the complete ordinary RHS; shared uses are inserted separately."""

    name: Identifier
    terms: tuple[EquationTerm, ...] = Field(max_length=32)


class ProcessDeclaration(StrictSchema):
    """A named law, local or shared; use checks wait for the whole draft."""

    name: Identifier
    depends_on: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    kind: Literal["transfer", "influence"]
    uses: tuple[signed_processes.SignedUse, ...] = Field(min_length=1, max_length=64)
    scientific_meaning: str = Field(min_length=1, max_length=650)


class Draft(StrictSchema):
    """Scientific declarations, without inferred roles or executable functions."""

    variables: tuple[GeneratedVariable, ...] = ()
    equations: tuple[EquationUpdate, ...] = ()
    processes: tuple[ProcessDeclaration, ...] = ()
    mechanism_bindings: tuple[MechanismBinding, ...] = ()

    @model_validator(mode="after")
    def unique_keys(self):
        for field in ("variables", "equations", "processes", "mechanism_bindings"):
            key = "requirement_id" if field == "mechanism_bindings" else "name"
            names = [getattr(v, key) for v in getattr(self, field)]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate keys in {field}")
        return self


class DraftPatch(Draft):
    """Explicit replacement/removal operations; omitted declarations survive."""

    remove_variables: tuple[Identifier, ...] = ()
    remove_equations: tuple[Identifier, ...] = ()
    remove_processes: tuple[Identifier, ...] = ()
    remove_bindings: tuple[Identifier, ...] = ()
    stage_complete: bool


def normalize_reply(
    brief: PublicScientificBrief, raw: object
) -> tuple[object, list[dict]]:
    """Canonicalize only quoted JSON null conversions; retain auditable raw replies."""
    value = deepcopy(raw)
    changes = []
    # A legal public covariate named null would make this spelling ambiguous.
    if any(
        v.name == "null" and v.data_role == "covariate" for v in brief.public_variables
    ):
        return value, changes
    processes = value.get("processes") if isinstance(value, dict) else None
    if not isinstance(processes, list):
        return value, changes
    for i, process in enumerate(processes):
        uses = process.get("uses") if isinstance(process, dict) else None
        if not isinstance(uses, list):
            continue
        for j, use in enumerate(uses):
            if isinstance(use, dict) and use.get("conversion") == "null":
                use["conversion"] = None
                changes.append(
                    {
                        "code": "quoted_null_conversion",
                        "path": f"processes[{i}].uses[{j}].conversion",
                        "before": "null",
                        "after": None,
                    }
                )
    return value, changes


def process_usage(draft: Draft) -> list[dict]:
    """Count distinct consumers, independently of process naming or prose."""
    return [
        {
            "name": p.name,
            "kind": p.kind,
            "consumers": sorted({u.target for u in p.uses}),
            "consumer_count": len({u.target for u in p.uses}),
            "scope": "shared" if len({u.target for u in p.uses}) >= 2 else "local",
        }
        for p in draft.processes
    ]


def inventory(
    brief: PublicScientificBrief, draft: Draft
) -> tuple[ScientificVariable, ...]:
    """Generate compiler input without deciding which available sources get used."""
    values = {
        p.name: ScientificVariable(
            name=p.name,
            definition="supplied",
            scientific_role=f"Public {p.data_role}; optional RHS source.",
        )
        for p in brief.public_variables
        if p.data_role != "target"
    }
    values.update(
        (v.name, ScientificVariable.model_validate(v.model_dump()))
        for v in draft.variables
    )
    for p in draft.processes:
        if p.name in values and values[p.name].definition == "differential":
            raise ValueError(f"{p.name}: shared instantaneous law conflicts with state")
        values[p.name] = ScientificVariable(
            name=p.name, definition="algebraic", scientific_role=p.scientific_meaning
        )
    return tuple(values.values())


def apply_patch(brief: PublicScientificBrief, draft: Draft, patch: DraftPatch) -> Draft:
    """Apply an unambiguous batch atomically; keep scientific incompleteness pending."""
    updated = {}
    for field in ("variables", "equations", "processes", "mechanism_bindings"):
        key = "requirement_id" if field == "mechanism_bindings" else "name"
        removal = (
            "remove_bindings" if field == "mechanism_bindings" else f"remove_{field}"
        )
        values = {getattr(v, key): v for v in getattr(draft, field)}
        remove = getattr(patch, removal)
        replacements = {getattr(v, key): v for v in getattr(patch, field)}
        if len(remove) != len(set(remove)):
            raise ValueError(f"duplicate names in {removal}; include each name once")
        conflicts = sorted(set(remove) & replacements.keys())
        if conflicts:
            raise ValueError(
                f"ambiguous replace/remove operations in {field}: {conflicts}. "
                f"To keep or revise an entry, include it in {field} and omit it "
                f"from {removal}. To delete it, include its name only in {removal} "
                f"and omit its declaration from {field}. Coordinate any affected "
                "equation/reference edits. No part of this reply was committed."
            )
        if set(remove) - values.keys():
            raise ValueError(
                f"cannot remove unknown {field}: {sorted(set(remove) - values.keys())}"
            )
        for name in remove:
            del values[name]
        values.update(replacements)
        updated[field] = tuple(values.values())
    candidate = Draft(**updated)
    # Types and public availability are not inferred from names or explanatory prose.
    inv = inventory(brief, candidate)
    public_roles = {v.name: v.data_role for v in brief.public_variables}
    for v in inv:
        role = public_roles.get(v.name)
        if role in {"external_input", "covariate", "time"} and v.definition in {
            "differential",
            "algebraic",
        }:
            raise ValueError(
                f"{v.name} is an already available public {role}, not a generated "
                "LHS. Omit its declaration from this reply's variables/processes "
                "and omit its equation; keep scientifically intended RHS references. "
                "It requires no activation or type declaration. "
                "No part of this reply was committed."
            )
    merge_variable_reply(brief, (), VariableReply(variables=inv))
    if {e.name for e in candidate.equations} & {p.name for p in candidate.processes}:
        raise ValueError(
            "a shared process already has one runtime-generated definition"
        )
    for p in candidate.processes:
        if len(p.depends_on) != len(set(p.depends_on)):
            raise ValueError(f"{p.name}: duplicate process drivers")
        for use in p.uses:
            if use.conversion is not None:
                covariates = {
                    v.name: 1.0
                    for v in brief.public_variables
                    if v.data_role == "covariate"
                }
                try:
                    signed_processes.conversion_value(use.conversion, covariates)
                except ValueError as exc:
                    raise ValueError(
                        f"{p.name} -> {use.target}: invalid fixed conversion "
                        f"{use.conversion!r}: {exc}. Allowed covariates: "
                        f"{sorted(covariates)}. Use null if unknown; kinetic/fitted "
                        "coefficients belong to the later function stage."
                    ) from exc
    if candidate == draft and not patch.stage_complete:
        raise ValueError("no draft change; make an explicit edit or finish this stage")
    return candidate


def pending(brief: PublicScientificBrief, draft: Draft) -> dict:
    """List unresolved references and LHS work without demanding future equations."""
    public = {v.name for v in brief.public_variables}
    targets = [v.name for v in brief.public_variables if v.data_role == "target"]
    generated = {v.name for v in draft.variables} | {p.name for p in draft.processes}
    defined = {e.name for e in draft.equations} | {p.name for p in draft.processes}
    references = {s for e in draft.equations for t in e.terms for s in t.sources}
    references.update(s for p in draft.processes for s in p.depends_on)
    references.update(u.target for p in draft.processes for u in p.uses)
    references.update(s for b in draft.mechanism_bindings for s in b.memory_states)
    unknown = references - public - generated
    undeclared = (defined | set(targets)) - public - generated
    undeclared |= set(targets) - generated
    supplied_consumers = sorted(
        {u.target for p in draft.processes for u in p.uses} & (public - generated)
    )
    missing = (generated | unknown | set(targets)) - defined
    # Public targets first, then dependency order, finally unreferenced declarations.
    graph = {e.name: [s for t in e.terms for s in t.sources] for e in draft.equations}
    for p in draft.processes:
        graph[p.name] = list(p.depends_on)
        for u in p.uses:
            graph.setdefault(u.target, []).append(p.name)
    queue, visited, ordered = list(targets), set(), []
    while queue:
        name = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)
        if name in missing:
            ordered.append(name)
        queue.extend(graph.get(name, []))
    ordered.extend(sorted(missing - set(ordered)))
    return {
        "missing_declarations": sorted(unknown | undeclared),
        "equations_to_define": ordered,
        "supplied_process_consumers": supplied_consumers,
        "required_memory_bindings": sorted(
            {r.id for r in brief.requirements if r.requires_dynamic_memory}
            - {b.requirement_id for b in draft.mechanism_bindings}
        ),
    }


def assembled_equations(
    brief: PublicScientificBrief, draft: Draft
) -> tuple[EquationDefinition, ...]:
    """Insert one shared definition and its linked uses into the visible skeleton."""
    bindings = [signed_processes.binding_for(p) for p in draft.processes]
    definitions = {v.name: v.definition for v in inventory(brief, draft)}
    equations = [shared_process_contract.definition(b) for b in bindings]
    for e in draft.equations:
        if definitions.get(e.name) not in {"differential", "algebraic"}:
            raise ValueError(f"{e.name}: LHS needs a generated-variable declaration")
        try:
            terms = signed_processes.assemble(bindings, e.name, e.terms)
        except ValueError as exc:
            raise ValueError(f"{e.name}: {exc}") from exc
        if not terms:
            raise ValueError(
                f"{e.name}: empty assembled RHS. terms=[] is valid only when "
                "declared process uses supply the equation. For an intended "
                "constant law use a term with sources=[]; do not invent another "
                "variable to represent its fitted coefficient."
            )
        equations.append(
            EquationDefinition(name=e.name, definition=definitions[e.name], terms=terms)
        )
    return tuple(equations)


def snapshot(brief: PublicScientificBrief, draft: Draft) -> dict:
    """Render current truth, including unresolved work, on every initial/repair call."""
    try:
        equations = [
            e.model_dump(mode="json") for e in assembled_equations(brief, draft)
        ]
        assembly_error = None
    except ValueError as exc:
        equations, assembly_error = None, str(exc)
    return {
        "declarations": draft.model_dump(mode="json"),
        "assembled_equations": equations,
        "assembly_pending_reason": assembly_error,
        "pending": pending(brief, draft),
        "process_usage": process_usage(draft),
        "already_included_process_uses": [
            {
                "process": p.name,
                "drivers": list(p.depends_on),
                **u.model_dump(mode="json"),
            }
            for p in draft.processes
            for u in p.uses
        ],
    }


def assess(
    brief: PublicScientificBrief,
    context: ValidationContext,
    target_definitions: dict[str, str],
    draft: Draft,
) -> dict:
    """Check a finished draft; no reference equations, semantic LLM or fitted data."""
    errors: list[dict] = []

    def fail(code: str, **details) -> None:
        errors.append({"code": code, **details})

    work = pending(brief, draft)
    for code, names in work.items():
        if names:
            fail(code, names=names)
    inv = inventory(brief, draft)
    definitions = {v.name: v.definition for v in inv}
    for name, expected in target_definitions.items():
        if definitions.get(name) != expected:
            fail(
                "public_target_type",
                name=name,
                expected=expected,
                actual=definitions.get(name),
            )
    for p in draft.processes:
        try:
            signed_processes.SignedProcess.model_validate(p.model_dump())
        except ValueError as exc:
            fail("process_uses", process=p.name, reason=str(exc))
    equations, topology, aliases = (), None, {}
    try:
        equations = assembled_equations(brief, draft)
        topology, aliases = lower_topology(brief, inv, equations, context)
    except (ValueError, ModelValidationError) as exc:
        fail("compiler", reason=str(exc))
    # Failed compilation is not evidence of a missing scientific pathway.
    # A partial/unassembled graph must never generate a false path verdict.
    graph_available = topology is not None
    checks = public_structure_checks(brief, equations) if graph_available else ()
    errors.extend({"code": "public_path", **v} for v in checks if not v["passed"])
    for e in equations:
        audit = audit_explicit_equation_polarity(
            e,
            compile_equation_polarity_policy(
                brief, e.name, proposer_owns_unfixed_sources=True
            ),
        )
        if not audit["passed"]:
            fail("public_fixed_sign", equation=e.name, audit=audit)
    graph = {e.name: {s for t in e.terms for s in t.sources} for e in equations}
    requirements = {r.id: r for r in brief.requirements}
    memory_checks = []
    for binding in draft.mechanism_bindings:
        r = requirements.get(binding.requirement_id)
        if r is None:
            fail(
                "unknown_memory_requirement",
                requirement=binding.requirement_id,
                known_requirements=sorted(requirements),
            )
            continue
        prior_errors = len(errors)
        for state in binding.memory_states:
            if definitions.get(state) != "differential" or (
                r.requires_dynamic_memory and state in {*r.targets, *r.drivers}
            ):
                fail("memory_type", requirement=r.id, state=state)
            for driver in r.drivers if graph_available else ():
                if driver not in _ancestors(state, graph) and not (
                    not r.requires_dynamic_memory and driver == state
                ):
                    fail(
                        "memory_driver_path",
                        requirement=r.id,
                        driver=driver,
                        memory=state,
                    )
        for target in r.targets if graph_available else ():
            ancestors = _ancestors(target, graph)
            if not r.requires_dynamic_memory:
                ancestors = ancestors | {target}
            if not set(binding.memory_states) & ancestors:
                fail(
                    "memory_target_path",
                    requirement=r.id,
                    target=target,
                    memories=list(binding.memory_states),
                )
        memory_checks.append(
            {
                "requirement": r.id,
                "mandatory": r.requires_dynamic_memory,
                "states": list(binding.memory_states),
                "status": "failed"
                if len(errors) > prior_errors
                else "unavailable_graph"
                if not graph_available
                else "passed"
                if r.drivers and r.targets
                else "unresolved_public_endpoints",
                "scope": (
                    "Declared types and available graph endpoints, not memory decay."
                ),
            }
        )
    return {
        "eligible": not errors,
        "errors": errors,
        "graph_check_status": "assessed" if graph_available else "unavailable",
        "graph_check_reason": None
        if graph_available
        else "Compile the complete draft before assessing public or memory pathways.",
        "unresolved_public_predicates": [
            r.id for r in brief.requirements if not r.drivers or not r.targets
        ],
        "public_structure_checks": list(checks),
        "memory_binding_checks": memory_checks,
        "topology": topology.model_dump(mode="json") if topology else None,
        "generated_auxiliary_aliases": aliases,
        "equations": [e.model_dump(mode="json") for e in equations],
        "shared_process_bindings": [
            signed_processes.binding_for(p) for p in draft.processes
        ],
        "process_usage": process_usage(draft),
        "scientific_adequacy": "not_assessed",
        "scope": (
            "Declared structural predicates only; functions, initialization and "
            "fitted mechanisms remain unassessed."
        ),
    }
