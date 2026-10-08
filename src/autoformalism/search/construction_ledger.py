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
from autoformalism.search import (
    construction_handoff,
    shared_process_contract,
    signed_processes,
)
from autoformalism.search import public_graph_obligations as graph_obligations
from autoformalism.search.variable_bindings import MechanismBinding
from autoformalism.staged_topology import (
    _ancestors,
    audit_explicit_equation_polarity,
    compile_equation_polarity_policy,
    content_hash,
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


class FeedbackBinding(StrictSchema):
    """Proposer-declared dynamic realization of an algebraic public target."""

    target: Identifier
    states: tuple[Identifier, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def unique_states(self):
        if len(set(self.states)) != len(self.states):
            raise ValueError("feedback coordinates must be distinct")
        return self


class OverlapConfirmation(StrictSchema):
    """A scientific distinction, bound to the exact contributions being retained."""

    overlap_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_distinction: str = Field(min_length=1, max_length=1000)


_KEYS = {
    "variables": "name",
    "equations": "name",
    "processes": "name",
    "mechanism_bindings": "requirement_id",
    "feedback_bindings": "target",
    "overlap_confirmations": "overlap_id",
}


class Draft(StrictSchema):
    """Scientific declarations, without inferred roles or executable functions."""

    variables: tuple[GeneratedVariable, ...] = ()
    equations: tuple[EquationUpdate, ...] = ()
    processes: tuple[ProcessDeclaration, ...] = ()
    mechanism_bindings: tuple[MechanismBinding, ...] = ()
    feedback_bindings: tuple[FeedbackBinding, ...] = ()
    overlap_confirmations: tuple[OverlapConfirmation, ...] = ()

    @model_validator(mode="after")
    def unique_keys(self):
        for field, key in _KEYS.items():
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
    remove_feedback_bindings: tuple[Identifier, ...] = ()
    remove_overlap_confirmations: tuple[str, ...] = ()
    stage_complete: bool


def normalize_reply(
    brief: PublicScientificBrief,
    raw: object,
    *,
    graph_contract: graph_obligations.PublicGraphContract | None = None,
    draft: Draft | None = None,
    ignore_definition_description: bool = False,
) -> tuple[object, list[dict]]:
    """Normalize delivery/namespace mistakes, without assigning scientific roles."""
    value = deepcopy(raw)
    changes = []
    if draft is not None:
        value, changes = construction_handoff.normalize_definition_repeats(
            value, draft, ignore_description=ignore_definition_description
        )
    if graph_contract is not None and isinstance(value, dict):
        graph_ids = {r.id for r in graph_contract.obligations}
        real_ids = {r.id for r in brief.requirements}
        bindings = value.get("mechanism_bindings")
        if isinstance(bindings, list):
            kept = []
            for binding in bindings:
                name = (
                    binding.get("requirement_id") if isinstance(binding, dict) else None
                )
                if isinstance(name, str) and name in graph_ids - real_ids:
                    changes.append(
                        {
                            "code": "automatic_check_is_not_memory_requirement",
                            "removed_binding": binding,
                            "replacement_assignment": None,
                        }
                    )
                else:
                    kept.append(binding)
            value["mechanism_bindings"] = kept
        # A repair may add the right binding while omitting the stale wrong one.
        # Remove only exact automatic-check IDs; do not map them to requirements.
        removed = value.get("remove_bindings", [])
        if isinstance(removed, list) and draft is not None:
            for binding in draft.mechanism_bindings:
                name = binding.requirement_id
                if name in graph_ids - real_ids and name not in removed:
                    removed = [*removed, name]
                    changes.append(
                        {
                            "code": "remove_stale_automatic_check_binding",
                            "removed_binding": binding.model_dump(mode="json"),
                            "replacement_assignment": None,
                        }
                    )
            if removed != value.get("remove_bindings", []):
                value["remove_bindings"] = removed
        # Out-of-scope readout annotations are not scientific requirements.
        # Discard only exact public target names, never map them to memory IDs.
        targets = {v.name for v in brief.public_variables if v.data_role == "target"}
        applicable = {
            r.target for r in graph_contract.obligations if r.kind == "target_feedback"
        }
        bindings = value.get("feedback_bindings", [])
        removed = value.get("remove_feedback_bindings", [])
        if isinstance(bindings, list) and isinstance(removed, list):
            kept = []
            for binding in bindings:
                name = binding.get("target") if isinstance(binding, dict) else None
                if isinstance(name, str) and name in targets - applicable:
                    changes.append(
                        {
                            "code": "inapplicable_feedback_binding",
                            "removed_binding": binding,
                            "replacement_assignment": None,
                        }
                    )
                else:
                    kept.append(binding)
            value["feedback_bindings"] = kept
            for binding in draft.feedback_bindings if draft is not None else ():
                if (
                    binding.target in targets - applicable
                    and binding.target not in removed
                ):
                    removed = [*removed, binding.target]
                    changes.append(
                        {
                            "code": "remove_stale_inapplicable_feedback_binding",
                            "removed_binding": binding.model_dump(mode="json"),
                            "replacement_assignment": None,
                        }
                    )
            if removed != value.get("remove_feedback_bindings", []):
                value["remove_feedback_bindings"] = removed
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


def process_usage(
    draft: Draft, equations: tuple[EquationDefinition, ...] | None = None
) -> list[dict]:
    """Separate declared consumers from uses actually present in assembled RHSs."""
    assembled = {e.name: e for e in equations or ()}
    process_names = {p.name for p in draft.processes}
    rows = []
    for p in draft.processes:
        consumers = sorted({u.target for u in p.uses})
        uses = []
        for u in p.uses:
            equation = assembled.get(u.target)
            count = (
                sum(
                    t.sources == (p.name,) and t.outer_weight_sign.value == u.sign
                    for t in equation.terms
                )
                if equation and u.target not in process_names
                else 0
            )
            uses.append(
                {
                    **u.model_dump(mode="json"),
                    "assembled_count": count if equations is not None else None,
                }
            )
        verified = (
            equations is not None
            and len(consumers) == len(p.uses)
            and all(u["assembled_count"] == 1 for u in uses)
        )
        rows.append(
            {
                "name": p.name,
                "kind": p.kind,
                "consumers": consumers,
                "consumer_count": len(consumers),
                "count_basis": "declared distinct consumers",
                "assembled_consumers": sorted(
                    {u["target"] for u in uses if u["assembled_count"] == 1}
                )
                if equations is not None
                else None,
                "uses": uses,
                "assembly_verified": verified,
                "scope": ("shared" if len(consumers) >= 2 else "local")
                if verified
                else "unverified",
            }
        )
    return rows


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
    for field, key in _KEYS.items():
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
                f"cannot remove unknown {field}: "
                f"{sorted(set(remove) - values.keys())}. "
                f"{removal} uses {key} keys; available keys={sorted(values)}. "
                "Empty replacement lists preserve existing entries."
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
    references.update(s for b in draft.feedback_bindings for s in b.states)
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
    issues = construction_handoff.consumer_destination_issues(draft)
    if issues:
        raise ValueError("process uses cannot target runtime-generated definitions")
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
    result = tuple(equations)
    for usage in process_usage(draft, result):
        if not usage["assembly_verified"]:
            raise ValueError(
                f"{usage['name']}: every declared consumer must have one assembled "
                f"use with its declared sign; accounting={usage['uses']}"
            )
    return result


def contribution_overlaps(
    draft: Draft, brief: PublicScientificBrief | None = None
) -> list[dict]:
    """Identify ambiguous repeated source sets, never infer physical duplication."""
    confirmations = {c.overlap_id: c for c in draft.overlap_confirmations}
    covariates = (
        {v.name for v in brief.public_variables if v.data_role == "covariate"}
        if brief
        else set()
    )
    rows = []
    for equation in draft.equations:
        for index, term in enumerate(equation.terms):
            for process in draft.processes:
                for use in process.uses:
                    ordinary, drivers = set(term.sources), set(process.depends_on)
                    exact = ordinary == drivers
                    dynamic = ordinary - covariates
                    if use.target != equation.name or not (
                        exact or (dynamic and dynamic == drivers - covariates)
                    ):
                        continue
                    evidence = {
                        "target": equation.name,
                        "ordinary_term_index": index,
                        "ordinary_term": term.model_dump(mode="json"),
                        "process": process.model_dump(mode="json"),
                        "variable_declarations": [
                            v.model_dump(mode="json") for v in draft.variables
                        ],
                    }
                    if not exact:
                        evidence["fixed_covariate_comparison"] = {
                            "matching_non_covariate_drivers": sorted(dynamic),
                            "ordinary_covariates": sorted(ordinary & covariates),
                            "process_covariates": sorted(drivers & covariates),
                            "public_covariates": sorted(covariates),
                        }
                    identity = content_hash(evidence)
                    confirmation = confirmations.get(identity)
                    rows.append(
                        {
                            "overlap_id": identity,
                            **evidence,
                            "status": "proposer_confirmed_distinct"
                            if confirmation
                            else "clarification_required",
                            "scientific_distinction": (
                                confirmation.scientific_distinction
                                if confirmation
                                else None
                            ),
                            "question": (
                                "Do the ordinary term and the inserted process use "
                                "represent the same effect? If yes, replace the "
                                "ordinary terms to omit repetition, or remove the "
                                "process and coordinate its consumers. If different, "
                                "retain both via overlap_confirmations with this "
                                "overlap_id and a scientific_distinction. "
                                "Matching drivers alone do not prove duplication."
                                " Differences only in fixed covariates may express "
                                "a conversion or a distinct effect; choose explicitly."
                            ),
                        }
                    )
    return rows


def equation_views(draft: Draft) -> list[dict]:
    """Show automatic contributions even before an ordinary-RHS declaration."""
    ordinary = {e.name: e for e in draft.equations}
    bindings = [signed_processes.binding_for(p) for p in draft.processes]
    views = []
    for variable in draft.variables:
        if variable.name in {p.name for p in draft.processes}:
            continue
        equation = ordinary.get(variable.name)
        terms = equation.terms if equation else ()
        try:
            assembled = [
                t.model_dump(mode="json")
                for t in signed_processes.assemble(bindings, variable.name, terms)
            ]
            error = None
        except ValueError as exc:
            assembled, error = None, str(exc)
        views.append(
            {
                **variable.model_dump(mode="json"),
                "ordinary_rhs_declared": equation is not None,
                "ordinary_terms": [t.model_dump(mode="json") for t in terms],
                "assembled_terms_including_process_uses": assembled,
                "assembly_error": error,
                "next_action": None
                if equation
                else (
                    "Declare only additional ordinary contributions; use terms=[] if "
                    "the displayed process uses already supply the whole equation. "
                    "An absent ordinary declaration does not mean the RHS is empty."
                ),
            }
        )
    return views


def snapshot(brief: PublicScientificBrief, draft: Draft) -> dict:
    """Render current truth, including unresolved work, on every initial/repair call."""
    try:
        assembled = assembled_equations(brief, draft)
        equations = [e.model_dump(mode="json") for e in assembled]
        assembly_error = None
    except ValueError as exc:
        assembled, equations, assembly_error = None, None, str(exc)
    return {
        "declarations": draft.model_dump(mode="json"),
        "assembled_equations": equations,
        "assembly_pending_reason": assembly_error,
        "pending": pending(brief, draft),
        "process_usage": process_usage(draft, assembled),
        "equation_views": equation_views(draft),
        "potential_contribution_overlaps": contribution_overlaps(draft, brief),
        "declared_process_uses": [
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
    *,
    graph_contract: graph_obligations.PublicGraphContract | None = None,
    clarify_overlaps: bool = False,
) -> dict:
    """Check a finished draft; no reference equations, semantic LLM or fitted data."""
    errors: list[dict] = []
    if graph_contract is not None:
        graph_contract.validate_public(brief)

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
            fail(
                "process_uses",
                process=p.name,
                reason=str(exc),
                repair_options=[
                    "If this is truly an internal pairwise transfer, choose two "
                    "scientifically appropriate opposite signs.",
                    "If it is an influence rather than an internal transfer, "
                    "replace kind with influence and retain appropriate signs.",
                    "If no common law is intended, remove the process and "
                    "declare ordinary contributions in its consumers.",
                ],
                warning=(
                    "Do not flip a sign or invent a consumer solely "
                    "to satisfy the schema."
                ),
            )
    errors.extend(construction_handoff.process_reference_issues(draft))
    errors.extend(construction_handoff.consumer_destination_issues(draft))
    equations, topology, aliases = (), None, {}
    try:
        equations = assembled_equations(brief, draft)
        errors.extend(construction_handoff.algebraic_cycle_issues(equations))
        topology, aliases = lower_topology(brief, inv, equations, context)
    except (ValueError, ModelValidationError) as exc:
        fail("compiler", reason=str(exc))
    # Failed compilation is not evidence of a missing scientific pathway.
    # A partial/unassembled graph must never generate a false path verdict.
    graph_available = topology is not None
    checks = public_structure_checks(brief, equations) if graph_available else ()
    errors.extend({"code": "public_path", **v} for v in checks if not v["passed"])
    feedback_targets = (
        {r.target for r in graph_contract.obligations if r.kind == "target_feedback"}
        if graph_contract is not None
        else set()
    )
    for binding in draft.feedback_bindings:
        if binding.target not in feedback_targets:
            fail(
                "unknown_feedback_target",
                target=binding.target,
                allowed_targets=sorted(feedback_targets),
                repair={"remove_feedback_bindings": [binding.target]},
                explanation=(
                    "This field applies only to listed target-feedback "
                    "checks, not memory assignments."
                ),
            )
        if definitions.get(binding.target) == "differential" and binding.states != (
            binding.target,
        ):
            fail(
                "differential_target_is_own_coordinate",
                target=binding.target,
                repair={"remove_feedback_bindings": [binding.target]},
                explanation=(
                    "Remove this annotation; the differential target "
                    "itself must have feedback."
                ),
            )
    reviewed_checks = (
        graph_obligations.check(
            graph_contract,
            equations if graph_available else None,
            feedback_coordinates={b.target: b.states for b in draft.feedback_bindings},
        )
        if graph_contract is not None
        else []
    )
    errors.extend(
        {"code": "reviewed_public_graph", **v}
        for v in reviewed_checks
        if v["passed"] is False
    )
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
                repair={"remove_bindings": [binding.requirement_id]},
                explanation=(
                    "This ID is not a public mechanism requirement. "
                    "Removing it does not assign a replacement memory state."
                ),
            )
            continue
        prior_errors = len(errors)
        binding_feedback = {
            "field": "mechanism_bindings",
            "entry": binding.model_dump(mode="json"),
            "mandatory": r.requires_dynamic_memory,
            "repair_options": [
                "Replace this mechanism_bindings entry with the scientifically "
                "intended state(s); feedback_bindings is a different field.",
                "If these states are intended, revise the necessary equation "
                "dependencies to establish the displayed driver/memory/target paths.",
            ]
            + (
                []
                if r.requires_dynamic_memory
                else [
                    f"If this optional assignment is unintended, remove it with "
                    f"remove_bindings=[{r.id!r}]. Empty lists preserve old entries."
                ]
            ),
        }
        for state in binding.memory_states:
            if definitions.get(state) != "differential" or (
                r.requires_dynamic_memory and state in {*r.targets, *r.drivers}
            ):
                fail(
                    "memory_type",
                    requirement=r.id,
                    state=state,
                    binding_feedback=binding_feedback,
                )
            for driver in r.drivers if graph_available else ():
                if driver not in _ancestors(state, graph) and not (
                    not r.requires_dynamic_memory and driver == state
                ):
                    fail(
                        "memory_driver_path",
                        requirement=r.id,
                        driver=driver,
                        memory=state,
                        binding_feedback=binding_feedback,
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
                    target_ancestors=sorted(_ancestors(target, graph)),
                    binding_feedback=binding_feedback,
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
    overlaps = contribution_overlaps(draft, brief)
    clarifications = (
        [r for r in overlaps if r["status"] == "clarification_required"]
        if clarify_overlaps
        else []
    )
    return {
        "eligible": not errors and not clarifications,
        "errors": errors,
        "clarification_requests": clarifications,
        "contribution_overlaps": overlaps,
        "graph_check_status": "assessed" if graph_available else "unavailable",
        "graph_check_reason": None
        if graph_available
        else "Compile the complete draft before assessing public or memory pathways.",
        "unresolved_public_predicates": [
            r.id for r in brief.requirements if not r.drivers or not r.targets
        ],
        "public_structure_checks": list(checks),
        "reviewed_public_graph_checks": reviewed_checks,
        "deferred_scientific_checks": list(graph_contract.deferred_scientific_checks)
        if graph_contract is not None
        else [],
        "memory_binding_checks": memory_checks,
        "topology": topology.model_dump(mode="json") if topology else None,
        "generated_auxiliary_aliases": aliases,
        "equations": [e.model_dump(mode="json") for e in equations],
        "shared_process_bindings": [
            signed_processes.binding_for(p) for p in draft.processes
        ],
        "process_usage": process_usage(draft, equations or None),
        "scientific_adequacy": "not_assessed",
        "scope": (
            "Declared structural predicates only; functions, initialization and "
            "fitted mechanisms remain unassessed."
        ),
    }
