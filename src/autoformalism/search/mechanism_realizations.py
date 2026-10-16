"""Opt-in mechanism declarations and structural checks, separate from behavior.

Historical construction schemas remain unchanged. A realization cites generated
quantities (their whole local RHS), named processes, or supplied public sources;
it never declares that a scientific mechanism has been verified.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy

from pydantic import Field, create_model, model_validator

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_checklist as checklist
from autoformalism.search import construction_handoff as handoff
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import signed_processes
from autoformalism.search.public_graph_obligations import PublicGraphContract
from autoformalism.search.variable_bindings import MechanismBinding
from autoformalism.staged_topology import _ancestors, content_hash

POLICY = "minimal-mechanism-realizations-1"
_COMMON = (
    "variables",
    "equations",
    "processes",
    "feedback_bindings",
    "overlap_confirmations",
)
_KEYS = {
    "variables": "name",
    "equations": "name",
    "processes": "name",
    "mechanism_realizations": "requirement_id",
    "feedback_bindings": "target",
    "overlap_confirmations": "overlap_id",
}
SCOPE = (
    "Declaration references and explicit structural predicates only. Functions, "
    "sign-dependent behavior, initialization, conservation, units, positivity, "
    "stability and fitted behavior remain unverified."
)


class MechanismRealization(StrictSchema):
    """One proposer-selected structural representation of a public mechanism."""

    requirement_id: Identifier
    components: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    memory_states: tuple[Identifier, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def consistent_references(self) -> MechanismRealization:
        """Reject ambiguous lists and memory annotations outside this record."""
        for field in ("components", "memory_states"):
            values = getattr(self, field)
            if len(values) != len(set(values)):
                raise ValueError(f"{field} must be unique")
        if set(self.memory_states) - set(self.components):
            raise ValueError("memory_states must be a subset of components")
        return self


class Draft(StrictSchema):
    """A new serialized contract with no historical memory-only field."""

    variables: tuple[ledger.GeneratedVariable, ...] = ()
    equations: tuple[ledger.EquationUpdate, ...] = ()
    processes: tuple[ledger.ProcessDeclaration, ...] = ()
    mechanism_realizations: tuple[MechanismRealization, ...] = ()
    feedback_bindings: tuple[ledger.FeedbackBinding, ...] = ()
    overlap_confirmations: tuple[ledger.OverlapConfirmation, ...] = ()

    @model_validator(mode="after")
    def unique_keys(self) -> Draft:
        """Allow at most one replacement for each editable identity."""
        for field, key in _KEYS.items():
            names = [getattr(value, key) for value in getattr(self, field)]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate keys in {field}")
        return self

    @property
    def mechanism_bindings(self) -> tuple[MechanismBinding, ...]:
        """Project selected memory into the unchanged exact-path checker."""
        return tuple(
            MechanismBinding(
                requirement_id=record.requirement_id,
                memory_states=record.memory_states,
            )
            for record in self.mechanism_realizations
            if record.memory_states
        )


class DraftPatch(Draft):
    """Atomic named replacements/removals for this opt-in contract only."""

    remove_variables: tuple[Identifier, ...] = ()
    remove_equations: tuple[Identifier, ...] = ()
    remove_processes: tuple[Identifier, ...] = ()
    remove_realizations: tuple[Identifier, ...] = ()
    remove_feedback_bindings: tuple[Identifier, ...] = ()
    remove_overlap_confirmations: tuple[str, ...] = ()
    stage_complete: bool

    @property
    def remove_bindings(self) -> tuple[str, ...]:
        """Support legacy scope inspection, without adding a serialized field."""
        return self.remove_realizations


def project(draft: Draft) -> ledger.Draft:
    """Build a disposable legacy view without changing either serialized schema."""
    return ledger.Draft(
        **{field: getattr(draft, field) for field in _COMMON},
        mechanism_bindings=draft.mechanism_bindings,
    )


def load_draft(raw: dict) -> Draft | ledger.Draft:
    """Read the declared artifact version; never silently migrate a saved draft."""
    if "external_exchanges" in raw:
        from autoformalism.search.construction_evidence import Draft as EvidenceDraft

        return EvidenceDraft.model_validate(raw)
    if "mechanism_realizations" in raw:
        if "mechanism_bindings" in raw:
            raise ValueError("mixed historical and mechanism realization contracts")
        return Draft.model_validate(raw)
    return ledger.Draft.model_validate(raw)


def from_legacy(draft: ledger.Draft) -> Draft:
    """Project only existing explicit memory selections for an offline preview.

    This is not a completed realization inventory or an on-disk migration. The
    preview caller records that provenance; uncited public mechanisms stay missing.
    """
    return Draft(
        **{field: getattr(draft, field) for field in _COMMON},
        mechanism_realizations=tuple(
            MechanismRealization(
                requirement_id=binding.requirement_id,
                components=binding.memory_states,
                memory_states=binding.memory_states,
            )
            for binding in draft.mechanism_bindings
        ),
    )


def edit_effects(before: Draft, after: Draft) -> dict:
    """Record full realization edits instead of only their memory projection."""
    result = handoff.edit_effects(project(before), project(after))
    old = {
        r.requirement_id: r.model_dump(mode="json")
        for r in before.mechanism_realizations
    }
    new = {
        r.requirement_id: r.model_dump(mode="json")
        for r in after.mechanism_realizations
    }
    result.pop("mechanism_bindings", None)
    result["mechanism_realizations"] = [
        {"key": name, "before": old.get(name), "after": new.get(name)}
        for name in sorted(old.keys() | new.keys())
        if old.get(name) != new.get(name)
    ]
    return result


def normalize_consumers(
    draft: Draft, patch: DraftPatch, previous: Draft
) -> tuple[Draft, list[dict]]:
    """Preserve the new records across existing unambiguous process normalization."""
    legacy_patch = ledger.DraftPatch(
        **{field: getattr(patch, field) for field in _COMMON},
        **{f"remove_{field}": getattr(patch, f"remove_{field}") for field in _COMMON},
        stage_complete=patch.stage_complete,
    )
    updated, changes = handoff.normalize_consumers(
        project(draft), legacy_patch, project(previous)
    )
    return (
        Draft(
            **{field: getattr(updated, field) for field in _COMMON},
            mechanism_realizations=draft.mechanism_realizations,
        ),
        changes,
    )


def response_model(stage: str) -> type[StrictSchema]:
    """Expose stage-specific edit fields with strict realization records."""
    if stage not in {"variables", "shared_laws", "equations", "repair"}:
        raise ValueError(f"unknown construction stage: {stage}")
    fields = set(DraftPatch.model_fields)
    if stage in {"variables", "shared_laws"}:
        fields -= {
            "equations",
            "remove_equations",
            "overlap_confirmations",
            "remove_overlap_confirmations",
        }
    if stage == "variables":
        fields -= {"processes", "remove_processes"}
    return create_model(
        "Realization" + stage.title().replace("_", "") + "Reply",
        __base__=StrictSchema,
        **{
            name: (field.annotation, deepcopy(field))
            for name, field in DraftPatch.model_fields.items()
            if name in fields
        },
    )


def normalize_reply(
    brief: PublicScientificBrief,
    raw: object,
    *,
    graph_contract: PublicGraphContract | None = None,
    draft: Draft | None = None,
    ignore_definition_description: bool = False,
) -> tuple[object, list[dict]]:
    """Keep existing mechanical normalization without assigning realizations.

    Legacy edit fields are deliberately left intact for strict schema rejection;
    automatic graph-check IDs are not silently treated as public mechanisms.
    """
    value = deepcopy(raw)
    realization_fields = {}
    if isinstance(value, dict):
        realization_fields = {
            field: value.pop(field)
            for field in ("mechanism_realizations", "remove_realizations")
            if field in value
        }
    value, changes = ledger.normalize_reply(
        brief,
        value,
        graph_contract=graph_contract,
        draft=project(draft) if draft is not None else None,
        ignore_definition_description=ignore_definition_description,
    )
    # The legacy normalizer may synthesize stale memory removals. Realization
    # ownership belongs to the proposer, so this adapter never synthesizes them.
    if isinstance(value, dict) and isinstance(raw, dict):
        value.update(realization_fields)
        for field in ("mechanism_bindings", "remove_bindings"):
            if field in raw:
                value[field] = deepcopy(raw[field])
            else:
                value.pop(field, None)
    changes = [
        change
        for change in changes
        if change["code"]
        not in {
            "automatic_check_is_not_memory_requirement",
            "remove_stale_automatic_check_binding",
        }
    ]
    return value, changes


def apply_patch(brief: PublicScientificBrief, draft: Draft, patch: DraftPatch) -> Draft:
    """Apply one atomic transaction; unresolved scientific references stay pending."""
    controls = {name for name in DraftPatch.model_fields if name.startswith("remove_")}
    controls.add("stage_complete")
    conflicts = [
        f"{field}[{entry.name}]"
        for field in ("variables", "equations", "processes")
        for entry in getattr(patch, field)
        if entry.name in controls
    ]
    if conflicts:
        public = {v.name for v in brief.public_variables}
        collisions = sorted(
            entry.name
            for field in ("variables", "equations", "processes")
            for entry in getattr(patch, field)
            if entry.name in controls & public
        )
        if collisions:
            raise ValueError(
                "public scientific names collide with reserved reply controls: "
                f"{collisions}. "
                "Do not automatically rename or delete public quantities. "
                "No part of this reply was committed."
            )
        raise ValueError(
            f"reply-control fields are not scientific declarations: {conflicts}. "
            "Keep stage_complete and remove_* at the reply's top level. A rejected "
            "declaration was never installed; omit it instead of removing it. "
            "No part of this reply was committed."
        )
    records = {r.requirement_id: r for r in draft.mechanism_realizations}
    replacements = {r.requirement_id: r for r in patch.mechanism_realizations}
    removed = patch.remove_realizations
    if len(set(removed)) != len(removed):
        raise ValueError("duplicate names in remove_realizations")
    conflicts = sorted(set(removed) & replacements.keys())
    if conflicts:
        raise ValueError(
            "ambiguous replace/remove operations in mechanism_realizations: "
            f"{conflicts}. "
            "To revise an entry, omit it from remove_realizations. "
            "To delete it, omit its replacement. No part of this reply was committed."
        )
    unknown = sorted(set(removed) - records.keys())
    if unknown:
        raise ValueError(
            f"cannot remove unknown mechanism_realizations: {unknown}. "
            "remove_realizations uses requirement_id keys; "
            f"available keys={sorted(records)}. Empty lists preserve entries."
        )
    for name in removed:
        del records[name]
    records.update(replacements)
    old_projection = project(draft)
    selected = tuple(
        MechanismBinding(requirement_id=r.requirement_id, memory_states=r.memory_states)
        for r in patch.mechanism_realizations
        if r.memory_states
    )
    old_memory = {r.requirement_id for r in old_projection.mechanism_bindings}
    deleted_memory = old_memory & (
        set(removed) | {name for name, r in replacements.items() if not r.memory_states}
    )
    legacy_patch = ledger.DraftPatch(
        **{field: getattr(patch, field) for field in _COMMON},
        **{f"remove_{field}": getattr(patch, f"remove_{field}") for field in _COMMON},
        mechanism_bindings=selected,
        remove_bindings=tuple(sorted(deleted_memory)),
        # Realization-only changes are invisible to this projection; the actual
        # no-op check is performed below on the full new contract.
        stage_complete=True,
    )
    updated = ledger.apply_patch(brief, old_projection, legacy_patch)
    candidate = Draft(
        **{field: getattr(updated, field) for field in _COMMON},
        mechanism_realizations=tuple(records.values()),
    )
    if candidate == draft and not patch.stage_complete:
        raise ValueError("no draft change; make an explicit edit or finish this stage")
    return candidate


def pending(brief: PublicScientificBrief, draft: Draft) -> dict:
    """Combine existing outstanding work with incomplete realization declarations."""
    work = ledger.pending(brief, project(draft))
    known = (
        {v.name for v in brief.public_variables}
        | {v.name for v in draft.variables}
        | {p.name for p in draft.processes}
    )
    references = {name for r in draft.mechanism_realizations for name in r.components}
    work["missing_declarations"] = sorted(
        set(work["missing_declarations"]) | (references - known)
    )
    work["required_mechanism_realizations"] = sorted(
        {r.id for r in brief.requirements}
        - {r.requirement_id for r in draft.mechanism_realizations}
    )
    work["required_memory_annotations"] = work.pop("required_memory_bindings")
    return work


def _declaration_rows(brief: PublicScientificBrief, draft: Draft) -> list[dict]:
    """Infer declaration facts from typed references, never scientific role prose."""
    definitions = {
        v.name: "supplied" for v in brief.public_variables if v.data_role != "target"
    }
    definitions.update((v.name, v.definition) for v in draft.variables)
    definitions.update((p.name, "algebraic") for p in draft.processes)
    requirements = {r.id: r for r in brief.requirements}
    records = {r.requirement_id: r for r in draft.mechanism_realizations}
    rows = []
    for name in sorted(requirements.keys() | records.keys()):
        requirement = requirements.get(name)
        record = records.get(name)
        components = record.components if record else ()
        states = record.memory_states if record else ()
        unknown = sorted(set(components) - definitions.keys())
        bad_states = [s for s in states if definitions.get(s) != "differential"]
        mandatory = bool(requirement and requirement.requires_dynamic_memory)
        excluded = (
            set(requirement.drivers + requirement.targets) if mandatory else set()
        )
        invalid_mediators = sorted(excluded & set(states))
        types = {definitions.get(c) for c in components} - {"supplied"}
        inferred = (
            "unresolved"
            if unknown or not types
            else "mixed"
            if types == {"algebraic", "differential"}
            else next(iter(types))
        )
        issues = []
        if requirement is None:
            issues.append("unknown_requirement")
        if unknown:
            issues.append("unknown_components")
        if bad_states:
            issues.append("memory_not_differential")
        if invalid_mediators:
            issues.append("mandatory_memory_endpoint")
        if mandatory and not states:
            issues.append("required_memory_annotation_missing")
        status = (
            "missing" if record is None else "inconsistent" if issues else "satisfied"
        )
        rows.append(
            {
                "requirement": name,
                "components": list(components),
                "memory_states": list(states),
                "component_definitions": {c: definitions.get(c) for c in components},
                "inferred_type": inferred,
                "declaration_status": status,
                "structural_status": "deferred",
                "behavior_status": "unverified",
                "requires_dynamic_memory": mandatory,
                "issues": issues,
                "unknown_components": unknown,
                "nondifferential_memory_states": bad_states,
                "excluded_mediator_endpoints": sorted(excluded),
                "invalid_mediators": invalid_mediators,
                "drivers": list(requirement.drivers) if requirement else [],
                "targets": list(requirement.targets) if requirement else [],
                "component_granularity": "whole generated RHS or process",
                "scope": SCOPE,
            }
        )
    return rows


def variable_checklist(
    brief: PublicScientificBrief,
    draft: Draft,
    target_definitions: dict[str, str],
    graph_contract: PublicGraphContract | None = None,
) -> dict:
    """Require each mechanism's declaration, while deferring topology and behavior."""
    base = checklist.variable_checklist(
        brief, project(draft), target_definitions, graph_contract
    )
    items = [
        item
        for item in base["items"]
        if item["code"] not in {"memory_assignment", "mechanism_topology"}
    ]
    rows = _declaration_rows(brief, draft)
    items.extend(
        {
            "id": f"mechanism_realization:{row['requirement']}",
            "code": "mechanism_realization",
            "name": row["requirement"],
            "status": row["declaration_status"],
            **row,
        }
        for row in rows
    )
    blocked = [
        item["id"] for item in items if item["status"] in {"missing", "inconsistent"}
    ]
    return {
        **base,
        "policy": POLICY,
        "items": items,
        "blocking_items": blocked,
        "status": "incomplete" if blocked else "ready",
        "mechanism_realization_checks": rows,
        "scope": "Declaration consistency only; no topology is required yet. " + SCOPE,
    }


def _component_relevance_graph(
    brief: PublicScientificBrief, draft: Draft, graph: dict[str, set[str]]
) -> dict[str, set[str]]:
    """Include validated fixed conversions for reference relevance only.

    These fixed factors are outside the historical topology dependency graph.
    They must not alter public-path or selected-memory verdicts.
    """
    result = {name: set(sources) for name, sources in graph.items()}
    covariates = {
        variable.name: 1.0
        for variable in brief.public_variables
        if variable.data_role == "covariate"
    }
    for process in draft.processes:
        for use in process.uses:
            if use.conversion is None or process.name not in graph.get(
                use.target, set()
            ):
                continue
            try:
                signed_processes.conversion_value(use.conversion, covariates)
                symbols = (
                    RestrictedParser()
                    .parse(use.conversion, location="realization fixed conversion")
                    .symbols
                )
            except (ValueError, ModelValidationError):
                continue  # Invalid expressions never supply reference evidence.
            result[use.target].update(symbols)
    return result


def _realization_paths(
    row: dict,
    graph: dict[str, set[str]],
    relevance_graph: dict[str, set[str]],
    public_checks: list[dict],
) -> tuple[list, list]:
    """Check public paths and component relevance without requiring one branch.

    Supporting covariates can enter a sibling branch. Listing endpoints alone is
    not a path witness; the assembled graph must actually connect the endpoints.
    """
    witnesses = []
    participants = {
        component
        for component in row["components"]
        if any(
            component == target or component in _ancestors(target, relevance_graph)
            for target in row["targets"]
        )
    }
    for check in public_checks:
        if check["requirement"] != row["requirement"] or check["kind"] != "driver_path":
            continue
        driver, target = check["driver"], check["target"]
        via = [
            component
            for component in row["components"]
            if check["passed"]
            and (component == driver or driver in _ancestors(component, graph))
            and (component == target or component in _ancestors(target, graph))
        ]
        witnesses.append(
            {
                "driver": driver,
                "target": target,
                "components_on_path": via,
                "passed": check["passed"],
            }
        )
    return witnesses, sorted(set(row["components"]) - participants)


def _realization_feedback(value: object, records: dict[str, dict]) -> object:
    """Translate legacy repair destinations without changing any graph verdict."""
    if isinstance(value, list):
        return [_realization_feedback(item, records) for item in value]
    if not isinstance(value, dict):
        return value
    translated = {}
    for key, item in value.items():
        updated_key = {
            "remove_bindings": "remove_realizations",
            "required_memory_bindings": "required_memory_annotations",
            "binding_feedback": "realization_feedback",
        }.get(key, key)
        translated[updated_key] = _realization_feedback(item, records)
    if translated.get("code") == "required_memory_bindings":
        translated["code"] = "required_memory_annotations"
    if translated.get("field") == "mechanism_bindings":
        translated["field"] = "mechanism_realizations"
        requirement = translated.get("entry", {}).get("requirement_id")
        if requirement in records:
            translated["entry"] = records[requirement]
        translated["repair_options"] = [
            "Replace the complete mechanism_realizations entry, preserving its "
            "intended components; memory_states must be differential components.",
            "Revise equation dependencies to establish the displayed exact "
            "driver/memory/target paths when scientifically intended.",
        ] + (
            []
            if translated.get("mandatory")
            else [
                "If this memory annotation is unintended, replace the complete "
                "realization with memory_states=[] and retain its intended components."
            ]
        )
    return translated


def assess(
    brief: PublicScientificBrief,
    context: ValidationContext,
    target_definitions: dict[str, str],
    draft: Draft,
    *,
    graph_contract: PublicGraphContract | None = None,
    clarify_overlaps: bool = False,
) -> dict:
    """Preserve old graph checks, then check realization references and pathways."""
    result = ledger.assess(
        brief,
        context,
        target_definitions,
        project(draft),
        graph_contract=graph_contract,
        clarify_overlaps=clarify_overlaps,
    )
    rows = _declaration_rows(brief, draft)
    graph = {
        equation["name"]: {
            source for term in equation["terms"] for source in term["sources"]
        }
        for equation in result["equations"]
    }
    relevance_graph = _component_relevance_graph(brief, draft, graph)
    memories = {row["requirement"]: row for row in result["memory_binding_checks"]}
    for row in rows:
        if row["declaration_status"] != "satisfied":
            row["structural_status"] = "deferred"
            result["errors"].append(
                {"code": "mechanism_realization_declaration", **row}
            )
        elif result["graph_check_status"] != "assessed":
            row["structural_status"] = "unavailable_graph"
        elif not row["drivers"] or not row["targets"]:
            row["structural_status"] = (
                "failed"
                if memories.get(row["requirement"], {}).get("status") == "failed"
                else "unresolved_public_endpoints"
            )
        else:
            paths, disconnected = _realization_paths(
                row, graph, relevance_graph, result["public_structure_checks"]
            )
            row["path_checks"] = paths
            row["public_path_check_status"] = "assessed" if paths else "not_requested"
            row["disconnected_components"] = disconnected
            failed = (
                bool(disconnected)
                or any(not path["passed"] for path in paths)
                or memories.get(row["requirement"], {}).get("status") == "failed"
                or any(
                    check["requirement"] == row["requirement"] and not check["passed"]
                    for check in result["public_structure_checks"]
                )
            )
            row["structural_status"] = (
                "failed"
                if failed
                else "passed"
                if paths
                or memories.get(row["requirement"], {}).get("status") == "passed"
                else "deferred"
            )
            if failed:
                result["errors"].append({"code": "mechanism_realization_path", **row})
    records = {
        r.requirement_id: r.model_dump(mode="json")
        for r in draft.mechanism_realizations
    }
    result["errors"] = _realization_feedback(result["errors"], records)
    result["mechanism_realization_checks"] = rows
    result["eligible"] = result["eligible"] and not result["errors"]
    result["behavior_status"] = "unverified"
    result["scope"] = SCOPE
    return result


def _memory_target_witness(
    graph: dict[str, set[str]],
    memory: str,
    target: str,
    excluded_intermediate: frozenset[str] = frozenset(),
) -> list[str] | None:
    """Find a deterministic path without excluding a legitimate final target.

    Exclusions apply only to intermediate nodes. In particular, a requirement
    for delayed self-feedback y -> memory -> y may end at its own public driver.
    """
    queue = deque([(memory, [memory])])
    visited = {memory}
    while queue:
        node, path = queue.popleft()
        for destination in sorted(graph.get(node, ())):
            if destination == target:
                return [*path, destination]
            if destination not in visited and destination not in excluded_intermediate:
                visited.add(destination)
                queue.append((destination, [*path, destination]))
    return None


def qualify_required_memory_paths(
    brief: PublicScientificBrief, draft: Draft, assessment: dict
) -> dict:
    """Opt in to history-to-effect qualification beyond current public drivers.

    Historical assessment is unchanged unless a caller explicitly invokes this
    helper. A required history mechanism must have, for each target, a route from
    at least one selected memory that does not merely return through a current
    driver before reaching that target. This is a qualified structural reading of
    an explicit memory requirement, not a universal restriction on feedback or a
    proof of function-level delayed behavior. Existing driver-to-memory checks and
    the collective selected-memory quantifier remain unchanged.
    """
    result = deepcopy(assessment)
    outgoing: dict[str, set[str]] = {}
    for equation in assessment["equations"]:
        for term in equation["terms"]:
            for source in term["sources"]:
                outgoing.setdefault(source, set()).add(equation["name"])
    records = {r.requirement_id: r for r in draft.mechanism_realizations}
    definitions = {v.name: v.definition for v in draft.variables}
    rows = []
    scope = (
        "Explicit required-memory realization only. The assigned history must "
        "reach the required effect beyond merely regenerating its current public "
        "drivers. Driver nodes are excluded only as intermediate nodes after "
        "memory; target endpoints, other feedback cycles and parallel direct "
        "effects remain allowed. Function-level delay and scientific correctness "
        "remain unverified."
    )
    for requirement in brief.requirements:
        if not requirement.requires_dynamic_memory:
            continue
        record = records.get(requirement.id)
        states = record.memory_states if record is not None else ()
        drivers = frozenset(requirement.drivers)
        row = {
            "requirement": requirement.id,
            "public_requirement": requirement.public_requirement,
            "drivers": list(requirement.drivers),
            "targets": list(requirement.targets),
            "selected_memory_states": list(states),
            "target_checks": [],
            "scope": scope,
        }
        if assessment["graph_check_status"] != "assessed":
            row["status"] = "unavailable_graph"
        elif not requirement.drivers or not requirement.targets:
            row["status"] = "unresolved_public_endpoints"
        elif not states or any(
            definitions.get(state) != "differential"
            or state in drivers
            or state in requirement.targets
            for state in states
        ):
            row["status"] = "unavailable_memory_declaration"
        else:
            for target in requirement.targets:
                witnesses = []
                for state in states:
                    ordinary = _memory_target_witness(outgoing, state, target)
                    qualified = _memory_target_witness(outgoing, state, target, drivers)
                    witnesses.append(
                        {
                            "memory": state,
                            "ordinary_path": ordinary,
                            "qualified_path": qualified,
                            "intermediate_current_drivers": [
                                node
                                for node in (ordinary or [])[1:-1]
                                if node in drivers
                            ],
                        }
                    )
                target_row = {
                    "target": target,
                    "passed": any(w["qualified_path"] for w in witnesses),
                    "memory_witnesses": witnesses,
                    "quantifier": "at least one selected memory per target",
                }
                row["target_checks"].append(target_row)
                if target_row["passed"] or not any(
                    witness["ordinary_path"] for witness in witnesses
                ):
                    continue  # Ordinary missing paths already have exact feedback.
                result["errors"].append(
                    {
                        "code": "required_memory_path_revisits_driver",
                        "requirement": requirement.id,
                        "public_requirement": requirement.public_requirement,
                        "target": target,
                        "drivers": list(requirement.drivers),
                        "memories": list(states),
                        "memory_witnesses": witnesses,
                        "excluded_intermediate_current_drivers": sorted(drivers),
                        "field_to_edit": "equations/processes/mechanism_realizations",
                        "action": (
                            "The displayed memory-to-target routes return through "
                            "current public driver quantities. Such feedback may "
                            "change those drivers, but does not identify a separate "
                            "history effect beyond their current values. For this "
                            "explicit required-memory mechanism, choose an intended "
                            "route from a selected memory to the target that does "
                            "not revisit a current driver before the target."
                        ),
                        "repair_options": [
                            "If the selected memory is intended, revise the "
                            "appropriate local/shared process dependencies so its "
                            "history affects the required target through the "
                            "scientifically intended mechanism. No particular "
                            "direct edge, sign or textbook equation is prescribed.",
                            "If another generated differential component carries "
                            "the required history, replace the complete realization "
                            "record with that intended assignment and coordinate "
                            "its actual topology. Keep other intended components.",
                            "Preserve valid driver generation and feedback. Do not "
                            "delete the driver, withdraw mandatory memory, or merely "
                            "rename a process to make the path appear different.",
                        ],
                        "scope": scope,
                    }
                )
            row["status"] = (
                "passed" if all(t["passed"] for t in row["target_checks"]) else "failed"
            )
        rows.append(row)
    failed = any(row["status"] == "failed" for row in rows)
    result["required_memory_path_qualification"] = {
        "policy": "current-driver-conditioned-memory-path-1",
        "status": "failed"
        if failed
        else "passed"
        if rows and all(row["status"] == "passed" for row in rows)
        else "unavailable"
        if rows
        else "not_applicable",
        "checks": rows,
        "scope": scope,
    }
    by_requirement = {row["requirement"]: row["status"] for row in rows}
    for row in result["mechanism_realization_checks"]:
        if row["requirement"] in by_requirement:
            row["required_memory_path_qualification_status"] = by_requirement[
                row["requirement"]
            ]
    result["eligible"] = result["eligible"] and not failed
    return result


def snapshot(brief: PublicScientificBrief, draft: Draft) -> dict:
    """Show the actual opt-in ledger while reusing ordinary equation views."""
    return {
        **ledger.snapshot(brief, project(draft)),
        "declarations": draft.model_dump(mode="json"),
        "pending": pending(brief, draft),
        "mechanism_realization_checks": _declaration_rows(brief, draft),
    }


def variable_snapshot(draft: Draft) -> dict:
    """Show variable declarations without premature topology requirements."""
    return {
        "declarations": draft.model_dump(mode="json"),
        "editing_contract": "Return changes only; omitted entries remain. Use "
        "remove_realizations with requirement_id to delete a realization. "
        "To withdraw optional memory, replace the record with memory_states=[]. "
        "No topology is required yet.",
    }


def final_handoff(draft: Draft, assessment: dict) -> dict:
    """Bind the realization map to its exact compiled structural artifact."""
    return {
        "policy": POLICY,
        "mechanism_realizations": [
            record.model_dump(mode="json") for record in draft.mechanism_realizations
        ],
        "mechanism_realization_checks": assessment["mechanism_realization_checks"],
        "topology_hash": content_hash(assessment["topology"])
        if assessment.get("topology") is not None
        else None,
        "structurally_eligible": assessment["eligible"],
        "behavior_status": "unverified",
        "scope": SCOPE,
    }
