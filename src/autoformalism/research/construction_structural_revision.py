"""Opt-in atomic revisions across dynamics, topology and latent initialization.

This interface never adopts a partial edit, changes public data access, fits a
parameter, or treats static compilation as scientific acceptance. Callers must
apply their authenticated candidate-specific model/data gates to the returned
problem before exporting it for fitting. Historical law-only policies are intact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError, model_validator

from autoformalism.expressions import ModelValidationError, compile_candidate
from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY
from autoformalism.fitting.fitter import _parameter_variable
from autoformalism.fitting.initialization_graph import InitializationGraph
from autoformalism.fitting.models import FitConfig
from autoformalism.research.construction_revision_edits import (
    LawEdit,
    RevisionValidationError,
    _bounded_ast,
    _canonical_rhs,
    _copy_problem,
    _definitions,
    _preflight_parent,
    _symbols,
    _validate_aliases,
    _validate_problem,
    public_path_checks,
)
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.candidate import ParameterSpec
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.staged_functions import FunctionParameter
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search.shared_model_revision import relationships
from autoformalism.staged_topology import content_hash

POLICY = "construction-structural-revision-1"


class InitialValueChoice(StrictSchema):
    """Select an independent scalar's scope without prescribing a fitted value."""

    mode: Literal["value"] = "value"
    scope: Literal["shared_training", "trajectory"]
    role: Literal["coefficient", "nonnegative_coefficient"] = "coefficient"


class StateInitialReference(StrictSchema):
    """Bind one initializer-local symbol to a generated state's initial value."""

    alias: Identifier = Field(
        description="Local symbol used in the initializer expression, e.g. root."
    )
    state: Identifier = Field(
        description="Existing or newly declared differential state, e.g. z1."
    )


class PublicInitialReference(StrictSchema):
    """Bind one initializer-local symbol to a permitted public initial reading."""

    alias: Identifier = Field(
        description="Local symbol used in the initializer expression, e.g. measured."
    )
    channel: Identifier = Field(
        description="Permitted public channel for the recorded time-zero value, e.g. y."
    )


class InitialMapChoice(StrictSchema):
    """A complete map; e.g. ``a*root+measured`` with aliases root->z1, measured->y.

    The alias lists identify initial values, not ongoing trajectory inputs.
    Parameter ``a`` is declared separately; the runtime chooses its initial guess.
    """

    mode: Literal["map"] = "map"
    expression: str = Field(min_length=1, max_length=4096)
    parameters: tuple[FunctionParameter, ...] = Field(default=(), max_length=32)
    state_initials: tuple[StateInitialReference, ...] = Field(default=(), max_length=64)
    public_initials: tuple[PublicInitialReference, ...] = Field(
        default=(), max_length=64
    )

    @model_validator(mode="after")
    def distinct_local_symbols(self) -> InitialMapChoice:
        """Do not let list-to-map lowering overwrite an ambiguous declaration."""
        names = (
            [parameter.name for parameter in self.parameters]
            + [reference.alias for reference in self.state_initials]
            + [reference.alias for reference in self.public_initials]
        )
        if len(names) != len(set(names)):
            raise ValueError("duplicate or ambiguous initializer alias/parameter")
        if set(names) & APPROVED_FUNCTION_ARITY.keys():
            raise ValueError("initializer aliases cannot shadow approved functions")
        return self


InitialChoice = Annotated[
    InitialValueChoice | InitialMapChoice, Field(discriminator="mode")
]


class NewLatentState(StrictSchema):
    """One new differential state, with its complete RHS and initialization."""

    name: Identifier
    description: str = Field(min_length=1, max_length=2000)
    rhs: str = Field(min_length=1, max_length=4096)
    sources: tuple[Identifier, ...] = Field(default=(), max_length=64)
    initial: InitialChoice


class NewProcess(StrictSchema):
    """One algebraic definition; its actual consumers determine whether shared."""

    name: Identifier
    description: str = Field(min_length=1, max_length=2000)
    expression: str = Field(min_length=1, max_length=4096)
    sources: tuple[Identifier, ...] = Field(default=(), max_length=64)


class InitializationEdit(StrictSchema):
    """Replace one editable latent state's complete initializer rule."""

    state: Identifier
    initial: InitialChoice


class StructuralRevisionReply(StrictSchema):
    """One bounded complete patch, or an explicit non-adoption decision."""

    decision: Literal["revise", "retain", "requires_wider_revision"]
    hypothesis: str = Field(min_length=1, max_length=3000)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    expected_observable_effect: str = Field(min_length=1, max_length=3000)
    edits: tuple[LawEdit, ...] = Field(default=(), max_length=32)
    new_parameters: tuple[FunctionParameter, ...] = Field(default=(), max_length=64)
    new_states: tuple[NewLatentState, ...] = Field(default=(), max_length=8)
    new_processes: tuple[NewProcess, ...] = Field(default=(), max_length=16)
    remove_states: tuple[Identifier, ...] = Field(default=(), max_length=8)
    remove_processes: tuple[Identifier, ...] = Field(default=(), max_length=16)
    initialization_edits: tuple[InitializationEdit, ...] = Field(
        default=(), max_length=16
    )

    @model_validator(mode="after")
    def unambiguous_patch(self) -> StructuralRevisionReply:
        """Do not guess the ordering or intent of duplicate/conflicting actions."""
        groups = {
            "edited component": [e.component for e in self.edits],
            "new parameter": [p.name for p in self.new_parameters],
            "new component": [s.name for s in self.new_states]
            + [p.name for p in self.new_processes],
            "removed component": [*self.remove_states, *self.remove_processes],
            "initializer target": [e.state for e in self.initialization_edits],
            "evidence ID": list(self.evidence_ids),
        }
        for label, names in groups.items():
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {label}; provide one complete declaration")
        if any(not value.strip() for value in self.evidence_ids):
            raise ValueError("evidence IDs cannot be blank")
        actions = any(
            (
                self.edits,
                self.new_parameters,
                self.new_states,
                self.new_processes,
                self.remove_states,
                self.remove_processes,
                self.initialization_edits,
            )
        )
        if actions and self.decision != "revise":
            raise ValueError("retain/requires_wider_revision must not contain edits")
        if not actions and self.decision == "revise":
            raise ValueError("revise requires an explicit edit")
        return self


def editable_initial_states(
    problem: PrefixFitProblem,
    scientific_aliases: Mapping[str, str] | None = None,
) -> list[str]:
    """Protect known initials and authoritative recorded-public identities."""
    aliases = _validate_aliases(problem, scientific_aliases)
    protected = set(problem.context.targets) | set(problem.context.auxiliaries)
    protected |= set(aliases.values())
    return sorted(
        name
        for name, rule in problem.graph.rules.items()
        if name not in protected and rule.initial.mode != "known"
    )


def _initial_rule(
    state: str, choice: InitialChoice, *, allow_trajectory_roots: bool
) -> dict[str, Any]:
    """Supply deterministic starts, leaving scopes and formula choices explicit."""
    payload = choice.model_dump(mode="json")
    if isinstance(choice, InitialValueChoice):
        if choice.scope == "trajectory" and not allow_trajectory_roots:
            raise RevisionValidationError(
                "INITIALIZATION_SCOPE",
                "this experiment has not authorized trajectory-specific roots",
                state=state,
            )
        payload["guess"] = 0.1
    else:
        expression = _canonical_rhs(choice.expression, state, "algebraic", "t")
        _bounded_ast(expression, f"initial:{state}")
        _symbols(expression, f"initial:{state}")
        payload["expression"] = expression
        payload["state_initials"] = {
            reference.alias: reference.state for reference in choice.state_initials
        }
        payload["public_initials"] = {
            reference.alias: reference.channel for reference in choice.public_initials
        }
        for parameter in payload["parameters"]:
            parameter["guess"] = 0.1
    return {"initial": payload}


def _normalize_law(
    expression: str,
    component: str,
    kind: str,
    sources: Iterable[str],
    *,
    time_symbol: str,
    parameters: set[str],
    allowed_sources: set[str],
    changes: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Normalize only redundant labels/sources and reject unavailable symbols."""
    rhs = _canonical_rhs(expression, component, kind, time_symbol)
    _bounded_ast(rhs, component)
    actual = _symbols(rhs, component) - parameters
    if unknown := actual - allowed_sources:
        raise RevisionValidationError(
            "UNKNOWN_FORMULA_SYMBOL",
            "every formula symbol must resolve in the resulting complete patch",
            component=component,
            unknown=sorted(unknown),
            allowed_variable_sources=sorted(allowed_sources),
            declared_parameters=sorted(parameters),
        )
    if rhs != expression or sorted(actual) != list(sources):
        changes.append(
            {
                "component": component,
                "expression_before": expression,
                "expression_after": rhs,
                "sources_before": list(sources),
                "sources_after": sorted(actual),
                "reason": "exact matching RHS label and parsed dependency metadata",
            }
        )
    return rhs, sorted(actual)


def _apply(
    problem: PrefixFitProblem,
    reply: StructuralRevisionReply,
    *,
    evidence_ids: Iterable[str],
    brief: PublicScientificBrief | None,
    scientific_aliases: Mapping[str, str] | None,
    allow_trajectory_roots: bool,
) -> dict[str, Any]:
    """Construct a detached candidate; all checks precede returning an adoption."""
    parent = _copy_problem(problem)
    _preflight_parent(parent)
    _validate_problem(parent)
    aliases = _validate_aliases(parent, scientific_aliases)
    if missing := set(reply.evidence_ids) - set(evidence_ids):
        raise RevisionValidationError(
            "UNKNOWN_EVIDENCE",
            "cite supplied training evidence only",
            unknown=sorted(missing),
        )
    before_paths = public_path_checks(
        parent, brief, scientific_aliases=scientific_aliases
    )
    audit: dict[str, Any] = {
        "protocol": POLICY,
        "parent_problem_sha256": content_hash(parent.model_dump(mode="json")),
        "reply": reply.model_dump(mode="json"),
        "public_path_checks_before": before_paths,
        "context_unchanged": True,
        "observation_mappings_unchanged": True,
        "constraints_unchanged": True,
        "parameter_fitting_performed": False,
        "numerical_rollout_performed": False,
        "scientific_correctness_established": False,
        "candidate_specific_data_checks_required": True,
        "trajectory_roots_authorized": allow_trajectory_roots,
    }
    if reply.decision != "revise":
        path_status = (
            "unassessed"
            if brief is None
            or any(row["status"] == "unresolved_identity" for row in before_paths)
            else "repair_required"
            if any(row["status"] != "pass" for row in before_paths)
            else "eligible"
        )
        audit.update(
            child_problem_sha256=audit["parent_problem_sha256"],
            public_path_checks_after=before_paths,
            static_eligibility="repair_required"
            if reply.decision == "requires_wider_revision"
            else path_status,
            initialization_unchanged=True,
        )
        return {
            "status": "retained" if reply.decision == "retain" else reply.decision,
            "problem": parent if reply.decision == "retain" else None,
            "audit": audit,
        }

    candidate, context = parent.base_candidate, parent.context
    definitions = _definitions(candidate)
    existing_parameters = {p.name for p in candidate.parameters}
    new_parameters = {p.name for p in reply.new_parameters}
    added = {s.name for s in reply.new_states} | {p.name for p in reply.new_processes}
    removed = set(reply.remove_states) | set(reply.remove_processes)
    edited = {e.component for e in reply.edits}
    reserved = (
        set(definitions)
        | existing_parameters
        | set(context.targets)
        | set(context.forcing_channels)
        | {context.time_symbol}
        | set(APPROVED_FUNCTION_ARITY)
    )
    collisions = (added & reserved) | (new_parameters & (reserved | added))
    collisions |= {
        name
        for name in added | new_parameters
        if name.startswith(("init_", "af_internal_"))
    }
    if collisions:
        raise RevisionValidationError(
            "DECLARATION_IDENTITY",
            "new names cannot redeclare or shadow existing identities; same-name "
            "state/process conversions require a wider revision",
            names=sorted(collisions),
        )
    if edited & removed:
        raise RevisionValidationError(
            "CONFLICTING_OPERATIONS",
            "cannot edit and remove one component",
            names=sorted(edited & removed),
        )
    latent = {s.name for s in candidate.states if s.kind.value == "latent"}
    processes = {p.name for p in candidate.processes}
    protected = (
        set(context.targets) | set(context.forcing_channels) | set(aliases.values())
    )
    protected |= {c.subject for c in candidate.constraints}
    protected |= {s.name for s in candidate.states if s.kind.value == "observed"}
    protected |= {
        name
        for name, rule in parent.graph.rules.items()
        if rule.initial.mode == "known"
    }
    if (
        set(reply.remove_states) - latent
        or set(reply.remove_processes) - processes
        or removed & protected
    ):
        raise RevisionValidationError(
            "REMOVAL_SCOPE",
            "only explicitly named unprotected latent states or processes "
            "may be removed",
            unavailable_states=sorted(set(reply.remove_states) - latent),
            unavailable_processes=sorted(set(reply.remove_processes) - processes),
            protected=sorted(removed & protected),
        )
    editable_initials = set(editable_initial_states(parent, aliases))
    if unavailable := {e.state for e in reply.initialization_edits} - (
        editable_initials - removed
    ):
        raise RevisionValidationError(
            "INITIALIZATION_IDENTITY",
            "initializer edits must name retained editable latent states; recorded "
            "public time-zero identities and known initials remain fixed",
            unavailable=sorted(unavailable),
            editable_initial_states=sorted(editable_initials - removed),
        )

    payload = candidate.model_dump(mode="json")
    payload["states"] = [s for s in payload["states"] if s["name"] not in removed]
    payload["state_equations"] = [
        e for e in payload["state_equations"] if e["state"] not in removed
    ]
    payload["processes"] = [p for p in payload["processes"] if p["name"] not in removed]
    payload["initial_conditions"] = [
        i for i in payload["initial_conditions"] if i["state"] not in removed
    ]
    graph_payload = parent.graph.model_dump(mode="json")
    graph_payload["rules"] = {
        name: rule
        for name, rule in graph_payload["rules"].items()
        if name not in removed
    }
    allowed_sources = (
        (set(definitions) - removed)
        | added
        | set(context.forcing_channels)
        | {context.time_symbol}
    )
    parameters = existing_parameters | new_parameters
    normalization: list[dict[str, Any]] = []
    revised = {
        name: value for name, value in definitions.items() if name not in removed
    }
    dependencies = []
    for edit in reply.edits:
        if (
            edit.component not in definitions
            or definitions[edit.component][0] != edit.kind
        ):
            raise RevisionValidationError(
                "COMPONENT_IDENTITY",
                "edit an existing component with its existing kind",
                component=edit.component,
                existing_components={
                    name: value[0] for name, value in definitions.items()
                },
            )
        rhs, sources = _normalize_law(
            edit.expression,
            edit.component,
            edit.kind,
            edit.sources,
            time_symbol=context.time_symbol,
            parameters=parameters,
            allowed_sources=allowed_sources,
            changes=normalization,
        )
        dependencies.append(
            {
                "component": edit.component,
                "before": sorted(
                    _symbols(definitions[edit.component][1], edit.component)
                    - existing_parameters
                ),
                "after": sources,
                "expression_before": definitions[edit.component][1],
                "expression_after": rhs,
            }
        )
        revised[edit.component] = (edit.kind, rhs)
    for item in reply.new_states:
        rhs, _ = _normalize_law(
            item.rhs,
            item.name,
            "dynamic",
            item.sources,
            time_symbol=context.time_symbol,
            parameters=parameters,
            allowed_sources=allowed_sources,
            changes=normalization,
        )
        revised[item.name] = ("dynamic", rhs)
        payload["states"].append(
            {"name": item.name, "kind": "latent", "description": item.description}
        )
        payload["state_equations"].append({"state": item.name, "rhs": rhs})
        # The base candidate's placeholder is never used by the scoped fitter;
        # the complete, validated graph supplies the actual initial expression.
        payload["initial_conditions"].append(
            {"state": item.name, "scope": "global", "fixed_value": 0.0}
        )
        graph_payload["rules"][item.name] = _initial_rule(
            item.name, item.initial, allow_trajectory_roots=allow_trajectory_roots
        )
    for item in reply.new_processes:
        rhs, _ = _normalize_law(
            item.expression,
            item.name,
            "algebraic",
            item.sources,
            time_symbol=context.time_symbol,
            parameters=parameters,
            allowed_sources=allowed_sources,
            changes=normalization,
        )
        revised[item.name] = ("algebraic", rhs)
        payload["processes"].append(
            {"name": item.name, "expression": rhs, "description": item.description}
        )
    for item in reply.initialization_edits:
        graph_payload["rules"][item.state] = _initial_rule(
            item.state, item.initial, allow_trajectory_roots=allow_trajectory_roots
        )
    for equation in payload["state_equations"]:
        equation["rhs"] = revised[equation["state"]][1]
    for process in payload["processes"]:
        process["expression"] = revised[process["name"]][1]

    # Unedited laws, observation maps, constraints and initializer edges cannot
    # silently lose references when a component is explicitly removed.
    expression_rows = [(name, value[1]) for name, value in revised.items()]
    expression_rows += [
        (f"output:{m.channel}", m.expression) for m in candidate.observation_mappings
    ]
    expression_rows += [
        (f"initial:{i['state']}", i["expression"])
        for i in payload["initial_conditions"]
        if i.get("expression")
    ]
    used: set[str] = set()
    dangling = []
    for location, expression in expression_rows:
        names = _symbols(expression, location)
        used |= names
        if names & removed:
            dangling.append({"location": location, "names": sorted(names & removed)})
    for state, rule in graph_payload["rules"].items():
        refs = set(rule["initial"].get("state_initials", {}).values()) & removed
        if refs:
            dangling.append(
                {"location": f"initialization_graph:{state}", "names": sorted(refs)}
            )
    if dangling:
        raise RevisionValidationError(
            "DANGLING_REFERENCE",
            "explicitly revise every consumer before removing a component",
            references=dangling,
        )
    if unused := new_parameters - used:
        raise RevisionValidationError(
            "UNUSED_NEW_PARAMETER",
            "new dynamics parameters must occur in the resulting model",
            names=sorted(unused),
        )
    removed_parameters = existing_parameters - used
    if constrained := removed_parameters & {c.subject for c in candidate.constraints}:
        raise RevisionValidationError(
            "FROZEN_CONSTRAINT",
            "cannot remove a constrained parameter",
            names=sorted(constrained),
        )
    payload["parameters"] = [
        p for p in payload["parameters"] if p["name"] not in removed_parameters
    ]
    payload["parameters"] += [
        ParameterSpec(name=p.name, scope="global", role=p.role).model_dump(mode="json")
        for p in reply.new_parameters
    ]
    payload["parent_candidate_id"] = candidate.candidate_id
    payload["candidate_id"] = (
        "structural_"
        + content_hash(
            {
                "parent": parent.model_dump(mode="json"),
                "reply": reply.model_dump(mode="json"),
            }
        )[:24]
    )
    payload["change_summary"] = reply.hypothesis
    child_candidate = CandidateModel.model_validate(payload)
    graph = InitializationGraph.model_validate(graph_payload)
    model = compile_candidate(child_candidate, context)
    guesses = {
        name: value
        for name, value in parent.parameter_guesses.items()
        if name not in removed_parameters
    }
    for spec in child_candidate.parameters:
        if spec.name in new_parameters:
            variable = _parameter_variable(model, spec, FitConfig())
            guesses[spec.name] = (variable.start_lower + variable.start_upper) / 2.0
    child = PrefixFitProblem(
        base_candidate=child_candidate,
        context=context,
        graph=graph,
        parameter_guesses=guesses,
    )
    _preflight_parent(child)
    _, initialization_audit = _validate_problem(child)
    after_paths = public_path_checks(
        child, brief, scientific_aliases=scientific_aliases
    )
    passing = {row["id"] for row in before_paths if row["status"] == "pass"}
    if regressed := [
        row for row in after_paths if row["id"] in passing and row["status"] != "pass"
    ]:
        raise RevisionValidationError(
            "PUBLIC_PATH_REGRESSION",
            "a previously present public mechanism path was removed",
            checks=regressed,
        )
    audit.update(
        child_problem_sha256=content_hash(child.model_dump(mode="json")),
        dependency_changes=dependencies,
        new_states=sorted(s.name for s in reply.new_states),
        new_processes=sorted(p.name for p in reply.new_processes),
        removed_states=list(reply.remove_states),
        removed_processes=list(reply.remove_processes),
        removed_unused_parameters=sorted(removed_parameters),
        new_parameters=sorted(new_parameters),
        initialization_edits=[e.state for e in reply.initialization_edits],
        initialization_unchanged=child.graph == parent.graph,
        initialization_audit=initialization_audit,
        normalization={
            "changes": normalization,
            "raw_reply": reply.model_dump(mode="json"),
        },
        shared_laws_before=relationships(candidate.model_dump(mode="json")),
        shared_laws_after=relationships(child_candidate.model_dump(mode="json")),
        public_path_checks_after=after_paths,
        scientific_aliases=aliases,
        alias_metadata_available=scientific_aliases is not None,
        static_eligibility="unassessed"
        if brief is None
        or any(row["status"] == "unresolved_identity" for row in after_paths)
        else "repair_required"
        if any(row["status"] != "pass" for row in after_paths)
        else "eligible",
    )
    return {"status": "revised", "problem": child, "audit": audit}


def apply_structural_revision(
    problem: PrefixFitProblem,
    reply: StructuralRevisionReply | Mapping[str, Any],
    *,
    evidence_ids: Iterable[str],
    brief: PublicScientificBrief | None = None,
    scientific_aliases: Mapping[str, str] | None = None,
    allow_trajectory_roots: bool = True,
) -> dict[str, Any]:
    """Return an all-or-nothing typed revision; no fitting or data reads occur."""
    try:
        raw = (
            reply.model_dump(mode="json")
            if isinstance(reply, StructuralRevisionReply)
            else deepcopy(reply)
        )
        parsed = StructuralRevisionReply.model_validate(raw)
        return _apply(
            problem,
            parsed,
            evidence_ids=evidence_ids,
            brief=brief,
            scientific_aliases=scientific_aliases,
            allow_trajectory_roots=allow_trajectory_roots,
        )
    except RevisionValidationError:
        raise
    except ValidationError as exc:
        raise RevisionValidationError(
            "REVISION_SCHEMA",
            "structural reply or candidate schema rejected",
            errors=[
                {"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()
            ],
        ) from exc
    except ModelValidationError as exc:
        raise RevisionValidationError(
            "MODEL_VALIDATION",
            "restricted model compilation rejected the complete patch",
            diagnostics=[
                {"code": d.code, "location": d.location, "message": d.message}
                for d in exc.diagnostics
            ],
        ) from exc
    except RecursionError as exc:
        raise RevisionValidationError(
            "EXPRESSION_COMPLEXITY", "model expansion exceeds safe recursion limits"
        ) from exc
    except (ValueError, TypeError) as exc:
        raise RevisionValidationError("REVISION_VALIDATION", str(exc)) from exc
