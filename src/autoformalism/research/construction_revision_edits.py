"""Atomic, construction-only law revisions with frozen scoped initializers.

This opt-in interface changes formulas and their explicitly declared sources on
an existing variable roster. It never fits parameters, integrates trajectories,
or turns static compilation into a scientific or numerical accuracy certificate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from autoformalism.expressions import (
    ModelValidationError,
    RestrictedParser,
    compile_candidate,
)
from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY
from autoformalism.fitting.fitter import _parameter_variable
from autoformalism.fitting.initialization_graph import apply_initialization_graph
from autoformalism.fitting.models import FitConfig
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.candidate import ParameterSpec
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.staged_functions import FunctionParameter
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search.shared_model_revision import relationships
from autoformalism.staged_topology import content_hash

POLICY = "training-construction-law-revision-1"


class LawEdit(StrictSchema):
    """Replace one existing complete RHS with explicit scientific dependencies."""

    component: Identifier
    kind: Literal["dynamic", "algebraic"]
    sources: tuple[Identifier, ...] = Field(max_length=64)
    expression: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def distinct_sources(self) -> LawEdit:
        """A dependency is a set, not repeated syntactic occurrences."""
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("declare each source exactly once")
        return self


class RevisionReply(StrictSchema):
    """One bounded coherent patch or an explicit reason to retain the parent."""

    hypothesis: str = Field(min_length=1, max_length=3000)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    expected_observable_effect: str = Field(min_length=1, max_length=3000)
    edits: tuple[LawEdit, ...] = Field(default=(), max_length=32)
    new_parameters: tuple[FunctionParameter, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def unambiguous_patch(self) -> RevisionReply:
        """Reject duplicate operations and parameter-only numeric adjustments."""
        for label, names in (
            ("component", [item.component for item in self.edits]),
            ("new parameter", [item.name for item in self.new_parameters]),
            ("evidence ID", list(self.evidence_ids)),
        ):
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {label}")
        if self.new_parameters and not self.edits:
            raise ValueError("new parameters require an explicit law edit")
        if any(not value.strip() for value in self.evidence_ids):
            raise ValueError("evidence IDs cannot be blank")
        return self


class RevisionValidationError(ValueError):
    """A bounded diagnostic that callers may return to a rejected proposer."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        self.code = code
        self.details = details
        super().__init__(f"{code}: {message}")

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable diagnostic without provider exception payloads."""
        return {"code": self.code, "message": str(self), "details": self.details}


def _definitions(candidate: CandidateModel) -> dict[str, tuple[str, str]]:
    """Keep named shared processes separate from their consuming equations."""
    return {
        **{e.state: ("dynamic", e.rhs) for e in candidate.state_equations},
        **{p.name: ("algebraic", p.expression) for p in candidate.processes},
    }


def _symbols(expression: str, location: str) -> set[str]:
    """Parse the restricted scalar grammar without executing provider text."""
    return set(RestrictedParser().parse(expression, location=location).symbols)


def _copy_problem(problem: PrefixFitProblem) -> PrefixFitProblem:
    """Detach nested mappings and revalidate even model_copy-created objects."""
    if not isinstance(problem, PrefixFitProblem):
        raise RevisionValidationError("PROBLEM_SCHEMA", "typed prefix problem required")
    return PrefixFitProblem.model_validate(problem.model_dump(mode="json"))


def _validate_problem(problem: PrefixFitProblem):
    """Check the unlowered model and exact scoped graph with no numerical work."""
    if problem.context.fitted_initialization or problem.context.lagged_targets:
        raise RevisionValidationError(
            "PROBLEM_CONTEXT", "an unlowered free-rollout context is required"
        )
    if any(p.scope.value != "global" for p in problem.base_candidate.parameters):
        raise RevisionValidationError(
            "PARAMETER_SCOPE", "all dynamics parameters must remain globally shared"
        )
    model = compile_candidate(problem.base_candidate, problem.context)
    if set(problem.parameter_guesses) != set(model.parameter_names):
        raise RevisionValidationError(
            "PARAMETER_STARTS", "runtime guesses must cover exactly dynamics parameters"
        )
    _, _, audit = apply_initialization_graph(model, problem.graph)
    return model, audit


def _path(graph: dict[str, set[str]], source: str, target: str) -> list[str] | None:
    """Find a declared directed path, allowing dynamic feedback loops."""
    queue, seen = [(source, [source])], {source}
    for current, path in queue:
        if current == target:
            return path
        for child in sorted(graph.get(current, set()) - seen):
            seen.add(child)
            queue.append((child, [*path, child]))
    return None


def _validate_aliases(
    problem: PrefixFitProblem, scientific_aliases: Mapping[str, str] | None
) -> dict[str, str]:
    """Check caller-authenticated identities without inferring them from initials.

    A bare mapping cannot prove which private generated auxiliary was originally
    declared. The source projector must authenticate that attribution; this gate
    checks namespace, uniqueness and fixed output identities only.
    """
    if scientific_aliases is None:
        return {}
    if not isinstance(scientific_aliases, Mapping):
        raise RevisionValidationError("SCIENTIFIC_ALIASES", "aliases must be a mapping")
    aliases = dict(scientific_aliases)
    if any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in aliases.items()
    ):
        raise RevisionValidationError(
            "SCIENTIFIC_ALIASES", "alias names must be strings"
        )
    context, candidate = problem.context, problem.base_candidate
    public = set(context.targets) | set(context.auxiliaries)
    generated = set(_definitions(candidate))
    unavailable_keys = set(aliases) - public
    unavailable_values = set(aliases.values()) - generated
    if unavailable_keys or unavailable_values:
        raise RevisionValidationError(
            "SCIENTIFIC_ALIASES",
            "aliases require public identities and generated definitions",
            unavailable_public=sorted(unavailable_keys),
            unavailable_generated=sorted(unavailable_values),
        )
    if len(set(aliases.values())) != len(aliases):
        raise RevisionValidationError(
            "SCIENTIFIC_ALIASES", "different scientific variables cannot share an alias"
        )
    for source, compiled in aliases.items():
        if compiled in public and compiled != source:
            raise RevisionValidationError(
                "SCIENTIFIC_ALIASES",
                "aliases cannot borrow another public identity",
                source=source,
                compiled=compiled,
            )
        if source in context.targets:
            mapping = next(
                m for m in candidate.observation_mappings if m.channel == source
            )
            if mapping.expression.strip() != compiled:
                raise RevisionValidationError(
                    "SCIENTIFIC_ALIASES",
                    "target alias differs from its fixed output mapping",
                    source=source,
                    compiled=compiled,
                )
    return aliases


def public_path_checks(
    problem: PrefixFitProblem,
    brief: PublicScientificBrief | None,
    *,
    scientific_aliases: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Check only authoritative public reachability, not mechanism sufficiency.

    Initial-only influence is displayed separately and never substitutes for a
    dynamic dependency. Nonzero effect, signs, conservation, and adequate delay
    cannot be certified from syntactic paths.
    """
    aliases = _validate_aliases(problem, scientific_aliases)
    if brief is None:
        return []
    brief = PublicScientificBrief.model_validate(brief.model_dump(mode="json"))
    context = problem.context
    expected = {
        "target": set(context.targets),
        "auxiliary": set(context.auxiliaries),
        "external_input": set(context.external_inputs),
        "covariate": set(context.fixed_covariates),
        "time": {context.time_symbol},
    }
    for role, names in expected.items():
        supplied = {v.name for v in brief.public_variables if v.data_role == role}
        if supplied != names:
            raise RevisionValidationError(
                "PUBLIC_BRIEF_CONTEXT",
                "brief roles must match the frozen public context",
                role=role,
                expected=sorted(names),
                supplied=sorted(supplied),
            )
    candidate = problem.base_candidate
    parameters = {p.name for p in candidate.parameters}
    graph: dict[str, set[str]] = {}
    for receiver, (_, expression) in _definitions(candidate).items():
        for source in _symbols(expression, receiver) - parameters:
            graph.setdefault(source, set()).add(receiver)
    outputs = {}
    for mapping in candidate.observation_mappings:
        target = f"output:{mapping.channel}"
        outputs[mapping.channel] = target
        for source in _symbols(mapping.expression, target) - parameters:
            graph.setdefault(source, set()).add(target)
    with_initials = {source: set(targets) for source, targets in graph.items()}
    for state, rule in problem.graph.rules.items():
        if rule.initial.mode == "map":
            for source in (
                *rule.initial.public_initials.values(),
                *rule.initial.state_initials.values(),
            ):
                with_initials.setdefault(source, set()).add(state)
    states = {s.name for s in candidate.states}
    checks = []

    def check(identifier, sources, target, memory):
        output = outputs.get(target, target)
        witnesses = []
        for source in sources:
            for actual in dict.fromkeys((source, aliases.get(source, source))):
                direct = _path(graph, actual, output)
                if direct is None:
                    continue
                identity = {
                    "source": source,
                    "actual_source": actual,
                    "source_mode": "direct" if actual == source else "modeled_alias",
                }
                if not memory:
                    witnesses.append({**identity, "path": direct, "memory_state": None})
                    continue
                for state in sorted(states - {source, actual}):
                    first = _path(graph, actual, state)
                    last = _path(graph, state, output)
                    if first is not None and last is not None:
                        witnesses.append(
                            {
                                **identity,
                                "path": first + last[1:],
                                "memory_state": state,
                            }
                        )
                        break
        initial_paths = [
            {"source": source, "path": path}
            for source in sources
            if (path := _path(with_initials, source, output)) is not None
            and _path(graph, source, output) is None
        ]
        identity_unknown = scientific_aliases is None and bool(
            set(sources) & set(context.auxiliaries)
        )
        checks.append(
            {
                "id": identifier,
                "sources": list(sources),
                "target": target,
                "requires_dynamic_memory": memory,
                "status": "pass"
                if witnesses
                else "unresolved_identity"
                if identity_unknown
                else "initialization_only"
                if initial_paths
                else "fail",
                "witnesses": witnesses,
                "initialization_only_paths": initial_paths,
                "alias_metadata_available": scientific_aliases is not None,
                "scientific_sufficiency_established": False,
            }
        )

    for requirement in brief.requirements:
        for source in requirement.drivers:
            for target in requirement.targets:
                check(
                    f"requirement:{requirement.id}:{source}:{target}",
                    [source],
                    target,
                    requirement.requires_dynamic_memory,
                )
    for index, dependency in enumerate(brief.target_dependencies):
        check(
            f"target_dependency:{index}:{dependency.target}",
            dependency.acceptable_sources,
            dependency.target,
            False,
        )
    return checks


def model_view(
    problem: PrefixFitProblem, *, scientific_aliases: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Expose complete assembled laws, dependencies and fixed initialization."""
    problem = _copy_problem(problem)
    _, audit = _validate_problem(problem)
    aliases = _validate_aliases(problem, scientific_aliases)
    candidate = problem.base_candidate
    parameters = {p.name for p in candidate.parameters}
    descriptions = {
        item.name: item.description
        for item in (*candidate.states, *candidate.processes)
    }
    return {
        "protocol": POLICY,
        "problem_sha256": content_hash(problem.model_dump(mode="json")),
        "candidate_id": candidate.candidate_id,
        "scientific_aliases": aliases,
        "alias_metadata_available": scientific_aliases is not None,
        "equations": [
            {
                "component": name,
                "kind": kind,
                "expression": expression,
                "display": (
                    f"d({name})/dt = {expression}"
                    if kind == "dynamic"
                    else f"{name} = {expression}"
                ),
                "scientific_role": descriptions[name],
                "sources": sorted(_symbols(expression, name) - parameters),
                "parameters": sorted(_symbols(expression, name) & parameters),
            }
            for name, (kind, expression) in _definitions(candidate).items()
        ],
        "states": [
            {
                "name": s.name,
                "kind": s.kind.value,
                **(
                    {"description": s.description}
                    if s.description != "unspecified"
                    else {}
                ),
            }
            for s in candidate.states
        ],
        "parameters": [
            {
                "name": p.name,
                "role": p.role.value,
                "domain": p.domain.value,
                **({"bounds": p.bounds.model_dump(mode="json")} if p.bounds else {}),
            }
            for p in candidate.parameters
        ],
        "observation_mappings": [
            {"channel": m.channel, "expression": m.expression}
            for m in candidate.observation_mappings
        ],
        "constraints": [c.model_dump(mode="json") for c in candidate.constraints],
        "shared_law_relationships": [
            {key: value for key, value in law.items() if key != "law"}
            for law in relationships(candidate.model_dump(mode="json"))
        ],
        "context": problem.context.model_dump(mode="json"),
        "initialization_graph": problem.graph.model_dump(mode="json"),
        "initialization_scopes": {
            "trajectory_root_parameters": audit["trajectory_root_parameters"],
            "shared_parameter_names": audit["shared_parameter_names"],
        },
        "frozen_fields": [
            "context",
            "states",
            "process_names",
            "initial_conditions",
            "initialization_graph",
            "observation_mappings",
            "constraints",
            "existing_parameter_roles_domains_bounds",
        ],
        "interpretation": (
            "Sources name model-generated variables or permitted forcing, never "
            "measured target histories. Shared laws are edited once; unedited "
            "consumer expressions and their signs remain exact. A complete RHS "
            "replacement changes signs only as explicitly written and is not a "
            "scientific sign or conservation certificate."
        ),
    }


def _apply(
    problem: PrefixFitProblem,
    reply: RevisionReply,
    evidence_ids: Iterable[str],
    brief: PublicScientificBrief | None,
    scientific_aliases: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Build a detached child and commit only after all static gates pass."""
    parent = _copy_problem(problem)
    _, initial_audit = _validate_problem(parent)
    _validate_aliases(parent, scientific_aliases)
    missing = set(reply.evidence_ids) - set(evidence_ids)
    if missing:
        raise RevisionValidationError(
            "UNKNOWN_EVIDENCE",
            "cite only the supplied training evidence IDs",
            unknown=sorted(missing),
        )
    candidate = parent.base_candidate
    definitions = _definitions(candidate)
    existing = {p.name for p in candidate.parameters}
    new = {p.name for p in reply.new_parameters}
    context = parent.context
    allowed_sources = (
        set(definitions) | set(context.forcing_channels) | {context.time_symbol}
    )
    reserved = allowed_sources | set(context.targets) | set(APPROVED_FUNCTION_ARITY)
    collisions = new & (existing | reserved)
    collisions |= {name for name in new if name.startswith(("af_internal_", "init_"))}
    if collisions:
        raise RevisionValidationError(
            "PARAMETER_IDENTITY",
            "new parameters cannot redeclare or shadow names",
            names=sorted(collisions),
        )
    revised_definitions = dict(definitions)
    dependency_changes = []
    for edit in reply.edits:
        old = definitions.get(edit.component)
        if old is None or old[0] != edit.kind:
            raise RevisionValidationError(
                "COMPONENT_IDENTITY",
                "edit an existing component with its existing kind",
                component=edit.component,
                expected_kind=old[0] if old else None,
            )
        unavailable = set(edit.sources) - allowed_sources
        if unavailable:
            raise RevisionValidationError(
                "UNAVAILABLE_SOURCE",
                "sources must be existing variables or permitted forcing",
                component=edit.component,
                unavailable=sorted(unavailable),
            )
        actual = _symbols(edit.expression, edit.component) - existing - new
        declared = set(edit.sources)
        if actual != declared:
            raise RevisionValidationError(
                "SOURCE_MISMATCH",
                "expression must use exactly its declared sources",
                component=edit.component,
                missing=sorted(declared - actual),
                undeclared=sorted(actual - declared),
            )
        revised_definitions[edit.component] = (edit.kind, edit.expression)
        dependency_changes.append(
            {
                "component": edit.component,
                "before": sorted(_symbols(old[1], edit.component) - existing),
                "after": sorted(declared),
                "expression_before": old[1],
                "expression_after": edit.expression,
            }
        )
    payload = candidate.model_dump(mode="json")
    for equation in payload["state_equations"]:
        equation["rhs"] = revised_definitions[equation["state"]][1]
    for process in payload["processes"]:
        process["expression"] = revised_definitions[process["name"]][1]
    # Remove obsolete dynamics coordinates only after scanning every remaining
    # formula, output, and base initializer. Scoped map parameters are local.
    expressions = [value[1] for value in revised_definitions.values()]
    expressions += [m.expression for m in candidate.observation_mappings]
    expressions += [i.expression for i in candidate.initial_conditions if i.expression]
    used = set().union(*(_symbols(value, "parameter_usage") for value in expressions))
    if unused := new - used:
        raise RevisionValidationError(
            "UNUSED_NEW_PARAMETER",
            "every new parameter must occur in the resulting model",
            names=sorted(unused),
        )
    removed = existing - used
    if constrained := removed & {c.subject for c in candidate.constraints}:
        raise RevisionValidationError(
            "FROZEN_CONSTRAINT",
            "an edit would remove a constrained parameter",
            names=sorted(constrained),
        )
    payload["parameters"] = [
        p for p in payload["parameters"] if p["name"] not in removed
    ]
    payload["parameters"] += [
        ParameterSpec(name=p.name, scope="global", role=p.role).model_dump(mode="json")
        for p in reply.new_parameters
    ]
    changed = any(
        definitions[name] != value for name, value in revised_definitions.items()
    )
    changed = changed or bool(new or removed)
    if changed:
        payload["parent_candidate_id"] = candidate.candidate_id
        payload["candidate_id"] = (
            "revision_"
            + content_hash(
                {
                    "parent": parent.model_dump(mode="json"),
                    "reply": reply.model_dump(mode="json"),
                }
            )[:24]
        )
        payload["change_summary"] = reply.hypothesis
    child_candidate = CandidateModel.model_validate(payload)
    model = compile_candidate(child_candidate, context)
    _, _, child_audit = apply_initialization_graph(model, parent.graph)
    if child_audit != initial_audit:
        raise RevisionValidationError(
            "INITIALIZATION_CHANGED",
            "law edits must preserve the complete initializer audit",
        )
    guesses = {
        name: value
        for name, value in parent.parameter_guesses.items()
        if name not in removed
    }
    settings = FitConfig()
    for spec in child_candidate.parameters:
        variable = _parameter_variable(model, spec, settings)
        if spec.name in new:
            guesses[spec.name] = (variable.start_lower + variable.start_upper) / 2.0
    child = PrefixFitProblem(
        base_candidate=child_candidate,
        context=context,
        graph=parent.graph,
        parameter_guesses=guesses,
    )
    _validate_problem(child)
    before_laws = relationships(candidate.model_dump(mode="json"))
    after_laws = relationships(child_candidate.model_dump(mode="json"))
    after_consumers = {r["process"]: set(r["direct_consumers"]) for r in after_laws}
    for law in before_laws:
        if len(law["direct_consumers"]) < 2 or not any(
            e.component == law["process"] for e in reply.edits
        ):
            continue
        if lost := set(law["direct_consumers"]) - after_consumers[law["process"]]:
            raise RevisionValidationError(
                "SHARED_CONSUMER_REMOVED",
                "a revised shared law must retain its consumers",
                process=law["process"],
                consumers=sorted(lost),
            )
    before_paths = public_path_checks(
        parent, brief, scientific_aliases=scientific_aliases
    )
    after_paths = public_path_checks(
        child, brief, scientific_aliases=scientific_aliases
    )
    passing = {row["id"] for row in before_paths if row["status"] == "pass"}
    regressed = [
        row for row in after_paths if row["id"] in passing and row["status"] != "pass"
    ]
    if regressed:
        raise RevisionValidationError(
            "PUBLIC_PATH_REGRESSION",
            "a previously present public mechanism path was removed",
            checks=regressed,
        )
    return {
        "status": "revised" if changed else "retained",
        "problem": child,
        "audit": {
            "protocol": POLICY,
            "parent_problem_sha256": content_hash(parent.model_dump(mode="json")),
            "child_problem_sha256": content_hash(child.model_dump(mode="json")),
            "reply": reply.model_dump(mode="json"),
            "dependency_changes": dependency_changes,
            "removed_unused_parameters": sorted(removed),
            "new_parameters": sorted(new),
            "shared_laws_before": before_laws,
            "shared_laws_after": after_laws,
            "public_path_checks_before": before_paths,
            "public_path_checks_after": after_paths,
            "scientific_aliases": dict(scientific_aliases or {}),
            "alias_metadata_available": scientific_aliases is not None,
            "static_eligibility": (
                "unassessed"
                if brief is None
                or any(row["status"] == "unresolved_identity" for row in after_paths)
                else "repair_required"
                if any(row["status"] != "pass" for row in after_paths)
                else "eligible"
            ),
            "initialization_unchanged": True,
            "context_unchanged": True,
            "parameter_fitting_performed": False,
            "numerical_rollout_performed": False,
            "scientific_correctness_established": False,
        },
    }


def apply_revision(
    problem: PrefixFitProblem,
    reply: RevisionReply | Mapping[str, Any],
    *,
    evidence_ids: Iterable[str],
    brief: PublicScientificBrief | None = None,
    scientific_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a valid detached child or an actionable error, never a partial edit."""
    try:
        raw = (
            reply.model_dump(mode="json") if isinstance(reply, RevisionReply) else reply
        )
        return _apply(
            problem,
            RevisionReply.model_validate(raw),
            evidence_ids,
            brief,
            scientific_aliases,
        )
    except RevisionValidationError:
        raise
    except ValidationError as exc:
        errors = [
            {"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()
        ]
        raise RevisionValidationError(
            "REVISION_SCHEMA", "reply or candidate schema rejected", errors=errors
        ) from exc
    except ModelValidationError as exc:
        raise RevisionValidationError(
            "MODEL_VALIDATION",
            "restricted model compilation rejected the patch",
            diagnostics=[
                {
                    "code": d.code,
                    "location": d.location,
                    "message": d.message,
                }
                for d in exc.diagnostics
            ],
        ) from exc
    except (ValueError, TypeError) as exc:
        raise RevisionValidationError("REVISION_VALIDATION", str(exc)) from exc
