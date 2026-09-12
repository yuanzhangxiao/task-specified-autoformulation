"""Bounded cross-level model edits with local salvage and atomic global commit."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.expressions import (
    ModelValidationError,
    RestrictedParser,
    ValidationContext,
    compile_candidate,
)
from autoformalism.fitting.initialization import (
    InitialCausalMap,
    InitialValueGuess,
    LatentInitializationPlan,
    LatentInitializationReply,
    apply_initialization_plan,
)
from autoformalism.llm.staged_topology import (
    StagedTopologyClient,
    atomic_json,
    visible_response,
)
from autoformalism.rebuttal.repair_evidence import (
    domain_findings,
    memory_path_findings,
    model_hash,
)
from autoformalism.rebuttal.revision_decision import (
    expressions,
    interaction_diff,
    nonlinear_target_paths,
    parameter_aliases,
    translate_names,
)
from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
    ComponentRevision,
    RevisionContractError,
    RevisionParameter,
    _effective_revision_parameter_roles,
)
from autoformalism.schemas import CandidateModel, ParameterSpec
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.staged_topology import content_hash


class EquationEdit(StrictSchema):
    """One complete definition; kind is needed only for inventory/type changes."""

    component: Identifier
    kind: Literal["preserve", "dynamic", "algebraic"] = "preserve"
    expression: str = Field(min_length=1, max_length=4096)
    parameters: tuple[RevisionParameter, ...] = Field(default=(), max_length=32)


class MappingEdit(StrictSchema):
    channel: Identifier
    expression: str = Field(min_length=1, max_length=4096)


class InitializerEdit(StrictSchema):
    """A training-fitted shared value or causal map, never validation fitting."""

    state: Identifier
    # Numeric guesses are deliberately runtime-owned in this experiment.
    causal_map: InitialCausalMap | None = None


class RepairAction(StrictSchema):
    """Compact intent plus actual edits, without duplicated candidate metadata."""

    scope: Literal["function", "model", "no_change"]
    hypothesis: str = Field(default="", max_length=800)
    equations: tuple[EquationEdit, ...] = Field(default=(), max_length=6)
    remove: tuple[Identifier, ...] = Field(default=(), max_length=2)
    mappings: tuple[MappingEdit, ...] = Field(default=(), max_length=3)
    initializers: tuple[InitializerEdit, ...] = Field(default=(), max_length=4)
    keep: tuple[Identifier, ...] = Field(default=(), max_length=24)

    @model_validator(mode="after")
    def unique_and_scoped(self):
        for items in (
            [e.component for e in self.equations],
            list(self.remove),
            [m.channel for m in self.mappings],
            [i.state for i in self.initializers],
        ):
            if len(items) != len(set(items)):
                raise ValueError("duplicate action target")
        changed = {e.component for e in self.equations}
        if changed & set(self.remove) or (changed | set(self.remove)) & set(self.keep):
            raise ValueError("a component cannot be changed/removed and kept together")
        if self.scope == "no_change" and (
            self.equations or self.remove or self.mappings or self.initializers
        ):
            raise ValueError("no_change contains edits")
        if self.scope == "function" and (
            self.remove
            or self.mappings
            or self.initializers
            or any(e.kind != "preserve" for e in self.equations)
        ):
            raise ValueError(
                "inventory, mapping and initialization edits require model scope"
            )
        return self


SYSTEM_PROMPT = """You revise a continuous-time scientific model. Runtime observations
are not causal explanations. Choose the scientific hypothesis and the smallest
coherent repair; the objective category is a priority, not a mandated component.
You may edit several related equations in one action, including another equation
than a finding names. Use scope=function only if every signed additive interaction
and nonparameter source set is unchanged. Use scope=model explicitly to add/remove
dependencies or change signs, add/remove a variable, change dynamic/algebraic kind,
target mappings or causal initializers. Supply each new variable's complete equation.
No current/future target measurements may drive RHSs; generated targets may.
Parent parameter aliases are automatically available, with immutable roles. Only
declare genuinely new parameters. Outer gains have nonnegative magnitudes; the
expression owns their + or - sign. New offsets are unrestricted. Ambiguous internal
shape parameters need a qualitative role; do not give numerical ranges or scopes.
Latent initial values are shared unknowns learned on training, not fixed at zero.
An initializer with causal_map=null restores that default. A map uses first public
measurements/inputs only; its local parameters are trained and frozen for validation.
Unknown denominator/log/sqrt domains are risks, not proof of reachable failure.
Numerical failure may motivate a safer functional law or a different hypothesis.
Scientific-judge concerns are advisory and separate from runtime contract findings.
Do not manufacture an edit: use no_change with a concise reason if no useful repair
is justified. Keep lists for known unchanged variables/inputs are harmless but
optional. On retry retain the displayed provisional edits and fix the named pending
components. No unseen data, arbitrary code, full model, or explanatory prose outside
the compact hypothesis field."""


def signed_tree(tree: ast.AST) -> ast.AST:
    """Expose (-k)*x as -(k*x) for both role inference and source comparison."""

    class Signs(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, (ast.Mult, ast.Div)):
                sign = 1
                for side in ("left", "right"):
                    value = getattr(node, side)
                    if isinstance(value, ast.UnaryOp) and isinstance(
                        value.op, ast.USub
                    ):
                        setattr(node, side, value.operand)
                        sign *= -1
                if sign == -1:
                    return ast.UnaryOp(op=ast.USub(), operand=node)
            return node

    return Signs().visit(tree)


def normalized_signs(candidate: CandidateModel) -> CandidateModel:
    """Canonicalize outer-sign syntax without changing scientific functions."""
    raw = candidate.model_dump(mode="json")
    for field, expression in (("state_equations", "rhs"), ("processes", "expression")):
        for entry in raw[field]:
            parsed = RestrictedParser().parse(entry[expression], location="signs")
            entry[expression] = ast.unparse(signed_tree(parsed.tree))
    return CandidateModel.model_validate(raw)


def default_initialization(
    candidate: CandidateModel, context: ValidationContext
) -> LatentInitializationPlan:
    """Opt in explicitly: numeric latent boundaries become shared trained values."""
    model = compile_candidate(candidate, context)
    direct = model.direct_state_observation_channels
    rules = {}
    for initial in candidate.initial_conditions:
        if initial.state not in direct:
            rules[initial.state] = LatentInitializationReply(
                initial=InitialValueGuess(guess=initial.fixed_value or 0.0)
                if initial.expression is None
                else InitialCausalMap(expression=initial.expression)
            )
    return LatentInitializationPlan(rules=rules)


def _local_edit(
    parent: CandidateModel,
    edit: EquationEdit,
    scope: str,
    context: ValidationContext,
    inventory: set[str],
    shared: tuple[RevisionParameter, ...] = (),
) -> tuple[dict, list[dict]]:
    old = expressions(parent)
    if edit.kind == "preserve" and edit.component not in old:
        raise ValueError("new variable requires dynamic or algebraic kind")
    if edit.component in context.forcing_channels:
        raise ValueError("this pilot cannot redefine a supplied forcing channel")
    parsed = RestrictedParser().parse(edit.expression, location=edit.component)
    parent_parameters = {p.name: p for p in parent.parameters}
    declared = {p.name for p in edit.parameters}
    if len(declared) != len(edit.parameters):
        raise ValueError("duplicate parameter declaration")
    allowed = inventory | set(context.forcing_channels) | {context.time_symbol}
    if declared & allowed or edit.component in parent_parameters:
        raise ValueError("scientific symbol/parameter collision")
    if declared - set(parsed.symbols):
        raise ValueError("unused new parameter declaration")
    inherited = tuple(
        p for p in shared if p.name in parsed.symbols and p.name not in declared
    )
    if inherited:
        edit = edit.model_copy(update={"parameters": (*edit.parameters, *inherited)})
        declared.update(p.name for p in inherited)
    if declared & allowed:
        raise ValueError("scientific symbol/parameter collision")
    unknown = set(parsed.symbols) - allowed - set(parent_parameters) - declared
    if unknown:
        raise RevisionContractError(
            "UNAVAILABLE_SYMBOL", "undeclared symbols", symbols=sorted(unknown)
        )
    role_tree = signed_tree(parsed.tree)
    roles, audit = _effective_revision_parameter_roles(
        ComponentRevision(
            component=edit.component,
            expression=ast.unparse(role_tree),
            parameters=edit.parameters,
        ),
        role_tree,
        scientific_symbols=allowed,
        parent_parameters=parent_parameters,
    )
    new_specs = [
        parent_parameters.get(p.name)
        or ParameterSpec(name=p.name, scope="global", role=roles[p.name])
        for p in edit.parameters
    ]
    if scope == "function":
        # Use a small temporary candidate only for syntactic interaction comparison.
        payload = parent.model_dump(mode="json")
        for entry in payload["state_equations"]:
            if entry["state"] == edit.component:
                entry["rhs"] = edit.expression
        for entry in payload["processes"]:
            if entry["name"] == edit.component:
                entry["expression"] = edit.expression
        payload["parameters"] += [
            p.model_dump(mode="json")
            for p in new_specs
            if p.name not in parent_parameters
        ]
        candidate = CandidateModel.model_validate(payload)
        diff = interaction_diff(
            normalized_signs(parent), normalized_signs(candidate), edit.component
        )
        if diff["removed_interactions"] or diff["added_interactions"]:
            raise RevisionContractError(
                "EXPLICIT_MODEL_SCOPE_REQUIRED",
                "source or signed-interaction change requires scope=model",
                **diff,
            )
    return {
        "edit": edit.model_dump(mode="json"),
        "parameters": [p.model_dump(mode="json") for p in new_specs],
    }, audit


def shared_declarations(
    edits: tuple[EquationEdit, ...],
) -> tuple[RevisionParameter, ...]:
    """A new parameter may be declared once and used across a coherent edit."""
    registry = {}
    for edit in edits:
        for parameter in edit.parameters:
            prior = registry.get(parameter.name)
            if (
                prior is not None
                and prior.role is not None
                and parameter.role is not None
                and prior.role != parameter.role
            ):
                raise ValueError(
                    f"conflicting new shared declarations: {parameter.name}"
                )
            if prior is None or parameter.role is not None:
                registry[parameter.name] = parameter
    return tuple(registry.values())


def commit_action(
    parent: CandidateModel,
    plan: LatentInitializationPlan,
    action: RepairAction,
    context: ValidationContext,
    nonlinear_targets: tuple[str, ...] = (),
    memory_targets: tuple[str, ...] = (),
) -> tuple[CandidateModel, LatentInitializationPlan, dict]:
    """Validate the whole dependency closure before any canonical mutation."""
    existing = expressions(parent)
    known = set(existing) | set(context.forcing_channels)
    unknown_keep = set(action.keep) - known
    if unknown_keep:
        raise ValueError(f"unknown kept symbols: {sorted(unknown_keep)}")
    if action.scope == "no_change":
        return (
            parent,
            plan,
            {
                "status": "no_change",
                "scope": "no_change",
                "ignored_known_keep": list(action.keep),
            },
        )
    if set(action.remove) - set(existing):
        raise ValueError("cannot remove a variable absent from the parent")
    new_names = {e.component for e in action.equations} - set(existing)
    if len(new_names) > 2 or len(set(existing) | new_names) - len(action.remove) > 12:
        raise ValueError("bounded variable budget exceeded")
    inventory = (set(existing) | new_names) - set(action.remove)
    local, role_audit = {}, []
    shared = shared_declarations(action.equations)
    for edit in action.equations:
        local[edit.component], audit = _local_edit(
            parent, edit, action.scope, context, inventory, shared
        )
        role_audit.extend(audit)
    payload = parent.model_dump(mode="json")
    payload["states"] = [s for s in payload["states"] if s["name"] not in action.remove]
    payload["state_equations"] = [
        s for s in payload["state_equations"] if s["state"] not in action.remove
    ]
    payload["processes"] = [
        p for p in payload["processes"] if p["name"] not in action.remove
    ]
    parameters = {p.name: p for p in parent.parameters}
    original_kinds = {s.name: "dynamic" for s in parent.states} | {
        p.name: "algebraic" for p in parent.processes
    }
    for name, accepted in local.items():
        edit = EquationEdit.model_validate(accepted["edit"])
        kind = original_kinds.get(name) if edit.kind == "preserve" else edit.kind
        payload["state_equations"] = [
            e for e in payload["state_equations"] if e["state"] != name
        ]
        payload["processes"] = [e for e in payload["processes"] if e["name"] != name]
        old_state = next((s for s in payload["states"] if s["name"] == name), None)
        payload["states"] = [s for s in payload["states"] if s["name"] != name]
        if kind == "dynamic":
            payload["states"].append(old_state or {"name": name, "kind": "latent"})
            payload["state_equations"].append({"state": name, "rhs": edit.expression})
        else:
            old_process = next(
                (p.model_dump(mode="json") for p in parent.processes if p.name == name),
                {},
            )
            payload["processes"].append(
                {**old_process, "name": name, "expression": edit.expression}
            )
        for raw in accepted["parameters"]:
            spec = ParameterSpec.model_validate(raw)
            if spec.name in parameters and parameters[spec.name] != spec:
                raise ValueError("shared parameter role conflict")
            parameters[spec.name] = spec
    mappings = {
        m.channel: m.model_dump(mode="json") for m in parent.observation_mappings
    }
    for mapping in action.mappings:
        if mapping.channel not in context.targets:
            raise ValueError("mapping must name a public target")
        mappings[mapping.channel] = mapping.model_dump(mode="json")
    payload["observation_mappings"] = list(mappings.values())
    direct = {
        m["expression"]: m["channel"]
        for m in mappings.values()
        if m["expression"] in {s["name"] for s in payload["states"]}
    }
    old_initials = {
        i.state: i.model_dump(mode="json") for i in parent.initial_conditions
    }
    payload["initial_conditions"] = []
    for state in payload["states"]:
        name = state["name"]
        state["kind"] = "observed" if name in direct else "latent"
        payload["initial_conditions"].append(
            {"state": name, "scope": "global", "expression": direct[name]}
            if name in direct
            else old_initials.get(
                name, {"state": name, "scope": "global", "fixed_value": 0.0}
            )
        )
        if name not in direct and any(
            s.name == name and s.kind.value == "observed" for s in parent.states
        ):
            payload["initial_conditions"][-1] = {
                "state": name,
                "scope": "global",
                "fixed_value": 0.0,
            }
    used = set()
    for e in (
        *payload["state_equations"],
        *payload["processes"],
        *payload["observation_mappings"],
        *payload["initial_conditions"],
    ):
        text = e.get("rhs", e.get("expression"))
        if text:
            used.update(RestrictedParser().parse(text, location="aggregate").symbols)
    payload["parameters"] = [
        p.model_dump(mode="json") for p in parameters.values() if p.name in used
    ]
    payload.update(
        candidate_id="repair_" + content_hash(action.model_dump(mode="json"))[:16],
        parent_candidate_id=parent.candidate_id,
        change_summary="runtime-derived bounded edit",
    )
    revised = CandidateModel.model_validate(payload)
    findings = domain_findings(revised, context)
    if any(f.blocking for f in findings):
        raise RevisionContractError(
            "DOMAIN_INVALID",
            "certified domain violation",
            findings=[f.model_dump(mode="json") for f in findings if f.blocking],
        )
    model = compile_candidate(revised, context)
    memory = memory_path_findings(revised, context, memory_targets)
    if memory:
        raise RevisionContractError(
            "INTERNAL_MEMORY_PATH_ABSENT",
            "required internal memory path absent",
            findings=[f.model_dump(mode="json") for f in memory],
        )
    if nonlinear_targets and not all(
        nonlinear_target_paths(revised, nonlinear_targets).values()
    ):
        raise RevisionContractError(
            "PUBLIC_NONLINEAR_OBLIGATION_LOST",
            "required nonlinear target pathway absent",
        )
    # Minimal structural obligation, not a mechanistic scientific verdict.
    from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
        _target_reachability,
    )

    reach = _target_reachability(revised, context)
    if any(
        not set(context.external_inputs).intersection(sources)
        for sources in reach.values()
    ):
        raise RevisionContractError(
            "INPUT_PATH_ABSENT", "target lacks any declared input pathway"
        )
    latent = set(model.state_names) - set(model.direct_state_observation_channels)
    rules = {k: v for k, v in plan.rules.items() if k in latent}
    for name in latent - set(rules):
        rules[name] = LatentInitializationReply(initial=InitialValueGuess(guess=0.0))
    for edit in action.initializers:
        if edit.state not in latent:
            raise ValueError("only latent initialization can be edited")
        rules[edit.state] = LatentInitializationReply(
            initial=edit.causal_map or InitialValueGuess(guess=0.0)
        )
    new_plan = LatentInitializationPlan(rules=rules)
    apply_initialization_plan(model, new_plan)
    unchanged = model_hash(revised) == model_hash(parent) and new_plan == plan
    return (
        (parent if unchanged else revised),
        new_plan,
        {
            "status": "no_change" if unchanged else "committed",
            "scope": action.scope,
            "changed_components": [e.component for e in action.equations],
            "added_variables": sorted(new_names),
            "removed_variables": list(action.remove),
            "mapping_changes": [m.channel for m in action.mappings],
            "initializer_changes": [i.state for i in action.initializers],
            "ignored_known_keep": list(action.keep),
            "role_audit": role_audit,
            "rechecks": [
                "schema",
                "symbols",
                "algebraic_cycles",
                "targets",
                "input_paths",
                "nonlinear_obligations",
                "domains",
                "initialization",
                "compile",
            ],
            "domain_findings": [f.model_dump(mode="json") for f in findings],
        },
    )


def request_repair(
    parent: CandidateModel,
    plan: LatentInitializationPlan,
    context: ValidationContext,
    report: dict,
    public_prompt: str,
    client: StagedTopologyClient,
    directory: Path,
    round_index: int,
    nonlinear_targets: tuple[str, ...] = (),
    memory_targets: tuple[str, ...] = (),
) -> tuple[CandidateModel, LatentInitializationPlan, dict]:
    """Keep locally valid expressions in a durable draft; commit once globally valid."""
    path = directory / "transaction.json"
    identity = content_hash(
        [
            model_hash(parent),
            plan.model_dump(mode="json"),
            report,
            public_prompt,
            nonlinear_targets,
            memory_targets,
        ]
    )
    draft = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "identity": identity,
            "accepted": {},
            "pending": [],
            "attempts": [],
            "status": "pending",
        }
    )
    if draft["identity"] != identity:
        raise ValueError("transaction provenance differs")
    if draft["status"] == "exhausted":
        return parent, plan, draft
    if draft["status"] in {"committed", "no_change"}:
        return (
            CandidateModel.model_validate(draft["candidate"]),
            LatentInitializationPlan.model_validate(draft["initialization"]),
            draft,
        )
    aliases = parameter_aliases(parent)
    for attempt in range(len(draft["attempts"]), client.settings.attempts_per_step):
        request = {
            "public_task": public_prompt,
            "report": report,
            "equations": expressions(parent),
            "kinds": {s.name: "dynamic" for s in parent.states}
            | {p.name: "algebraic" for p in parent.processes},
            "parent_parameters": [
                {"name": p.name, "role": p.role.value} for p in parent.parameters
            ],
            "available_forcing": sorted(context.forcing_channels),
            "target_mappings": [
                m.model_dump(mode="json") for m in parent.observation_mappings
            ],
            "initialization_plan": plan.model_dump(mode="json"),
            "provisional_edits": draft["accepted"],
            "provisional_related_edits": draft.get("extras", {}),
            "pending_components": draft["pending"],
            "retry_findings": draft["attempts"][-1:],
        }
        record = client.call(
            system=SYSTEM_PROMPT,
            user=json.dumps(translate_names(request, aliases), sort_keys=True),
            response_model=RepairAction,
            step=f"repair_round_{round_index}",
            attempt=attempt,
        )
        diagnostics, raw = [], None
        try:
            raw = visible_response(record)
            action = RepairAction.model_validate(
                translate_names(raw, {v: k for k, v in aliases.items()})
            )
            previous_pending = set(draft["pending"])
            provided = (
                {e.component for e in action.equations}
                | set(action.keep)
                | set(action.remove)
            )
            if action.scope != "no_change":
                for name in previous_pending - provided:
                    diagnostics.append(
                        {
                            "component": name,
                            "code": "PENDING_ACTION_OMITTED",
                            "message": "repair the pending edit or explicitly keep it",
                            "details": {},
                        }
                    )
            merged = dict(draft["accepted"])
            for name in action.keep:
                merged.pop(name, None)
            for edit in action.equations:
                merged[edit.component] = edit.model_dump(mode="json")
            extras = draft.get(
                "extras", {"remove": [], "mappings": [], "initializers": []}
            )
            if action.scope == "no_change":
                merged = {}
                extras = {"remove": [], "mappings": [], "initializers": []}
            else:
                extras["remove"] = sorted(
                    (set(extras["remove"]) | set(action.remove)) - set(action.keep)
                )
                for field, key in (("mappings", "channel"), ("initializers", "state")):
                    entries = {e[key]: e for e in extras[field]}
                    entries.update(
                        {
                            getattr(e, key): e.model_dump(mode="json")
                            for e in getattr(action, field)
                        }
                    )
                    extras[field] = list(entries.values())
            draft["extras"] = extras
            inventory = set(expressions(parent)) | set(merged)
            shared = shared_declarations(
                tuple(EquationEdit.model_validate(e) for e in merged.values())
            )
            for name, payload in list(merged.items()):
                try:
                    accepted, _ = _local_edit(
                        parent,
                        EquationEdit.model_validate(payload),
                        action.scope,
                        context,
                        inventory,
                        shared,
                    )
                    merged[name] = accepted["edit"]
                except (ValueError, ModelValidationError) as exc:
                    diagnostics.append(
                        {
                            "component": name,
                            "code": getattr(exc, "code", "ACTION_CONTRACT"),
                            "message": str(exc),
                            "details": getattr(exc, "details", {}),
                        }
                    )
                    merged.pop(name)
            draft["accepted"] = merged
            draft["pending"] = [d["component"] for d in diagnostics]
            if diagnostics:
                raise ValueError("local checks left pending components")
            if action.scope == "no_change":
                # An explicit abstention abandons a draft, never commits hidden edits.
                merged = {}
            combined = RepairAction.model_validate(
                {
                    **action.model_dump(mode="json"),
                    **extras,
                    "equations": list(merged.values()),
                    "keep": [n for n in action.keep if n not in merged],
                }
            )
            revised, initial, audit = commit_action(
                parent, plan, combined, context, nonlinear_targets, memory_targets
            )
            draft.update(
                status=audit["status"],
                candidate=revised.model_dump(mode="json"),
                initialization=initial.model_dump(mode="json"),
                audit=audit,
                hypothesis=action.hypothesis,
            )
        except (ValueError, ModelValidationError) as exc:
            if not diagnostics:
                diagnostics.append(
                    {
                        "component": None,
                        "code": getattr(exc, "code", "ACTION_CONTRACT"),
                        "message": str(exc),
                        "details": getattr(exc, "details", {}),
                    }
                )
                draft["pending"] = list(draft["accepted"])
        draft["attempts"].append(
            {
                "attempt": attempt,
                "request_hash": record["request_hash"],
                "response": raw,
                "diagnostics": diagnostics,
            }
        )
        atomic_json(path, draft)
        if draft["status"] != "pending":
            return revised, initial, draft
    draft["status"] = "exhausted"
    atomic_json(path, draft)
    return parent, plan, draft
