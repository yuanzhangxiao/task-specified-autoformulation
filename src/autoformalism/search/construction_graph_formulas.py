"""Opt-in analytic initializer construction with a sealed M43 fitting handoff.

This module performs no fitting. Executable maps receive only permitted initial
values. Independent root scope and all dependencies belong to the upstream
topology contract. Only root roles and scalar formulas are requested from the provider.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.expressions import (
    CompiledModel,
    ModelValidationError,
    ValidationContext,
    compile_candidate,
)
from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY, RestrictedParser
from autoformalism.fitting.fitter import _parameter_variable
from autoformalism.fitting.initialization import InitialMapParameter
from autoformalism.fitting.initialization_graph import (
    InitializationGraph,
    apply_initialization_graph,
)
from autoformalism.fitting.models import FitConfig
from autoformalism.llm.staged_topology import (
    StagedTopologyClient,
    atomic_json,
    visible_response,
)
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.staged_functions import FunctionParameter
from autoformalism.search.causal_initialization import _restore_attempt_costs
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
    GraphDependencyRecord,
    GraphMapDependency,
    initializer_aliases,
    validate_handoff,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-initializer-formulas-1"
START_POLICY = "existing-role-runtime-range-midpoint-1"


class GraphRootReply(StrictSchema):
    """Qualitative domain only; root scope and numerical starts are runtime-owned."""

    role: Literal["coefficient", "nonnegative_coefficient"]


class GraphMapReply(StrictSchema):
    """A scalar RHS using exactly the supplied aliases and local parameters."""

    expression: str = Field(min_length=1, max_length=4096)
    parameters: tuple[FunctionParameter, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def unique_parameters(self) -> GraphMapReply:
        """Reject duplicate local names before interpreting any formula."""
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("duplicate initializer parameter declaration")
        return self


SYSTEM_PROMPT = """Construct the selected state's initial value, not its derivative.
The initializer graph and selected dependency record are fixed upstream. For an
independent value root, return only its qualitative role. The runtime chooses
numerical starts. A shared_training root is one learned constant c reused across
trajectories; a trajectory root is an independent xi_j for each training row and
is inferred for an unseen row from its declared observation prefix. Scope cannot
be changed here. Known preparation values and observed identities are runtime-owned.
For a dependent map, return a scalar RHS and local parameter names/roles. For
example z2_j(0) = a*state_1 + b*public_1 uses state_1 = z1_j(0) and
public_1 = y_j(0). Use those supplied aliases, not raw state or channel names.
state_1 and public_1 remain distinct even if their source names coincide.
Every declared dependency alias and local parameter must occur in the formula.
Map coefficients are shared across trajectories and frozen for prediction; the
map output is calculated and is never a separate fitted coordinate. An ancestor
trajectory root can make a dependent output vary by trajectory. Do not impose
equal outputs from equal t0 readings when a trajectory root is an ancestor.
The displayed differential equations give scientific context, not a claim of
initial equilibrium. Supply no assignment, derivative, mode, scope, guesses,
fitted numbers, parameter bounds, state identifiers, or dependency edits.
Use +, -, *, /, integer-literal ** (absolute exponent at most 16), abs, exp, log,
sigmoid, sin, softplus, sqrt, tanh, min or max in the restricted scalar grammar.
No indexing, attributes, arbitrary calls, implicit equation parameters, hidden
states, trajectory identifiers, future readings, or future schedules are inputs
to these formulas. Public aliases denote permitted t0 readings/covariates/time
only; they never denote an observation prefix or a full time series. Prefix data
are used by the separate numerical root estimator, never by an initializer map.
Return only the selected schema. Revise dependencies explicitly upstream when
needed; this formula reply cannot change the selected graph.
"""


def _receipt(
    contract: GraphConstructionContract,
    handoff: GraphDependencyHandoff,
    state_aliases: Mapping[str, str] | None,
) -> dict:
    """Retain scientific declarations separately from compiler state aliases."""
    return {
        "contract": contract.model_dump(mode="json"),
        "handoff": handoff.model_dump(mode="json"),
        "state_aliases": dict(state_aliases or {}),
    }


def runtime_observed_initial_rules(
    model: CompiledModel,
    contract: GraphConstructionContract,
    state_aliases: Mapping[str, str] | None,
) -> tuple[dict[str, dict], list[dict]]:
    """Lower attested generated auxiliary t0 identities without supervision.

    The historical topology compiler aliases generated auxiliary states and
    deliberately omits auxiliary observation mappings. A reviewed same-channel
    initial identity can still be represented by an exact, parameter-free graph
    map. This does not expose auxiliary histories or create provider choices.
    """
    aliases = state_aliases or {}
    observed = {
        aliases.get(state, state): channel
        for state, channel in contract.observed_initial_channels.items()
    }
    direct = dict(model.direct_state_observation_channels)
    if any(observed.get(state) != channel for state, channel in direct.items()):
        raise ValueError("graph observed identities differ from compiled model")
    states = {state.name: state for state in model.validated.candidate.states}
    context = model.validated.context
    rules, provenance = {}, []
    for scientific_state, channel in contract.observed_initial_channels.items():
        state = aliases.get(scientific_state, scientific_state)
        if state in direct:
            continue
        if (
            scientific_state != channel
            or state == scientific_state
            or channel not in context.auxiliaries
            or channel not in contract.permitted_public_initials
            or state not in states
            or states[state].kind.value != "latent"
        ):
            raise ValueError("graph observed identities differ from compiled model")
        rules[state] = {
            "initial": {
                "mode": "map",
                "expression": "public_1",
                "parameters": [],
                "state_initials": {},
                "public_initials": {"public_1": channel},
            }
        }
        provenance.append(
            {
                "scientific_state": scientific_state,
                "compiled_state": state,
                "channel": channel,
                "identity": "runtime_observed_initial",
                "scope": "recorded_t0_only",
                "independent_parameter": False,
                "state_declaration_sha256": contract.state_declaration_sha256[
                    scientific_state
                ],
                "contract_sha256": content_hash(contract.model_dump(mode="json")),
                "public_context_sha256": contract.public_context_sha256,
                "provenance": contract.provenance,
            }
        )
    return rules, provenance


def _resolve(
    model: CompiledModel,
    context: ValidationContext,
    contract: GraphConstructionContract,
    handoff: GraphDependencyHandoff,
    state_aliases: Mapping[str, str] | None,
    *,
    scientific_context: str | None = None,
) -> GraphDependencyHandoff:
    """Validate compiled identities without inferring any missing topology rule."""
    resolved = validate_handoff(
        handoff,
        contract,
        state_names=model.state_names,
        context=context,
        state_aliases=state_aliases,
        scientific_context=scientific_context,
    )
    runtime_observed_initial_rules(model, contract, state_aliases)
    if context.fitted_initialization or context.lagged_targets:
        raise ValueError(
            "graph construction requires an unlowered free-rollout context"
        )
    if any(
        parameter.scope.value != "global"
        for parameter in model.validated.candidate.parameters
    ):
        raise ValueError("graph construction requires shared dynamics parameters")
    return resolved


def _choice_rule(
    record: GraphDependencyRecord,
    choice: dict,
    contract: GraphConstructionContract,
    state_aliases: Mapping[str, str] | None,
) -> dict:
    """Bind a narrow provider reply to the immutable graph dependency record."""
    dependency = record.rule
    if dependency.mode == "known":
        known = {
            (state_aliases or {}).get(state, state): value
            for state, value in contract.known_initials.items()
        }
        if choice:
            raise ValueError("known initializer has no provider-owned choice")
        return {"initial": known[record.state].model_dump(mode="json")}
    if dependency.mode == "value":
        reply = GraphRootReply.model_validate(choice)
        return {
            "initial": {
                "mode": "value",
                "scope": dependency.scope,
                "role": reply.role,
                "guess": 0.0,
            }
        }
    reply = GraphMapReply.model_validate(choice)
    aliases = initializer_aliases(dependency)
    parsed = RestrictedParser().parse(
        reply.expression, location=f"initial:{record.state}"
    )
    # The general parser supports legacy x(t) notation; an initializer alias is
    # a scalar, so even that compatibility notation is unavailable at this boundary.
    source = ast.parse(reply.expression, mode="eval")
    if any(
        isinstance(node, ast.Call)
        and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in APPROVED_FUNCTION_ARITY
        )
        for node in ast.walk(source)
    ):
        raise ValueError("initializer aliases are scalars, not callable histories")
    local = {parameter.name for parameter in reply.parameters}
    required = set(aliases["state_initials"]) | set(aliases["public_initials"])
    if local & required or parsed.symbols != required | local:
        raise ValueError(
            "INITIALIZER_GRAPH_DEPENDENCIES: use exactly the assigned aliases and "
            f"local parameters; missing={sorted((required | local) - parsed.symbols)}, "
            f"undeclared={sorted(parsed.symbols - required - local)}, "
            f"ambiguous={sorted(local & required)}. Revise dependencies upstream."
        )
    return {
        "initial": {
            "mode": "map",
            "expression": reply.expression,
            "parameters": [
                InitialMapParameter(
                    name=parameter.name, role=parameter.role
                ).model_dump(mode="json")
                for parameter in reply.parameters
            ],
            **aliases,
        }
    }


def _default_choice(record: GraphDependencyRecord) -> dict:
    """Supply validation-only placeholders for not-yet-constructed sibling nodes."""
    if record.rule.mode == "known":
        return {}
    if record.rule.mode == "value":
        return {"role": "coefficient"}
    aliases = initializer_aliases(record.rule)
    names = [*aliases["state_initials"], *aliases["public_initials"]]
    return {"expression": " + ".join(names) or "0", "parameters": []}


def _lower(
    model: CompiledModel,
    resolved: GraphDependencyHandoff,
    contract: GraphConstructionContract,
    choices: Mapping[str, dict],
    state_aliases: Mapping[str, str] | None,
    *,
    complete: bool,
) -> tuple[
    CompiledModel, InitializationGraph, dict[str, float], dict, dict[str, float]
]:
    """Build the actual M43 graph and choose deterministic role-policy starts."""
    expected = {
        record.state for record in resolved.records if record.rule.mode != "known"
    }
    if set(choices) - expected or (complete and set(choices) != expected):
        raise ValueError(
            "graph choices must cover exactly the requested formula states"
        )
    runtime_rules, _ = runtime_observed_initial_rules(model, contract, state_aliases)
    graph = InitializationGraph.model_validate(
        {
            "rules": {
                record.state: _choice_rule(
                    record,
                    choices.get(record.state, _default_choice(record)),
                    contract,
                    state_aliases,
                )
                for record in resolved.records
            }
            | runtime_rules
        }
    )
    compiled, _, audit = apply_initialization_graph(model, graph)
    settings = FitConfig()
    guesses = {}
    for parameter in compiled.validated.candidate.parameters:
        variable = _parameter_variable(compiled, parameter, settings)
        guesses[parameter.name] = (variable.start_lower + variable.start_upper) / 2.0
    payload = graph.model_dump(mode="json")
    for state, binding in audit["bindings"].items():
        initial = payload["rules"][state]["initial"]
        if initial["mode"] == "value":
            initial["guess"] = guesses[binding["parameters"][0]]
        elif initial["mode"] == "map":
            for parameter, name in zip(
                initial["parameters"], binding["parameters"], strict=True
            ):
                parameter["guess"] = guesses[name]
    graph = InitializationGraph.model_validate(payload)
    compiled, generated, audit = apply_initialization_graph(model, graph)
    dynamics = {name: guesses[name] for name in model.parameter_names}
    return compiled, graph, {**dynamics, **generated}, audit, dynamics


def _result_fields(
    candidate: CandidateModel,
    context: ValidationContext,
    model: CompiledModel,
    contract: GraphConstructionContract,
    handoff: GraphDependencyHandoff,
    choices: Mapping[str, dict],
    state_aliases: Mapping[str, str] | None,
) -> tuple[CompiledModel, dict]:
    """Produce only verified compiled artifacts and the unlowered M43 input."""
    resolved = _resolve(model, context, contract, handoff, state_aliases)
    compiled, graph, guesses, audit, dynamics = _lower(
        model, resolved, contract, choices, state_aliases, complete=True
    )
    problem = PrefixFitProblem(
        base_candidate=candidate,
        context=context,
        graph=graph,
        parameter_guesses=dynamics,
    )
    policy_required = (
        bool(audit["trajectory_parameter_names"]) and contract.prefix_policy is None
    )
    _, runtime_initials = runtime_observed_initial_rules(
        model, contract, state_aliases
    )
    fields = {
        "base_candidate": candidate.model_dump(mode="json"),
        "base_context": context.model_dump(mode="json"),
        "candidate": compiled.validated.candidate.model_dump(mode="json"),
        "context": compiled.validated.context.model_dump(mode="json"),
        "initialization_dependency_contract": _receipt(
            contract, handoff, state_aliases
        ),
        "graph": graph.model_dump(mode="json"),
        "parameter_guesses": guesses,
        "audit": audit,
        "prefix_fit_problem": problem.model_dump(mode="json"),
        "prefix_policy": contract.prefix_policy.model_dump(mode="json")
        if contract.prefix_policy
        else None,
        "start_policy": START_POLICY,
        "parameter_fitting_performed": False,
        "fitting_handoff": "m43_prefix_initialization_only",
        "window_policy_unresolved": policy_required,
        "fitting_policy_required": policy_required,
    }
    if runtime_initials:
        fields["runtime_observed_initials"] = runtime_initials
    return compiled, fields


def compile_graph_formula_result(
    candidate: CandidateModel,
    context: ValidationContext,
    result: dict,
    *,
    contract: GraphConstructionContract | None = None,
    handoff: GraphDependencyHandoff | None = None,
    state_aliases: Mapping[str, str] | None = None,
) -> CompiledModel:
    """Verify a sealed result by deterministic M43 lowering, never by fitting.

    The returned model is for inspection/replay. The accompanying PrefixFitProblem
    is the fitting boundary; trajectory coordinates must not enter legacy fitters.
    """
    payload = {key: value for key, value in result.items() if key != "result_sha256"}
    if (
        result.get("result_sha256") != content_hash(payload)
        or result.get("protocol") != PROTOCOL
    ):
        raise ValueError("graph formula result seal or protocol differs")
    receipt = result["initialization_dependency_contract"]
    retained_contract = GraphConstructionContract.model_validate(receipt["contract"])
    retained_handoff = GraphDependencyHandoff.model_validate(receipt["handoff"])
    aliases = receipt["state_aliases"]
    if (contract is None) != (handoff is None):
        raise ValueError("supply graph contract and handoff together")
    if contract is not None and receipt != _receipt(contract, handoff, state_aliases):
        raise ValueError("graph formula result differs from requested handoff")
    if (
        contract is None
        and state_aliases is not None
        and dict(state_aliases) != aliases
    ):
        raise ValueError("graph formula result state aliases differ")
    model = compile_candidate(candidate, context)
    compiled, expected = _result_fields(
        candidate,
        context,
        model,
        retained_contract,
        retained_handoff,
        result["choices"],
        aliases,
    )
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"graph formula artifact differs from deterministic lowering: {key}"
            )
    return compiled


def _save(path: Path, state: dict) -> None:
    """Seal accepted choices and consumed attempt receipts atomically."""
    atomic_json(path, {**state, "state_sha256": content_hash(state)})


def _prompt(
    candidate: CandidateModel,
    brief: dict,
    record: GraphDependencyRecord,
    resolved: GraphDependencyHandoff,
    contract: GraphConstructionContract,
    choices: Mapping[str, dict],
    state_aliases: Mapping[str, str] | None,
) -> str:
    """Render equations and the selected fixed dependencies in distinct namespaces."""
    dependency = record.rule
    aliases = (
        initializer_aliases(dependency)
        if isinstance(dependency, GraphMapDependency)
        else {}
    )
    return "Initializer graph formulas\n" + json.dumps(
        {
            "protocol": PROTOCOL,
            "public_brief": brief,
            "equations": [
                f"d({equation.state})/dt = {equation.rhs}"
                for equation in candidate.state_equations
            ],
            "selected_state": record.state,
            "selected_initializer_dependency": record.model_dump(mode="json"),
            "dependency_aliases": aliases,
            "initializer_graph": [
                item.model_dump(mode="json") for item in resolved.records
            ],
            "observed_initial_channels": {
                (state_aliases or {}).get(state, state): channel
                for state, channel in contract.observed_initial_channels.items()
            },
            "prefix_policy": contract.prefix_policy.model_dump(mode="json")
            if contract.prefix_policy
            else None,
            "accepted_formulas": choices,
        },
        sort_keys=True,
    )


def construct_graph_formulas(
    candidate: CandidateModel,
    context: ValidationContext,
    brief: dict,
    contract: GraphConstructionContract,
    handoff: GraphDependencyHandoff,
    client: StagedTopologyClient,
    directory: Path,
    *,
    state_aliases: Mapping[str, str] | None = None,
) -> dict:
    """Construct bounded, cached formulas for an already selected graph.

    Interrupted requests become terminal, visibly charged uncertain outcomes on
    resume. Accepted siblings and exhausted attempt limits are never reset.
    """
    from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash

    contract = GraphConstructionContract.model_validate(
        contract.model_dump(mode="json")
    )
    handoff = GraphDependencyHandoff.model_validate(handoff.model_dump(mode="json"))
    model = compile_candidate(candidate, context)
    resolved = _resolve(
        model,
        context,
        contract,
        handoff,
        state_aliases,
        scientific_context=brief.get("scientific_context"),
    )
    identity = content_hash(
        {
            "protocol": PROTOCOL,
            "start_policy": START_POLICY,
            "candidate": candidate.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "brief": brief,
            "contract": _receipt(contract, handoff, state_aliases),
            "system": SYSTEM_PROMPT,
            "schemas": [
                GraphRootReply.model_json_schema(),
                GraphMapReply.model_json_schema(),
            ],
            "client_namespace": client.namespace,
            "settings": client.settings.model_dump(mode="json"),
            "seed": client.seed,
            "request_policy": client.request_policy(),
            "runtime_sha256": runtime_source_hash(),
        }
    )
    path = directory / "state.json"
    state = {"identity": identity, "choices": {}, "attempts": [], "status": "running"}
    if path.exists():
        saved = json.loads(path.read_text())
        digest = saved.pop("state_sha256", None)
        if digest != content_hash(saved) or saved.get("identity") != identity:
            raise ValueError("graph formula checkpoint identity or digest differs")
        state = saved
    _save(path, state)
    _restore_attempt_costs(client, state["attempts"])
    if state["status"] == "uncertain":
        raise ValueError(
            "graph formula request interrupted: uncertain outcome remains charged"
        )
    if state["status"] == "complete":
        compile_graph_formula_result(
            candidate,
            context,
            state["result"],
            contract=contract,
            handoff=handoff,
            state_aliases=state_aliases,
        )
        return state["result"]
    _lower(model, resolved, contract, state["choices"], state_aliases, complete=False)
    for record in resolved.records:
        name = record.state
        if record.rule.mode == "known" or name in state["choices"]:
            continue
        attempts = [event for event in state["attempts"] if event["state"] == name]
        response_model = (
            GraphRootReply if record.rule.mode == "value" else GraphMapReply
        )
        for attempt in range(len(attempts), client.settings.attempts_per_step):
            prompt = _prompt(
                candidate,
                brief,
                record,
                resolved,
                contract,
                state["choices"],
                state_aliases,
            )
            payload = json.loads(prompt.split("\n", 1)[1])
            payload["construction_identity"] = identity
            payload["diagnostics"] = attempts[-1:]
            outcome = client.call(
                system=SYSTEM_PROMPT,
                user="Initializer graph formulas\n"
                + json.dumps(payload, sort_keys=True),
                response_model=response_model,
                step=f"graph_initial_{name}",
                attempt=attempt,
            )
            event = {
                "state": name,
                "attempt": attempt,
                "request_hash": outcome["request_hash"],
                "accepted": False,
            }
            try:
                choice = response_model.model_validate(
                    visible_response(outcome)
                ).model_dump(mode="json")
                choices = {**state["choices"], name: choice}
                _lower(
                    model, resolved, contract, choices, state_aliases, complete=False
                )
            except (ValueError, TypeError, KeyError, ModelValidationError) as exc:
                event["error"] = str(exc)[:2000]
            else:
                state["choices"] = choices
                event["accepted"] = True
            state["attempts"].append(event)
            attempts.append(event)
            if outcome["status"] == "uncertain":
                state["status"] = "uncertain"
            _save(path, state)
            if state["status"] == "uncertain":
                raise ValueError(
                    "graph formula request interrupted: uncertain outcome "
                    "remains charged"
                )
            if event["accepted"]:
                break
        if name not in state["choices"]:
            raise ValueError(f"bounded graph formula repair exhausted for {name}")
    _, fields = _result_fields(
        candidate, context, model, contract, handoff, state["choices"], state_aliases
    )
    result = {
        "protocol": PROTOCOL,
        "identity": identity,
        **fields,
        "choices": state["choices"],
        "attempts": state["attempts"],
    }
    result["result_sha256"] = content_hash(result)
    compile_graph_formula_result(
        candidate,
        context,
        result,
        contract=contract,
        handoff=handoff,
        state_aliases=state_aliases,
    )
    state.update(status="complete", result=result)
    _save(path, state)
    return result
