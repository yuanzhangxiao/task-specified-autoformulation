"""Construct transferable latent boundaries using only initial public information."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from autoformalism.expressions import (
    CompiledModel,
    ModelValidationError,
    ValidationContext,
    compile_candidate,
)
from autoformalism.fitting.initialization import (
    InitialCausalMap,
    InitialMapParameter,
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
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.staged_functions import FunctionParameter
from autoformalism.staged_topology import content_hash

PROTOCOL = "causal-initializer-construction-2"
EXPRESSION_NORMALIZATION = "initial-assignment-normalization-1"


class SharedBoundary(StrictSchema):
    """A common preparation hypothesis; the runtime fits its unknown value."""

    mode: Literal["shared_value"] = "shared_value"
    role: Literal["coefficient", "nonnegative_coefficient"] = "coefficient"


class MappedBoundary(StrictSchema):
    """One common function, potentially different values on different trajectories."""

    mode: Literal["causal_map"] = "causal_map"
    expression: str = Field(
        min_length=1,
        max_length=4096,
        description=(
            "Scalar RHS only for the selected state's initial value. "
            "No assignment, left-hand side, or derivative."
        ),
    )
    parameters: tuple[FunctionParameter, ...] = Field(default=(), max_length=32)


class InitializerChoice(StrictSchema):
    """No state IDs, fitted numbers, optimizer guesses, or redundant scope."""

    initial: Annotated[SharedBoundary | MappedBoundary, Field(discriminator="mode")]


SYSTEM_PROMPT = """Choose the initial-boundary policy for one selected latent state.
The first public target measurements and declared initial input/auxiliary values,
fixed covariates and initial time are available. Later target observations and
trajectory identifiers are unavailable for validation and testing. Input schedules
may drive the ODE when permitted but may not be used as future initializer features.
Choose shared_value only as a common-preparation hypothesis. Its unknown value is
fitted once on training. Choose causal_map to express different latent initials
as one function of the allowed initial information. Each channel symbol denotes
only its first measurement. Map coefficients are shared, fitted on training and
frozen for prediction; the evaluated initial state can differ between trajectories.
Identical initial information must produce identical deterministic initialization.
Different unexplained responses need not be resolvable by a causal point initializer.
Zero input alone does not imply zero latent state. Directly measured state initials
are runtime-bound to their own first measurements and are not part of this call.
The requested quantity is the selected state's initial value, not its derivative
or an algebraic law for every time. For causal_map, put only the scalar RHS in
expression, for example "a + b*u01" with a and b declared as parameters when
u01 is allowed. Do not include an assignment, left-hand side, or derivative.
The selected state and its differential definition are already fixed by topology;
this call chooses only its initialization policy, not its equation type or dynamics.
Use +, -, *, /, integer-literal ** (absolute exponent at most 16), abs, exp, log,
sigmoid, softplus, sqrt, tanh, min or max in the restricted scalar grammar. No
indexing, attributes, arbitrary calls, future values or other generated states.
Declare every map-local parameter once with its qualitative role, and no unused
parameters. Equation parameters are not implicitly available to the map.
Return only the schema. Numerical guesses and parameter fitting belong to runtime.
"""


def _normalize_choice(
    choice: InitializerChoice, state: str, reserved: set[str]
) -> tuple[InitializerChoice, dict | None]:
    """Strip only a matching initial-value assignment; never execute supplied text."""
    if (
        isinstance(choice.initial, SharedBoundary)
        or "=" not in choice.initial.expression
    ):
        return choice, None
    original = choice.initial.expression
    source = original.strip()
    targets = {state}
    alias = f"{state}_0"
    if alias not in reserved | {p.name for p in choice.initial.parameters}:
        targets.add(alias)
    guidance = (
        f"This call sets only the initial value of {state}. Supply only its scalar "
        "RHS in expression, without an assignment, left-hand side, or derivative. "
        "The state and equation type are already fixed. Preserve the intended "
        "RHS formula when correcting notation."
    )
    try:
        parsed = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise ValueError(
            f"INITIALIZER_ASSIGNMENT_SYNTAX at initial:{state}: {guidance}"
        ) from exc
    if len(parsed.body) == 1 and isinstance(parsed.body[0], ast.Expr):
        # Comparisons and named expressions still face the restricted RHS parser.
        return choice, None
    if (
        len(parsed.body) != 1
        or not isinstance(parsed.body[0], ast.Assign)
        or len(parsed.body[0].targets) != 1
        or not isinstance(parsed.body[0].targets[0], ast.Name)
    ):
        raise ValueError(f"INITIALIZER_ASSIGNMENT_FORM at initial:{state}: {guidance}")
    assignment = parsed.body[0]
    target = ast.get_source_segment(source, assignment.targets[0])
    if target not in targets:
        raise ValueError(
            f"INITIALIZER_ASSIGNMENT_TARGET at initial:{state}: "
            f"received {target!r}. {guidance}"
        )
    statement = ast.get_source_segment(source, assignment)
    if statement is None:  # pragma: no cover - parsed nodes have source positions
        raise ValueError("initializer assignment RHS has no source span")
    # Keep outer parentheses, including grouping that permits multiline RHS text.
    expression = statement.split("=", 1)[1].strip()
    initial = choice.initial.model_copy(update={"expression": expression})
    return choice.model_copy(update={"initial": initial}), {
        "policy": EXPRESSION_NORMALIZATION,
        "kind": "matching_initial_assignment",
        "target": target,
        "quantity": "initial_value",
        "original_expression": original,
        "normalized_expression": expression,
    }


def _rule(choice: InitializerChoice) -> LatentInitializationReply:
    """Lower scientific choices with fixed runtime-owned optimizer starts."""
    if isinstance(choice.initial, SharedBoundary):
        initial = InitialValueGuess(guess=0.0, role=choice.initial.role)
    else:
        initial = InitialCausalMap(
            expression=choice.initial.expression,
            parameters=tuple(
                InitialMapParameter(name=p.name, role=p.role)
                for p in choice.initial.parameters
            ),
        )
    return LatentInitializationReply(initial=initial)


def _save(path: Path, state: dict) -> None:
    """Checkpoint accepted scientific choices and consumed attempts atomically."""
    atomic_json(path, {**state, "state_sha256": content_hash(state)})


def _restore_attempt_costs(client: StagedTopologyClient, attempts: list[dict]) -> None:
    """Skipping accepted states must not erase consumed provider budgets on resume."""
    records = {r["request_hash"]: r for r in client.records}
    for attempt in attempts:
        key = attempt["request_hash"]
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(c not in "0123456789abcdef" for c in key)
        ):
            raise ValueError("invalid initializer request hash")
        record = json.loads((client.directory / f"{key}.json").read_text())
        if (
            record["request_hash"] != key
            or record["request"]["namespace"] != client.namespace
            or content_hash(record["request"]) != key
        ):
            raise ValueError("initializer request provenance differs")
        records[key] = record
    client.records = list(records.values())


def compile_initialization_result(
    candidate: CandidateModel,
    context: ValidationContext,
    result: dict,
) -> CompiledModel:
    """Reconstruct the artifact and reject changed or incomplete boundaries."""
    model = compile_candidate(candidate, context)
    if result["base_candidate"] != candidate.model_dump(mode="json"):
        raise ValueError("initializer base candidate differs")
    if result["base_context"] != context.model_dump(mode="json"):
        raise ValueError("initializer public context differs")
    plan = LatentInitializationPlan.model_validate(result["plan"])
    latent = set(model.state_names) - set(model.direct_state_observation_channels)
    if set(plan.rules) != latent:
        raise ValueError("initializer plan must cover every latent state exactly once")
    compiled, guesses, audit = apply_initialization_plan(model, plan)
    if (
        result["candidate"] != compiled.validated.candidate.model_dump(mode="json")
        or result["context"] != compiled.validated.context.model_dump(mode="json")
        or result["guesses"] != guesses
        or result["audit"] != audit
    ):
        raise ValueError("initializer artifact differs from deterministic lowering")
    return compiled


def construct_initializers(
    candidate: CandidateModel,
    context: ValidationContext,
    public_brief: dict,
    client: StagedTopologyClient,
    output: Path,
) -> dict:
    """Cache bounded state-local choices; failed entries retain accepted siblings."""
    from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash

    model = compile_candidate(candidate, context)
    identity = content_hash(
        {
            "protocol": PROTOCOL,
            "expression_normalization": EXPRESSION_NORMALIZATION,
            "candidate": candidate.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "brief": public_brief,
            "runtime": runtime_source_hash(),
            "system": SYSTEM_PROMPT,
            "schema": InitializerChoice.model_json_schema(),
            "client_namespace": client.namespace,
            "settings": client.settings.model_dump(mode="json"),
            "seed": client.seed,
        }
    )
    path = output / "state.json"
    state = {
        "identity": identity,
        "rules": {},
        "choices": {},
        "attempts": [],
        "status": "running",
    }
    if path.exists():
        saved = json.loads(path.read_text())
        digest = saved.pop("state_sha256", None)
        if digest != content_hash(saved) or saved.get("identity") != identity:
            raise ValueError("initializer checkpoint identity or digest differs")
        state = saved
    _save(path, state)
    _restore_attempt_costs(client, state["attempts"])
    if state["status"] == "complete":
        compile_initialization_result(candidate, context, state["result"])
        return state["result"]
    allowed = sorted(
        set(context.targets) | set(context.forcing_channels) | {context.time_symbol}
    )
    reserved = (
        set(allowed)
        | set(context.unavailable_observed_channels)
        | set(model.state_names)
        | set(model.parameter_names)
        | {p.name for p in candidate.processes}
    )
    for name in model.state_names:
        if name in model.direct_state_observation_channels or name in state["rules"]:
            continue
        attempts = [a for a in state["attempts"] if a["state"] == name]
        for attempt in range(len(attempts), client.settings.attempts_per_step):
            payload = {
                "protocol": PROTOCOL,
                "public_brief": public_brief,
                "selected_state": {
                    "name": name,
                    "rhs": next(
                        e.rhs for e in candidate.state_equations if e.state == name
                    ),
                },
                "allowed_initial_symbols": allowed,
                "observed_initial_channels": dict(
                    model.direct_state_observation_channels
                ),
                "accepted_initializers": state["choices"],
                "diagnostics": attempts[-1:] if attempts else [],
            }
            record = client.call(
                system=SYSTEM_PROMPT,
                user="Causal initialization\n" + json.dumps(payload),
                response_model=InitializerChoice,
                step=f"causal_initial_{name}",
                attempt=attempt,
            )
            event = {
                "state": name,
                "attempt": attempt,
                "request_hash": record["request_hash"],
                "accepted": False,
            }
            try:
                reply = visible_response(record)
                choice = InitializerChoice.model_validate(reply)
                choice, normalization = _normalize_choice(choice, name, reserved)
                if normalization is not None:
                    normalization["original_expression"] = reply["initial"][
                        "expression"
                    ]
                    event["normalization"] = normalization
                rule = _rule(choice)
                rules = {**state["rules"], name: rule.model_dump(mode="json")}
                plan = LatentInitializationPlan.model_validate({"rules": rules})
                apply_initialization_plan(model, plan)
            except (ValueError, TypeError, KeyError, ModelValidationError) as exc:
                event["error"] = str(exc)[:2000]
            else:
                state["rules"] = rules
                state["choices"][name] = choice.model_dump(mode="json")
                event["accepted"] = True
            state["attempts"].append(event)
            attempts.append(event)
            _save(path, state)
            if event["accepted"]:
                break
        if name not in state["rules"]:
            raise ValueError(f"bounded causal-initializer repair exhausted for {name}")
    plan = LatentInitializationPlan.model_validate({"rules": state["rules"]})
    compiled, guesses, audit = apply_initialization_plan(model, plan)
    result = {
        "protocol": PROTOCOL,
        "expression_normalization": EXPRESSION_NORMALIZATION,
        "identity": identity,
        "base_candidate": candidate.model_dump(mode="json"),
        "base_context": context.model_dump(mode="json"),
        "candidate": compiled.validated.candidate.model_dump(mode="json"),
        "context": compiled.validated.context.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "guesses": guesses,
        "audit": audit,
        "attempts": state["attempts"],
        "parameter_fitting_performed": False,
    }
    compile_initialization_result(candidate, context, result)
    state.update(status="complete", result=result)
    _save(path, state)
    return result
