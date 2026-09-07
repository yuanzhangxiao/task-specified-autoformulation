"""Checkpointed one-term function construction for an immutable reviewed topology."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autoformalism.construction import finalize_functional_draft
from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.llm.staged_topology import (
    StagedTopologyClient,
    atomic_json,
    visible_response,
)
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.candidate import StateKind
from autoformalism.schemas.construction import FunctionalDraft
from autoformalism.schemas.staged_functions import (
    InteractionFunctionReply,
    LatentInitialReply,
)
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search.staged_function_prompts import (
    render_interaction_function_system_prompt,
    render_interaction_function_user_prompt,
    render_latent_initial_system_prompt,
    render_latent_initial_user_prompt,
)
from autoformalism.staged_functions import (
    apply_function_reply,
    apply_initial_reply,
    initial_symbols,
)
from autoformalism.staged_topology import content_hash, lower_topology
from autoformalism.staging import topology_commitment_sha256


def run_staged_functions(
    brief: PublicScientificBrief,
    context: ValidationContext,
    source: dict[str, Any],
    client: StagedTopologyClient,
    output: Path,
) -> dict[str, Any]:
    """Assign functions and causal initializers without fitting or topology edits."""
    inventory = tuple(
        ScientificVariable.model_validate(item) for item in source["inventory"]
    )
    equations = tuple(
        EquationDefinition.model_validate(item) for item in source["equations"]
    )
    topology, aliases = lower_topology(brief, inventory, equations, context)
    if (
        not source.get("complete_topology")
        or topology.model_dump(mode="json") != source["topology"]
    ):
        raise ValueError(
            "source topology is incomplete or differs from its scientific declarations"
        )
    commitment = topology_commitment_sha256(topology)
    draft = FunctionalDraft(topology_commitment_sha256=commitment)
    events: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    registry: dict[str, str] = {}
    common = {
        "public_brief_json": brief.model_dump_json(),
        "inventory_json": json.dumps(source["inventory"]),
        "equation_sketch_json": json.dumps(source["equations"]),
    }

    def checkpoint() -> None:
        atomic_json(
            output / "progress.json",
            {
                "source_topology_result_sha256": content_hash(source),
                "draft": draft.model_dump(mode="json"),
                "accepted_functions": accepted,
                "events": events,
            },
        )

    def request(
        step: str,
        system: str,
        render: Callable[[str | None], str],
        model: type[StrictSchema],
        validate: Callable[[Any], FunctionalDraft],
    ) -> tuple[Any, FunctionalDraft]:
        diagnostic = None
        for attempt in range(client.settings.attempts_per_step):
            rejected: object = None
            record = client.call(
                system=system,
                user=render(diagnostic),
                response_model=model,
                step=step,
                attempt=attempt,
            )
            try:
                rejected = visible_response(record)
                reply = model.model_validate(rejected)
                result = validate(reply)
            except (ValueError, TypeError, KeyError, ModelValidationError) as error:
                if rejected is None:
                    choices = record.get("raw_response", {}).get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        rejected = choices[0].get("message", {}).get("content")
                diagnostic = json.dumps(
                    {"rejected_response": rejected, "error": str(error)[:6000]}
                )
                events.append(
                    {
                        "step": step,
                        "attempt": attempt,
                        "accepted": False,
                        "request_hash": record["request_hash"],
                        "error": str(error)[:6000],
                    }
                )
                checkpoint()
                continue
            events.append(
                {
                    "step": step,
                    "attempt": attempt,
                    "accepted": True,
                    "request_hash": record["request_hash"],
                }
            )
            return reply, result
        raise ValueError(f"bounded local repair exhausted for {step}")

    error = None
    expansion = None
    try:
        for equation_index, equation in enumerate(equations):
            for term_index, term in enumerate(equation.terms):
                identifier = f"term_{equation_index}_{term_index}"
                selected = {
                    "lhs": equation.name,
                    "definition": equation.definition,
                    **term.model_dump(mode="json"),
                    "assembly_template": (
                        f"d({equation.name})/dt"
                        if equation.definition == "differential"
                        else equation.name
                    )
                    + " = ... "
                    + ("+" if term.outer_sign == "add" else "-")
                    + " (FUNCTION)",
                }
                reply, draft = request(
                    f"function_{identifier}",
                    render_interaction_function_system_prompt(),
                    lambda diagnostic,
                    selected=selected: render_interaction_function_user_prompt(
                        **common,
                        selected_term_json=json.dumps(selected),
                        accepted_functions_json=json.dumps(accepted),
                        parameter_registry_json=json.dumps(registry),
                        diagnostics_json=diagnostic,
                    ),
                    InteractionFunctionReply,
                    lambda reply,
                    identifier=identifier,
                    draft=draft: apply_function_reply(
                        topology, draft, identifier, reply, context, aliases
                    ),
                )
                accepted.append(
                    {"selected_term": selected, **reply.model_dump(mode="json")}
                )
                registry.update(
                    {item.name: item.role.value for item in reply.parameters}
                )
                checkpoint()
        inverse = {value: key for key, value in aliases.items()}
        for state in topology.states:
            if state.kind is not StateKind.LATENT:
                continue
            selected_state = {
                "name": inverse.get(state.name, state.name),
                "scientific_role": state.description,
            }
            _, draft = request(
                f"initial_{state.name}",
                render_latent_initial_system_prompt(),
                lambda diagnostic,
                selected_state=selected_state: render_latent_initial_user_prompt(
                    **common,
                    selected_state_json=json.dumps(selected_state),
                    allowed_symbols_json=json.dumps(initial_symbols(context, aliases)),
                    accepted_functions_json=json.dumps(accepted),
                    diagnostics_json=diagnostic,
                ),
                LatentInitialReply,
                lambda reply, state=state, draft=draft: apply_initial_reply(
                    topology, draft, state.name, reply, context, aliases
                ),
            )
            checkpoint()
        expansion = finalize_functional_draft(topology, draft, context)
    except (ValueError, ModelValidationError) as exc:
        error = str(exc)[:6000]
    result = {
        "protocol": "scientific-staged-functions-1",
        "status": "complete" if expansion is not None else "failed",
        "error": error,
        "complete_model": expansion is not None,
        "source_topology_result_sha256": content_hash(source),
        "topology_commitment_sha256": commitment,
        "draft": draft.model_dump(mode="json"),
        "accepted_functions": accepted,
        "candidate": expansion.candidate.model_dump(mode="json") if expansion else None,
        "events": events,
        "physical_requests": len(client.records),
        "budget_charge": sum(item.get("budget_charge", 0) for item in client.records),
        "observed_total_tokens": sum(
            item.get("observed_total_tokens") or 0 for item in client.records
        ),
        "unmeasured_requests": sum(
            item.get("observed_total_tokens") is None for item in client.records
        ),
        "provider_seconds": sum(
            item.get("latency_seconds", 0) for item in client.records
        ),
        "function_generation_performed": True,
        "parameter_fitting_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    checkpoint()
    atomic_json(output / "result.json", result)
    return result
