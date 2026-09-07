"""Checkpointed one-term function construction for an immutable reviewed topology."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

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
    EquationFunctionBatchReply,
    InteractionFunctionObligation,
    InteractionFunctionReply,
    LatentInitialReply,
)
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search.staged_function_prompts import (
    render_equation_function_batch_system_prompt,
    render_equation_function_batch_user_prompt,
    render_interaction_function_system_prompt,
    render_interaction_function_user_prompt,
    render_latent_initial_system_prompt,
    render_latent_initial_user_prompt,
)
from autoformalism.staged_functions import (
    apply_equation_function_reply,
    apply_function_reply,
    apply_initial_reply,
    bind_function_reply,
    derive_interaction_function_obligation,
    has_nonlinear_source_dependence,
    initial_symbols,
    rename_expression,
)
from autoformalism.staged_topology import content_hash, lower_topology
from autoformalism.staging import topology_commitment_sha256

FunctionGenerationGranularity = Literal[
    "atomic_interaction",
    "equation_batch",
    "equation_batch_atomic_repair",
]


def run_staged_functions(
    brief: PublicScientificBrief,
    context: ValidationContext,
    source: dict[str, Any],
    client: StagedTopologyClient,
    output: Path,
    *,
    generation_granularity: FunctionGenerationGranularity = "atomic_interaction",
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
    batch_term_audits: list[dict[str, Any]] = []
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
                "batch_term_audits": batch_term_audits,
                "events": events,
            },
        )

    def request(
        step: str,
        system: str,
        render: Callable[[str | None], str],
        model: type[StrictSchema],
        validate: Callable[[Any], FunctionalDraft],
        initial_diagnostic: str | None = None,
    ) -> tuple[Any, FunctionalDraft]:
        diagnostic = initial_diagnostic
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

    if generation_granularity not in {
        "atomic_interaction",
        "equation_batch",
        "equation_batch_atomic_repair",
    }:
        raise ValueError(
            f"unsupported function generation granularity: {generation_granularity}"
        )
    error = None
    expansion = None
    try:
        for equation_index, equation in enumerate(equations):
            parameter_identity_policy = (
                "interaction_local"
                if generation_granularity == "equation_batch_atomic_repair"
                else "preserve"
            )
            selected_terms = tuple(
                _selected_term(
                    equation,
                    term,
                    parameter_identity_policy=parameter_identity_policy,
                )
                for term in equation.terms
            )
            identifiers = tuple(
                f"term_{equation_index}_{term_index}"
                for term_index in range(len(equation.terms))
            )
            if generation_granularity == "atomic_interaction":
                for identifier, selected in zip(
                    identifiers, selected_terms, strict=True
                ):
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
                        draft=draft,
                        selected=selected: apply_function_reply(
                            topology,
                            draft,
                            identifier,
                            reply,
                            context,
                            aliases,
                            _obligation(selected),
                        ),
                    )
                    accepted.append(
                        {"selected_term": selected, **reply.model_dump(mode="json")}
                    )
                    registry.update(
                        {item.name: item.role.value for item in reply.parameters}
                    )
                    checkpoint()
            elif generation_granularity == "equation_batch":
                selected_equation = {
                    "lhs": equation.name,
                    "definition": equation.definition,
                    "terms": list(selected_terms),
                }
                reply, draft = request(
                    f"equation_functions_{equation_index}",
                    render_equation_function_batch_system_prompt(),
                    lambda diagnostic,
                    selected_equation=selected_equation: (
                        render_equation_function_batch_user_prompt(
                            **common,
                            selected_equation_json=json.dumps(selected_equation),
                            accepted_functions_json=json.dumps(accepted),
                            parameter_registry_json=json.dumps(registry),
                            diagnostics_json=diagnostic,
                        )
                    ),
                    EquationFunctionBatchReply,
                    lambda reply,
                    identifiers=identifiers,
                    draft=draft: apply_equation_function_reply(
                        topology,
                        draft,
                        identifiers,
                        reply,
                        context,
                        aliases,
                    ),
                )
                for selected, function in zip(
                    selected_terms, reply.functions, strict=True
                ):
                    accepted.append(
                        {
                            "selected_term": selected,
                            **function.model_dump(mode="json"),
                        }
                    )
                    registry.update(
                        {
                            item.name: item.role.value
                            for item in function.parameters
                        }
                    )
                checkpoint()
            else:
                selected_equation = {
                    "lhs": equation.name,
                    "definition": equation.definition,
                    "terms": list(selected_terms),
                }

                def validate_batch_shape(
                    reply: EquationFunctionBatchReply,
                    *,
                    expected_count: int = len(identifiers),
                    unchanged: FunctionalDraft = draft,
                ) -> FunctionalDraft:
                    if len(reply.functions) != expected_count:
                        raise ValueError(
                            "equation function count mismatch: "
                            f"expected={expected_count}, actual={len(reply.functions)}"
                        )
                    return unchanged

                reply, _ = request(
                    f"equation_functions_{equation_index}",
                    render_equation_function_batch_system_prompt(),
                    lambda diagnostic,
                    selected_equation=selected_equation: (
                        render_equation_function_batch_user_prompt(
                            **common,
                            selected_equation_json=json.dumps(selected_equation),
                            accepted_functions_json=json.dumps(accepted),
                            parameter_registry_json=json.dumps(registry),
                            diagnostics_json=diagnostic,
                        )
                    ),
                    EquationFunctionBatchReply,
                    validate_batch_shape,
                )
                for identifier, selected, function in zip(
                    identifiers,
                    selected_terms,
                    reply.functions,
                    strict=True,
                ):
                    audit = {
                        "interaction_id": identifier,
                        "lhs": equation.name,
                        "functional_obligation": selected["functional_obligation"],
                        "batch_function": function.model_dump(mode="json"),
                        "batch_accepted": False,
                        "batch_error": None,
                        "atomic_repair_attempted": False,
                        "atomic_repair_succeeded": False,
                        "final_source": None,
                    }
                    batch_term_audits.append(audit)
                    try:
                        draft, _ = bind_function_reply(
                            topology,
                            draft,
                            identifier,
                            function,
                            context,
                            aliases,
                            _obligation(selected),
                        )
                    except (
                        ValueError,
                        TypeError,
                        KeyError,
                        ModelValidationError,
                    ) as exc:
                        audit["batch_error"] = str(exc)[:6000]
                        audit["atomic_repair_attempted"] = True
                        checkpoint()
                        diagnostic = json.dumps(
                            {
                                "repair_scope": "selected_term_only",
                                "rejected_batch_function": function.model_dump(
                                    mode="json"
                                ),
                                "deterministic_error": str(exc)[:6000],
                            }
                        )
                        _, draft = request(
                            f"atomic_repair_{identifier}",
                            render_interaction_function_system_prompt(),
                            lambda runtime_diagnostic,
                            selected=selected: render_interaction_function_user_prompt(
                                **common,
                                selected_term_json=json.dumps(selected),
                                accepted_functions_json=json.dumps(accepted),
                                parameter_registry_json=json.dumps(registry),
                                diagnostics_json=runtime_diagnostic,
                            ),
                            InteractionFunctionReply,
                            lambda repaired,
                            identifier=identifier,
                            selected=selected,
                            draft=draft: bind_function_reply(
                                topology,
                                draft,
                                identifier,
                                repaired,
                                context,
                                aliases,
                                _obligation(selected),
                            )[0],
                            initial_diagnostic=diagnostic,
                        )
                        audit["atomic_repair_succeeded"] = True
                        audit["final_source"] = "atomic_repair"
                    else:
                        audit["batch_accepted"] = True
                        audit["final_source"] = "equation_batch"
                    stored = _accepted_function_record(
                        draft,
                        identifier,
                        selected,
                        aliases,
                    )
                    accepted.append(stored)
                    registry.update(
                        {
                            item["name"]: item["role"]
                            for item in stored["parameters"]
                        }
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
        "generation_granularity": generation_granularity,
        "status": "complete" if expansion is not None else "failed",
        "error": error,
        "complete_model": expansion is not None,
        "source_topology_result_sha256": content_hash(source),
        "topology_commitment_sha256": commitment,
        "draft": draft.model_dump(mode="json"),
        "accepted_functions": accepted,
        "batch_term_audits": batch_term_audits,
        "candidate": expansion.candidate.model_dump(mode="json") if expansion else None,
        "scientific_review_facts": (
            _scientific_review_facts(brief, accepted) if expansion else None
        ),
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


def _selected_term(
    equation: EquationDefinition,
    term: Any,
    *,
    parameter_identity_policy: str = "preserve",
) -> dict[str, Any]:
    """Render one runtime-owned inner function slot."""
    obligation = derive_interaction_function_obligation(
        term.scientific_role,
        parameter_identity_policy=parameter_identity_policy,
    )
    return {
        "lhs": equation.name,
        "definition": equation.definition,
        **term.model_dump(mode="json"),
        "functional_obligation": obligation.model_dump(mode="json"),
        "assembly_template": (
            f"d({equation.name})/dt"
            if equation.definition == "differential"
            else equation.name
        )
        + " = ... "
        + ("+" if term.outer_sign == "add" else "-")
        + " (FUNCTION)",
    }


def _obligation(selected: dict[str, Any]) -> InteractionFunctionObligation:
    """Recover one typed runtime obligation from its provider-visible slot."""
    return InteractionFunctionObligation.model_validate(
        selected["functional_obligation"]
    )


def _accepted_function_record(
    draft: FunctionalDraft,
    interaction_id: str,
    selected: dict[str, Any],
    aliases: dict[str, str],
) -> dict[str, Any]:
    """Expose one accepted normalized function using public scientific names."""
    function = next(
        item
        for item in draft.interaction_functions
        if item.interaction_id == interaction_id
    )
    inverse = {value: key for key, value in aliases.items()}
    return {
        "selected_term": selected,
        "expression": rename_expression(function.expression, inverse),
        "parameters": [
            item.model_dump(mode="json") for item in function.parameters
        ],
    }


def _scientific_review_facts(
    brief: PublicScientificBrief,
    accepted: list[dict[str, Any]],
) -> dict[str, Any]:
    """Expose syntax facts and unresolved human judgments without scoring them."""
    parameter_uses: dict[str, list[str]] = {}
    parameter_lhs: dict[str, set[str]] = {}
    relaxation_terms = []
    identity_terms = []
    nonlinear_source_terms = []
    for index, item in enumerate(accepted):
        selected = item["selected_term"]
        lhs = selected["lhs"]
        label = f"{lhs}[{index}]"
        parameter_names = {parameter["name"] for parameter in item["parameters"]}
        for name in parameter_names:
            parameter_uses.setdefault(name, []).append(label)
            parameter_lhs.setdefault(name, set()).add(lhs)
        expression = item["expression"]
        tree = ast.parse(expression, mode="eval")
        sources = set(selected["sources"])
        if has_nonlinear_source_dependence(tree, sources):
            nonlinear_source_terms.append(label)
        if (
            len(sources) == 1
            and not parameter_names
            and isinstance(tree.body, ast.Name)
            and tree.body.id in sources
        ):
            identity_terms.append(label)
        role = selected["scientific_role"].lower()
        if any(
            phrase in role
            for phrase in (
                "decay",
                "relax",
                "clearance",
                "return toward baseline",
                "toward baseline",
            )
        ):
            relaxation_terms.append(
                {
                    "term": label,
                    "lhs": lhs,
                    "outer_sign": selected["outer_sign"],
                    "sources": selected["sources"],
                    "expression": expression,
                    "subtractive_outer_sign": selected["outer_sign"] == "subtract",
                }
            )
    public_text = " ".join(
        [
            brief.scientific_context,
            *(item.public_requirement for item in brief.requirements),
            *(
                positive
                for item in brief.requirements
                for positive in item.positive_requirements
            ),
        ]
    ).lower()
    requires_nonlinearity = "nonlinear" in public_text
    return {
        "schema_version": "staged-function-scientific-review-facts-1",
        "required_nonlinearity_mentioned_publicly": requires_nonlinearity,
        "source_nonlinear_term_count": len(nonlinear_source_terms),
        "source_nonlinear_terms": nonlinear_source_terms,
        "required_nonlinearity_has_syntax_evidence": (
            bool(nonlinear_source_terms) if requires_nonlinearity else None
        ),
        "relaxation_terms": relaxation_terms,
        "all_tagged_relaxation_outer_signs_subtractive": (
            all(item["subtractive_outer_sign"] for item in relaxation_terms)
            if relaxation_terms
            else None
        ),
        "identity_source_terms": identity_terms,
        "shared_parameters": {
            name: uses for name, uses in sorted(parameter_uses.items()) if len(uses) > 1
        },
        "cross_lhs_shared_parameters": {
            name: sorted(lhs_values)
            for name, lhs_values in sorted(parameter_lhs.items())
            if len(lhs_values) > 1
        },
        "human_review_required": [
            "unit_consistency",
            "scientific_justification_of_parameter_sharing",
            "scientific_direction_inside_grouped_source_laws",
            "mechanistic_adequacy_beyond_syntax",
        ],
    }
