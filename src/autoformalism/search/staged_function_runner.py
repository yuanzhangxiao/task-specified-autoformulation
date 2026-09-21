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
    DeterministicFunctionRepair,
    EquationFunctionBatchReply,
    InteractionFunctionObligation,
    InteractionFunctionReply,
    LatentInitialReply,
)
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    EquationTerm,
    LegacyEquationTerm,
    OuterWeightSign,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search import function_dependencies as dependencies
from autoformalism.search import process_assembly_contract as assembly
from autoformalism.search import shared_process_contract as shared
from autoformalism.search import signed_processes as signed
from autoformalism.search.causal_initialization import construct_initializers
from autoformalism.search.shared_process_guidance import system_prompt
from autoformalism.search.staged_function_prompts import (
    render_equation_function_batch_system_prompt,
    render_equation_function_batch_user_prompt,
    render_interaction_function_system_prompt,
    render_interaction_function_user_prompt,
    render_latent_initial_system_prompt,
    render_latent_initial_user_prompt,
)
from autoformalism.search.training_evidence import TrainingEvidence, evidence_brief
from autoformalism.staged_functions import (
    apply_equation_function_reply,
    apply_function_reply,
    apply_initial_reply,
    bind_function_reply,
    derive_interaction_function_obligation,
    has_nonlinear_source_dependence,
    initial_symbols,
    rename_expression,
    repair_certified_outer_gain_role,
)
from autoformalism.staged_topology import content_hash, lower_topology
from autoformalism.staging import topology_commitment_sha256

FunctionGenerationGranularity = Literal[
    "atomic_interaction",
    "equation_batch",
    "equation_batch_atomic_repair",
]
FunctionRepairPolicy = Literal["legacy", "certified_outer_gain"]


def run_staged_functions(
    brief: PublicScientificBrief,
    context: ValidationContext,
    source: dict[str, Any],
    client: StagedTopologyClient,
    output: Path,
    *,
    generation_granularity: FunctionGenerationGranularity = "atomic_interaction",
    function_repair_policy: FunctionRepairPolicy = "legacy",
    initialization_policy: Literal["legacy", "causal_training"] = "legacy",
    training_evidence: TrainingEvidence | None = None,
    shared_process_guidance: bool = False,
    dependency_policy: str = "strict",
    assembly_policy: str = "legacy",
) -> dict[str, Any]:
    """Assign functions; optional local source edits are checked transactions."""
    if dependency_policy not in {"strict", dependencies.POLICY}:
        raise ValueError("unknown function dependency policy")
    flexible = dependency_policy == dependencies.POLICY
    assembly.validate_policy(assembly_policy)
    owned_assembly = assembly_policy == assembly.POLICY
    if owned_assembly and (
        not flexible or function_repair_policy != "certified_outer_gain"
    ):
        raise ValueError(
            "assembly ownership requires local dependencies and certified role repair"
        )
    if flexible and generation_granularity != "equation_batch_atomic_repair":
        raise ValueError("local dependency repair requires hybrid construction")
    if initialization_policy not in {"legacy", "causal_training"}:
        raise ValueError("unknown function initialization policy")
    enriched = evidence_brief(brief.model_dump(mode="json"), context, training_evidence)
    if (
        shared_process_guidance
        or flexible
        or source.get("shared_process_contract")
        or initialization_policy == "causal_training"
        or training_evidence is not None
        or (output / "construction_contract.json").exists()
    ):
        contract = {
            "source": content_hash(source),
            "brief": brief.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "granularity": generation_granularity,
            "repair": function_repair_policy,
            "initialization": initialization_policy,
        }
        if shared_process_guidance:
            contract["shared_process_guidance"] = True
        if flexible:
            contract["dependency_policy"] = dependency_policy
        if owned_assembly:
            contract["assembly_policy"] = assembly_policy
        if training_evidence is not None:
            contract["training_evidence_sha256"] = training_evidence.packet_sha256
        path = output / "construction_contract.json"
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError("function construction contract differs")
        if not path.exists() and (output / "result.json").exists():
            raise ValueError("causal initialization requires a new construction root")
        atomic_json(path, contract)
    inventory = tuple(
        ScientificVariable.model_validate(item) for item in source["inventory"]
    )
    equations = tuple(
        EquationDefinition.model_validate(item) for item in source["equations"]
    )
    process_contract = source.get("shared_process_contract")
    if (process_contract or {}).get(
        "protocol"
    ) == signed.POLICY and generation_granularity != "equation_batch_atomic_repair":
        raise ValueError(
            "signed process compilation requires hybrid function construction"
        )
    process_bindings = shared.validate_contract(process_contract, equations, inventory)
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
    provider_accepted: list[dict[str, Any]] = []
    registry: dict[str, str] = {}
    effective_source = source
    dependency_revisions: list[dict] = []
    assembly_decisions: list[dict] = []
    common = {
        "public_brief_json": json.dumps(enriched)
        if training_evidence is not None
        else brief.model_dump_json(),
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
                "provider_visible_accepted_functions": provider_accepted,
                "batch_term_audits": batch_term_audits,
                "events": events,
                **(
                    {
                        "assembly_policy": assembly_policy,
                        "assembly_decisions": assembly_decisions,
                    }
                    if owned_assembly
                    else {}
                ),
                **(
                    {
                        "dependency_revisions": dependency_revisions,
                        "effective_source": effective_source,
                    }
                    if flexible
                    else {}
                ),
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
        function_request = flexible and step.startswith(
            ("equation_functions_", "atomic_repair_")
        )
        if function_request:
            system = dependencies.prompt(
                system, atomic=step.startswith("atomic_repair_")
            )
            if owned_assembly:
                system = assembly.prompt(system)
        for attempt in range(client.settings.attempts_per_step):
            rejected: object = None
            record = client.call(
                system=system,
                user=dependencies.request_context(
                    render(diagnostic), brief, context, effective_source
                )
                if function_request
                else render(diagnostic),
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
    if function_repair_policy not in {"legacy", "certified_outer_gain"}:
        raise ValueError(
            f"unsupported function repair policy: {function_repair_policy}"
        )

    def accepted_context() -> list[dict[str, Any]]:
        """Keep runtime namespaces outside prospective provider requests."""
        return (
            provider_accepted
            if function_repair_policy == "certified_outer_gain"
            else accepted
        )

    def registry_context() -> dict[str, str]:
        """Interaction-local names do not form a cross-term provider registry."""
        return {} if function_repair_policy == "certified_outer_gain" else registry

    def prepare_provider_reply(
        reply: InteractionFunctionReply,
        selected: dict[str, Any],
    ) -> tuple[InteractionFunctionReply, tuple[DeterministicFunctionRepair, ...]]:
        """Apply only the versioned, AST-certified provider-side repair."""
        shared.validate_function(reply.expression, selected.get("shared_process_use"))
        if function_repair_policy == "legacy":
            return reply, ()
        return repair_certified_outer_gain_role(
            reply,
            set(selected["sources"]),
            outer_weight_sign=selected["outer_weight_sign"],
        )

    def bind_hybrid(reply, identifier, selected):
        """Validate all changes before mutating topology, selected slot or ledger."""
        nonlocal topology, aliases, commitment, effective_source, process_contract
        decision = None
        if owned_assembly:
            normalized, decision = assembly.prepare(reply, selected)
            reply = dependencies.DependencyFunctionReply(
                **normalized.model_dump(mode="json"),
                revise_dependencies=bool(getattr(reply, "revise_dependencies", False)),
            )
        updated, event = (
            dependencies.prepare(
                brief,
                context,
                effective_source,
                identifier,
                reply,
                preserve_process_paths=owned_assembly,
            )
            if flexible
            else (effective_source, None)
        )
        next_topology, next_aliases = (topology, aliases)
        next_selected = dict(selected)
        if event:
            next_topology, next_aliases = lower_topology(
                brief,
                inventory,
                tuple(
                    EquationDefinition.model_validate(e) for e in updated["equations"]
                ),
                context,
            )
            next_selected["sources"] = event["after_sources"]
        local, repairs = prepare_provider_reply(
            dependencies.plain(reply), next_selected
        )
        next_commitment = topology_commitment_sha256(next_topology)
        next_draft, _ = bind_function_reply(
            next_topology,
            draft.model_copy(update={"topology_commitment_sha256": next_commitment}),
            identifier,
            local,
            context,
            next_aliases,
            _obligation(next_selected),
        )
        if event:
            effective_source = updated
            topology, aliases, commitment = next_topology, next_aliases, next_commitment
            process_contract = updated.get("shared_process_contract")
            selected["sources"] = event["after_sources"]
            dependency_revisions.append(event)
        if decision is not None:
            assembly_decisions.append({"interaction_id": identifier, **decision})
        return next_draft, local, repairs

    error = None
    expansion = None
    initialization = None
    try:
        for equation_index, equation in enumerate(equations):
            parameter_identity_policy = (
                "interaction_local"
                if generation_granularity == "equation_batch_atomic_repair"
                else "preserve"
            )
            selected_terms = tuple(
                {
                    **_selected_term(
                        equation,
                        term,
                        parameter_identity_policy=parameter_identity_policy,
                    ),
                    **(
                        {"deterministic_role_repair_policy": ("certified_outer_gain")}
                        if function_repair_policy == "certified_outer_gain"
                        else {}
                    ),
                }
                for term in equation.terms
            )
            selected_terms = tuple(
                shared.decorate_function_term(
                    selected,
                    process_bindings,
                    process_review=source.get("process_review")
                    if source.get("signed_process_handoff")
                    == "signed-process-handoff-2"
                    else None,
                )
                for selected in selected_terms
            )
            if owned_assembly:
                selected_terms = tuple(
                    assembly.decorate(s, process_bindings) for s in selected_terms
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
                        system_prompt(
                            render_interaction_function_system_prompt(),
                            "functions",
                            shared_process_guidance,
                        ),
                        lambda diagnostic,
                        selected=selected: render_interaction_function_user_prompt(
                            **common,
                            selected_term_json=json.dumps(selected),
                            accepted_functions_json=json.dumps(accepted_context()),
                            parameter_registry_json=json.dumps(registry_context()),
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
                            prepare_provider_reply(reply, selected)[0],
                            context,
                            aliases,
                            _obligation(selected),
                        ),
                    )
                    local_reply, _ = prepare_provider_reply(reply, selected)
                    stored = _accepted_function_record(
                        draft,
                        identifier,
                        selected,
                        aliases,
                    )
                    accepted.append(stored)
                    provider_accepted.append(
                        {
                            "selected_term": selected,
                            **local_reply.model_dump(mode="json"),
                        }
                    )
                    registry.update(
                        {item["name"]: item["role"] for item in stored["parameters"]}
                    )
                    checkpoint()
            elif generation_granularity == "equation_batch":

                def validate_shared_batch(
                    reply,
                    selected_terms=selected_terms,
                    draft=draft,
                    identifiers=identifiers,
                ):
                    if len(reply.functions) != len(selected_terms):
                        raise ValueError("equation function count mismatch")
                    for function, selected in zip(
                        reply.functions, selected_terms, strict=True
                    ):
                        shared.validate_function(
                            function.expression, selected.get("shared_process_use")
                        )
                    return apply_equation_function_reply(
                        topology, draft, identifiers, reply, context, aliases
                    )

                selected_equation = {
                    "lhs": equation.name,
                    "definition": equation.definition,
                    "terms": list(selected_terms),
                }
                reply, draft = request(
                    f"equation_functions_{equation_index}",
                    system_prompt(
                        render_equation_function_batch_system_prompt(),
                        "functions",
                        shared_process_guidance,
                    ),
                    lambda diagnostic, selected_equation=selected_equation: (
                        render_equation_function_batch_user_prompt(
                            **common,
                            selected_equation_json=json.dumps(selected_equation),
                            accepted_functions_json=json.dumps(accepted_context()),
                            parameter_registry_json=json.dumps(registry_context()),
                            diagnostics_json=diagnostic,
                        )
                    ),
                    EquationFunctionBatchReply,
                    validate_shared_batch,
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
                    provider_accepted.append(
                        {
                            "selected_term": selected,
                            **function.model_dump(mode="json"),
                        }
                    )
                    registry.update(
                        {item.name: item.role.value for item in function.parameters}
                    )
                checkpoint()
            else:
                automatic = [signed.automatic_function(s) for s in selected_terms]
                requested_terms = [
                    s
                    for s, a in zip(selected_terms, automatic, strict=True)
                    if a is None
                ]
                selected_equation = {
                    "lhs": equation.name,
                    "definition": equation.definition,
                    "terms": requested_terms,
                }

                def validate_batch_shape(
                    reply: EquationFunctionBatchReply,
                    *,
                    expected_count: int = len(requested_terms),
                    unchanged: FunctionalDraft = draft,
                ) -> FunctionalDraft:
                    if len(reply.functions) != expected_count:
                        raise ValueError(
                            "equation function count mismatch: "
                            f"expected={expected_count}, actual={len(reply.functions)}"
                        )
                    return unchanged

                reply, _ = (
                    request(
                        f"equation_functions_{equation_index}",
                        system_prompt(
                            render_equation_function_batch_system_prompt(),
                            "functions",
                            shared_process_guidance,
                        ),
                        lambda diagnostic, selected_equation=selected_equation: (
                            render_equation_function_batch_user_prompt(
                                **common,
                                selected_equation_json=json.dumps(selected_equation),
                                accepted_functions_json=json.dumps(accepted_context()),
                                parameter_registry_json=json.dumps(registry_context()),
                                diagnostics_json=diagnostic,
                            )
                        ),
                        EquationFunctionBatchReply,
                        validate_batch_shape,
                    )
                    if requested_terms
                    else (None, draft)
                )
                responses = iter(reply.functions if reply else ())
                all_functions = tuple(
                    a if a is not None else next(responses) for a in automatic
                )
                for identifier, selected, function in zip(
                    identifiers,
                    selected_terms,
                    all_functions,
                    strict=True,
                ):
                    audit = {
                        "interaction_id": identifier,
                        "lhs": equation.name,
                        "functional_obligation": selected["functional_obligation"],
                        "batch_function": function.model_dump(mode="json"),
                        "batch_accepted": False,
                        "batch_error": None,
                        "deterministic_role_repairs": [],
                        "atomic_deterministic_role_repairs": [],
                        "atomic_repair_attempted": False,
                        "atomic_repair_succeeded": False,
                        "final_source": None,
                        **(
                            {"runtime_generated": "signed_process_identity"}
                            if signed.automatic_function(selected) is not None
                            else {}
                        ),
                    }
                    batch_term_audits.append(audit)
                    local_function = function
                    deterministic_repairs: tuple[DeterministicFunctionRepair, ...] = ()
                    try:
                        draft, local_function, deterministic_repairs = bind_hybrid(
                            function, identifier, selected
                        )
                        audit["deterministic_role_repairs"] = [
                            item.model_dump(mode="json")
                            for item in deterministic_repairs
                        ]
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
                                "rejected_batch_function": local_function.model_dump(
                                    mode="json"
                                ),
                                "deterministic_error": str(exc)[:6000],
                            }
                        )
                        atomic_reply, draft = request(
                            f"atomic_repair_{identifier}",
                            system_prompt(
                                render_interaction_function_system_prompt(),
                                "functions",
                                shared_process_guidance,
                            ),
                            lambda runtime_diagnostic,
                            selected=selected: render_interaction_function_user_prompt(
                                **common,
                                selected_term_json=json.dumps(selected),
                                accepted_functions_json=json.dumps(accepted_context()),
                                parameter_registry_json=json.dumps(registry_context()),
                                diagnostics_json=runtime_diagnostic,
                            ),
                            assembly.AssemblyFunctionReply
                            if owned_assembly
                            else dependencies.DependencyFunctionReply
                            if flexible
                            else InteractionFunctionReply,
                            lambda repaired,
                            identifier=identifier,
                            selected=selected: bind_hybrid(
                                repaired, identifier, selected
                            )[0],
                            initial_diagnostic=diagnostic,
                        )
                        normalized_atomic = (
                            assembly.prepare(atomic_reply, selected)[0]
                            if owned_assembly
                            else dependencies.plain(atomic_reply)
                        )
                        repaired_reply, atomic_deterministic_repairs = (
                            prepare_provider_reply(normalized_atomic, selected)
                        )
                        audit["atomic_deterministic_role_repairs"] = [
                            item.model_dump(mode="json")
                            for item in atomic_deterministic_repairs
                        ]
                        audit["atomic_repair_succeeded"] = True
                        audit["final_source"] = "atomic_repair"
                        provider_function = repaired_reply
                    else:
                        audit["batch_accepted"] = True
                        audit["final_source"] = (
                            "equation_batch_deterministic_repair"
                            if deterministic_repairs
                            else "equation_batch"
                        )
                        provider_function = local_function
                    stored = _accepted_function_record(
                        draft,
                        identifier,
                        selected,
                        aliases,
                    )
                    accepted.append(stored)
                    provider_accepted.append(
                        {
                            "selected_term": selected,
                            **provider_function.model_dump(mode="json"),
                        }
                    )
                    registry.update(
                        {item["name"]: item["role"] for item in stored["parameters"]}
                    )
                    checkpoint()
        inverse = {value: key for key, value in aliases.items()}
        for state in topology.states:
            if state.kind is not StateKind.LATENT:
                continue
            if initialization_policy == "causal_training":
                # Temporary completeness scaffold. It is never emitted as the
                # final physical boundary: every latent state receives a plan.
                draft = apply_initial_reply(
                    topology,
                    draft,
                    state.name,
                    LatentInitialReply(initial={"fixed_value": 0.0}),
                    context,
                    aliases,
                )
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
                    accepted_functions_json=json.dumps(accepted_context()),
                    diagnostics_json=diagnostic,
                ),
                LatentInitialReply,
                lambda reply, state=state, draft=draft: apply_initial_reply(
                    topology, draft, state.name, reply, context, aliases
                ),
            )
            checkpoint()
        if flexible:
            failed_paths = [
                c
                for c in dependencies.required_checks(
                    brief,
                    tuple(
                        EquationDefinition.model_validate(e)
                        for e in effective_source["equations"]
                    ),
                )
                if not c["passed"]
            ]
            if failed_paths:
                raise ValueError("public pathways failed: " + json.dumps(failed_paths))
        expansion = finalize_functional_draft(topology, draft, context)
        if initialization_policy == "causal_training":
            initialization = construct_initializers(
                expansion.candidate,
                context,
                enriched,
                client,
                output / "initialization",
            )
    except (ValueError, ModelValidationError) as exc:
        error = str(exc)[:6000]
        expansion = None
    result = {
        "protocol": "scientific-staged-functions-2"
        if initialization_policy == "causal_training"
        else "scientific-staged-functions-1",
        "generation_granularity": generation_granularity,
        "function_repair_policy": function_repair_policy,
        "status": "complete" if expansion is not None else "failed",
        "error": error,
        "complete_model": expansion is not None,
        "source_topology_result_sha256": content_hash(source),
        "topology_commitment_sha256": commitment,
        "draft": draft.model_dump(mode="json"),
        "accepted_functions": accepted,
        "provider_visible_accepted_functions": provider_accepted,
        "batch_term_audits": batch_term_audits,
        "candidate": (
            initialization["candidate"]
            if initialization
            else expansion.candidate.model_dump(mode="json")
        )
        if expansion
        else None,
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
    if initialization_policy == "causal_training":
        result["initialization_policy"] = initialization_policy
        result["initialization"] = initialization
    if process_contract:
        result["shared_process_contract"] = process_contract
    if owned_assembly:
        result.update(
            assembly_policy=assembly_policy, assembly_decisions=assembly_decisions
        )
    if flexible:
        result.update(
            dependency_policy=dependency_policy,
            dependency_revisions=dependency_revisions,
            effective_source=effective_source,
            equation_connectivity=dependencies.connectivity(result["candidate"]),
        )
    checkpoint()
    atomic_json(output / "result.json", result)
    return result


def _selected_term(
    equation: EquationDefinition,
    term: EquationTerm | LegacyEquationTerm,
    *,
    parameter_identity_policy: str = "preserve",
) -> dict[str, Any]:
    """Render one runtime-owned inner function slot."""
    obligation = derive_interaction_function_obligation(
        term.scientific_role,
        parameter_identity_policy=parameter_identity_policy,
    )
    outer_weight_sign = _outer_weight_sign(term)
    slot = {
        OuterWeightSign.POSITIVE: "+ (FUNCTION)",
        OuterWeightSign.NEGATIVE: "- (FUNCTION)",
        OuterWeightSign.UNRESTRICTED: "+ (SIGNED_FUNCTION)",
    }[outer_weight_sign]
    selected = {
        "lhs": equation.name,
        "definition": equation.definition,
        **term.model_dump(mode="json"),
        "outer_weight_sign": outer_weight_sign.value,
        "functional_obligation": obligation.model_dump(mode="json"),
        "assembly_template": (
            f"d({equation.name})/dt"
            if equation.definition == "differential"
            else equation.name
        )
        + " = ... "
        + slot,
    }
    selected.pop("outer_sign", None)
    return selected


def _outer_weight_sign(
    term: EquationTerm | LegacyEquationTerm,
) -> OuterWeightSign:
    """Normalize new proposer-owned signs and replay-only legacy signs."""
    if isinstance(term, LegacyEquationTerm):
        return (
            OuterWeightSign.POSITIVE
            if term.outer_sign == "add"
            else OuterWeightSign.NEGATIVE
        )
    return term.outer_weight_sign


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
            {"name": item.name, "role": item.role.value} for item in function.parameters
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
                    "outer_weight_sign": selected["outer_weight_sign"],
                    "sources": selected["sources"],
                    "expression": expression,
                    "negative_outer_weight_sign": (
                        selected["outer_weight_sign"] == "negative"
                    ),
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
        "all_tagged_relaxation_outer_weight_signs_negative": (
            all(item["negative_outer_weight_sign"] for item in relaxation_terms)
            if relaxation_terms
            else None
        ),
        # Replay-only name retained for already frozen function campaigns.
        "all_tagged_relaxation_outer_signs_subtractive": (
            all(item["negative_outer_weight_sign"] for item in relaxation_terms)
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
