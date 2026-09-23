"""Advisory model size, explicit state capabilities, and safe declaration cleanup."""

from __future__ import annotations

from autoformalism.expressions import (
    RestrictedParser,
    ValidationContext,
    compile_candidate,
)
from autoformalism.rebuttal.repair_transactions import normalized_signs
from autoformalism.rebuttal.revision_decision import (
    expressions,
    interaction_contract,
    parameter_aliases,
    translate_names,
)
from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
    RevisionContractError,
)
from autoformalism.schemas import CandidateModel
from autoformalism.search import review_revision_v4 as previous

ScientificRevision = previous.ScientificRevision
SYSTEM_PROMPT = (
    previous.SYSTEM_PROMPT
    + """

Revision policy: initial-construction size limits are advisory references, NOT
acceptance limits for the accumulated model. current_model_size reports actual
counts. Prefer a parsimonious scientific explanation; adding justified variables
or terms is allowed. Extra complexity receives no extra fitting or request budget.
The six-definition/two-new-variable patch limits still apply to each reply.

Read state_capabilities before proposing any new differential equation. In the
no-latent control, only states directly observed as public targets may be dynamic;
other generated variables must be algebraic. Never turn a memory ODE into an
algebraic expression just to pass: propose a scientifically meaningful allowed
alternative, or return no change. Existing algebraic variables may be revised.

Declare parameters used by initial causal maps locally in causal_map.parameters.
The runtime drops genuinely unused NEW declarations and records that cleanup.
It never invents a missing source, changes a used parameter role, or guesses an
initial causal map. Each new hidden dynamic state still needs an initializer:
use {"state": "name", "causal_map": null} for a shared train-fitted initial value,
or supply a causal map. Initial observed states remain as observed.
"""
)


def model_size(bundle: dict) -> dict:
    """Measure outer additive terms with the same grammar as historical checks."""
    candidate = CandidateModel.model_validate(
        bundle["initialization"]["base_candidate"]
    )
    normalized = normalized_signs(candidate)
    terms = {
        name: len(interaction_contract(normalized, name))
        for name in expressions(candidate)
    }
    limits = bundle["brief"]["limits"]
    counts = {
        "dynamic_states": len(candidate.states),
        "algebraic_processes": len(candidate.processes),
        "generated_variables": len(terms),
        "terms_per_equation": terms,
        "total_terms": sum(terms.values()),
        "equation_parameters": len(candidate.parameters),
        "fitted_parameters": len(bundle["candidate"]["parameters"]),
    }
    exceeded = []
    for name in ("generated_variables", "total_terms"):
        if counts[name] > limits[name]:
            exceeded.append(
                {"quantity": name, "count": counts[name], "reference": limits[name]}
            )
    exceeded.extend(
        {
            "quantity": "terms_per_equation",
            "component": name,
            "count": count,
            "reference": limits["terms_per_equation"],
        }
        for name, count in terms.items()
        if count > limits["terms_per_equation"]
    )
    return {
        "counts": counts,
        "initial_construction_references": dict(limits),
        "exceeded_references": exceeded,
        "revision_enforcement": "advisory",
    }


def state_capabilities(bundle: dict) -> dict:
    """Expose the actual control and observed-state mapping, including aliases."""
    model = compile_candidate(
        CandidateModel.model_validate(bundle["initialization"]["base_candidate"]),
        ValidationContext.model_validate(bundle["context"]),
    )
    restricted = bundle["source_task"].get("arm") == "no_latent"
    targets = set(bundle["context"]["targets"])
    observed = sorted(
        s for s, ch in model.direct_state_observation_channels.items() if ch in targets
    )
    return {
        "control": "no_persistent_latent" if restricted else "latent_states_allowed",
        "public_targets": sorted(targets),
        "existing_target_states": observed,
        "new_hidden_dynamic_states_allowed": not restricted,
        "new_algebraic_processes_allowed": True,
        "dynamic_rule": (
            "Only states directly mapped to public targets; any new "
            "target-state name needs its direct output mapping in the same patch."
        )
        if restricted
        else (
            "New hidden dynamic states need an explicit shared fitted "
            "initial value or causal map."
        ),
    }


def payload(bundle: dict, packet: dict, parameters: dict, retry=None) -> dict:
    """Separate advisory complexity from scientific requirements and hard controls."""
    value = previous.payload(bundle, packet, parameters, retry)
    value["protocol"] = "scientific-content-revision-5"
    value["public_brief"].pop("limits", None)
    value["current_model_size"] = model_size(bundle)
    value["state_capabilities"] = state_capabilities(bundle)
    return value


def _cleanup(
    bundle: dict, reply: ScientificRevision, *, output_mappings: tuple | None = None
) -> tuple[dict, list[dict]]:
    """Drop only unused new metadata; never drop a symbol used by an initializer."""
    parent = CandidateModel.model_validate(bundle["initialization"]["base_candidate"])
    inverse = {v: k for k, v in parameter_aliases(parent).items()}
    existing = {p.name for p in parent.parameters}
    definitions = expressions(parent)
    for name in reply.remove_variables:
        definitions.pop(name, None)
    definitions.update({e.component: e.expression for e in reply.equations})
    values = list(definitions.values())
    if output_mappings is None:
        values += (
            [reply.output_expression]
            if reply.output_expression is not None
            else [m.expression for m in parent.observation_mappings]
        )
    else:
        mappings = {m.channel: m.expression for m in parent.observation_mappings}
        mappings.update({m.channel: m.expression for m in output_mappings})
        values += list(mappings.values())
    # Conservative: preserve references in unchanged and replacement boundaries.
    maps = [
        r["initial"]["expression"]
        for r in bundle["initialization"]["plan"]["rules"].values()
        if r["initial"]["mode"] == "map"
    ]
    maps += [i.causal_map.expression for i in reply.initializers if i.causal_map]

    def symbols(items):
        return set().union(
            *(
                set(
                    RestrictedParser()
                    .parse(translate_names(x, inverse), location="parameter_cleanup")
                    .symbols
                )
                for x in items
            )
        )

    used, initial_used = symbols(values), symbols(maps)
    generated = set(definitions) | set(bundle["context"]["targets"])
    context = ValidationContext.model_validate(bundle["context"])
    generated |= set(context.forcing_channels) | {context.time_symbol}
    kept, discarded = [], []
    for declaration in reply.new_parameters:
        name = inverse.get(declaration.name, declaration.name)
        if name in initial_used and name not in used and name not in existing:
            raise RevisionContractError(
                "INITIALIZER_PARAMETER_SCOPE",
                "A parameter used only by an initial causal map must be declared "
                "locally in causal_map.parameters, not new_parameters.",
                parameter=name,
            )
        if name in used | initial_used | existing | generated:
            kept.append(declaration.model_dump(mode="json"))
        else:
            discarded.append(
                {
                    "parameter": name,
                    "declaration": declaration.model_dump(mode="json"),
                    "reason": "unused_in_equations_outputs_and_initializers",
                }
            )
    return {**reply.model_dump(mode="json"), "new_parameters": kept}, discarded


def apply_edits(bundle: dict, packet: dict, raw: dict) -> dict:
    """Compile unchanged scientific content without the historical size gate."""
    reply = ScientificRevision.model_validate(raw)
    cleaned, discarded = _cleanup(bundle, reply)
    result = previous.apply_edits(bundle, packet, cleaned, enforce_size_limits=False)
    result["provenance"].update(
        protocol="scientific-content-revision-5",
        unused_new_declarations_removed=discarded,
        size_audit={
            "before": model_size(bundle),
            "after": model_size(result["bundle"] or bundle),
        },
    )
    if result["bundle"]:
        result["bundle"]["revision_provenance"] = result["provenance"]
    return result


def feedback(bundle, packet, parameters, raw, error) -> dict:
    """Restate task-specific state options and current capacity on every retry."""
    result = previous.feedback(bundle, packet, parameters, raw, error)
    result.update(
        current_model_size=model_size(bundle),
        state_capabilities=state_capabilities(bundle),
    )
    if "new latent states require explicit initializer definitions" in str(error):
        result["initializer_guidance"] = (
            "For each new hidden dynamic state, add an initializer with "
            "causal_map=null to learn a shared initial value, or provide its "
            "causal map. The no-latent control forbids such states."
        )
    return result
