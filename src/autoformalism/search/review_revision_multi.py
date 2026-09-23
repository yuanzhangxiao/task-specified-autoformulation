"""Joint-output content patches and unfitted public-contract repairs."""

from __future__ import annotations

from pydantic import Field

from autoformalism.rebuttal.repair_transactions import MappingEdit
from autoformalism.rebuttal.revision_decision import parameter_aliases, translate_names
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.search import review_model_edits as legacy
from autoformalism.search import review_revision_v3 as v3
from autoformalism.search import review_revision_v4 as v4
from autoformalism.search import review_revision_v5 as v5
from autoformalism.staged_topology import content_hash

POLICY = "scientific-content-multi-revision-1"


class ScientificRevision(StrictSchema):
    """Explicit channel-addressed mappings; omitted mappings remain unchanged."""

    hypothesis: str = Field(min_length=1, max_length=3000)
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=16)
    equations: tuple[v4.EquationContent, ...] = Field(default=(), max_length=6)
    remove_variables: tuple[Identifier, ...] = Field(default=(), max_length=2)
    output_mappings: tuple[MappingEdit, ...] = Field(default=(), max_length=3)
    initializers: tuple[legacy.InitializerContent, ...] = Field(
        default=(), max_length=4
    )
    new_parameters: tuple[v4.RevisionParameter, ...] = Field(default=(), max_length=32)

    @property
    def output_expression(self) -> None:
        """Internal compatibility for shared helpers, absent from the wire schema."""
        return None


SYSTEM_PROMPT = (
    v5.SYSTEM_PROMPT.replace(
        "remove_variables, output_expression,", "remove_variables, output_mappings,"
    ).replace(
        "output_expression optionally replaces the expression for the one "
        "PUBLIC target\nshown in editable_objects. It does not define an internal "
        "variable; use equations\n"
        "for that. Omit it to retain the output mapping.",
        "output_mappings is a list of {channel, expression} replacements for declared\n"
        "PUBLIC target channels. Omitted mappings stay unchanged. "
        "Mapping edits do not\ndefine internal variables: use equations for those. "
        "All targets remain required.",
    )
    + """

Read public_target_contract before editing: it lists every enforced output mapping,
composition dependency and representation requirement with its public provenance.
For a representation repair, explicitly choose kind=dynamic or algebraic and supply
the scientifically intended RHS. The runtime never silently reinterprets a derivative.
When stage is construction_repair there are NO fitted residuals or fitted parameters.
Use the public task, descriptive training evidence and failed public predicates;
do not invent numerical results or cite evidence_refs absent from evidence_catalog.
An empty patch cannot resolve a failing public predicate. Unrelated valid equations,
outputs and initializers should remain unchanged. Before any fit, all public checks
must pass. Scientific claims remain unverified even when their citations resolve.
"""
)


def payload(
    bundle: dict, packet: dict | None, parameters: dict | None, retry=None
) -> dict:
    """Keep unfitted drafts distinct from training-residual-driven revisions."""
    if packet is not None:
        value = v5.payload(bundle, packet, parameters, retry)
    else:
        base = CandidateModel.model_validate(bundle["initialization"]["base_candidate"])
        value = translate_names(
            {
                "public_brief": bundle["brief"],
                "model": {
                    "equations": [
                        {"component": e.state, "kind": "dynamic", "expression": e.rhs}
                        for e in base.state_equations
                    ]
                    + [
                        {
                            "component": p.name,
                            "kind": "algebraic",
                            "expression": p.expression,
                        }
                        for p in base.processes
                    ],
                    "parameters": [p.model_dump(mode="json") for p in base.parameters],
                    "outputs": [
                        m.model_dump(mode="json") for m in base.observation_mappings
                    ],
                },
                "initialization_plan": bundle["initialization"]["plan"],
                "evidence_catalog": [],
                "training_evidence": None,
                "retained_fitted_parameters": None,
                "retry_feedback": retry,
                "editable_objects": {
                    "variables": {
                        **{s.name: "dynamic" for s in base.states},
                        **{p.name: "algebraic" for p in base.processes},
                    },
                    "latent_initializers": sorted(
                        bundle["initialization"]["plan"]["rules"]
                    ),
                },
                "current_model_size": v5.model_size(bundle),
                "state_capabilities": v5.state_capabilities(bundle),
            },
            parameter_aliases(base),
        )
    value["protocol"] = POLICY
    value["stage"] = (
        "construction_repair" if packet is None else "fitted_model_revision"
    )
    value["editable_objects"].pop("public_output", None)
    value["editable_objects"]["public_outputs"] = list(bundle["context"]["targets"])
    return value


def apply_edits(
    bundle: dict,
    packet: dict | None,
    raw: dict,
    *,
    reference_catalog=None,
    reject_existing_role_conflicts: bool = False,
) -> dict:
    """Compile all outputs atomically; unresolved citations never become evidence."""
    reply = ScientificRevision.model_validate(raw)
    mappings, duplicates = v3._unique(reply.output_mappings, "channel")
    cleaned, discarded = v5._cleanup(bundle, reply, output_mappings=tuple(mappings))
    reply = ScientificRevision.model_validate(cleaned)
    specs, audit = v4._resolve(
        bundle,
        reply,
        output_mappings=tuple(mappings),
        reject_existing_role_conflicts=reject_existing_role_conflicts,
    )
    if reject_existing_role_conflicts:
        audit["existing_parameter_policy"] = "reject-conflicting-existing-role-1"
    # This binding is compiler provenance only, never a fabricated ResidualEvidence.
    binding = (
        packet
        if packet is not None
        else {"candidate_sha256": content_hash(bundle["candidate"])}
    )
    result = v3.apply_content(
        bundle,
        binding,
        reply,
        parameter_specs=specs,
        enforce_size_limits=False,
        output_mappings=[m.model_dump(mode="json") for m in mappings],
        reference_catalog={} if packet is None else reference_catalog,
    )
    result["provenance"].update(
        protocol=POLICY,
        evidence_stage="unfitted_public_contract"
        if packet is None
        else "training_residuals",
        parameter_declaration_audit=audit,
        unused_new_declarations_removed=discarded,
        identical_output_mappings=duplicates,
        size_audit={
            "before": v5.model_size(bundle),
            "after": v5.model_size(result["bundle"] or bundle),
        },
    )
    if result["bundle"]:
        result["bundle"]["revision_provenance"] = result["provenance"]
    return result


def feedback(bundle, packet, parameters, raw, error) -> dict:
    """Report executable failures separately from advisory reference quality."""
    return {
        "code": getattr(error, "code", "SCIENTIFIC_CONTENT_CONTRACT"),
        "message": str(error)[:6000],
        "rejected_patch": raw,
        "incumbent_unchanged": True,
        "editable_objects": payload(bundle, packet, parameters)["editable_objects"],
        "available_evidence_refs": list(v3.references(packet))
        if packet is not None
        else [],
        "details": getattr(error, "details", {}),
        "state_capabilities": v5.state_capabilities(bundle),
    }
