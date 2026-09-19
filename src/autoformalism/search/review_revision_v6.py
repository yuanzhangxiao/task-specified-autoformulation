"""Whole-model scientific revisions without historical per-patch count quotas.

Opt-in adapter for the next construction campaign. Historical campaigns retain
their original schemas and identities; this module does not change their dispatch.
"""

from pydantic import Field

from autoformalism.rebuttal.staged_multiround_feedback_campaign import RevisionParameter
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import Identifier
from autoformalism.search import review_model_edits as legacy
from autoformalism.search import review_revision_v3 as compiler
from autoformalism.search import review_revision_v4 as declarations
from autoformalism.search import review_revision_v5 as previous


class ScientificRevision(previous.ScientificRevision):
    """Supply all definitions and boundaries needed for one coherent revision."""

    equations: tuple[declarations.EquationContent, ...] = ()
    remove_variables: tuple[Identifier, ...] = ()
    initializers: tuple[legacy.InitializerContent, ...] = ()
    new_parameters: tuple[RevisionParameter, ...] = ()


class CheckedContent(legacy.WholeModelEdits):
    """Keep optional evidence references separate from executable content."""

    evidence_ids: tuple[str, ...] = Field(default=(), max_length=16)


SYSTEM_PROMPT = previous.SYSTEM_PROMPT.replace(
    "At most six definitions and two new\nvariables per visit.",
    "Supply all related definitions needed for a coherent revision.",
).replace(
    "The six-definition/two-new-variable patch limits still apply to each reply.",
    "Historical per-reply equation/new-variable and twelve-total-variable limits\n"
    "are removed, along with patch removal/initializer/new-parameter list quotas.\n"
    "The model serializer's capacities are reported in serialization_capacity.\n"
    "Existing expression grammar, provider context, request and fitting budgets\n"
    "still apply. Prefer concise changes; complexity is reported explicitly.",
)


def serialization_capacity() -> dict[str, int]:
    """Expose existing model-storage capacities without inventing new patch caps."""
    return {
        name: item.max_length
        for name in ("states", "processes", "parameters", "state_equations")
        for item in CandidateModel.model_fields[name].metadata
        if getattr(item, "max_length", None) is not None
    }


def payload(bundle, packet, parameters, retry=None) -> dict:
    """Expose the opt-in policy without leaking private references or test data."""
    value = previous.payload(bundle, packet, parameters, retry)
    value["protocol"] = "scientific-content-revision-6"
    value["patch_count_limits"] = None
    value["serialization_capacity"] = serialization_capacity()
    return value


def apply_edits(bundle: dict, packet: dict, raw: dict) -> dict:
    """Preserve roles, signs, complete boundaries and atomic compiler validation."""
    reply = ScientificRevision.model_validate(raw)
    cleaned, discarded = previous._cleanup(bundle, reply)
    reply = ScientificRevision.model_validate(cleaned)
    specs, audit = declarations._resolve(bundle, reply)
    result = compiler.apply_content(
        bundle,
        packet,
        reply,
        parameter_specs=specs,
        enforce_size_limits=False,
        content_model=CheckedContent,
        enforce_patch_limits=False,
    )
    result["provenance"].update(
        protocol="scientific-content-revision-6",
        patch_count_limits=None,
        parameter_declaration_audit=audit,
        unused_new_declarations_removed=discarded,
        size_audit={
            "before": previous.model_size(bundle),
            "after": previous.model_size(result["bundle"] or bundle),
        },
    )
    if result["bundle"]:
        result["bundle"]["revision_provenance"] = result["provenance"]
    return result


def feedback(bundle, packet, parameters, raw, error) -> dict:
    """Retain actionable compiler feedback and advisory actual model counts."""
    result = previous.feedback(bundle, packet, parameters, raw, error)
    result["patch_count_limits"] = None
    result["serialization_capacity"] = serialization_capacity()
    return result
