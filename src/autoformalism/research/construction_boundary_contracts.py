"""Reviewed public external-loss obligations for the next construction study."""

from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search.construction_evidence import BoundaryContract
from autoformalism.staged_topology import content_hash

from . import construction_obligations as public


def reviewed_contract(case: str, brief: PublicScientificBrief) -> BoundaryContract:
    """Select reviewed public wording; never infer mechanisms from hidden truth."""
    if case not in {
        *public.DALLA,
        public.CSTR,
        public.ALIEN,
        public.COUPLED,
        public.INDEPENDENT,
    }:
        raise ValueError(f"no reviewed public boundary profile for {case}")
    obligations = []
    if case in {public.COUPLED, public.INDEPENDENT}:
        obligations.append(
            {
                "id": "downstream_external_loss",
                "target": "h_down",
                "public_quote": (
                    "Represent threshold-dependent overflow and the "
                    "downstream discharge."
                    if case == public.COUPLED
                    else "Represent downstream accumulation, a threshold-dependent "
                    "outlet, "
                    "nonnegative storage and its local water balance."
                ),
                "interpretation": (
                    "Identify a distinct downstream discharge leaving the modeled "
                    "system, "
                    "in the depth balance or equivalent storage coordinates. Internal "
                    "transfer between modeled basins does not realize this external "
                    "loss. A local process is sufficient; no named shared process, "
                    "outlet formula "
                    "or outer sign is prescribed. Function-level direction, threshold "
                    "behavior and quantitative balance remain unverified."
                ),
            }
        )
    contract = BoundaryContract(
        source_brief_sha256=content_hash(brief.model_dump(mode="json")),
        obligations=obligations,
    )
    contract.validate_public(brief)
    return contract
