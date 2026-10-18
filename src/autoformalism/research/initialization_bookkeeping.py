"""Opt-in initializer bookkeeping after a validated topology transaction.

Callers select this behavior only for the V6 policy. These helpers neither
assemble topology edits nor validate initializer choices; ordinary transaction
validation must still succeed before anything is committed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from autoformalism.research import repair_stagnation
from autoformalism.staged_topology import content_hash

if TYPE_CHECKING:
    from autoformalism.search.construction_initialized_repair import Draft, DraftPatch
    from autoformalism.search.topology_initialization import (
        InitializerDependencyHandoff,
    )


def cleanup_inherited_handoff(
    before: Draft,
    candidate: Draft,
    retained: InitializerDependencyHandoff,
    patch: DraftPatch,
) -> InitializerDependencyHandoff:
    """Drop inherited records only for explicit, realized state removals/retypes.

    A replacement handoff remains untouched for normal validation, including
    stale or otherwise invalid choices. Unrelated pre-existing stale records
    also remain visible to validation; this is not a general handoff repair.
    """
    if patch.initialization is not None:
        return patch.initialization
    explicitly_changed = set(patch.remove_variables) | {
        variable.name
        for variable in patch.variables
        if variable.definition != "differential"
    }
    before_states = {
        variable.name
        for variable in before.variables
        if variable.definition == "differential"
    }
    candidate_states = {
        variable.name
        for variable in candidate.variables
        if variable.definition == "differential"
    }
    removed_states = explicitly_changed & (before_states - candidate_states)
    if not removed_states:
        return retained
    return retained.model_copy(
        update={
            "records": tuple(
                record
                for record in retained.records
                if record.state not in removed_states
            )
        }
    )


def rejected_initializer_attempt_signature(
    draft: Draft,
    check: dict,
    report: dict,
    candidate: Draft,
    *,
    retained: dict,
    proposed: object,
    error: str,
) -> str:
    """Identify a rejected initializer edit by its assembled topology effect.

    Use only after the topology candidate has passed assembly, normalization,
    and public-binding validation, and initializer validation has failed. Earlier
    failures retain the caller's ordinary rejected-payload fingerprint. Ignoring
    redundant topology restatements does not discard different proposed handoffs.
    """
    if not error or error.startswith("no draft change;"):
        raise ValueError("a rejected initializer validation error is required")
    base = content_hash(
        [
            repair_stagnation.model_signature(draft, check, report),
            repair_stagnation.scientific(candidate.model_dump(mode="json")),
        ]
    )
    return repair_stagnation.initializer_attempt_signature(
        base, retained, proposed, error=error
    )
