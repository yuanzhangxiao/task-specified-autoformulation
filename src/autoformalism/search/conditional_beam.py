"""Deterministic beam state for topology-conditioned functional children."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.construction import (
    assess_functional_compatibility,
    finalize_functional_draft,
    finalize_topology_draft,
    functional_draft_sha256,
    select_conditional_beam,
    topology_draft_sha256,
)
from autoformalism.expressions import ValidationContext
from autoformalism.schemas import (
    ConditionalBeamEntry,
    FunctionalCompatibilityReport,
    FunctionalDraft,
    TopologyCandidate,
    TopologyDraft,
)
from autoformalism.schemas.base import FiniteFloat, NonEmptyText, StrictSchema
from autoformalism.schemas.staged import Sha256Digest
from autoformalism.staging import StagedCandidateExpansion

FunctionalBranchStatus = Literal[
    "incomplete",
    "rejected_incompatible",
    "scored",
    "scoring_failed",
]


class TopologyBranchRecord(StrictSchema):
    """One complete topology retained independently of its function children."""

    topology_sha256: Sha256Digest
    draft: TopologyDraft
    topology: TopologyCandidate
    functional_attempt_count: int = Field(default=0, ge=0)


class FunctionalBranchRecord(StrictSchema):
    """One localized function child under exactly one topology."""

    topology_sha256: Sha256Digest
    functional_sha256: Sha256Digest
    draft: FunctionalDraft
    compatibility: FunctionalCompatibilityReport
    status: FunctionalBranchStatus
    expansion: StagedCandidateExpansion | None = None
    score: FiniteFloat | None = None
    scoring_error: NonEmptyText | None = None


class ConditionalBeamState(StrictSchema):
    """Checkpointable coupled topology/function beam state."""

    schema_version: Literal["conditional-construction-beam-state-1"] = (
        "conditional-construction-beam-state-1"
    )
    revision: int = Field(default=0, ge=0)
    topology_branches: tuple[TopologyBranchRecord, ...] = ()
    functional_branches: tuple[FunctionalBranchRecord, ...] = ()
    selected: tuple[ConditionalBeamEntry, ...] = ()


ScoreExpansion = Callable[[StagedCandidateExpansion], float]


class ConditionalConstructionBeam:
    """Maintain topology-conditioned candidates with deterministic resume."""

    def __init__(
        self,
        *,
        checkpoint_path: Path | None = None,
        maximum_functions_per_topology: int,
    ) -> None:
        if maximum_functions_per_topology < 1:
            raise ValueError("maximum_functions_per_topology must be at least one")
        self._checkpoint_path = checkpoint_path
        self._maximum_functions_per_topology = maximum_functions_per_topology
        self._state = (
            _load_state(checkpoint_path)
            if checkpoint_path is not None and checkpoint_path.exists()
            else ConditionalBeamState()
        )

    @property
    def state(self) -> ConditionalBeamState:
        """Return the current immutable beam state."""
        return self._state

    def register_topology(
        self,
        draft: TopologyDraft,
        context: ValidationContext,
    ) -> tuple[TopologyBranchRecord, bool]:
        """Register one complete topology, collapsing canonical duplicates."""
        topology = finalize_topology_draft(draft, context)
        topology_sha256 = topology_draft_sha256(draft)
        for branch in self._state.topology_branches:
            if branch.topology_sha256 == topology_sha256:
                return branch, False
        branch = TopologyBranchRecord(
            topology_sha256=topology_sha256,
            draft=draft,
            topology=topology,
        )
        self._replace_state(
            topology_branches=tuple(
                sorted(
                    (*self._state.topology_branches, branch),
                    key=lambda item: item.topology_sha256,
                )
            )
        )
        return branch, True

    def evaluate_function_child(
        self,
        *,
        topology_sha256: str,
        draft: FunctionalDraft,
        context: ValidationContext,
        score_expansion: ScoreExpansion,
    ) -> tuple[FunctionalBranchRecord, bool]:
        """Validate and score one function child without pruning its topology."""
        topology_branch = self._topology(topology_sha256)
        functional_sha256 = functional_draft_sha256(draft)
        for child in self._state.functional_branches:
            if child.functional_sha256 == functional_sha256:
                if child.topology_sha256 != topology_sha256:
                    raise ValueError(
                        "functional hash collision across topology commitments"
                    )
                return child, False
        if topology_branch.functional_attempt_count >= (
            self._maximum_functions_per_topology
        ):
            raise ValueError(
                "functional expansion allowance exhausted for topology: "
                f"{topology_sha256}"
            )

        compatibility = assess_functional_compatibility(
            topology_branch.topology, draft
        )
        expansion: StagedCandidateExpansion | None = None
        score: float | None = None
        error: str | None = None
        if compatibility.status == "incompatible":
            status: FunctionalBranchStatus = "rejected_incompatible"
        elif compatibility.status == "incomplete":
            status = "incomplete"
        else:
            expansion = finalize_functional_draft(
                topology_branch.topology, draft, context
            )
            try:
                score = float(score_expansion(expansion))
                if not (-float("inf") < score < float("inf")):
                    raise ValueError("scorer returned a nonfinite score")
            except Exception as exc:
                status = "scoring_failed"
                error = f"{type(exc).__name__}: {exc}"
                score = None
            else:
                status = "scored"

        child = FunctionalBranchRecord(
            topology_sha256=topology_sha256,
            functional_sha256=functional_sha256,
            draft=draft,
            compatibility=compatibility,
            status=status,
            expansion=expansion,
            score=score,
            scoring_error=error,
        )
        updated_topologies = tuple(
            item.model_copy(
                update={
                    "functional_attempt_count": item.functional_attempt_count + 1
                }
            )
            if item.topology_sha256 == topology_sha256
            else item
            for item in self._state.topology_branches
        )
        self._replace_state(
            topology_branches=updated_topologies,
            functional_branches=tuple(
                sorted(
                    (*self._state.functional_branches, child),
                    key=lambda item: (
                        item.topology_sha256,
                        item.functional_sha256,
                    ),
                )
            ),
        )
        return child, True

    def select(
        self,
        *,
        beam_size: int,
        require_attempt_per_topology: bool = True,
        require_exhaustion_without_scored_child: bool = True,
    ) -> tuple[ConditionalBeamEntry, ...]:
        """Select scored children with topology diversity and bounded fanout."""
        if require_attempt_per_topology:
            unattempted = sorted(
                item.topology_sha256
                for item in self._state.topology_branches
                if item.functional_attempt_count == 0
            )
            if unattempted:
                raise ValueError(
                    "cannot prune topologies before one functional attempt: "
                    f"{unattempted}"
                )
        if require_exhaustion_without_scored_child:
            scored_topologies = {
                item.topology_sha256
                for item in self._state.functional_branches
                if item.status == "scored"
            }
            unresolved = sorted(
                item.topology_sha256
                for item in self._state.topology_branches
                if item.topology_sha256 not in scored_topologies
                and item.functional_attempt_count
                < self._maximum_functions_per_topology
            )
            if unresolved:
                raise ValueError(
                    "cannot prune unscored topologies while function allowance "
                    f"remains: {unresolved}"
                )
        entries = tuple(
            ConditionalBeamEntry(
                topology_sha256=item.topology_sha256,
                functional_sha256=item.functional_sha256,
                score=item.score,
            )
            for item in self._state.functional_branches
            if item.status == "scored" and item.score is not None
        )
        selected = select_conditional_beam(
            entries,
            beam_size=beam_size,
            maximum_functions_per_topology=(
                self._maximum_functions_per_topology
            ),
        )
        self._replace_state(selected=selected)
        return selected

    def remaining_function_allowance(self, topology_sha256: str) -> int:
        """Return the bounded number of further children allowed."""
        branch = self._topology(topology_sha256)
        return max(
            0,
            self._maximum_functions_per_topology
            - branch.functional_attempt_count,
        )

    def selected_records(self) -> tuple[FunctionalBranchRecord, ...]:
        """Resolve selected identities to their full checkpointed records."""
        selected = {
            (item.topology_sha256, item.functional_sha256)
            for item in self._state.selected
        }
        return tuple(
            item
            for item in self._state.functional_branches
            if (item.topology_sha256, item.functional_sha256) in selected
        )

    def _topology(self, topology_sha256: str) -> TopologyBranchRecord:
        for branch in self._state.topology_branches:
            if branch.topology_sha256 == topology_sha256:
                return branch
        raise ValueError(f"unknown topology branch: {topology_sha256}")

    def _replace_state(self, **updates: object) -> None:
        self._state = self._state.model_copy(
            update={"revision": self._state.revision + 1, **updates}
        )
        if self._checkpoint_path is not None:
            _write_state(self._checkpoint_path, self._state)


def _write_state(path: Path, state: ConditionalBeamState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        f"{state.model_dump_json()}\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_state(path: Path) -> ConditionalBeamState:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid conditional beam checkpoint {path}: {exc}") from exc
    return ConditionalBeamState.model_validate(payload)


def scored_entries(
    records: Iterable[FunctionalBranchRecord],
) -> tuple[ConditionalBeamEntry, ...]:
    """Expose scored identities for audit or an external selection policy."""
    return tuple(
        ConditionalBeamEntry(
            topology_sha256=item.topology_sha256,
            functional_sha256=item.functional_sha256,
            score=item.score,
        )
        for item in records
        if item.status == "scored" and item.score is not None
    )


__all__ = [
    "ConditionalBeamState",
    "ConditionalConstructionBeam",
    "FunctionalBranchRecord",
    "TopologyBranchRecord",
    "scored_entries",
]
