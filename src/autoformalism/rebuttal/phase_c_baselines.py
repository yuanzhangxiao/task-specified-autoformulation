"""Phase C public cells in the shape the external baselines already take.

Phase B baselines locate data through the legacy benchmark registry, which the
Phase C release deliberately leaves unchanged. This adapter reads one released
public cell and returns the development dataset, validation context and prompt
that the registry path produces, so every baseline method runs unchanged.

Only files listed in the release receipt under ``public/<cell>`` are opened;
evaluator diagnostics are never traversed. The roster is the one the Phase C
construction baseline uses, so baselines and construction see the same cells.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from autoformalism.baselines.core import baseline_validation_context
from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.benchmarks.phase_c_release import load_public_cell
from autoformalism.data.models import (
    DatasetSplit,
    DerivativeProvenance,
    DevelopmentDataset,
    TierRoles,
)
from autoformalism.expressions import ValidationContext
from autoformalism.fitting.public_fitting import unpack_split
from autoformalism.research.phase_c_inputs import ROSTER

PROTOCOL = "phase-c-development-2"
RELEASE_CELLS = 28
PUBLIC_FILES = ("specification.json", "proposer_prompt.txt", "train.json", "val.json")


@dataclass(frozen=True)
class _DeclaredChannels:
    """The three registry fields that ``baseline_validation_context`` reads.

    Every Phase B benchmark was registered with ``one_step_target_history``
    false, so no target history entered the forcing bounds; Phase C keeps it.
    """

    external_inputs: tuple[str, ...]
    fixed_covariates: tuple[str, ...]
    one_step_target_history: bool = False


@dataclass(frozen=True)
class PhaseCBaselineCell:
    """One public cell, ready for ``run_baseline_development``."""

    dataset: DevelopmentDataset
    context: ValidationContext
    prompt: str
    identity: dict[str, str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_release(release: Path, cells: Iterable[str]) -> str:
    """Check the publication receipt and each public file a baseline will read.

    Returns the receipt's digest, which identifies the release a model was
    developed on. A changed, missing or unlisted file is refused.
    """
    summary = read_seal(release / "summary.json")
    if (
        summary["protocol"] != PROTOCOL
        or not summary["whole_phase_c_roster_ready"]
        or summary["ready_cells"] != RELEASE_CELLS
        or summary["test_generated"]
    ):
        raise ValueError("requires the qualified 28-cell development release")
    for name in cells:
        for filename in PUBLIC_FILES:
            key = f"public/{name}/{filename}"
            listed = summary["files"].get(key)
            if listed is None or _sha256(release / key) != listed:
                raise ValueError(f"published public file changed or unlisted: {key}")
    return _sha256(release / "summary.json")


def with_estimated_derivatives(split: DatasetSplit) -> DatasetSplit:
    """Attach the derivative estimates the Phase B registry loader attached.

    Public cells carry no derivative labels. Derivative-regression baselines
    such as SINDy read them, and Phase B estimated them from the observations
    alone with this exact rule (``BenchmarkLoader._make_tidy_trajectory``), so
    no privileged information enters and the method sees what it saw before.
    The split's fingerprint is kept: the estimate is a function of its content.
    """
    trajectories = []
    for trajectory in split.trajectories:
        order = 2 if len(trajectory.time) >= 3 else 1
        observed = {**trajectory.targets, **trajectory.auxiliaries}
        trajectories.append(
            replace(
                trajectory,
                derivatives={
                    name: np.gradient(values, trajectory.time, edge_order=order)
                    for name, values in observed.items()
                },
                derivative_provenance=DerivativeProvenance.ESTIMATED,
            )
        )
    return DatasetSplit(split.name, tuple(trajectories), split.fingerprint)


def tier_of(name: str) -> str:
    """Return the observability tier in a cell name.

    Basin cells have one fixed observation design, so they carry no tier.
    """
    for tier in ("easy", "hard"):
        if f"_{tier}_" in f"{name}_":
            return tier
    return "fixed"


def load_cell(release: Path, name: str) -> PhaseCBaselineCell:
    """Load one roster cell's train, validation and prompt; never test data."""
    if name not in ROSTER:
        raise ValueError(f"{name} is outside the Phase C baseline roster")
    receipt = verify_release(release, (name,))
    directory = release / "public" / name
    spec, train, validation = load_public_cell(directory)
    prompt_path = directory / "proposer_prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    if spec["protocol"] != PROTOCOL or spec["test_released"]:
        raise ValueError("cell is not a development-only Phase C cell")
    if prompt != spec["public_prompt"] + "\n":
        raise ValueError("proposer prompt differs from the sealed public prompt")
    dataset = DevelopmentDataset(
        benchmark_id=name,
        tier=tier_of(name),
        roles=TierRoles(
            targets=tuple(spec["targets"]), auxiliaries=tuple(spec["auxiliaries"])
        ),
        train=with_estimated_derivatives(unpack_split(train)),
        validation=with_estimated_derivatives(unpack_split(validation)),
    )
    context = baseline_validation_context(
        dataset,
        _DeclaredChannels(
            external_inputs=tuple(spec["external_inputs"]),
            fixed_covariates=tuple(spec["fixed_covariates"]),
        ),
    )
    identity = {
        "protocol": PROTOCOL,
        "release_summary_sha256": receipt,
        "benchmark_id": name,
        "train": dataset.train.fingerprint,
        "validation": dataset.validation.fingerprint,
        "prompt": _sha256(prompt_path),
    }
    return PhaseCBaselineCell(dataset, context, prompt, identity)
