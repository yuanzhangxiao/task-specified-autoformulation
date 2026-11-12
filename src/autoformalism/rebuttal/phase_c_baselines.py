"""Phase C public cells in the shape the external baselines already take.

Phase B baselines locate data through the legacy benchmark registry, which the
Phase C release deliberately leaves unchanged. This adapter reads one released
public cell and returns the development dataset, validation context and prompt
that the registry path produces, so every baseline method runs unchanged.

Only files listed in the release receipt under ``public/<cell>`` are opened;
evaluator diagnostics are never traversed. The roster is the one the Phase C
construction baseline uses, so baselines and construction see the same cells.

The T2-hard measurement-noise study publishes its four acquisitions in a
release of their own, identified by ``release.json``; such a cell is read with
the same rules, from that release only.
"""

from __future__ import annotations

import hashlib
import json
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
#: Cells a baseline may run on beyond the eight-cell roster, each for one
#: study. CSTR-hard serves the reactor intervention study, which scores models
#: on that cell; a plan for it keeps every other setting of its frozen source.
INTERVENTION_CELLS = frozenset(
    {"phase_c_cstr_controlled_reactor_mechanism_canonical_named_hard_reset_v1"}
)
#: The release of the T2-hard measurement-noise study.
NOISE_PROTOCOL = "t2-noise-public-1"
#: Its four acquisitions: the T2-hard cell's sixteen training and four
#: validation trajectories with 0, 1, 5 and 10% measurement noise.
NOISE_CELLS = frozenset(f"acquisition_{index:02d}" for index in range(4))
#: Every cell a Phase C baseline plan may name.
BASELINE_CELLS = frozenset(ROSTER) | INTERVENTION_CELLS | NOISE_CELLS
#: The releases a plan may read.
RELEASE_PROTOCOLS = frozenset({PROTOCOL, NOISE_PROTOCOL})


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


def release_protocol(cells: Iterable[str]) -> str:
    """The release a set of cells is read from; one plan never mixes two."""
    names = set(cells)
    noisy = names & NOISE_CELLS
    if noisy and noisy != names:
        raise ValueError(
            "noise acquisitions and Phase C cells are in different releases"
        )
    return NOISE_PROTOCOL if noisy else PROTOCOL


def release_receipt(release: Path, protocol: str) -> str:
    """The digest of the receipt file that identifies a release."""
    if protocol not in RELEASE_PROTOCOLS:
        raise ValueError(f"unknown release protocol {protocol!r}")
    name = "release.json" if protocol == NOISE_PROTOCOL else "summary.json"
    return _sha256(release / name)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _verify_noise_release(release: Path, cells: tuple[str, ...]) -> str:
    """Check the noise study's content-addressed record and each listed file."""
    record = json.loads((release / "release.json").read_bytes())
    if (
        set(record) != {"sha256", "value"}
        or hashlib.sha256(_canonical(record["value"])).hexdigest() != record["sha256"]
    ):
        raise ValueError("the noise release record differs from its digest")
    value = record["value"]
    if (
        value.get("format") != NOISE_PROTOCOL
        or set(value.get("cells", {})) != NOISE_CELLS
    ):
        raise ValueError("requires the four-acquisition T2 noise release")
    for name in cells:
        listed = value["cells"][name]["public_files"]
        if set(listed) != set(PUBLIC_FILES):
            raise ValueError(f"unexpected public files listed for {name}")
        for filename in PUBLIC_FILES:
            key = f"public/{name}/{filename}"
            if _sha256(release / key) != listed[filename]:
                raise ValueError(f"published public file changed or unlisted: {key}")
    return release_receipt(release, NOISE_PROTOCOL)


def verify_release(release: Path, cells: Iterable[str]) -> str:
    """Check the publication receipt and each public file a baseline will read.

    Returns the receipt's digest, which identifies the release a model was
    developed on. A changed, missing or unlisted file is refused.
    """
    cells = tuple(cells)
    if release_protocol(cells) == NOISE_PROTOCOL:
        return _verify_noise_release(release, cells)
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

    Basin cells have one fixed observation design, so they carry no tier. The
    noise acquisitions are the T2-hard cell's trajectories, so they are hard.
    """
    if name in NOISE_CELLS:
        return "hard"
    for tier in ("easy", "hard"):
        if f"_{tier}_" in f"{name}_":
            return tier
    return "fixed"


def load_cell(release: Path, name: str) -> PhaseCBaselineCell:
    """Load one roster cell's train, validation and prompt; never test data."""
    if name not in BASELINE_CELLS:
        raise ValueError(f"{name} is outside the Phase C baseline roster")
    receipt = verify_release(release, (name,))
    directory = release / "public" / name
    spec, train, validation = load_public_cell(directory)
    prompt_path = directory / "proposer_prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    if spec["protocol"] != PROTOCOL or spec["test_released"]:
        raise ValueError("cell is not a development-only Phase C cell")
    noisy = name in NOISE_CELLS
    # The noise release writes the sealed prompt with exactly one final newline.
    sealed = spec["public_prompt"].rstrip("\n") if noisy else spec["public_prompt"]
    if prompt != sealed + "\n":
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
        "protocol": NOISE_PROTOCOL if noisy else PROTOCOL,
        "release_summary_sha256": receipt,
        "benchmark_id": name,
        "train": dataset.train.fingerprint,
        "validation": dataset.validation.fingerprint,
        "prompt": _sha256(prompt_path),
    }
    return PhaseCBaselineCell(dataset, context, prompt, identity)
