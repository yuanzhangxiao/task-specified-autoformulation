"""The nine Phase B cases of the main comparison, staged for further baselines.

The comparison's campaign read its public inputs from ``public/phase_b_v1``
on the cluster and recorded the SHA-256 of every file it read. The digests of
the training and validation tables and the proposer prompt of each of the
nine cells are in ``configs/phase_b_nine_case_inputs_v1.json``. ``stage``
rebuilds that input tree from local copies and accepts a file only when its
digest is the recorded one, so a baseline staged here sees exactly the
trajectories and prompt the compared methods saw.

Two local copies differ from the recorded files in one documented way: the
two obfuscated T1 tables were saved before their input column was renamed
from ``u06`` to ``u01``. Such a file is accepted only when renaming that
column in its header line gives the recorded digest; the numbers are not
touched. Nothing else is transformed.

Only these three files per cell are staged. No manifest is staged, because
the registered one also fingerprints the sealed test table, and no test data
are staged or read. ``load_cell`` reads a staged cell through the same table
reader the registered loader uses, so derivatives are estimated as they were
for the compared baselines.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from autoformalism.baselines.core import baseline_validation_context
from autoformalism.baselines.raw_data_agent import raw_agent_validation_context
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry, DevelopmentDataset
from autoformalism.data.models import SplitName
from autoformalism.rebuttal import phase_c_baselines
from autoformalism.rebuttal.phase_c_baselines import PhaseCBaselineCell
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write

PROTOCOL = "phase-b-nine-case-inputs-1"
REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs" / "phase_b_nine_case_inputs_v1.json"
FILES = ("train.csv", "validation.csv", "proposer_prompt.txt")
SUBDIRECTORY = "phase_b_v1"
#: A plan may instead name the Phase C release. It then reads one of the Phase C
#: baseline cells from that release, as Phase C D3 does, with that cell's
#: public context.
PHASE_C = phase_c_baselines.PROTOCOL


class StagedCell(BaseModel):
    """One cell's recorded digests and the one rename a local copy may need."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: Literal["easy", "hard"]
    files: dict[str, str]
    header_renames: dict[str, str] = Field(default_factory=dict)


class NineCaseInputs(BaseModel):
    """The recorded public inputs of the nine-case comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-nine-case-inputs-1"]
    purpose: str = Field(min_length=1)
    source_campaign_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cells: dict[str, StagedCell] = Field(min_length=9, max_length=9)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_inputs(path: Path | None = None) -> NineCaseInputs:
    """Read the recorded digests, which must name the three files of each cell."""
    path = CONFIG if path is None else path
    inputs = NineCaseInputs.model_validate_json(path.read_text(encoding="utf-8"))
    for name, cell in inputs.cells.items():
        if set(cell.files) != set(FILES):
            raise ValueError(f"{name} must record exactly {FILES}")
        registry = BenchmarkRegistry().get(name)
        if cell.tier not in registry.tier_roles:
            raise ValueError(f"{name} has no {cell.tier} tier")
    return inputs


def _renamed(data: bytes, renames: dict[str, str]) -> bytes:
    """Rename columns in the header line only; every other byte is kept."""
    header, newline, rest = data.partition(b"\n")
    text = header.decode("utf-8")
    ending = "\r" if text.endswith("\r") else ""
    columns = text.removesuffix(ending).split(",")
    renamed = ",".join(renames.get(column, column) for column in columns) + ending
    return renamed.encode("utf-8") + newline + rest


def stage(sources: list[Path], output: Path, config: Path | None = None) -> dict:
    """Copy each recorded file from the first source holding it; refuse otherwise.

    A source is a directory with one folder per cell, as the local public
    copies are laid out. The output gets ``phase_b_v1/<cell>/<file>`` and a
    sealed ``receipt.json``. Staging again into the same output only verifies
    it.
    """
    config = CONFIG if config is None else config
    inputs = load_inputs(config)
    output = output.expanduser().resolve()
    sources = [source.expanduser().resolve() for source in sources]
    if any(output == s or output.is_relative_to(s) for s in sources):
        raise ValueError("stage into a new directory outside every source")
    chosen: list[tuple[str, str, bytes, str]] = []
    for name, cell in inputs.cells.items():
        for filename in FILES:
            expected = cell.files[filename]
            found = None
            tried = []
            for source in sources:
                path = source / name / filename
                tried.append(str(path))
                if not path.is_file():
                    continue
                data = path.read_bytes()
                if _sha256(data) == expected:
                    found = (data, "exact")
                    break
                if cell.header_renames and filename.endswith(".csv"):
                    renamed = _renamed(data, cell.header_renames)
                    if _sha256(renamed) == expected:
                        found = (renamed, "header_renamed")
                        break
            if found is None:
                raise ValueError(
                    f"no source holds the recorded {name}/{filename}; tried "
                    + ", ".join(tried)
                )
            chosen.append((name, filename, *found))
    receipt_path = output / "receipt.json"
    for name, filename, data, _ in chosen:
        path = output / SUBDIRECTORY / name / filename
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"{path} exists and differs; use a new output")
    for name, filename, data, _ in chosen:
        path = output / SUBDIRECTORY / name / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    receipt = {
        "protocol": PROTOCOL,
        "inputs_config_sha256": _sha256(config.read_bytes()),
        "source_campaign_plan_sha256": inputs.source_campaign_plan_sha256,
        "cells": {
            name: {"tier": cell.tier, "files": dict(cell.files)}
            for name, cell in inputs.cells.items()
        },
        "transforms": {
            f"{name}/{filename}": how
            for name, filename, _, how in chosen
            if how != "exact"
        },
        "test_data_staged": False,
        "manifest_staged": False,
    }
    if receipt_path.exists():
        kept = sealed_read(receipt_path)
        if {k: v for k, v in kept.items() if k != "artifact_sha256"} != receipt:
            raise ValueError(f"{receipt_path} differs; use a new output")
        return kept
    return sealed_write(receipt_path, receipt)


def verify(root: Path, config: Path | None = None) -> str:
    """Check the receipt and every staged file; return the receipt's digest."""
    inputs = load_inputs(config)
    receipt = sealed_read(root / "receipt.json")
    if (
        receipt["protocol"] != PROTOCOL
        or receipt["test_data_staged"] is not False
        or receipt["source_campaign_plan_sha256"] != inputs.source_campaign_plan_sha256
    ):
        raise ValueError("not a staged nine-case input tree")
    for name, cell in inputs.cells.items():
        if receipt["cells"].get(name) != {"tier": cell.tier, "files": cell.files}:
            raise ValueError(f"the receipt's record of {name} differs")
        for filename, expected in cell.files.items():
            path = root / SUBDIRECTORY / name / filename
            if not path.is_file() or _sha256(path.read_bytes()) != expected:
                raise ValueError(f"{path} differs from the recorded file")
    return _sha256((root / "receipt.json").read_bytes())


def load_cell(
    root: Path,
    name: str,
    *,
    context: str = "raw_agent",
    config: Path | None = None,
) -> PhaseCBaselineCell:
    """Load one staged cell's training, validation and prompt; never test data.

    `context` picks the validation context each compared baseline was built
    with: ``raw_agent`` for D3, as the Phase B D3 campaign built it, and
    ``baseline`` for the regression baselines. `config` is the record of the
    inputs, by default the committed one.
    """
    inputs = load_inputs(config)
    if name not in inputs.cells:
        raise ValueError(f"{name} is not one of the nine cases")
    receipt = verify(root, config)
    tier = inputs.cells[name].tier
    spec = BenchmarkRegistry().get(name)
    roles = spec.tier_roles[tier]
    directory = root / SUBDIRECTORY / name
    loader = BenchmarkLoader()
    train = loader._load_tidy_split(
        spec, roles, SplitName.TRAIN, directory / "train.csv"
    )
    validation = loader._load_tidy_split(
        spec, roles, SplitName.VALIDATION, directory / "validation.csv"
    )
    dataset = DevelopmentDataset(
        benchmark_id=name, tier=tier, roles=roles, train=train, validation=validation
    )
    if context == "raw_agent":
        validation_context = raw_agent_validation_context(dataset, spec)
    elif context == "baseline":
        validation_context = baseline_validation_context(dataset, spec)
    else:
        raise ValueError(f"unknown context {context!r}")
    prompt_path = directory / "proposer_prompt.txt"
    identity = {
        "protocol": PROTOCOL,
        "inputs_receipt_sha256": receipt,
        "benchmark_id": name,
        "train": train.fingerprint,
        "validation": validation.fingerprint,
        "prompt": _sha256(prompt_path.read_bytes()),
    }
    return PhaseCBaselineCell(
        dataset, validation_context, prompt_path.read_text(encoding="utf-8"), identity
    )


#: Task outcomes that mean the harness, not the method, went wrong. A
#: rollout that failed, or a fit stopped at its wall-clock limit, is the
#: method's own outcome and is scored as such.
HARNESS_FAILURES = frozenset({"discovery_failed", "native_check_failed", "failed"})


def smoke_problems(summaries: dict[str, dict]) -> list[str]:
    """Why a set of smoke summaries does not clear the full campaigns.

    Each summary is a run's ``summary.json``, the campaign report with one row
    per task. A smoke clears when its report is complete and no task ended in
    a harness failure.
    """
    problems = []
    for name, summary in summaries.items():
        if summary.get("status") != "complete":
            problems.append(f"{name}: report status {summary.get('status')}")
        rows = summary.get("rows")
        if not rows:
            problems.append(f"{name}: the report has no task rows")
            continue
        for row in rows:
            if row["status"] in HARNESS_FAILURES:
                problems.append(
                    f"{name}: task {row['index']} {row['status']}: "
                    f"{str(row.get('error', ''))[:300]}"
                )
    return problems


def check_plan_cell(protocol: str, name: str, tier: str, prompt_sha256: str) -> None:
    """A plan's cell: one of the nine at its tier and prompt, or a Phase C cell.

    A Phase C cell's prompt is checked against the release when it is prepared.
    """
    if protocol == PHASE_C:
        if name not in phase_c_baselines.BASELINE_CELLS:
            raise ValueError(f"{name} is not a Phase C baseline cell")
        if tier != phase_c_baselines.tier_of(name):
            raise ValueError(f"{name} declares the wrong tier")
        return
    recorded = load_inputs().cells.get(name)
    if recorded is None:
        raise ValueError(f"{name} is not one of the nine cases")
    if tier != recorded.tier:
        raise ValueError(f"{name} declares the wrong tier")
    if prompt_sha256 != recorded.files["proposer_prompt.txt"]:
        raise ValueError(f"{name} declares another prompt")


def verify_inputs(protocol: str, root: Path, names) -> str:
    """The receipt digest of the inputs a plan reads: staged cases or release."""
    if protocol == PHASE_C:
        return phase_c_baselines.verify_release(root, tuple(names))
    return verify(root)


def load_plan_cell(
    protocol: str,
    root: Path,
    name: str,
    *,
    context: str = "raw_agent",
    config: Path | None = None,
) -> PhaseCBaselineCell:
    """Load a plan's cell from the staged cases or from the Phase C release."""
    if protocol == PHASE_C:
        return phase_c_baselines.load_cell(root, name)
    return load_cell(root, name, context=context, config=config)


def receipt_of(cell: PhaseCBaselineCell) -> str:
    """The receipt a loaded cell was verified against."""
    identity = cell.identity
    if "inputs_receipt_sha256" in identity:
        return identity["inputs_receipt_sha256"]
    return identity["release_summary_sha256"]


def scope_of(protocol: str) -> str:
    """How a report names the cells a plan ran on."""
    if protocol == PHASE_C:
        return "the Phase C cells its plan names"
    return "the nine Phase B cases of the main comparison"
