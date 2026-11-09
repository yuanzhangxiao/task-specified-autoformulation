"""The nine-case inputs are staged only from files whose digests were recorded.

A synthetic source tree stands in for the local public copies: every one of
the nine cells has its three files, and the alien-device cell's tables are
real tidy tables so that a staged cell can be loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from autoformalism.rebuttal import phase_b_nine_case as nine

ALIEN = "phase_b_alien_device_unknown_device_mechanism_canonical_functional_easy"
OBFUSCATED = "phase_b_anonymous_system_t1_canonical_obfuscated_easy"
SAVED_HEADER = b"trajectory_id,t,v01,v02,v03,v04,v05,u06\r\nx,0,1,2,3,4,5,0\r\n"
RECORDED_HEADER = b"trajectory_id,t,v01,v02,v03,v04,v05,u01\r\nx,0,1,2,3,4,5,0\r\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(prefix: str, count: int, rows: int = 21) -> bytes:
    """A tidy alien-device table: a decaying target pushed by its input."""
    lines = ["trajectory_id,t,v01,v02,v03,u01"]
    for k in range(count):
        value = 1.0 + 0.5 * k
        for i in range(rows):
            time = round(0.1 * i, 10)
            push = 0.5 if i >= 5 + k else 0.0
            lines.append(
                f"{prefix}_{k:03d},{time!r},{value!r},{math.sin(time)!r},"
                f"{math.cos(time)!r},{push!r}"
            )
            value += 0.1 * (-0.5 * value + push)
    return ("\n".join(lines) + "\n").encode()


def build(tmp_path: Path) -> tuple[Path, Path]:
    """A source holding all nine cells, and a config recording its digests."""
    source = tmp_path / "source"
    cells = {}
    for name, cell in nine.load_inputs().cells.items():
        directory = source / name
        directory.mkdir(parents=True)
        files = {}
        for filename in nine.FILES:
            data = recorded = f"{name} {filename}\n".encode()
            if name == ALIEN and filename == "train.csv":
                data = recorded = table("train", 3)
            elif name == ALIEN and filename == "validation.csv":
                data = recorded = table("val", 1)
            elif name == OBFUSCATED and filename == "train.csv":
                data, recorded = SAVED_HEADER, RECORDED_HEADER
            (directory / filename).write_bytes(data)
            files[filename] = _sha256(recorded)
        entry: dict = {"tier": cell.tier, "files": files}
        if cell.header_renames:
            entry["header_renames"] = dict(cell.header_renames)
        cells[name] = entry
    config = tmp_path / "inputs.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "phase-b-nine-case-inputs-1",
                "purpose": "Tests.",
                "source_campaign_plan_sha256": "0" * 64,
                "cells": cells,
            }
        )
    )
    return source, config


@pytest.fixture
def staged(tmp_path, monkeypatch) -> Path:
    """A staged copy of the synthetic source, read through its own config."""
    source, config = build(tmp_path)
    monkeypatch.setattr(nine, "CONFIG", config)
    output = tmp_path / "staged"
    nine.stage([source], output)
    return output


def test_the_committed_record_names_the_nine_compared_cases():
    inputs = nine.load_inputs()
    assert len(inputs.cells) == 9
    assert {cell.tier for cell in inputs.cells.values()} == {"easy", "hard"}
    renamed = sorted(n for n, c in inputs.cells.items() if c.header_renames)
    assert renamed == [
        "phase_b_anonymous_system_t1_canonical_obfuscated_easy",
        "phase_b_anonymous_system_t1_perturbed_obfuscated_easy",
    ]


def test_recorded_files_are_staged_and_only_the_declared_header_is_renamed(staged):
    receipt = json.loads((staged / "receipt.json").read_text())
    assert receipt["transforms"] == {f"{OBFUSCATED}/train.csv": "header_renamed"}
    assert receipt["test_data_staged"] is False
    copied = staged / "phase_b_v1" / OBFUSCATED / "train.csv"
    assert copied.read_bytes() == RECORDED_HEADER
    assert len(nine.verify(staged)) == 64
    assert not list(staged.rglob("test*"))
    assert not list(staged.rglob("manifest.json"))


def test_staging_again_only_verifies(staged, tmp_path):
    first = (staged / "receipt.json").read_bytes()
    nine.stage([tmp_path / "source"], staged)
    assert (staged / "receipt.json").read_bytes() == first


def test_a_file_with_another_digest_is_refused(tmp_path, monkeypatch):
    source, config = build(tmp_path)
    monkeypatch.setattr(nine, "CONFIG", config)
    (source / ALIEN / "validation.csv").write_bytes(table("val", 2))
    with pytest.raises(ValueError, match="no source holds"):
        nine.stage([source], tmp_path / "staged")


def test_a_rename_that_does_not_give_the_recorded_file_is_refused(
    tmp_path, monkeypatch
):
    source, config = build(tmp_path)
    monkeypatch.setattr(nine, "CONFIG", config)
    (source / OBFUSCATED / "train.csv").write_bytes(
        SAVED_HEADER.replace(b"1,2,3", b"1,2,4")
    )
    with pytest.raises(ValueError, match="no source holds"):
        nine.stage([source], tmp_path / "staged")


def test_a_later_source_supplies_a_file_the_first_lacks(tmp_path, monkeypatch):
    source, config = build(tmp_path)
    monkeypatch.setattr(nine, "CONFIG", config)
    prompts = tmp_path / "prompts"
    (prompts / ALIEN).mkdir(parents=True)
    moved = source / ALIEN / "proposer_prompt.txt"
    (prompts / ALIEN / "proposer_prompt.txt").write_bytes(moved.read_bytes())
    moved.unlink()
    nine.stage([source, prompts], tmp_path / "staged")


def test_staging_into_a_source_is_refused(tmp_path, monkeypatch):
    source, config = build(tmp_path)
    monkeypatch.setattr(nine, "CONFIG", config)
    with pytest.raises(ValueError, match="outside every source"):
        nine.stage([source], source / "staged")


def test_a_changed_staged_file_fails_verification(staged):
    path = staged / "phase_b_v1" / ALIEN / "train.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differs from the recorded file"):
        nine.verify(staged)


def test_a_cell_loads_its_development_data_and_identity(staged):
    cell = nine.load_cell(staged, ALIEN)
    assert cell.dataset.tier == "easy"
    assert len(cell.dataset.train.trajectories) == 3
    assert len(cell.dataset.validation.trajectories) == 1
    assert cell.context.targets == ("v01",)
    assert cell.context.auxiliaries == ("v02", "v03")
    assert cell.context.external_inputs == ("u01",)
    assert cell.identity["inputs_receipt_sha256"] == nine.verify(staged)
    assert cell.prompt == f"{ALIEN} proposer_prompt.txt\n"
    regression = nine.load_cell(staged, ALIEN, context="baseline")
    assert regression.context.targets == cell.context.targets


def test_a_cell_outside_the_nine_is_refused(staged):
    with pytest.raises(ValueError, match="not one of the nine"):
        nine.load_cell(staged, "phase_b_dalla_man_t3_canonical_named_easy")
