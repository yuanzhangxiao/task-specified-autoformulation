"""Public-only, nested training subsets and immutable matched follow-up controls."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import iclr_pp8w_followup as study
from scripts import iclr_stress_data as data


def fixture_source(root):
    cells = []
    for key in study.SOURCES.values():
        rows = [
            {
                "trajectory_id": f"train_{i:03d}",
                "time": list(range(17)),
                "targets": {"y": [1.0 + i + t / 10 for t in range(17)]},
                "auxiliaries": {},
                "external_inputs": {"u": [max(0.0, 3 - abs(t - 6)) for t in range(17)]},
            }
            for i in range(16)
        ]
        spec = {
            "targets": ["y"],
            "auxiliaries": [],
            "external_inputs": ["u"],
            "public_prompt": "Frozen public specification.\n",
        }
        val = [{**deepcopy(rows[0]), "trajectory_id": "val_000"}]
        cells.append(data.write_cell(root, key, spec, rows, val) | {"source": key})
    sealed_write(root / "manifest.json", {"cells": cells})


def test_nested_subsets_do_not_depend_on_observed_responses():
    rows = [
        {"trajectory_id": f"train_{i:03d}", "targets": {"y": [i]}} for i in range(16)
    ]
    for family in study.SOURCES:
        ids = [
            {r["trajectory_id"] for r in study.selected_rows(rows, family, n)}
            for n in (4, 8, 16)
        ]
        assert ids[0] < ids[1] < ids[2]
        assert [len(s) for s in ids] == [4, 8, 16]
        changed = [{**r, "targets": {"y": [-999]}} for r in reversed(rows)]
        assert [
            r["trajectory_id"] for r in study.selected_rows(changed, family, 4)
        ] == [r["trajectory_id"] for r in study.selected_rows(rows, family, 4)]
    with pytest.raises(ValueError, match="sixteen"):
        study.selected_rows(rows[:-1], "cstr", 4)
    with pytest.raises(ValueError, match="unsupported"):
        study.selected_rows(rows, "cstr", 5)


def test_prepare_resume_retains_validation_prompts_controls_and_input_schedules(
    tmp_path,
):
    source, out = tmp_path / "source", tmp_path / "output"
    fixture_source(source)
    before = {str(p): data.digest(p) for p in source.rglob("*") if p.is_file()}
    manifest = study.prepare(source, out)
    assert len(manifest["runs"]) == 18
    assert study.prepare(source, out) == manifest
    assert not manifest["test_data_opened"] and manifest["provider_calls"] == 0
    for family, key in study.SOURCES.items():
        original = source / "public" / key
        for count in (4, 16):
            new = out / "public" / f"{family}_trajectories_{count:02d}"
            assert (new / "val.json").read_bytes() == (
                original / "val.json"
            ).read_bytes()
            assert (new / "proposer_prompt.txt").read_bytes() == (
                original / "proposer_prompt.txt"
            ).read_bytes()
            assert len(read_seal(new / "train.json")["rows"]) == count
        assert (
            out / "public" / f"{family}_trajectories_16" / "train.json"
        ).read_bytes() == (original / "train.json").read_bytes()
    clean = read_seal(source / "public/sensor_00/train.json")["rows"]
    noisy = read_seal(out / "public/cstr_noise_10pct/train.json")["rows"]
    assert noisy[0]["targets"] != clean[0]["targets"]
    assert all(
        a["external_inputs"] == b["external_inputs"]
        for a, b in zip(clean, noisy, strict=True)
    )
    assert before == {str(p): data.digest(p) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        study.prepare(source, out, include_eight=True)


def test_modified_source_is_rejected(tmp_path):
    source = tmp_path / "source"
    fixture_source(source)
    (source / "public/sensor_00/proposer_prompt.txt").write_text("changed")
    with pytest.raises(ValueError, match="source public files"):
        study.prepare(source, tmp_path / "output")
