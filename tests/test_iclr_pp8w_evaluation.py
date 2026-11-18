"""Frozen-roster gates and the matched held-out acquisition contract."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import iclr_pp8w_evaluation as study


def test_noisy_initialization_is_paired_and_future_targets_are_hidden():
    context = {
        "targets": ["T"],
        "auxiliaries": [],
        "external_inputs": ["Cf", "Tf", "Tjf"],
    }
    record = {
        "time": np.arange(9).tolist(),
        "T": [350 + i for i in range(9)],
        "Cf": [1, 1, 1, 2, 3, 3, 3, 3, 3],
        "Tf": [350] * 9,
        "Tjf": [340] * 9,
    }
    cell = {"noise": 0.1, "stride": 4, "scales": {"T": 10}, "mask": []}
    a = study.cstr_input(context, record, cell, "paired_case")
    other = deepcopy(record)
    other["T"][1:] = [9999] * 8
    b = study.cstr_input(context, other, cell, "paired_case")
    assert np.array_equal(a.targets["T"], b.targets["T"])
    assert np.all(a.targets["T"] == a.targets["T"][0])
    assert a.targets["T"][0] != record["T"][0]
    assert np.array_equal(a.external_inputs["Cf"], record["Cf"])
    clean = study.cstr_input(context, record, cell | {"noise": 0}, "paired_case")
    assert np.all(clean.targets["T"] == record["T"][0])
    independent = study.cstr_input(context, record, cell, "another_case")
    assert independent.targets["T"][0] != a.targets["T"][0]


def fixtures(tmp_path):
    names = [
        "cstr_trajectories_04",
        "cstr_trajectories_16",
        "cstr_noise_10pct",
        "cstr_stride_04",
        "t2_trajectories_04",
        "t2_trajectories_16",
    ]
    rows = [
        {
            "benchmark_id": c,
            "repetition": seed,
            "method": "full",
            "status": "no_retained_model",
            "public_file_sha256": {"train": c},
        }
        for c in names
        for seed in range(3)
    ]
    manifest = {
        "runs": rows,
        "cells": [
            {
                "cell": c,
                "family": "cstr" if c.startswith("cstr") else "t2",
                "public_files": {"train": c},
                "noise": 0.1 if "noise" in c else 0,
            }
            for c in names
        ],
    }
    bundle = {
        "protocol": "iclr-controlled-search-models-1",
        "selection_freeze_sha256": "synthetic",
        "rows": rows,
    }
    sealed_write(tmp_path / "design/manifest.json", manifest)
    return bundle


def test_incomplete_roster_is_rejected_before_test_panel_is_opened(tmp_path):
    bundle = fixtures(tmp_path)
    bundle["rows"] = bundle["rows"][:-1]
    sealed_write(tmp_path / "models.json", bundle)
    with pytest.raises(ValueError, match="eighteen-model"):
        study.freeze(
            tmp_path / "models.json",
            tmp_path / "design",
            tmp_path / "absent-test-panel.json",
            tmp_path / "out",
        )


def test_freeze_keeps_failed_seeds_and_separates_clean_initial_diagnostic(
    tmp_path, monkeypatch
):
    bundle = fixtures(tmp_path)
    sealed_write(tmp_path / "models.json", bundle)
    sealed_write(
        tmp_path / "panels.json",
        {"references": {"cstr": {}, "t2": {}}, "training_sd": {}, "cstr_schedule": {}},
    )
    monkeypatch.setattr(study.cstr, "source_identity", lambda: {})
    plan = study.freeze(
        tmp_path / "models.json",
        tmp_path / "design",
        tmp_path / "panels.json",
        tmp_path / "out",
    )
    assert len(plan["rows"]) == 21 and all(r["endpoint"] is None for r in plan["rows"])
    primary = [r for r in plan["rows"] if r["assay"] == "matched_acquisition"]
    secondary = [r for r in plan["rows"] if r["assay"] == "clean_initial_diagnostic"]
    assert len(primary) == 18 and len(secondary) == 3
    assert all(r["acquisition"]["noise"] == 0 for r in secondary)
    assert all(
        r["acquisition"]["noise"] == 0.1
        for r in primary
        if r["condition"] == "cstr_noise_10pct"
    )
    assert not plan["test_used_for_selection"]
