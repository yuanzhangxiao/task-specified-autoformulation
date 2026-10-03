"""Phase C public cells reach the unchanged baselines, and only public cells do."""

import hashlib
import json

import numpy as np
import pytest

from autoformalism.baselines.models import BaselineConfig
from autoformalism.baselines.runner import run_baseline_development
from autoformalism.benchmarks.audited_release import seal
from autoformalism.data.models import DerivativeProvenance
from autoformalism.rebuttal import phase_c_baselines as pcb

CELL = "phase_c_detention_coupled_noise0_v1"
PROMPT = "Predict the downstream depth from the measured inflows."
COVARIATES = {"area_up": 2.0, "area_down": 3.0}


def _trajectory(identifier, start):
    time = np.linspace(0.0, 3.0, 31)
    return {
        "trajectory_id": identifier,
        "time": time.tolist(),
        "targets": {"h_down": (start * np.exp(-time)).tolist()},
        "auxiliaries": {},
        "external_inputs": {
            "inflow_up": np.zeros_like(time).tolist(),
            "inflow_down": np.zeros_like(time).tolist(),
        },
        "fixed_covariates": COVARIATES,
    }


def _release(root, *, test_generated=False, prompt=PROMPT, cell=CELL):
    directory = root / "public" / cell
    directory.mkdir(parents=True)
    seal(
        directory / "specification.json",
        {
            "protocol": pcb.PROTOCOL,
            "benchmark_id": cell,
            "targets": ["h_down"],
            "auxiliaries": [],
            "external_inputs": ["inflow_up", "inflow_down"],
            "fixed_covariates": sorted(COVARIATES),
            "public_prompt": PROMPT,
            "test_released": False,
        },
    )
    starts = {"train": (1.0, 2.0), "val": (1.5,)}
    for name, values in starts.items():
        rows = [_trajectory(f"{name}_{i:03d}", s) for i, s in enumerate(values)]
        seal(
            directory / f"{name}.json",
            {"name": name, "fingerprint": f"toy-{name}", "rows": rows},
        )
    (directory / "proposer_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    files = {
        f"public/{cell}/{f}": hashlib.sha256((directory / f).read_bytes()).hexdigest()
        for f in pcb.PUBLIC_FILES
    }
    seal(
        root / "summary.json",
        {
            "protocol": pcb.PROTOCOL,
            "whole_phase_c_roster_ready": True,
            "ready_cells": pcb.RELEASE_CELLS,
            "test_generated": test_generated,
            "files": files,
        },
    )
    return root


def test_cell_takes_the_registry_shape_with_phase_b_derivatives(tmp_path):
    cell = pcb.load_cell(_release(tmp_path), CELL)
    assert cell.dataset.tier == "fixed"
    assert cell.context.targets == ("h_down",)
    assert cell.context.lagged_targets == ()
    assert set(cell.context.fixed_covariates) == set(COVARIATES)
    assert cell.prompt == PROMPT + "\n"
    trajectory = cell.dataset.train.trajectories[0]
    assert trajectory.derivative_provenance is DerivativeProvenance.ESTIMATED
    expected = np.gradient(trajectory.targets["h_down"], trajectory.time, edge_order=2)
    np.testing.assert_array_equal(trajectory.derivatives["h_down"], expected)
    assert cell.identity["benchmark_id"] == CELL
    assert cell.identity["release_summary_sha256"] == hashlib.sha256(
        (tmp_path / "summary.json").read_bytes()
    ).hexdigest()


def test_unchanged_sindy_runs_on_a_phase_c_cell_without_test_data(tmp_path):
    cell = pcb.load_cell(_release(tmp_path), CELL)
    result = run_baseline_development(
        BaselineConfig(method="sindy"), cell.dataset, cell.context
    )
    assert result.test_data_opened is False
    assert result.benchmark_id == CELL
    assert result.validation_normalized_mse < 1e-2


def test_changed_public_file_is_refused(tmp_path):
    release = _release(tmp_path)
    path = release / "public" / CELL / "train.json"
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="changed or unlisted"):
        pcb.load_cell(release, CELL)


def test_release_with_test_data_is_refused(tmp_path):
    with pytest.raises(ValueError, match="28-cell development release"):
        pcb.load_cell(_release(tmp_path, test_generated=True), CELL)


def test_prompt_differing_from_sealed_prompt_is_refused(tmp_path):
    release = _release(tmp_path, prompt="A different task.")
    with pytest.raises(ValueError, match="proposer prompt differs"):
        pcb.load_cell(release, CELL)


def test_cell_outside_the_roster_is_refused(tmp_path):
    name = "phase_c_detention_coupled_noise1_v1"
    with pytest.raises(ValueError, match="outside the Phase C baseline roster"):
        pcb.load_cell(_release(tmp_path, cell=name), name)


def test_receipt_never_lists_evaluator_files(tmp_path):
    release = _release(tmp_path)
    summary = json.loads((release / "summary.json").read_text())["value"]
    assert all(key.startswith("public/") for key in summary["files"])


@pytest.mark.parametrize(
    ("name", "tier"),
    [
        ("phase_c_dalla_man_t1_canonical_named_easy_rates_v1", "easy"),
        ("phase_c_cstr_controlled_reactor_named_hard_reset_v1", "hard"),
        ("phase_c_detention_independent_noise0_v1", "fixed"),
    ],
)
def test_tier_comes_from_the_cell_name(name, tier):
    assert pcb.tier_of(name) == tier
