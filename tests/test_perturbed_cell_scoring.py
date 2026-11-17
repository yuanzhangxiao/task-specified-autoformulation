"""A perturbed cell is scored like its base cell, against perturbed references."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from autoformalism.benchmarks.phase_b_generation import CSTR_PERTURBATION
from scripts import iclr_anchor_fidelity as anchor
from scripts import score_baseline_interventions as scoring

CSTR = anchor.fits.CSTR
PERTURBED = anchor.CSTR_PERTURBED
DATA_ROOT = Path("data_raw")
SPEC = (
    DATA_ROOT
    / "benchmark5_anonymous_nonlinear_process/private/system_specification.json"
)
MAIN = Path("/Users/yuanzhangxiao/Projects/autoformalism")
RELEASE = MAIN / "artifacts/phase-c-development-final-v2"
BUILD = MAIN / "outputs/phase-c-cstr-perturbed-hard-v1"
SUITE = Path("configs/cstr_combined_interventions_v1.json")


def _suite() -> dict:
    return json.loads(SUITE.read_text())


def test_a_perturbed_cell_reads_as_its_base_cell_with_perturbed_dynamics():
    assert PERTURBED != CSTR and anchor.base_cell(PERTURBED) == CSTR
    assert anchor.dynamics_of(PERTURBED) == "perturbed"
    assert anchor.dynamics_of(CSTR) == "canonical"
    assert anchor.dynamics_of(anchor.fits.T1P) == "perturbed"


def test_the_base_reactor_suite_serves_its_perturbed_cell():
    suite = _suite()
    assert scoring.suite_schedules(PERTURBED, suite) == scoring.suite_schedules(
        CSTR, suite
    )


def test_a_perturbed_cell_needs_its_sealed_build(tmp_path):
    with pytest.raises(ValueError, match="perturbed build"):
        scoring.public_root(PERTURBED, tmp_path, None)
    assert scoring.public_root(CSTR, tmp_path, None) == tmp_path / "public"
    with pytest.raises(ValueError, match="cells outside the study"):
        scoring.freeze(
            tmp_path / "root", tmp_path, {}, [PERTURBED], study_models=tmp_path
        )


@pytest.mark.skipif(not SPEC.exists(), reason="private CSTR specification absent")
def test_perturbed_references_multiply_the_reaction_heat():
    table = scoring.suite_schedules(PERTURBED, _suite())
    spec = json.loads(SPEC.read_text())
    p, t_eq = spec["parameters"], spec["equilibrium"]["T"]
    with scoring.composite_inputs.extended():
        basal = next(k for k, v in table.items() if v["role"] == "control")
        a = anchor.reference(PERTURBED, table[basal], DATA_ROOT, "Radau")
        b = anchor.reference(CSTR, table[basal], DATA_ROOT, "Radau")
        assert np.allclose(a["T"], b["T"], atol=1e-9)
        probe = next(k for k, v in table.items() if v["role"] == "contrast")
        a = anchor.reference(PERTURBED, table[probe], DATA_ROOT, "Radau")
        b = anchor.reference(CSTR, table[probe], DATA_ROOT, "Radau")
        assert np.max(np.abs(np.subtract(a["T"], b["T"]))) > 0.1
    temperature = np.maximum(np.asarray(a["T"]), 250)
    multiplier = 1 - CSTR_PERTURBATION["amplitude"] * np.tanh(
        (temperature - t_eq) / CSTR_PERTURBATION["width_kelvin"]
    )
    expected = (
        p["source_gain"]
        * p["k0"]
        * np.exp(-p["E_over_R"] / temperature)
        * np.maximum(np.asarray(a["C"]), 0)
        * multiplier
    )
    assert np.allclose(a["reaction_heat"], expected, rtol=1e-12)


@pytest.mark.skipif(
    not (BUILD / "summary.json").exists() or not RELEASE.exists(),
    reason="perturbed build or release absent",
)
def test_the_perturbed_build_is_verified_against_its_seal_and_base_cell(tmp_path):
    identity = scoring.perturbed_identity(BUILD, RELEASE, [PERTURBED])
    assert identity["cells"] == [PERTURBED]
    assert scoring.perturbed_identity(BUILD, RELEASE, [CSTR]) is None
    copy = tmp_path / "build"
    shutil.copytree(BUILD / "public", copy / "public")
    shutil.copy(BUILD / "summary.json", copy / "summary.json")
    shutil.copytree(BUILD / "diagnostic" / PERTURBED, copy / "diagnostic" / PERTURBED)
    prompt = copy / "public" / PERTURBED / "proposer_prompt.txt"
    prompt.write_text(prompt.read_text() + " ")
    with pytest.raises(ValueError, match="file changed"):
        scoring.perturbed_identity(copy, RELEASE, [PERTURBED])
