"""The T2-hard measurement-noise plans: the frozen sources' settings, other cells."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from autoformalism.rebuttal import phase_b_nine_case as nine
from autoformalism.rebuttal import phase_c_baselines as pcb
from autoformalism.rebuttal.phase_b_nine_case_d3 import NineCaseD3Plan
from autoformalism.rebuttal.phase_b_nine_case_spec_sindy import SpecSindyPlan
from autoformalism.rebuttal.phase_c_baseline_plan import load_phase_c_baseline_plan
from autoformalism.rebuttal.phase_c_d3 import PhaseCD3Plan
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    load_phase_c_vendored_plan,
    verify_plan_release,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RECEIPT = "30c50333df09ad24e0454086a96d821e7375d48564e8c1f2325086509b61ee29"
PAIRS = {
    "t2_noise_classical_jetstream_cpu_v1.json": (
        "phase_c_cstr_hard_public_baseline_jetstream_cpu_v1.json",
        {"purpose"},
    ),
    "t2_noise_d3_hosted_120b_v2.json": (
        "phase_c_cstr_hard_d3_hosted_120b_v2.json",
        {"purpose"},
    ),
    "t2_noise_hdtwingen_hosted_120b_v1.json": (
        "phase_c_cstr_hard_hdtwingen_hosted_120b_v1.json",
        {"purpose"},
    ),
    "t2_noise_spec_sindy_hosted_120b_v1.json": (
        "phase_c_cstr_hard_spec_sindy_hosted_120b_v1.json",
        {"purpose"},
    ),
    "t2_noise_llm_ode_hosted_120b_v1.json": (
        "phase_c_cstr_hard_llm_ode_hosted_120b_v1.json",
        {"scope_note", "budget", "full_design_conditions"},
    ),
    "t2_noise_llm_ode_published_prompt_hosted_120b_v1.json": (
        "phase_c_cstr_hard_llm_ode_published_prompt_hosted_120b_v1.json",
        {"scope_note", "budget", "full_design_conditions"},
    ),
    "t2_noise_llm_sr_expression_hosted_120b_v1.json": (
        "phase_c_cstr_hard_llm_sr_expression_hosted_120b_v1.json",
        {"scope_note", "budget", "full_design_conditions"},
    ),
    "t2_noise_latent_ode_v1.json": ("latent_ode_intervention_v1.json", {"purpose"}),
}
RELEASE_KEYS = {
    "cells",
    "release_protocol",
    "inputs_protocol",
    "release_summary_sha256",
}


def _load(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text())


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_only_the_cells_release_and_their_wording_change(name):
    source_name, worded = PAIRS[name]
    noise, source = _load(name), _load(source_name)
    changed = {k for k in noise.keys() | source.keys() if noise.get(k) != source.get(k)}
    assert changed <= RELEASE_KEYS | worded
    if "budget" in changed:
        assert {
            k for k in noise["budget"] if noise["budget"][k] != source["budget"][k]
        } == {"rationale"}
    assert noise["release_summary_sha256"] == RECEIPT
    assert noise.get("repetitions", [0, 1, 2]) == [0, 1, 2]
    names = [
        cell if isinstance(cell, str) else cell["benchmark_id"]
        for cell in noise["cells"]
    ]
    assert names == sorted(pcb.NOISE_CELLS)
    assert "status" in noise and "frozen" in noise["status"]


def test_every_plan_model_accepts_the_noise_plans():
    load_phase_c_baseline_plan(CONFIGS / "t2_noise_classical_jetstream_cpu_v1.json")
    PhaseCD3Plan.model_validate(_load("t2_noise_d3_hosted_120b_v2.json"))
    NineCaseD3Plan.model_validate(_load("t2_noise_hdtwingen_hosted_120b_v1.json"))
    SpecSindyPlan.model_validate(_load("t2_noise_spec_sindy_hosted_120b_v1.json"))
    for name in (
        "t2_noise_llm_ode_hosted_120b_v1.json",
        "t2_noise_llm_ode_published_prompt_hosted_120b_v1.json",
        "t2_noise_llm_sr_expression_hosted_120b_v1.json",
    ):
        plan = load_phase_c_vendored_plan(CONFIGS / name)
        assert plan.expected_task_count == 12
        assert plan.full_design_conditions == 4


def test_a_plan_cannot_mix_releases_or_misname_one():
    value = _load("t2_noise_d3_hosted_120b_v2.json")
    wrong = {**value, "release_protocol": "phase-c-development-2"}
    with pytest.raises(ValueError, match="release"):
        PhaseCD3Plan.model_validate(wrong)
    mixed = {
        **value,
        "cells": [
            *value["cells"],
            {**value["cells"][0], "benchmark_id": next(iter(pcb.INTERVENTION_CELLS))},
        ],
    }
    with pytest.raises(ValueError, match="different releases"):
        PhaseCD3Plan.model_validate(mixed)


def test_noise_cells_are_hard_and_read_from_their_own_release():
    assert {pcb.tier_of(name) for name in pcb.NOISE_CELLS} == {"hard"}
    assert pcb.release_protocol(pcb.NOISE_CELLS) == pcb.NOISE_PROTOCOL
    assert pcb.release_protocol(pcb.INTERVENTION_CELLS) == pcb.PROTOCOL
    with pytest.raises(ValueError):
        nine.check_plan_cell(pcb.PROTOCOL, "acquisition_00", "hard", "")
    nine.check_plan_cell(nine.NOISE, "acquisition_00", "hard", "")


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fake_release(root: Path) -> Path:
    cells = {}
    for name in sorted(pcb.NOISE_CELLS):
        directory = root / "public" / name
        directory.mkdir(parents=True)
        files = {}
        for filename in pcb.PUBLIC_FILES:
            (directory / filename).write_text(f"{name} {filename}\n")
            files[filename] = hashlib.sha256(
                (directory / filename).read_bytes()
            ).hexdigest()
        cells[name] = {"public_files": files}
    value = {"format": pcb.NOISE_PROTOCOL, "cells": cells}
    record = {"sha256": hashlib.sha256(_canonical(value)).hexdigest(), "value": value}
    (root / "release.json").write_text(json.dumps(record))
    return root


def test_the_noise_release_is_checked_file_by_file(tmp_path):
    release = _fake_release(tmp_path)
    receipt = pcb.verify_release(release, ["acquisition_02"])
    assert (
        receipt == hashlib.sha256((release / "release.json").read_bytes()).hexdigest()
    )
    assert pcb.release_receipt(release, pcb.NOISE_PROTOCOL) == receipt
    (release / "public" / "acquisition_02" / "train.json").write_text("changed\n")
    with pytest.raises(ValueError, match="changed or unlisted"):
        pcb.verify_release(release, ["acquisition_02"])
    pcb.verify_release(release, ["acquisition_01"])
    record = json.loads((release / "release.json").read_text())
    record["value"]["format"] = "other"
    (release / "release.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="digest"):
        pcb.verify_release(release, ["acquisition_01"])


FIT = os.environ.get("T2_NOISE_FIT")


@pytest.mark.skipif(not FIT, reason="set T2_NOISE_FIT to the extracted public release")
def test_the_real_release_loads_as_the_handoff_reads_it():
    import sys

    root = Path(FIT)
    sys.path.insert(0, str(root))
    from noise_handoff import native
    from noise_handoff.exchange import read

    listed = read(root / "release.json")["cells"]
    for name in sorted(pcb.NOISE_CELLS):
        ours = pcb.load_cell(root, name)
        theirs = native.baseline_cell(
            {"cell": name, "public_files": listed[name]["public_files"]},
            root / "public" / name,
        )
        assert ours.identity == theirs.identity
        assert ours.context == theirs.context
        assert ours.prompt == theirs.prompt
    assert pcb.verify_release(root, pcb.NOISE_CELLS) == RECEIPT
    for name in (
        "t2_noise_llm_ode_hosted_120b_v1.json",
        "t2_noise_llm_sr_expression_hosted_120b_v1.json",
    ):
        assert verify_plan_release(
            load_phase_c_vendored_plan(CONFIGS / name), root
        ) == (RECEIPT)
