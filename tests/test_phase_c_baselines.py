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


# The frozen Phase C classical matrix.

from pathlib import Path  # noqa: E402

from autoformalism.rebuttal import phase_c_baseline_plan as plan_module  # noqa: E402
from autoformalism.rebuttal.baseline_pilot import BaselinePilotTask  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "configs" / "phase_c_public_baseline_delta_cpu_v1.json"
PHASE_B = REPO / "configs" / "phase_b_public_baseline_full_delta_cpu_v1.json"


def _plan_payload(release, **overrides):
    payload = json.loads(SHIPPED.read_text())
    prompt = release / "public" / CELL / "proposer_prompt.txt"
    payload["release_summary_sha256"] = hashlib.sha256(
        (release / "summary.json").read_bytes()
    ).hexdigest()
    payload["cells"] = [
        {
            "benchmark_id": CELL,
            "tier": "fixed",
            "public_prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        }
    ]
    return {**payload, **overrides}


def _write_plan(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_shipped_plan_is_the_roster_with_phase_b_settings():
    plan = plan_module.load_phase_c_baseline_plan(SHIPPED)
    assert [cell.benchmark_id for cell in plan.cells] == list(pcb.ROSTER)
    assert plan.repetitions == (0, 1)
    phase_b = {
        item["method"]: item for item in json.loads(PHASE_B.read_text())["methods"]
    }
    for method in plan.methods:
        assert method.model_dump(mode="json") == phase_b[method.method]
    tasks = plan_module.build_phase_c_baseline_tasks(plan)
    assert len(tasks) == 2 * 8 * 2
    assert [task.task_index for task in tasks] == list(range(len(tasks)))
    assert {task.tier for task in tasks if "detention" in task.benchmark_id} == {
        "fixed"
    }


def test_freeze_writes_a_task_ledger_the_summarizer_reads(tmp_path):
    release = _release(tmp_path / "release")
    config = _write_plan(tmp_path / "plan.json", _plan_payload(release))
    manifest = plan_module.freeze_phase_c_baseline_plan(
        config, tmp_path / "frozen", release
    )
    assert manifest["task_count"] == 4
    assert manifest["tasks_by_method"] == {"sindy": 2, "pysr": 2}
    lines = (tmp_path / "frozen" / "task_plan.jsonl").read_text().splitlines()
    tasks = [BaselinePilotTask.model_validate_json(line) for line in lines]
    assert {(task.method, task.tier, task.repetition) for task in tasks} == {
        ("sindy", "fixed", 0), ("sindy", "fixed", 1),
        ("pysr", "fixed", 0), ("pysr", "fixed", 1),
    }
    # Freezing again with the same inputs is a no-op, not a second plan.
    assert plan_module.freeze_phase_c_baseline_plan(
        config, tmp_path / "frozen", release
    ) == manifest


def test_freeze_refuses_a_different_release(tmp_path):
    release = _release(tmp_path / "release")
    payload = _plan_payload(release, release_summary_sha256="0" * 64)
    config = _write_plan(tmp_path / "plan.json", payload)
    with pytest.raises(ValueError, match="release receipt differs"):
        plan_module.freeze_phase_c_baseline_plan(config, tmp_path / "frozen", release)
    assert not (tmp_path / "frozen" / "task_plan.jsonl").exists()


def test_freeze_refuses_a_different_prompt(tmp_path):
    release = _release(tmp_path / "release")
    payload = _plan_payload(release)
    payload["cells"][0]["public_prompt_sha256"] = "0" * 64
    config = _write_plan(tmp_path / "plan.json", payload)
    with pytest.raises(ValueError, match="public prompt differs"):
        plan_module.freeze_phase_c_baseline_plan(config, tmp_path / "frozen", release)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda p: p["cells"][0].update(
                benchmark_id="phase_c_detention_coupled_noise1_v1"
            ),
            "outside the Phase C roster",
        ),
        (lambda p: p["cells"][0].update(tier="hard"), "wrong tier"),
        (lambda p: p["methods"][0].update(platform="aces_cpu"), "Delta CPUs only"),
        (lambda p: p.update(repetitions=[0, 0]), "unique and nonnegative"),
    ],
)
def test_plan_refuses_cells_tiers_and_methods_it_does_not_cover(change, message):
    payload = json.loads(SHIPPED.read_text())
    change(payload)
    with pytest.raises(ValueError, match=message):
        plan_module.PhaseCBaselinePlan.model_validate(payload)
