"""Attainability controls, oracle isolation, provenance and deterministic resume."""

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.benchmarks.phase_b_generation import (
    PhaseBProtocol,
    _scalar_input,
    _simulate_alien,
    phase_b_protocols,
)
from autoformalism.rebuttal import attainability_campaign as campaign
from autoformalism.rebuttal.attainability_controls import (
    fixed_state_fit,
    generated_problem,
    ordinary_start,
    simulate_split,
    system_for,
)
from autoformalism.rebuttal.attainability_reference import (
    export_reference,
    reference_problem,
    reference_skeleton,
    shared_boundary_lower_bound,
)
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.staged_topology import content_hash


def toy_spec():
    """Independent artificial reference grammar; no benchmark coefficients."""
    return {
        "grammar_version": "alien_grammar_v1",
        "n_latent": 5,
        "decay": [0.2] * 5,
        "skew": [[0.0] * 5 for _ in range(5)],
        "tanh_terms": [
            [{"source": (i + 1) % 5, "coefficient": 0.1, "scale": 1.0, "bias": 0.3}]
            for i in range(5)
        ],
        "product_terms": [
            [
                {
                    "source_1": 1,
                    "source_2": 3,
                    "coefficient": -0.2,
                    "scale_1": 0.8,
                    "scale_2": 0.9,
                }
            ],
            [],
            [],
            [],
            [],
        ],
        "input_vector": [1.0, 0.5, 0.0, 0.0, 0.0],
        "input_scale": 1.0,
        "output_decay": 0.3,
        "output_terms": [{"source": 0, "coefficient": 0.4, "scale": 0.7}],
        "output_product_terms": [
            {
                "source_1": 0,
                "source_2": 2,
                "coefficient": 0.2,
                "scale_1": 1.0,
                "scale_2": 1.0,
            }
        ],
    }


def source_at(root, count=4):
    problem, _ = synthetic_problem("shared", 0.0, 0)
    write_json(root / "problems/000.json", problem)
    frozen = {
        "plan": {"protocol": "fitter-feasibility-comparison-1"},
        "cases": [
            {"index": 0, "label": f"public{i}", "synthetic": False}
            for i in range(count)
        ],
        "assets": {"problems/000.json": sha256(root / "problems/000.json")},
        "test_data_opened": False,
    }
    write_json(root / "freeze.json", {**frozen, "identity": content_hash(frozen)})
    return problem


def reference_template():
    problem, _ = synthetic_problem("shared", 0.0, 0)
    for key, split in (("train", "train"), ("val", "validation")):
        rows = []
        for i, p in enumerate(
            x for x in phase_b_protocols("alien_device") if x.split == split
        ):
            t = [0.0, 0.1, 60.0]
            rows.append(
                {
                    "trajectory_id": f"{split}_{i:03d}",
                    "time": t,
                    "targets": {"v01": [0.0, 0.1, 0.2]},
                    "auxiliaries": {},
                    "external_inputs": {
                        "u01": [_scalar_input(x, p.specification) for x in t]
                    },
                    "fixed_covariates": {},
                }
            )
        problem["splits"][key]["rows"] = rows
    return problem


def test_reference_translation_and_independent_generator_agree(tmp_path):
    spec = toy_spec()
    spec["skew"][0][1], spec["skew"][1][0] = 0.2, -0.2
    candidate, context, truth = reference_skeleton(spec)
    assert len(candidate["states"]) == 6
    assert len([n for n in truth if n.startswith("skew")]) == 1
    assert sum(n.endswith("b") for n in truth) == 5
    bundle = {"candidate": candidate, "context": context, "spec": spec}
    problem = reference_problem(reference_template(), bundle)
    assert problem["splits"]["train"]["rows"][14]["fixed_covariates"]["known_z1"] == 0.5
    system, _ = system_for(problem)
    theta = np.array([truth[n] for n in system.names])
    for x in (np.zeros(6), np.arange(6) * 0.3, np.arange(6) * -0.4):
        z, y, u = x[:5], x[5], -0.7
        expected = -0.2 * z + np.asarray(spec["skew"]) @ z
        expected += 0.1 * np.tanh(np.roll(z, -1) + 0.3)
        expected[0] -= 0.2 * np.tanh(0.8 * z[1]) * np.tanh(0.9 * z[3])
        expected += np.array([1.0, 0.5, 0.0, 0.0, 0.0]) * np.tanh(u)
        dy = -0.3 * y + 0.4 * np.tanh(0.7 * z[0]) + 0.2 * np.tanh(z[0]) * np.tanh(z[2])
        forcing = [u if name == "u01" else 0.0 for name in system.inputs]
        actual = np.asarray(system.rhs(0.0, x, theta, forcing)).ravel()
        np.testing.assert_allclose(actual, np.r_[expected, dy], atol=1e-12)
    protocol = PhaseBProtocol(
        protocol_id="train_toy",
        family="alien_device",
        split="train",
        duration=0.1,
        dt=0.1,
        input_names=("u",),
        specification={"kind": "constant", "value": 0.0},
    )
    native = _simulate_alien(protocol, spec, {})
    row = deepcopy(problem["splits"]["train"]["rows"][0])
    row.update(
        time=native.time.tolist(),
        targets={"v01": native.states[:, -1].tolist()},
        external_inputs={"u01": [0.0, 0.0]},
    )
    problem["splits"]["train"]["rows"] = [row]
    y, states = simulate_split(problem, truth, "train")
    assert np.asarray(states["train_000"]).shape == (2, 6)
    np.testing.assert_allclose(y["train_000"], native.states[:, -1], atol=1e-8)
    initial = system.initial_for(
        unpack_split(problem["splits"]["train"]).trajectories[0],
        np.array([truth[n] for n in system.names]),
    )
    np.testing.assert_array_equal(initial, 0)
    spec["skew"][1][0] = 0.1
    with pytest.raises(ValueError, match="skew"):
        reference_skeleton(spec)


def test_reference_export_hashes_and_forcing_mismatch(tmp_path):
    spec_path = tmp_path / "spec.json"
    write_json(spec_path, toy_spec())
    bundle = export_reference(spec_path, tmp_path / "reference.json")
    assert bundle["proposer_access"] is False
    assert bundle["test_data_opened"] is False
    template = reference_template()
    template["splits"]["train"]["rows"][0]["external_inputs"]["u01"][1] = 2
    with pytest.raises(ValueError, match="forcing"):
        reference_problem(template, bundle)


def test_shared_boundary_error_floor_ignores_oracle_covariates():
    problem, _ = synthetic_problem("shared", 0.0, 0)
    row = problem["splits"]["train"]["rows"][0]
    second = deepcopy(row)
    second["trajectory_id"] = "different_hidden_boundary"
    second["targets"]["v01"] = [row["targets"]["v01"][0]] + [
        v + 2 for v in row["targets"]["v01"][1:]
    ]
    second["fixed_covariates"]["known_z1"] = 0.5
    row["fixed_covariates"]["known_z1"] = 0.0
    problem["splits"]["train"]["rows"] = [row, second]
    floor = shared_boundary_lower_bound(problem)
    assert floor["training_nmse_lower_bound"] > 0
    assert len(floor["conflicting_groups"]) == 1
    second["external_inputs"]["u01"][0] += 1
    assert shared_boundary_lower_bound(problem)["training_nmse_lower_bound"] == 0


def test_freeze_matrix_provenance_and_non_numerical_prepare(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "out"
    source_at(source)

    def forbidden(*a, **k):
        raise AssertionError("prepare must not run a simulator")

    monkeypatch.setattr(campaign, "simulate_split", forbidden)
    frozen = campaign.prepare(campaign.AttainabilityPlan(), output, source)
    assert len(frozen["tasks"]) == 28
    assert campaign.prepare(campaign.AttainabilityPlan(), output, source) == frozen
    assert campaign.verify(output) == frozen
    write_json(output / "problems/000.json", {})
    with pytest.raises(ValueError, match="frozen input"):
        campaign.verify(output)


def test_oracle_matrix_has_separate_known_and_shared_initials(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    source_at(source)
    problem = reference_template()
    write_json(source / "problems/000.json", problem)
    frozen = read_json(source / "freeze.json")
    frozen["assets"]["problems/000.json"] = sha256(source / "problems/000.json")
    frozen["identity"] = content_hash(
        {k: v for k, v in frozen.items() if k != "identity"}
    )
    write_json(source / "freeze.json", frozen)
    write_json(tmp_path / "spec.json", toy_spec())
    export_reference(tmp_path / "spec.json", tmp_path / "ref.json")
    result = campaign.prepare(
        campaign.AttainabilityPlan(), output, source, tmp_path / "ref.json"
    )
    assert len(result["cases"]) == 5 and len(result["tasks"]) == 39
    assert result["private_reference_opened"] is True
    assert {t["dataset"] for t in result["tasks"] if t["case_index"] == 4} == {
        "synthetic",
        "actual",
        "shared_initials",
    }


def test_generation_does_not_resample_on_validation_failure(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "out"
    source_at(source, 1)
    campaign.prepare(campaign.AttainabilityPlan(), output, source)
    calls = []

    def chosen(*a):
        calls.append("training")
        return {}, {}, {}, []

    def fail(*a):
        calls.append("validation")
        raise ValueError("validation integration failed")

    monkeypatch.setattr(campaign, "choose_truth", chosen)
    monkeypatch.setattr(campaign, "simulate_split", fail)
    result = campaign.generate(output, 0)
    assert result["status"] == "generation_failed"
    assert calls == ["training", "validation"]
    assert campaign.generate(output, 0) == result
    assert calls == ["training", "validation"]
    assert campaign.execute(output, 0)["status"] == "generation_unavailable"


def test_interrupted_fit_and_stdlib_summary(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    source_at(source, 1)
    frozen = campaign.prepare(campaign.AttainabilityPlan(), output, source)
    task = frozen["tasks"][0]
    identity = content_hash([frozen["identity"], task])
    write_json(output / "results/task_000/fit_started.json", {"identity": identity})
    assert campaign.execute(output, 0)["status"] == "interrupted"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/run_attainability_campaign.py"
    )
    subprocess.run(
        [sys.executable, "-S", str(script), "summarize", "--output", str(output)],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert read_json(output / "summary.json")["statuses"] == {
        "interrupted": 1,
        "missing": 6,
    }


def test_saved_actual_fit_is_reused_without_reading_generating_truth(
    tmp_path, monkeypatch
):
    source, output = tmp_path / "source", tmp_path / "out"
    source_at(source, 1)
    frozen = campaign.prepare(campaign.AttainabilityPlan(), output, source)
    task = frozen["tasks"][-1]
    identity = content_hash([frozen["identity"], task])
    directory = output / f"results/task_{task['index']:03d}"
    write_json(
        directory / "fit.json",
        {"identity": identity, "fit": {"status": "fit_failed", "parameters": None}},
    )

    def forbidden(*a, **k):
        raise AssertionError("saved fit must not run again")

    monkeypatch.setattr(campaign, "fit_collocation_forward_sensitivity", forbidden)
    result = campaign.execute(output, task["index"])
    assert result["status"] == "fit_failed"
    assert result["oracle_start"] is False
    assert campaign.execute(output, task["index"]) == result


def test_fixed_node_optimization_does_not_fit_parameters(tmp_path):
    problem, _ = synthetic_problem("shared", 0.0, 1)
    truth = ordinary_start(problem)
    clean = {k: simulate_split(problem, truth, k)[0] for k in ("train", "val")}
    generated = generated_problem(problem, clean)
    assert generated["candidate"] == problem["candidate"]
    assert (
        generated["splits"]["train"]["rows"][0]["external_inputs"]
        == problem["splits"]["train"]["rows"][0]["external_inputs"]
    )
    result = fixed_state_fit(generated, truth, None, tmp_path, 20)
    assert result["success"]
    assert result["parameters_optimized"] is False
    assert result["initials_optimized"] is False
    assert result["hidden_trajectory_labels_used"] is False
    assert result["collocation_training_nmse"] < 1e-5
    assert result["last_iteration"]["worst_defects"]


def test_launcher_schedules_generation_before_fitting_and_deduplicates(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    source_at(source)
    write_json(source / "problems/000.json", reference_template())
    frozen = read_json(source / "freeze.json")
    frozen["assets"]["problems/000.json"] = sha256(source / "problems/000.json")
    frozen["identity"] = content_hash(
        {k: v for k, v in frozen.items() if k != "identity"}
    )
    write_json(source / "freeze.json", frozen)
    write_json(tmp_path / "spec.json", toy_spec())
    export_reference(tmp_path / "spec.json", tmp_path / "ref.json")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(
        '#!/bin/sh\ncase "$1" in\nstatus) exit 0;;\n'
        "rev-parse) echo test-commit;;\n*) exit 2;;\nesac\n"
    )
    batch = bin_dir / "sbatch"
    batch.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$AF_BATCH_TEST_LOG"\n'
        'wc -l < "$AF_BATCH_TEST_LOG" | tr -d " "\n'
    )
    git.chmod(0o755)
    batch.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "AF_REPO_ROOT": str(root),
        "AF_PYTHON": sys.executable,
        "AF_CASADI_ROOT": str(tmp_path),
        "AF_SOURCE_ROOT": str(source),
        "AF_OUTPUT_ROOT": str(output),
        "AF_REFERENCE_FILE": str(tmp_path / "ref.json"),
        "AF_BATCH_TEST_LOG": str(tmp_path / "batch.log"),
    }
    cmd = ["bash", str(root / "scripts/hpc/submit_attainability_delta.sh")]
    subprocess.run(cmd, env=env, check=True, capture_output=True)
    log = (tmp_path / "batch.log").read_text().splitlines()
    assert len(log) == 4
    assert "--dependency=afterok:1" in log[1] and "--array=0-4%2" in log[1]
    assert "--dependency=afterok:1,afterany:2" in log[2]
    assert "--array=0-38%2" in log[2]
    assert "--dependency=afterany:3" in log[3]
    assert read_json(output / "submission.json")["submission_complete"] is True
    subprocess.run(cmd, env=env, check=True, capture_output=True)
    assert (tmp_path / "batch.log").read_text().splitlines() == log
