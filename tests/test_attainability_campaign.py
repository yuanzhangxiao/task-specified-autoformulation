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
from autoformalism.rebuttal import reference_audit as audit
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
from autoformalism.rebuttal.attainability_restart import prepare_reference_recovery
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


def recovery_source_at(tmp_path, count=1):
    """Materialize completed public controls and a failed, unfitted reference."""
    raw, source = tmp_path / "raw", tmp_path / "original"
    source_at(raw, count)
    write_json(raw / "problems/000.json", reference_template())
    frozen = read_json(raw / "freeze.json")
    frozen["assets"]["problems/000.json"] = sha256(raw / "problems/000.json")
    frozen["identity"] = content_hash(
        {k: v for k, v in frozen.items() if k != "identity"}
    )
    write_json(raw / "freeze.json", frozen)
    write_json(tmp_path / "spec.json", toy_spec())
    export_reference(tmp_path / "spec.json", tmp_path / "reference.json")
    frozen = campaign.prepare(
        campaign.AttainabilityPlan(), source, raw, tmp_path / "reference.json"
    )
    for case in frozen["cases"]:
        reference = case["reference_skeleton"]
        write_json(
            source / f"generated/{case['index']:03d}/result.json",
            {
                "identity": content_hash(
                    [frozen["identity"], "generate", case["index"]]
                ),
                "case": case,
                "status": "generation_failed" if reference else "complete",
                "error": "ValueError: original generator does not reproduce public data"
                if reference
                else None,
            },
        )
    for task in frozen["tasks"]:
        if frozen["cases"][task["case_index"]]["reference_skeleton"]:
            continue
        write_json(
            source / f"results/task_{task['index']:03d}/result.json",
            {
                "identity": content_hash([frozen["identity"], task]),
                "task": task,
                "case": frozen["cases"][task["case_index"]],
                "status": "complete",
            },
        )
    return source


def recovery_plan():
    return campaign.AttainabilityPlan(
        protocol="fitter-attainability-2", reference_audit=audit.ReferenceAuditConfig()
    )


def test_recovery_retains_original_identities_and_does_not_fit(tmp_path, monkeypatch):
    source = recovery_source_at(tmp_path)
    output = tmp_path / "recovered"
    before = {str(p.relative_to(source)): sha256(p) for p in source.rglob("*.json")}
    frozen = prepare_reference_recovery(source, output, recovery_plan())
    assert frozen["selected_tasks"] == list(range(7, 18))
    assert frozen["selected_generations"] == [1]
    assert len(frozen["retained_results"]) == 7
    assert prepare_reference_recovery(source, output, recovery_plan()) == frozen

    def forbidden(*a, **k):
        raise AssertionError("retained work must never run again")

    monkeypatch.setattr(campaign, "fit_collocation_forward_sensitivity", forbidden)
    monkeypatch.setattr(campaign, "choose_truth", forbidden)
    assert campaign.execute(output, 0) == read_json(
        source / "results/task_000/result.json"
    )
    assert campaign.generate(output, 0) == read_json(
        source / "generated/000/result.json"
    )
    assert {
        str(p.relative_to(source)): sha256(p) for p in source.rglob("*.json")
    } == before
    path = output / frozen["retained_results"]["0"]
    write_json(path, {"changed": True})
    with pytest.raises(ValueError, match="frozen input"):
        campaign.verify(output)


def test_recovery_rejects_old_attempt_and_changed_fit_budget(tmp_path):
    source = recovery_source_at(tmp_path)
    plan = recovery_plan()
    changed = plan.model_copy(
        update={"fit": plan.fit.model_copy(update={"refinement_seconds": 601})}
    )
    with pytest.raises(ValueError, match="fitting settings"):
        prepare_reference_recovery(source, tmp_path / "bad_budget", changed)
    write_json(source / "results/task_007/fit_started.json", {})
    with pytest.raises(ValueError, match="prior numerical attempt"):
        prepare_reference_recovery(source, tmp_path / "bad_attempt", plan)


def test_recovery_rejects_changed_source_and_reference_translation(tmp_path):
    source = recovery_source_at(tmp_path)
    write_json(source / "problems/000.json", {})
    with pytest.raises(ValueError, match="input hash"):
        prepare_reference_recovery(source, tmp_path / "out", recovery_plan())
    # Updating outer provenance does not bypass the specification/translation check.
    source = recovery_source_at(tmp_path / "different")
    bundle = read_json(source / "reference_input.json")
    bundle["truth"]["decay0"] += 1
    bundle["identity"] = content_hash(
        {k: v for k, v in bundle.items() if k != "identity"}
    )
    write_json(source / "reference_input.json", bundle)
    frozen = read_json(source / "freeze.json")
    frozen["assets"]["reference_input.json"] = sha256(source / "reference_input.json")
    frozen["identity"] = content_hash(
        {k: v for k, v in frozen.items() if k != "identity"}
    )
    write_json(source / "freeze.json", frozen)
    with pytest.raises(ValueError, match="generating specification"):
        prepare_reference_recovery(source, tmp_path / "bad_reference", recovery_plan())


def test_stdlib_report_retains_source_and_explains_blocked_generation(tmp_path):
    source = recovery_source_at(tmp_path)
    output = tmp_path / "recovered"
    frozen = prepare_reference_recovery(source, output, recovery_plan())
    write_json(
        output / "generated/001/result.json",
        {
            "identity": content_hash([frozen["identity"], "generate", 1]),
            "status": "generation_failed",
            "error": "numerical agreement failed",
        },
    )
    script = (
        Path(__file__).resolve().parents[1] / "scripts/run_attainability_campaign.py"
    )
    subprocess.run(
        [sys.executable, "-S", str(script), "summarize", "--output", str(output)],
        check=True,
        capture_output=True,
    )
    report = read_json(output / "summary.json")
    assert report["statuses"] == {"complete": 7, "blocked_by_generation": 11}
    assert sum(r["retained"] for r in report["records"]) == 7
    assert (
        report["records"][0]["identity"]
        == read_json(source / "results/task_000/result.json")["identity"]
    )
    assert "Strict recovery" in (output / "summary.md").read_text()
    assert len(read_json(output / "diagnostics.json")["records"]) == 18


def test_scale_aware_gate_accepts_numerical_drift_and_rejects_wrong_trajectory():
    config = audit.ReferenceAuditConfig()
    small = audit.agreement(
        [1, 2],
        [1.000022, 2],
        3.8,
        config.saved_absolute_tolerance,
        config.saved_scale_tolerance,
    )
    assert small["pass"] and small["maximum_absolute_difference"] > 1e-5
    assert small["normalized_mse"] < 1e-10
    assert not audit.agreement([1, 2], [1.1, 2], 3.8, 1e-8, 1e-4)["pass"]
    assert not audit.agreement([1, 2], [1.000022, 2], 3.8, 1e-10, 1e-6)["pass"]


@pytest.mark.parametrize(
    "left,right,scale",
    [([np.nan], [0], 1), ([0], [0, 1], 1), ([], [], 1), ([1], [1], 0)],
)
def test_audit_rejects_nonfinite_and_misaligned_comparisons(left, right, scale):
    with pytest.raises(ValueError, match="aligned finite"):
        audit.agreement(left, right, scale, 1e-8, 1e-4)


def test_tight_audit_uses_continuous_forcing_and_checks_independent_solvers(
    tmp_path, monkeypatch
):
    spec = toy_spec()
    candidate, context, truth = reference_skeleton(spec)
    protocols = [
        PhaseBProtocol(
            protocol_id=f"{split}_toy",
            family="alien_device",
            split=split,
            duration=1.0,
            dt=0.1,
            input_names=("u",),
            specification={"kind": "constant", "value": 1.0},
        )
        for split in ("train", "validation")
    ]
    monkeypatch.setattr(audit, "phase_b_protocols", lambda _: protocols)
    monkeypatch.setattr(
        "autoformalism.rebuttal.attainability_reference.phase_b_protocols",
        lambda _: protocols,
    )
    problem, _ = synthetic_problem("shared", 0, 0)
    problem.update(
        candidate=candidate, context=context, initialization_plan={"rules": {}}
    )
    for key, protocol in zip(("train", "val"), protocols, strict=True):
        generated = _simulate_alien(protocol, spec, {})
        problem["splits"][key]["rows"] = [
            {
                "trajectory_id": f"{protocol.split}_000",
                "time": generated.time.tolist(),
                "targets": {"v01": generated.states[:, -1].tolist()},
                "auxiliaries": {},
                "external_inputs": {"u01": generated.inputs[:, 0].tolist()},
                "fixed_covariates": {f"known_z{i + 1}": 0.0 for i in range(5)},
            }
        ]
    bundle = {"spec": spec, "truth": truth}
    result = audit.scaled_reference_audit(problem, bundle, audit.ReferenceAuditConfig())
    assert result["pass"]
    original = audit.continuous_rollout

    def inconsistent(*args):
        value = original(*args)
        return (
            value + 1e-5 * result["training_output_scale"]
            if args[4] == "BDF"
            else value
        )

    monkeypatch.setattr(audit, "continuous_rollout", inconsistent)
    failed = audit.scaled_reference_audit(problem, bundle, audit.ReferenceAuditConfig())
    assert not failed["pass"]
    assert all(
        not rows[0]["tight_solver_agreement"]["pass"]
        for rows in failed["checks"].values()
    )
    assert all(
        all(c["pass"] for c in rows[0]["tight_vs_saved"].values())
        for rows in failed["checks"].values()
    )


def test_continuous_rollout_preserves_unsampled_input_pulse():
    from time import monotonic

    problem, _ = synthetic_problem("shared", 0, 0)
    problem.update(
        candidate={
            "candidate_id": "pulse_control",
            "parent_candidate_id": None,
            "states": [{"name": "v01", "kind": "observed"}],
            "state_equations": [{"state": "v01", "rhs": "k*u01"}],
            "observation_mappings": [{"channel": "v01", "expression": "v01"}],
            "parameters": [{"name": "k", "scope": "global", "role": "coefficient"}],
            "initial_conditions": [
                {"state": "v01", "scope": "global", "expression": "v01"}
            ],
        },
        context={
            "targets": ["v01"],
            "external_inputs": ["u01"],
            "fitted_initialization": True,
        },
        initialization_plan={"rules": {}},
    )
    row = problem["splits"]["train"]["rows"][0]
    row.update(
        time=[0.0, 0.5, 1.0],
        targets={"v01": [0.0, 0.0, 0.0]},
        external_inputs={"u01": [0.0, 0.0, 0.0]},
    )
    problem["splits"]["train"]["rows"] = [row]
    protocol = PhaseBProtocol(
        protocol_id="train_pulse",
        family="alien_device",
        split="train",
        duration=1.0,
        dt=0.5,
        input_names=("u",),
        specification={"kind": "pulse", "start": 0.123, "end": 0.127, "amplitude": 1.0},
    )
    system, _ = system_for(problem)
    trajectory = unpack_split(problem["splits"]["train"]).trajectories[0]
    for method in ("Radau", "BDF"):
        values = audit.continuous_rollout(
            system,
            trajectory,
            protocol,
            {"k": 1.0},
            method,
            audit.ReferenceAuditConfig(),
            monotonic() + 10,
        )
        np.testing.assert_allclose(values, [0, 0.004, 0.004], atol=1e-10)


def test_scaled_generation_failure_is_saved_and_blocks_fitting(tmp_path, monkeypatch):
    source = recovery_source_at(tmp_path)
    output = tmp_path / "recovered"
    prepare_reference_recovery(source, output, recovery_plan())
    calls = []

    def failed(*args):
        calls.append("audit")
        return {"pass": False, "native_generator_audit": {"train": [], "val": []}}

    def forbidden(*args, **kwargs):
        raise AssertionError("failed reference must not generate or fit")

    monkeypatch.setattr(campaign, "scaled_reference_audit", failed)
    monkeypatch.setattr(campaign, "simulate_split", forbidden)
    monkeypatch.setattr(campaign, "fit_collocation_forward_sensitivity", forbidden)
    result = campaign.generate(output, 1)
    assert result["status"] == "generation_failed"
    assert "scaled reference replay agreement failed" in result["error"]
    assert campaign.generate(output, 1) == result
    assert calls == ["audit"]
    assert campaign.execute(output, 7)["status"] == "generation_unavailable"


def test_recovery_launcher_selects_only_eleven_reference_arms(tmp_path):
    source = recovery_source_at(tmp_path, 4)
    output = tmp_path / "recovered"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "git").write_text(
        '#!/bin/sh\ncase "$1" in\nstatus) exit 0;;\n'
        "rev-parse) echo test-commit;;\n*) exit 2;;\nesac\n"
    )
    (bin_dir / "sbatch").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$AF_BATCH_TEST_LOG"\n'
        'wc -l < "$AF_BATCH_TEST_LOG" | tr -d " "\n'
    )
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "AF_REPO_ROOT": str(root),
        "AF_PYTHON": sys.executable,
        "AF_CASADI_ROOT": str(tmp_path),
        "AF_SOURCE_ROOT": str(source),
        "AF_OUTPUT_ROOT": str(output),
        "AF_BATCH_TEST_LOG": str(tmp_path / "batch.log"),
    }
    command = ["bash", str(root / "scripts/hpc/submit_reference_recovery_delta.sh")]
    subprocess.run(command, env=env, check=True, capture_output=True)
    log = (tmp_path / "batch.log").read_text().splitlines()
    assert len(log) == 4
    assert "--array=4%2" in log[1]
    assert "--dependency=afterok:1,afterany:2" in log[2]
    assert "--array=" + ",".join(map(str, range(28, 39))) + "%2" in log[2]
    assert "--dependency=afterany:3" in log[3]
    subprocess.run(command, env=env, check=True, capture_output=True)
    assert (tmp_path / "batch.log").read_text().splitlines() == log
    assert len(read_json(output / "freeze.json")["retained_results"]) == 28


def test_reference_verification_budget_is_explicit_and_legacy_stays_sixty(monkeypatch):
    from types import SimpleNamespace

    from autoformalism.rebuttal import piecewise_campaign

    problem, _ = synthetic_problem("shared", 0, 0)
    split = unpack_split(problem["splits"]["train"])
    calls = []

    def simulated(model, row, parameters, initials, settings, **kwargs):
        calls.append((settings.maximum_wall_time_seconds, kwargs["deadline"]))
        return SimpleNamespace(success=True, predictions={"v01": row.targets["v01"]})

    monkeypatch.setattr(piecewise_campaign, "simulate_trajectory", simulated)
    monkeypatch.setattr(piecewise_campaign, "monotonic", lambda: 100.0)
    assert piecewise_campaign.replay(None, split, {}, 1.0)["pass"]
    assert all(seconds == 60 and deadline == 160 for seconds, deadline in calls)
    calls.clear()
    assert campaign.checked_replay(None, split, {}, 1.0, None, seconds=240)["pass"]
    assert all(seconds == 240 and deadline == 340 for seconds, deadline in calls)

    def timed_out(*args, **kwargs):
        raise TimeoutError("verification exhausted")

    monkeypatch.setattr(campaign, "replay", timed_out)
    result = campaign.checked_replay(None, split, {}, 1.0, None, seconds=240)
    assert result == {
        "pass": False,
        "status": "verification_timeout",
        "error": "verification exhausted",
    }
    with pytest.raises(ValueError, match="protocol 2"):
        campaign.AttainabilityPlan(reference_rollout_seconds=240)
