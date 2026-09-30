"""Matched scopes, frozen evidence, failures, reporting and scheduler receipts."""

import hashlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.benchmarks.fitting_qualification_inputs import experiment_request
from autoformalism.benchmarks.reference_qualification import cstr_request
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import qualification as q
from autoformalism.fitting.simulation import trajectory_initial_state


@pytest.fixture
def inputs(tmp_path):
    spec = {
        "parameters": {
            "k0": math.exp(25),
            "E_over_R": 8750,
            "flow_rate": 1,
            "source_gain": 80,
            "exchange_rate": 2.5,
            "secondary_flow_rate": 1.5,
            "secondary_exchange_rate": 1.5,
        },
        "equilibrium": {"C": 0.26, "T": 365, "Tj": 347},
    }
    req, truth = cstr_request("hard", "generic", spec)
    payload = req.model_dump(mode="json")
    payload["initialization_plan"]["rules"] = {
        s: {"initial": {"mode": "value", "guess": g}}
        for s, g in (("C", 0.25), ("Tj", 345))
    }
    truth = {n: v for n, v in truth.items() if not n.startswith("init_")}
    truth.update(init_C_value=0.26, init_Tj_value=347)
    case = {
        "request": payload,
        "reference_parameters": truth,
        "state_channel_proxies": {"T": "T", "Tj": "Tjf", "C": "Cf"},
    }
    for name, field in (("train", "training"), ("val", "validation")):
        case[field] = {
            "name": name,
            "fingerprint": name,
            "rows": [
                {
                    "trajectory_id": name,
                    "time": [0, 1],
                    "targets": {"T": [365, 367]},
                    "auxiliaries": {},
                    "external_inputs": {
                        "Cf": [1, 1],
                        "Tf": [350, 350],
                        "Tjf": [330, 330],
                    },
                }
            ],
        }
    path = tmp_path / "inputs.json"
    # Other roster fixtures deliberately reuse the small valid equation fixture.
    seal(
        path,
        {
            "protocol": "phase-c-fitting-inputs-1",
            "cases": dict.fromkeys(("cstr_easy", "cstr_hard", "basin_coupled"), case),
        },
    )
    return path


def test_factorial_preserves_physical_starts_and_domains(inputs, tmp_path):
    root = tmp_path / "campaign"
    first = q.prepare(inputs, root, q.QualificationConfig())
    assert first["tasks"] == 42
    assert q.prepare(inputs, root, q.QualificationConfig()) == first
    plan, _ = q.verify(root)
    tasks = [
        t
        for t in plan["tasks"]
        if t["case"] == "cstr_hard" and t["scope"] == "joint" and t["seed"] == 0
    ]
    assert len(tasks) == 4 and all(t["start"] == tasks[0]["start"] for t in tasks)
    models = []
    for task in tasks:
        model, start, _ = public._lower(
            q.PublicFitRequest.model_validate(task["request"])
        )
        models.append(model)
        assert (
            model.validated.candidate.state_equations
            == models[0].validated.candidate.state_equations
        )
        p = next(
            p for p in model.validated.candidate.parameters if p.name == "init_C_value"
        )
        assert p.domain.value == (
            "nonnegative" if task["arm"] in {"domain", "scaled_domain"} else "real"
        )
        row = public.unpack_split(
            q.PublicSplit.model_validate(
                read_seal(inputs)["cases"]["cstr_hard"]["training"]
            )
        ).trajectories[0]
        np.testing.assert_allclose(
            trajectory_initial_state(model, row, {}, parameters=start),
            trajectory_initial_state(models[0], row, {}, parameters=tasks[0]["start"]),
        )
    report = q.report(root)
    assert report["status_counts"] == {"missing": 42}
    assert len(report["paired_comparisons"]) == 33


def test_known_blocks_and_truth_do_not_set_generic_guesses(inputs):
    case = read_seal(inputs)["cases"]["cstr_hard"]
    p, _ = experiment_request(case, "parameters_only", 0, False)
    i, _ = experiment_request(case, "initials_only", 0, False)
    j, truth = experiment_request(case, "joint", 0, False)
    assert all(not n.startswith("init_") for n in public._lower(p)[0].parameter_names)
    assert all(n.startswith("init_") for n in public._lower(i)[0].parameter_names)
    assert public._lower(j)[1] != truth
    changed = {
        **case,
        "reference_parameters": {
            n: v * 2 for n, v in case["reference_parameters"].items()
        },
    }
    assert (
        public._lower(experiment_request(changed, "joint", 0, False)[0])[1]
        == public._lower(j)[1]
    )


def test_export_verifies_release_assets_and_does_not_open_test(inputs, tmp_path):
    from autoformalism.benchmarks import fitting_qualification_inputs as source

    case = read_seal(inputs)["cases"]["cstr_hard"]
    release, data = tmp_path / "release", tmp_path / "data"
    spec_path = data / source.SPEC_PATH
    spec_path.parent.mkdir(parents=True)
    public._write(
        spec_path,
        {
            "parameters": {
                "k0": math.exp(25),
                "E_over_R": 8750,
                "flow_rate": 1,
                "source_gain": 80,
                "exchange_rate": 2.5,
                "secondary_flow_rate": 1.5,
                "secondary_exchange_rate": 1.5,
            },
            "equilibrium": {"C": 0.26, "T": 365, "Tj": 347},
        },
    )
    plan = {
        "private_spec_sha256": {
            "cstr": hashlib.sha256(spec_path.read_bytes()).hexdigest()
        }
    }
    records = []
    for cell in source.CELLS.values():
        directory = release / "public" / cell
        for name, value in (
            ("train.json", case["training"]),
            ("val.json", case["validation"]),
            ("specification.json", {}),
        ):
            seal(directory / name, value)
        (directory / "proposer_prompt.txt").write_text("Public task")
        # Deliberately unreadable as JSON; test data must never be consumed.
        (directory / "test.json").write_text("DO NOT OPEN")
        records.append(
            {
                "cell": cell,
                "ready_for_development": True,
                "files": {
                    str(p.relative_to(release)): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in directory.iterdir()
                    if p.name != "test.json"
                },
            }
        )
    summary = {
        "protocol": "phase-c-development-2",
        "whole_phase_c_roster_ready": True,
        "plan_sha256": public.content_sha256(plan),
        "test_generated": False,
        "cells": records,
    }
    seal(release / "plan.json", plan)
    seal(release / "summary.json", summary)
    exported = tmp_path / "exported.json"
    assert source.export_inputs(release, data, exported)["cases"] == 3
    assert not read_seal(exported)["test_data_opened"]
    with pytest.raises(ValueError, match="outside"):
        source.export_inputs(release, data, release / "inputs.json")
    path = release / "public" / source.CELLS["cstr_easy"] / "proposer_prompt.txt"
    path.write_text("Modified prompt")
    with pytest.raises(ValueError, match="release asset differs"):
        source.export_inputs(release, data, tmp_path / "changed.json")


def test_drift_and_interrupted_budget(inputs, tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    q.prepare(inputs, root, q.QualificationConfig(starts=1))
    plan, _ = q.verify(root)
    seal(
        root / "qualification/result.json",
        {"passed": True, "plan_sha256": public.content_sha256(plan)},
    )
    task = plan["tasks"][0]
    seal(root / "results" / task["task_id"] / "started.json", {"started": True})
    monkeypatch.setattr(
        q,
        "fit_collocation_forward_sensitivity",
        lambda *a, **k: pytest.fail("budget reset"),
    )
    result = q.run_task(root, 0)
    assert result["status"] == "interrupted"
    assert q.run_task(root, 0) == result
    assert q.report(root)["recorded"] == 1
    monkeypatch.setattr(public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="source/runtime"):
        q.run_task(root, 1)


def test_complete_resume_and_backend_tampering(inputs, tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    q.prepare(inputs, root, q.QualificationConfig(starts=1))
    plan, _ = q.verify(root)
    seal(
        root / "qualification/result.json",
        {"passed": True, "plan_sha256": public.content_sha256(plan)},
    )
    task = plan["tasks"][0]
    monkeypatch.setattr(
        q,
        "fit_collocation_forward_sensitivity",
        lambda *a, **k: {
            "status": "complete",
            "parameters": task["reference_parameters"],
            "initializer": {},
            "refinement": {},
        },
    )
    monkeypatch.setattr(
        q,
        "replay",
        lambda *a, **k: {
            "complete": True,
            "metrics": {"train": 0.0, "val": 0.0},
            "maximum_solver_difference": 0,
        },
    )
    result = q.run_task(root, 0)
    assert result["accuracy_passed"]
    monkeypatch.setattr(
        q,
        "fit_collocation_forward_sensitivity",
        lambda *a, **k: pytest.fail("refitted"),
    )
    assert q.run_task(root, 0) == result
    backend = root / "results" / task["task_id"] / "backend.json"
    backend.unlink()
    seal(backend, {"different": True})
    with pytest.raises(ValueError, match="backend evidence"):
        q.report(root)


@pytest.mark.parametrize("confirmed", [True, False])
def test_scheduler_receipts_no_duplicate_or_gpu(
    inputs, tmp_path, monkeypatch, confirmed
):
    spec = importlib.util.spec_from_file_location(
        "qualification_submit", "scripts/submit_phase_c_fitting.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("AF_PYTHON", __import__("sys").executable)
    calls = []

    def scheduler(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(
            returncode=0,
            stdout=f"{1000 + len(calls)}\n" if confirmed else "\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", scheduler)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "a" * 40)
    root = tmp_path / "campaign"
    config = Path("configs/phase_c_fitting_v1.json")
    for _ in range(2):
        if confirmed:
            result = module.submit(
                root, inputs, config, account="test-cpu", concurrency=2
            )
            assert result["array_tasks"] == 42
        else:
            with pytest.raises(ValueError, match=r"[Uu]nconfirmed"):
                module.submit(root, inputs, config, account="test-cpu", concurrency=2)
    assert len(calls) == (3 if confirmed else 1)
    assert all(
        "--partition=cpu" in cmd and not any("--gres" in s for s in cmd)
        for cmd in calls
    )
    if confirmed:
        assert "--array=0-41%2" in calls[1]
