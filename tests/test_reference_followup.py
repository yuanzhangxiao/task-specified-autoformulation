"""Matched starts, public-only initializer coordinates, and frozen follow-up resume."""

import math
from pathlib import Path

import numpy as np
import pytest

from autoformalism.benchmarks import reference_followup as followup
from autoformalism.benchmarks import reference_qualification as qualification
from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.simulation import trajectory_initial_state
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    specification = {
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
    seal(
        root / "plan.json",
        {
            "protocol": "phase-c-cstr-qualification-1",
            "source_sha256": public._source_identity(),
            "assistance": "unit fixture with evaluator equations",
            "other_families": {},
        },
    )
    tasks = []
    for tier in ("easy", "hard"):
        for start in qualification.STARTS:
            request, truth = qualification.cstr_request(tier, start, specification)
            task = {"task": f"cstr_{tier}_{start}", "tier": tier, "start": start}
            rows = [
                {
                    "trajectory_id": f"train_{i}",
                    "time": [0, 0.01],
                    "targets": {"T": [365 + 5 * i, 365 + 5 * i]},
                    "auxiliaries": {"C": [0.26 + 0.08 * i] * 2, "Tj": [347 + 3 * i] * 2}
                    if tier == "easy"
                    else {},
                    "external_inputs": {
                        "Cf": [1] * 2,
                        "Tf": [350] * 2,
                        "Tjf": [330] * 2,
                    },
                }
                for i in range(2)
            ]
            train = PublicSplit(name="train", fingerprint="fixture", rows=rows)
            val = PublicSplit(name="val", fingerprint="fixture-val", rows=rows[:1])
            directory = root / task["task"]
            frozen = public.prepare_fit(request, train, val, directory / "fit")
            seal(
                directory / "qualification.json",
                {
                    "status": "complete",
                    "fit": {"identity": frozen["identity"]},
                },
            )
            seal(directory / "reference_parameters.json", truth)
            tasks.append(task)
    seal(root / "tasks.json", {"tasks": tasks})
    return root


def test_matched_roster_original_starts_and_no_test_access(
    source, tmp_path, monkeypatch
):
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    original = Path.open

    def guarded(self, *args, **kwargs):
        assert "test.csv" not in str(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    root = tmp_path / "followup"
    first = followup.prepare(root, source)
    assert first["tasks"] == 14
    assert followup.prepare(root, source) == first
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert followup.report(root)["status"] == "incomplete"
    for task in read_seal(root / "tasks.json")["tasks"]:
        old = public._read(source / task["source_task"] / "fit/freeze.json")
        new = public._read(root / task["task"] / "fit/freeze.json")
        assert new["training"] == old["training"]
        assert new["validation"] == old["validation"]
        assert new["request"]["base_candidate"] == old["request"]["base_candidate"]
        assert (
            new["request"]["parameter_guesses"] == old["request"]["parameter_guesses"]
        )
        settings = old["settings"].copy()
        if task["arm"].startswith("local"):
            settings["poll_policy"] = "local"
        elif task["arm"] == "wide_long":
            settings.update(
                initializer_seconds=300,
                refinement_seconds=900,
                maximum_function_evaluations=1200,
            )
        assert new["settings"] == settings


def test_initializer_values_preserved_and_concentration_positive(source):
    frozen = public._read(source / "cstr_hard_reference_near/fit/freeze.json")
    old = PublicFitRequest.model_validate(frozen["request"])
    train = PublicSplit.model_validate(frozen["training"])
    truth = read_seal(source / "cstr_hard_reference_near/reference_parameters.json")
    new, new_truth, audit = followup.physical_initializers(old, truth, train)
    old_model, old_guesses, _ = public._lower(old)
    new_model, new_guesses, _ = public._lower(new)
    trajectories = public.unpack_split(train).trajectories
    for row in trajectories:
        for a, b in ((truth, new_truth), (old_guesses, new_guesses)):
            old_y = trajectory_initial_state(old_model, row, {}, parameters=a)
            new_y = trajectory_initial_state(new_model, row, {}, parameters=b)
            np.testing.assert_allclose(old_y, new_y, atol=1e-12)
    assert audit["concentration_nonnegative_by_construction"]
    assert not audit["jacket_initial_domain_constrained"]
    domains = {p.name: p.domain.value for p in new_model.validated.candidate.parameters}
    assert domains["init_C_lo"] == domains["init_C_hi"] == "nonnegative"
    for temperature in (0, 364, 365, 367, 370, 1000):
        data = train.model_dump(mode="json")
        data["rows"][0]["targets"]["T"] = [temperature] * 2
        row = public.unpack_split(PublicSplit.model_validate(data)).trajectories[0]
        y = trajectory_initial_state(
            new_model,
            row,
            {},
            parameters={**new_guesses, "init_C_lo": 0, "init_C_hi": 1},
        )
        assert 0 <= y[list(new_model.state_names).index("C")] <= 1


@pytest.mark.parametrize("change", ["freeze", "result", "truth", "roster"])
def test_source_tampering_rejected(source, tmp_path, change):
    paths = {
        "freeze": "cstr_easy_generic/fit/freeze.json",
        "result": "cstr_easy_generic/qualification.json",
        "truth": "cstr_easy_generic/reference_parameters.json",
        "roster": "tasks.json",
    }
    path = source / paths[change]
    value = public._read(path)
    if change == "freeze":
        value["initial_parameters"]["flow"] = 9
    else:
        value["value"]["tamper"] = True
    public._write(path, value)
    with pytest.raises(ValueError):
        followup.prepare(tmp_path / "followup", source)


def test_followup_execution_resume_and_interrupted_budget(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "followup"
    followup.prepare(root, source)
    tasks = read_seal(root / "tasks.json")["tasks"]
    calls = []

    def backend(request, model, training, validation, guesses, settings, directory):
        calls.append(request.source.task_id)
        metric = {
            "normalized_mse": 0.0,
            "per_target_normalized_mse": {"T": 0.0},
            "failed_trajectories": [],
            "trajectory_initial_conditions": {"train_0": {"C": 0.2}},
        }
        return {
            "parameters": guesses,
            "training": metric,
            "validation": metric,
            "refinement": {"stages": [], "budget_exhausted": False},
        }

    monkeypatch.setattr(public, "_run_backend", backend)
    monkeypatch.setattr(
        qualification,
        "replay",
        lambda *a: {
            "complete": True,
            "metrics": {s: {"nmse": 0, "complete": True} for s in ("train", "val")},
            "rows": [{"solver_scaled_max_difference": 0}],
        },
    )
    qualification.run_task(root, 0)
    qualification.run_task(root, 0)
    assert len(calls) == 1
    second = root / tasks[1]["task"] / "fit"
    public._write(
        second / "started.json",
        {"identity": public._read(second / "freeze.json")["identity"]},
    )
    result = qualification.run_task(root, 1)
    assert result["fit"]["status"] == "interrupted"
    assert len(calls) == 1
    report = followup.report(root)
    assert report["expected"] == 14 and report["status"] == "incomplete"
    assert report["rows"][0]["training_initial_concentration_minimum"] == 0.2
    assert report["rows"][1]["fit_status"] == "interrupted"


def test_preparing_another_worker_does_not_lock_an_active_fit(source, tmp_path):
    root = tmp_path / "followup"
    prepared = followup.prepare(root, source)
    task = read_seal(root / "tasks.json")["tasks"][0]["task"]
    with public._lock(root / task / "fit"):
        assert followup.prepare(root, source) == prepared
    freeze_path = root / task / "fit/freeze.json"
    frozen = public._read(freeze_path)
    frozen["settings"]["refinement_seconds"] = 1000
    public._write(freeze_path, frozen)
    with pytest.raises(ValueError, match="freeze differs"):
        followup.prepare(root, source)
