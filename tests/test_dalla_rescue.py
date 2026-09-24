"""Separate allocations, exact retained parameters, pruning and scheduler resume."""

import copy
import json

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.rebuttal import dalla_rescue as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest
from scripts import smoke_process_pruning as smoke
from scripts import submit_dalla_rescue as submitter
from scripts.smoke_review_multi import example
from tests.test_process_pruning_campaign import backend
from tests.test_review_integrity import patch


def packet(tmp_path):
    toy = smoke.fixture(tmp_path / "toy")
    old = toy["rows"][0]
    source = tmp_path / "inputs.json"
    sealed_write(
        source,
        {
            "protocol": campaign.INPUT_PROTOCOL,
            "test_data_opened": False,
            "cells": toy["cells"],
            "rows": [
                {
                    "task": old["task"],
                    "request": old["parent"]["request"],
                    "parameters": old["parent"]["fit"]["parameters"],
                }
            ],
        },
    )
    return source, tmp_path / "out"


def test_opt_in_budget_preserves_existing_profiles(tmp_path):
    source, _ = packet(tmp_path)
    raw = sealed_read(source)["rows"][0]["request"]
    original = public.profile_settings(PublicFitRequest.model_validate(raw))
    larger = public.profile_settings(campaign._request(raw))
    expected = {
        "initializer_seconds": 300.0,
        "refinement_seconds": 900.0,
        "maximum_function_evaluations": 1200,
        "node_warmup_seconds": 30.0,
        "recovery_probe_seconds": 60.0,
    }
    assert larger == {**original, **expected}
    assert original["refinement_seconds"] == 180
    assert original["recovery_probe_seconds"] == 10


def test_real_seed_replay_then_equal_pruning_allocations_and_resume(
    tmp_path, monkeypatch
):
    source, root = packet(tmp_path)
    plan = campaign.freeze(source, root)
    before = source.read_bytes()
    calls = []
    monkeypatch.setattr(public, "_run_backend", backend(calls))
    result = campaign.run_one(root, 0)
    assert len(calls) == 3
    assert calls[0]["parameters"] == plan["rows"][0]["parameters"]
    assert all(c["settings"] == calls[0]["settings"] for c in calls)
    assert result["seed"]["training"]["normalized_mse"] < 1e-8
    assert not result["seed"]["validation_initials_fitted"]
    assert result["pre_pruning_origin"] == "seed"
    assert result["pruning"]["selection"]["selected"] == "pruned"
    snapshots = {str(p): p.read_bytes() for p in (root / "results").rglob("*.json")}
    assert campaign.run_one(root, 0) == result
    assert len(calls) == 3 and source.read_bytes() == before
    assert snapshots == {
        str(p): p.read_bytes() for p in (root / "results").rglob("*.json")
    }
    report = campaign.report(root)
    assert report["status"] == "complete"
    models = json.loads((root / "models.json").read_text())
    assert (
        models["models"][0]["result"]["selected_request"] == result["selected_request"]
    )
    assert not models["interventions_evaluated"]


def test_training_selection_does_not_rank_by_validation(tmp_path):
    source, root = packet(tmp_path)
    plan = campaign.freeze(source, root)
    row = plan["rows"][0]
    seed = campaign.replay_seed(row, plan["cells"][row["task"]["cell"]])
    seed["training"]["normalized_mse"] = 0.2
    fitted = copy.deepcopy(seed)
    fitted["training"]["normalized_mse"] = 0.3
    fitted["validation"]["normalized_mse"] = 0.0
    assert campaign.retain(seed, fitted)[0] == "seed"
    fitted["training"]["normalized_mse"] = 0.1
    fitted["validation"]["normalized_mse"] = 100.0
    assert campaign.retain(seed, fitted)[0] == "refit"
    fitted["training"]["normalized_mse"] = 0.2
    assert campaign.retain(seed, fitted)[0] == "seed"


@pytest.mark.parametrize("stage", ["seed", "refit"])
def test_interrupted_allocations_are_consumed(tmp_path, monkeypatch, stage):
    source, root = packet(tmp_path)
    plan = campaign.freeze(source, root)
    row = plan["rows"][0]
    directory = root / "results" / row["task"]["task_id"]
    directory.mkdir(parents=True)
    if stage == "seed":
        sealed_write(
            directory / "seed_started.json", {"identity": plan["artifact_sha256"]}
        )
        monkeypatch.setattr(
            campaign, "replay_seed", lambda *a: pytest.fail("replayed consumed seed")
        )
    else:
        cell = plan["cells"][row["task"]["cell"]]
        request = PublicFitRequest.model_validate(row["request"])
        frozen = sibling_fit.prepare_child_fit(
            request,
            request,
            row["parameters"],
            campaign.PublicSplit.model_validate(cell["training"]),
            campaign.PublicSplit.model_validate(cell["validation"]),
            directory / "refit",
            lineage={"campaign": plan["artifact_sha256"], "task": row["task"]},
        )
        public._write(
            directory / "refit/started.json", {"identity": frozen["identity"]}
        )
    calls = []
    monkeypatch.setattr(public, "_run_backend", backend(calls))
    result = campaign.run_one(root, 0)
    assert result[stage]["status"] == "interrupted"
    assert len(calls) == (3 if stage == "seed" else 2)
    assert campaign.run_one(root, 0) == result


def test_failed_fits_are_terminal_not_successful_models(tmp_path, monkeypatch):
    source, root = packet(tmp_path)
    campaign.freeze(source, root)
    monkeypatch.setattr(
        campaign,
        "replay_seed",
        lambda *a: {"status": "rollout_failed", "parameters": None},
    )
    monkeypatch.setattr(public, "_run_backend", lambda *a: {})
    result = campaign.run_one(root, 0)
    assert result["status"] == "no_finite_model" and result["selected_fit"] is None
    assert campaign.report(root)["rows"][0]["status"] == "no_finite_model"


def test_freeze_drift_and_missing_report(tmp_path):
    source, root = packet(tmp_path)
    assert campaign.report(root)["status"] == "not_prepared"
    assert (root / "summary.json").is_file()
    first = campaign.freeze(source, root)
    assert campaign.freeze(source, root) == first
    assert campaign.report(root)["recorded"] == 0
    with pytest.raises(ValueError, match="index"):
        campaign.run_one(root, -1)
    source.write_text(source.read_text().replace('"eps": 0.0', '"eps": 1.0'))
    with pytest.raises(ValueError, match="digest"):
        campaign.verify(root)


@pytest.mark.parametrize(
    "mutation", ["missing_parameter", "unsafe_id", "test", "missing_target"]
)
def test_bad_packets_fail_before_numerical_work(tmp_path, mutation):
    source, root = packet(tmp_path)
    raw = sealed_read(source)
    raw.pop("artifact_sha256")
    if mutation == "missing_parameter":
        raw["rows"][0]["parameters"].pop("a")
    elif mutation == "unsafe_id":
        raw["rows"][0]["task"]["task_id"] = "../wrong"
    elif mutation == "test":
        raw["test_data_opened"] = True
    else:
        raw["rows"][0]["request"]["base_candidate"]["observation_mappings"] = []
    other = tmp_path / "bad.json"
    sealed_write(other, raw)
    with pytest.raises(ValueError):
        campaign.freeze(other, root)
    assert not (root / "plan.json").exists()


def test_explicit_collision_correction_keeps_incumbent_roles():
    bundle, *_ = example()
    before = copy.deepcopy(bundle)
    result = campaign.correct_declaration(bundle, patch(), "c", "tau_delay_rescue")
    assert bundle == before
    assert not result["verified_scientific_citations"]
    params = result["revision"]["bundle"]["initialization"]["base_candidate"][
        "parameters"
    ]
    assert next(p for p in params if p["name"] == "c")["role"] == "rate"
    assert (
        next(p for p in params if p["name"] == "tau_delay_rescue")["role"]
        == "time_constant"
    )
    with pytest.raises(ValueError, match="occupied"):
        campaign.correct_declaration(bundle, patch(), "c", "i")
    with pytest.raises(ValueError, match="no identified"):
        campaign.correct_declaration(bundle, patch("fresh"), "fresh", "other")


def test_submission_partial_resume_and_adoption(tmp_path, monkeypatch):
    source, root = packet(tmp_path)
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setenv("AF_PYTHON", str(python))
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    monkeypatch.setattr(submitter, "source_commit", lambda _: "a" * 40)
    calls = []

    def submit(directory, key, options, worker, stage, index):
        argv = ["sbatch", "--parsable", *options, str(worker), stage, str(index)]
        public._write(directory / f"{key}.intent.json", {"argv": argv})
        calls.append(key)
        if key == "fit":
            raise ValueError("uncertain scheduler reply")
        (directory / f"{key}.id").write_text(str(100 + len(calls)))
        return str(100 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", submit)
    with pytest.raises(ValueError, match="uncertain"):
        submitter.submit(source, root)
    with pytest.raises(ValueError, match="Unconfirmed fit"):
        submitter.submit(source, root)
    assert calls == ["prepare", "fit"]
    verified = []
    monkeypatch.setattr(
        submitter,
        "scheduler_record",
        lambda job, argv, **kw: verified.append((job, argv)) or {"verified": True},
    )
    result = submitter.submit(source, root, {"fit": "777"})
    assert result["jobs"]["fit"] == "777" and len(verified) == 1
    assert "--array=0-0%3" in verified[0][1]
    assert result["gpus"] == 0 and calls == ["prepare", "fit", "report"]
    assert submitter.submit(source, root) == result
