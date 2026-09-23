"""Pilot matrix and all-target evidence through the production revision boundary."""

import json
import subprocess
from pathlib import Path

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.schemas.public_fitting import PublicFitResult
from autoformalism.search.training_evidence import build_training_evidence
from scripts.smoke_multi_target_fitting import control
from tests.test_review_deadline_submission import submission_fixture

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/dalla_mechanism_pilots_v1.json").read_text())


def test_pilot_has_six_full_lineages_three_visits_and_all_required_outputs():
    plan = io.DeadlineConfig.model_validate(config())
    tasks = io.tasks(plan)
    assert len(tasks) == 6 and plan.rounds == 3
    assert {t["arm"] for t in tasks} == {"full"}
    assert all(t["shared_round_zero"] is None for t in tasks)
    assert [io.public_validation_context(c).targets for c in plan.public_cells] == [
        ("Gp",),
        ("Gp", "I", "U"),
        ("Gp", "I"),
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"fit_profile": "collocation-single-target-v2"},
        {"protocol": "review-deadline-3"},
        {"no_latent_cells": ["phase_b_dalla_man_t1_canonical_named_hard"]},
    ],
)
def test_invalid_pilot_combinations_fail_before_provider_calls(changes):
    with pytest.raises(ValueError):
        io.DeadlineConfig.model_validate({**config(), **changes})


def test_training_and_revision_evidence_include_each_target(tmp_path):
    request, train, val, truth = control()
    data = public.unpack_split(train)
    evidence = build_training_evidence(data, request.context)
    assert set(evidence.trajectories[0].targets) == {"Gp", "I", "U"}
    public.prepare_fit(request, train, val, tmp_path / "fit")
    frozen = public._read(tmp_path / "fit/freeze.json")
    result = PublicFitResult(
        **public._result_base(frozen, request),
        status="complete",
        message="synthetic retained truth",
        parameters=truth,
        training={
            "available": True,
            "normalized_mse": 0,
            "per_target_normalized_mse": dict.fromkeys(request.context.targets, 0),
        },
    )
    packet = pipeline.replay_packet(
        tmp_path, tmp_path / "replay", request, result, train
    )
    assert {r["target"] for r in packet["rows"]} == {"Gp", "I", "U"}
    assert len(packet["rows"]) == 6
    assert (
        pipeline.replay_packet(tmp_path, tmp_path / "replay", request, result, train)
        == packet
    )


def test_full_only_launcher_prints_matrix_and_submits_only_its_fit_indices(tmp_path):
    root, env, command, log = submission_fixture(tmp_path, io.CONTENT_PROTOCOL)
    path = root / "plan.json"
    plan = json.loads(path.read_text())
    plan["config"].update(full_only=True, fit_profile="collocation-multi-target-v1")
    plan["tasks"] = [t for t in plan["tasks"] if t["arm"] == "full"]
    path.write_text(json.dumps(plan))
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=True
    )
    preflight, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert preflight["lineages"] == 1 and preflight["planned_task_visits"] == 3
    assert preflight["fit_profile"] == "collocation-multi-target-v1"
    assert preflight["public_asset_hashes"] == {
        "cell": {"train.csv": "synthetic-digest"}
    }
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    fits = [call for call in calls if "--job-name=review-v2-fit-0" in call]
    assert len(fits) == 1 and "--array=0%16" in fits[0]
    saved = log.read_bytes()
    subprocess.run(command, env=env, capture_output=True, check=True)
    assert log.read_bytes() == saved
