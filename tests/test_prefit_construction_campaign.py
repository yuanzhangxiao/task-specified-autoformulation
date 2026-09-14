"""Matched freeze, bounded failure, stage resume and public-data separation."""

import fcntl
import json
from pathlib import Path

import pytest

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal import prefit_construction_campaign as campaign
from autoformalism.schemas.staged_topology import PublicScientificBrief
from scripts.smoke_prefit_construction import CELL, client_for, synthetic_fixture


def test_pilot_config_is_two_cells_three_seeds_with_shared_fitter():
    config = campaign.ConstructionCampaignConfig.model_validate_json(
        Path("configs/prefit_matched_construction_v1.json").read_text()
    )
    assert len(config.public_cells) == 2 and config.seeds == (0, 1, 2)
    assert not config.fit.allow_derivative_regression
    assert config.scientific_judge == "off"
    changed = config.model_dump(mode="json")
    changed["fit"]["allow_derivative_regression"] = True
    with pytest.raises(ValueError, match="bounded general rollout"):
        campaign.ConstructionCampaignConfig.model_validate(changed)


def test_freeze_copies_only_public_assets_and_rejects_stale_source(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    fixture = synthetic_fixture(source)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(fixture["config"]))
    # This test's prose is synthetic; production still enforces exact prompt hashes.
    monkeypatch.setattr(
        campaign,
        "build_scientific_brief",
        lambda *a, **k: PublicScientificBrief.model_validate(
            fixture["cells"][CELL]["brief"]
        ),
    )
    original = Path.read_bytes

    def guarded(path):
        assert path.name != "test.csv"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    root = tmp_path / "frozen"
    plan = campaign.freeze(config_path, source / "public", root)
    assert campaign.freeze(config_path, source / "public", root) == plan
    assert campaign.verify(root) == plan
    assert {p.name for p in (root / "public/phase_b_v1" / CELL).iterdir()} == set(
        campaign.PUBLIC_FILES
    )
    assert [(t["arm"], t["seed"]) for t in plan["tasks"]] == [
        ("brief_only", 0),
        ("training_evidence", 0),
    ]
    source_file = source / "public/phase_b_v1" / CELL / "train.csv"
    source_file.write_text(source_file.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen public asset differs"):
        campaign.freeze(config_path, source / "public", root)
    assert campaign.verify(root) == plan
    monkeypatch.setattr(campaign, "runtime_source_hash", lambda: "changed")
    with pytest.raises(ValueError, match="runtime differs"):
        campaign.verify(root)


def test_only_packet_differs_and_every_call_and_failure_remain_counted(tmp_path):
    plan = synthetic_fixture(tmp_path)
    results = []
    calls_by_arm = {}
    for task in plan["tasks"]:
        calls = []
        client = client_for(tmp_path, plan, task, calls)
        result = campaign.construct_task(tmp_path, plan, task, client)
        assert result["status"] == "complete", result
        calls_by_arm[task["arm"]] = calls
        for request in calls:
            brief = request["public_brief"]
            assert ("training_observations" in brief) == (
                task["arm"] == "training_evidence"
            )
            assert "validation_" not in json.dumps(request)
        assert result["cost"]["physical_requests"] == len(calls)
        assert result["cost"]["observed_tokens"] == 100 * len(calls)
        before = len(calls)
        assert (
            campaign.construct_task(
                tmp_path, plan, task, client_for(tmp_path, plan, task, calls)
            )
            == result
        )
        assert len(calls) == before
        results.append(result)
    assert results[0]["functions"]["candidate"] == results[1]["functions"]["candidate"]
    for a, b in zip(
        calls_by_arm["brief_only"], calls_by_arm["training_evidence"], strict=True
    ):
        b = json.loads(json.dumps(b))
        b["public_brief"].pop("training_observations")
        assert a == b
    summary = campaign.summarize(tmp_path)
    assert not summary["pairs"][0]["topology_differs"]
    assert summary["pairs"][0]["validation_nmse_evidence_minus_brief"] is None


def test_deferred_calls_resume_with_one_cross_stage_budget(tmp_path):
    plan = synthetic_fixture(tmp_path)
    task = plan["tasks"][0]
    calls = []
    client = client_for(tmp_path, plan, task, calls)
    client.can_start = lambda: len(calls) < 3
    with pytest.raises(DeferredCall):
        campaign.construct_task(tmp_path, plan, task, client)
    assert campaign.summarize(tmp_path)["arms"]["brief_only"]["physical_requests"] == 3
    resumed = client_for(tmp_path, plan, task, calls)
    assert (
        campaign.construct_task(tmp_path, plan, task, resumed)["status"] == "complete"
    )
    assert len(resumed.records) == len(calls)


def test_exhausted_budget_is_terminal_without_fitting_or_budget_reset(
    tmp_path, monkeypatch
):
    plan = synthetic_fixture(tmp_path, request_budget=1)
    task = plan["tasks"][0]
    calls = []
    first = campaign.construct_task(
        tmp_path, plan, task, client_for(tmp_path, plan, task, calls)
    )
    assert first["status"] == "construction_failed"
    assert first["cost"]["physical_requests"] == 1
    monkeypatch.setattr(
        campaign,
        "fit_candidate",
        lambda *a, **k: pytest.fail("failed construction was fitted"),
    )
    assert campaign.fit_task(tmp_path, plan, task)["status"] == "not_run"
    assert (
        campaign.construct_task(
            tmp_path, plan, task, client_for(tmp_path, plan, task, calls)
        )
        == first
    )
    assert len(calls) == 1
    assert campaign.summarize(tmp_path)["arms"]["brief_only"]["expected"] == 1


def test_killed_fit_is_terminal_and_altered_checkpoint_rejected(tmp_path, monkeypatch):
    plan = synthetic_fixture(tmp_path)
    task = plan["tasks"][0]
    campaign.construct_task(tmp_path, plan, task, client_for(tmp_path, plan, task, []))
    directory = tmp_path / "results" / task["task_id"]
    campaign._sealed_write(
        directory / "fit_started.json", {"identity": campaign._identity(plan, task)}
    )
    monkeypatch.setattr(
        campaign,
        "fit_candidate",
        lambda *a, **k: pytest.fail("consumed fit budget reset"),
    )
    fitted = campaign.fit_task(tmp_path, plan, task)
    assert fitted["status"] == "interrupted"
    assert campaign.fit_task(tmp_path, plan, task) == fitted
    result = campaign.read(directory / "construction.json")
    result["status"] = "altered"
    (directory / "construction.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="artifact digest differs"):
        campaign.summarize(tmp_path)


def test_arm_lock_and_deadline_do_not_block_or_spend_other_task_budget(
    tmp_path, monkeypatch
):
    plan = synthetic_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(
        campaign,
        "_validated_client",
        lambda config, task, *a: client_for(tmp_path, plan, task, calls),
    )
    first = tmp_path / "results" / plan["tasks"][0]["task_id"]
    first.mkdir(parents=True)
    with (first / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = campaign.run(tmp_path, "construct")
    assert result["arms"]["brief_only"]["constructed"] == 0
    assert result["arms"]["training_evidence"]["constructed"] == 1
    before = len(calls)
    campaign.run(tmp_path, "construct", arm="brief_only", wall_seconds=0)
    assert len(calls) == before
    with pytest.raises(ValueError, match="unknown stage or arm"):
        campaign.run(tmp_path, "construct", arm="typo")


def test_changed_launcher_and_provider_cache_are_rejected(tmp_path, monkeypatch):
    plan = synthetic_fixture(tmp_path)
    task = plan["tasks"][0]
    campaign.construct_task(tmp_path, plan, task, client_for(tmp_path, plan, task, []))
    cache = next((tmp_path / "results" / task["task_id"] / "calls").glob("*.json"))
    record = campaign.read(cache)
    record["request"]["namespace"] = "other"
    cache.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="provider request provenance"):
        campaign.summarize(tmp_path)
    monkeypatch.setattr(campaign, "launcher_hash", lambda: "changed")
    with pytest.raises(ValueError, match="launcher differs"):
        campaign.verify(tmp_path)


def test_missing_cache_cannot_reset_budget_or_silently_reduce_reported_cost(tmp_path):
    plan = synthetic_fixture(tmp_path)
    task = plan["tasks"][0]
    campaign.construct_task(tmp_path, plan, task, client_for(tmp_path, plan, task, []))
    cache = next((tmp_path / "results" / task["task_id"] / "calls").glob("*.json"))
    cache.unlink()
    with pytest.raises(ValueError, match="missing a recorded request"):
        campaign.summarize(tmp_path)
    with pytest.raises(ValueError, match="missing a recorded request"):
        campaign.construct_task(
            tmp_path, plan, task, client_for(tmp_path, plan, task, [])
        )


def test_finite_fit_is_not_reported_as_optimizer_convergence(tmp_path):
    plan = synthetic_fixture(tmp_path)
    task = plan["tasks"][0]
    campaign._sealed_write(
        tmp_path / "results" / task["task_id"] / "fit.json",
        {
            "identity": campaign._identity(plan, task),
            "status": "evaluated",
            "fit": {
                "success": True,
                "best_start_index": 1,
                "validation_normalized_mse": 0.2,
                "validation_failed_trajectories": [],
                "diagnostics": [
                    {"start_index": 0, "success": True, "status": 1},
                    {
                        "start_index": 1,
                        "success": False,
                        "status": 0,
                        "message": "evaluation budget reached",
                    },
                ],
            },
        },
    )
    row = campaign.summarize(tmp_path)["rows"][0]
    assert row["fit_success"] is True
    assert row["optimizer_success"] is False
    assert row["optimizer_status"] == 0
    assert row["first_validation_nmse"] == 0.2
