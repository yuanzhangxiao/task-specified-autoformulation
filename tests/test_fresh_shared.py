"""Fresh composition keeps public-only gates, multi-output fitting and old policies."""

import json

import pytest

from autoformalism.llm.response_revision import ResponseRevisionClient
from autoformalism.rebuttal import fresh_shared as campaign
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.rebuttal.repair_comparison import BudgetedRepairClient
from autoformalism.search import fresh_shared as construction
from autoformalism.search.review_integrity import selection_policy
from scripts import smoke_fresh_shared as smoke


def test_new_matrix_is_fresh_full_only_fifteen_rounds():
    config = io.DeadlineConfig.model_validate_json(
        (io.REPO / "configs/dalla_shared_multi_pruning_v1.json").read_text()
    )
    assert config.rounds == 15 and len(io.tasks(config)) == 6
    assert config.fit_profile == "collocation-multi-target-v1"
    assert config.protocol not in io.CONTINUATION_PROTOCOLS
    assert config.scientific_judge == "off"
    assert campaign.policy()["selection"] == selection_policy()
    assert campaign.policy()["pruning"]["maximum_candidates_per_parent"] == 1
    for changes in (
        {"full_only": False},
        {"fit_profile": "collocation-single-target-v2"},
        {"protocol": io.INTEGRITY_PROTOCOL},
    ):
        with pytest.raises(ValueError):
            io.DeadlineConfig.model_validate({**config.model_dump(), **changes})


def test_frozen_policy_cannot_be_silently_changed(tmp_path):
    plan = smoke.fixture(tmp_path)
    io.verify(tmp_path)
    plan.pop("artifact_sha256")
    plan["fresh_shared_policy"]["pruning"]["validation_relative_tolerance"] = 0.5
    (tmp_path / "plan.json").unlink()
    sealed_write(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="policy differs"):
        io.verify(tmp_path)


def test_public_contract_visible_before_shared_inventory(tmp_path, monkeypatch):
    plan = smoke.fixture(tmp_path)
    task = plan["tasks"][0]
    cell = plan["cells"][task["cell"]]
    before = json.dumps(cell, sort_keys=True)

    def build(visible, got_task, client, directory, **kwargs):
        assert (
            "Runtime public target contract" in visible["brief"]["scientific_context"]
        )
        assert "Generate w causally" in visible["brief"]["scientific_context"]
        assert kwargs["retain_failed_draft"]
        assert got_task == task
        return {"status": "construction_failed", "bundle": None}

    monkeypatch.setattr(construction, "construct", build)
    assert (
        construction.fresh(cell, task, None, tmp_path)["status"]
        == "construction_failed"
    )
    assert json.dumps(cell, sort_keys=True) == before


def test_unbuilt_lineage_can_reconstruct_with_construction_budget(tmp_path):
    plan = smoke.fixture(tmp_path)
    task = plan["tasks"][0]
    sealed_write(
        io.round_path(tmp_path, task, 0) / "result.json",
        {
            "task": task,
            "round": 0,
            "selected": None,
            "construction_draft": None,
        },
    )
    client = pipeline._client(tmp_path, plan, task, 1, "http://unused", lambda: True)
    assert isinstance(client, BudgetedRepairClient)
    assert not isinstance(client, ResponseRevisionClient)


def test_missing_or_unfitted_parent_does_not_prune(tmp_path):
    plan = smoke.fixture(tmp_path)
    task = plan["tasks"][0]
    with pytest.raises(ValueError, match="final search round missing"):
        campaign.run_one(tmp_path, 0)
    sealed_write(
        io.round_path(tmp_path, task, 1) / "result.json",
        {
            "task": task,
            "round": 1,
            "selected": None,
            "status": "construction_failed",
        },
    )
    result = campaign.run_one(tmp_path, 0)
    assert result["status"] == "skipped_no_finite_parent"
    assert result == campaign.run_one(tmp_path, 0)
    assert not list((tmp_path / "pruning").rglob("started.json"))
    assert campaign.report(tmp_path)["pruning_status_counts"] == {
        "skipped_no_finite_parent": 1
    }


def test_real_fresh_multi_output_shared_revision_pruning_and_resume(tmp_path):
    result = smoke.run(tmp_path)
    assert result["observed_targets"] == 2
    assert result["pruning_fit_arms"] == ["control", "pruned"]
    assert not result["test_data_opened"]
    plan = io.verify(tmp_path)
    row = campaign.final_row(tmp_path, plan, plan["tasks"][0])
    receipt = (
        io.round_path(tmp_path, row["task"], row["parent"]["origin_round"])
        / "fit/result.json"
    )
    receipt.write_text("{}")
    with pytest.raises((ValueError, KeyError), match=r"receipt|result|sha256"):
        campaign.run_one(tmp_path, 0)


def test_unfitted_draft_token_preflight_preserves_null_residual_evidence(tmp_path):
    from autoformalism.llm.response_revision import ResponseRevisionClient
    from autoformalism.llm.staged_topology import StagedModelSettings
    from autoformalism.search.review_revision_multi import ScientificRevision

    shown = []

    def transport(url, body, timeout):
        shown.append(json.loads(body["messages"][1]["content"]))
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {"hypothesis": "Repair the public mapping."}
                        )
                    },
                }
            ],
            "usage": {"total_tokens": 10},
        }

    client = ResponseRevisionClient(
        settings=StagedModelSettings(),
        seed=0,
        namespace="draft-repair",
        directory=tmp_path,
        base_url="http://offline",
        can_start=lambda: True,
        transport=transport,
        token_transport=lambda *a: {"count": 1000, "max_model_len": 32768},
    )
    payload = {
        "training_evidence": None,
        "stage": "construction_repair",
        "public_target_contract": {"targets": ["a", "b"]},
    }
    client.call(
        system="Repair",
        user=json.dumps(payload),
        response_model=ScientificRevision,
        step="repair_public_construction",
        attempt=0,
    )
    assert shown == [payload]
