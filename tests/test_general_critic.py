"""General critic isolation, bounded repair, hard gates and historical handoffs."""

import copy
import json

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import general_critic as campaign
from autoformalism.rebuttal import general_critic_io as io
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest
from scripts import smoke_general_critic as smoke
from scripts import submit_general_critic as submitter
from tests.test_process_pruning_campaign import backend


def ready(root, monkeypatch):
    plan = smoke.fixture(root)
    assert campaign.prepare_evidence(root, 0)["status"] == "available"
    reviews = []

    def review(*args):
        reviews.append(args[0])
        return smoke.prescribed_review(*args)

    monkeypatch.setattr(campaign.judge, "perform_review", review)
    campaign.review_one(root, 0, "parent", "http://offline")
    return plan, reviews


def test_symbolic_judge_contract_is_unchanged_and_numerically_blind(tmp_path):
    plan = smoke.fixture(tmp_path)
    row = plan["rows"][0]
    request = campaign.critic_request(plan, row)
    assert request["protocol"] == campaign.judge.judge_protocol()
    assert set(request) == {
        "parent",
        "candidate",
        "context",
        "public_prompt",
        "seed",
        "model_revision",
        "protocol",
    }
    assert request["parent"] == row["bundle"]["candidate"]
    # A change of estimates, convergence, external certificates or observations
    # must not change a single byte of the calibrated judge request.
    changed = copy.deepcopy(row)
    changed["fit"] = {
        "parameters": {"secret": 827364},
        "validation": "VALIDATION_SENTINEL",
    }
    changed["certificate"] = {"all_public_graph_requirements_certified": False}
    altered = copy.deepcopy(plan)
    altered["cells"]["toy"]["training"] = "TRAINING_SENTINEL"
    altered["cells"]["toy"]["validation"] = "VALIDATION_SENTINEL"
    assert campaign.critic_request(altered, changed) == request
    assert "SENTINEL" not in json.dumps(request)


def test_changed_law_reviewed_then_fitted_and_resume_has_no_new_work(
    tmp_path, monkeypatch
):
    _, reviews = ready(tmp_path, monkeypatch)
    calls, fits = [], []
    monkeypatch.setattr(public, "_run_backend", backend(fits))
    proposal = campaign.propose_one(
        tmp_path, 0, "http://offline", transport=smoke.transport_for(calls)
    )
    assert proposal["status"] == "committed", proposal
    assert calls[0]["scientific_advice"]["used_for_selection"] is False
    assert "validation" not in calls[0] and "test" not in calls[0]
    campaign.review_one(tmp_path, 0, "child", "http://offline")
    assert len(reviews) == 2 and reviews[1]["parent"] != reviews[1]["candidate"]
    assert len(fits) == 0
    result = campaign.fit_one(tmp_path, 0)
    assert len(fits) == 1 and result["status"] == "complete"
    assert not result["critic_used_for_selection"]
    assert (
        campaign.propose_one(
            tmp_path, 0, "http://offline", transport=smoke.transport_for(calls)
        )
        == proposal
    )
    assert campaign.fit_one(tmp_path, 0) == result
    campaign.review_one(tmp_path, 0, "child", "http://offline")
    assert len(calls) == 1 and len(fits) == 1 and len(reviews) == 2
    report = campaign.report(tmp_path)
    assert report["proposer_cost"]["physical_requests"] == 1
    assert report["judge_cost"]["unique_review_requests"] == 2


def test_no_change_reuses_review_and_does_not_grant_extra_fit(tmp_path, monkeypatch):
    _, reviews = ready(tmp_path, monkeypatch)
    proposal = campaign.propose_one(
        tmp_path, 0, "http://offline", transport=smoke.transport_for([], no_change=True)
    )
    assert proposal["status"] == "no_change", proposal
    child = campaign.review_one(tmp_path, 0, "child", "http://offline")
    assert child["reused_parent_review"] and len(reviews) == 1
    monkeypatch.setattr(
        public, "_run_backend", lambda *a: pytest.fail("unrequested refit")
    )
    assert campaign.fit_one(tmp_path, 0)["selected"] == "parent"


def test_indeterminate_review_is_not_a_failure_or_approval(tmp_path, monkeypatch):
    plan = smoke.fixture(tmp_path)
    campaign.prepare_evidence(tmp_path, 0)

    def unavailable(*args):
        return {**smoke.prescribed_review(*args), "status": "indeterminate"}

    monkeypatch.setattr(campaign.judge, "perform_review", unavailable)
    campaign.review_one(tmp_path, 0, "parent", "http://offline")
    calls = []
    result = campaign.propose_one(
        tmp_path, 0, "http://offline", transport=smoke.transport_for(calls)
    )
    assert result["status"] == "committed"
    status = calls[0]["scientific_advice"]["availability"]
    assert status["advice_available"] is False
    assert status["no_findings_does_not_imply_approval"] is True
    campaign.review_one(tmp_path, 0, "child", "http://offline")
    monkeypatch.setattr(public, "_run_backend", backend([]))
    assert campaign.fit_one(tmp_path, 0)["status"] == "complete"
    assert plan["policy"]["critic_can_veto"] is False


def test_proposer_never_gets_validation_and_stale_advice_is_rejected(
    tmp_path, monkeypatch
):
    plan, _ = ready(tmp_path, monkeypatch)
    row = copy.deepcopy(plan["rows"][0])
    row["fit"]["validation"] = "DO_NOT_SHOW_VALIDATION"
    row["original_parent"] = {"test": "DO_NOT_SHOW_TEST"}
    work = campaign.directory(tmp_path, row)
    packet = sealed_read(work / "evidence.json")["packet"]
    review = sealed_read(work / "parent_review.json")
    payload = campaign.advisory_payload(row, packet, review)
    assert "DO_NOT_SHOW" not in json.dumps(payload)
    assert payload["fitting_status"]["status"] == "complete"
    review["candidate_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another candidate"):
        campaign.advisory_payload(row, packet, review)


def test_failed_public_checks_cannot_be_overruled_by_positive_judge(
    tmp_path, monkeypatch
):
    ready(tmp_path, monkeypatch)
    original = campaign.certificates

    def reject(*args):
        return {**original(*args), "eligible_for_development_selection": False}

    monkeypatch.setattr(campaign, "certificates", reject)
    calls = []
    result = campaign.propose_one(
        tmp_path, 0, "http://offline", transport=smoke.transport_for(calls)
    )
    assert result["status"] == "revision_failed" and len(calls) == 3
    assert all(not a["accepted"] for a in result["attempts"])
    assert "PUBLIC_MODEL_REQUIREMENTS" in calls[1]["retry_feedback"]["message"]
    campaign.review_one(tmp_path, 0, "child", "http://offline")
    monkeypatch.setattr(
        public, "_run_backend", lambda *a: pytest.fail("invalid child fit")
    )
    assert campaign.fit_one(tmp_path, 0)["selected"] == "parent"


def test_numeric_selector_has_no_critic_input_and_keeps_parent_on_tie(tmp_path):
    plan = smoke.fixture(tmp_path)
    row = plan["rows"][0]
    parent = copy.deepcopy(row["fit"])
    parent["validation"]["normalized_mse"] = 1.0
    child = copy.deepcopy(parent)
    request = PublicFitRequest.model_validate(row["request"])
    assert (
        campaign.select(parent, child, row["certificate"], request, request) == "parent"
    )
    child["validation"]["normalized_mse"] = 0.9
    assert (
        campaign.select(parent, child, row["certificate"], request, request) == "child"
    )
    assert (
        campaign.select(
            parent,
            child,
            {"eligible_for_development_selection": False},
            request,
            request,
        )
        == "parent"
    )
    child["status"] = "fit_failed"
    assert (
        campaign.select(parent, child, row["certificate"], request, request) == "parent"
    )


def test_interrupted_review_consumes_budget_without_network(tmp_path):
    plan = smoke.fixture(tmp_path)
    request = campaign.critic_request(plan, plan["rows"][0])
    cache = tmp_path / "judge_cache" / campaign.content_hash(request)
    cache.mkdir(parents=True)
    public._write(
        cache / "review_started.json",
        {"request_sha256": campaign.content_hash(request)},
    )
    result = campaign.review_one(tmp_path, 0, "parent", "http://not-a-server")
    assert result["review"]["status"] == "interrupted"
    assert campaign.review_one(tmp_path, 0, "parent", "http://not-a-server") == result


def test_execution_seal_and_missing_rows(tmp_path):
    smoke.fixture(tmp_path)
    assert campaign.report(tmp_path)["status_counts"] == {"missing": 1}
    (tmp_path / "evaluation_freeze.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed for evaluation"):
        io.verify(tmp_path)
    (tmp_path / "evaluation_freeze.json").unlink()
    raw = json.loads((tmp_path / "plan.json").read_text())
    raw["policy"]["critic_can_veto"] = True
    (tmp_path / "plan.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="digest"):
        io.verify(tmp_path)


def test_interrupted_stages_keep_observed_usage_before_publication(
    tmp_path, monkeypatch
):
    proposer_root, judge_root = tmp_path / "proposer", tmp_path / "judge"
    ready(proposer_root, monkeypatch)

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign.revision, "apply_edits", interrupted)
    with pytest.raises(KeyboardInterrupt):
        campaign.propose_one(
            proposer_root, 0, "http://offline", transport=smoke.transport_for([])
        )
    report = campaign.report(proposer_root)
    assert report["rows"][0]["proposal"] is None
    assert report["proposer_cost"]["physical_requests"] == 1
    assert report["proposer_cost"]["observed_tokens"] > 0

    smoke.fixture(judge_root)

    def partial_review(request, cache, base_url):
        (cache / "events.jsonl").write_text(
            json.dumps(
                {
                    "event": "llm_response",
                    "request_hash": "partial-call",
                    "provider_attempts": 1,
                    "usage": {"total_tokens": 51},
                }
            )
            + "\n"
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign.judge, "perform_review", partial_review)
    with pytest.raises(KeyboardInterrupt):
        campaign.review_one(judge_root, 0, "parent", "http://offline")
    report = campaign.report(judge_root)
    assert report["review_status_counts"] == {"unpublished": 1}
    assert report["judge_cost"]["physical_requests"] == 1
    assert report["judge_cost"]["observed_total_tokens"] == 51
    assert report["judge_cost"]["usage_complete"] is False


def submission_source(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "out"
    plan = sealed_write(
        source / "plan.json",
        {"protocol": io.pruning.PROTOCOL, "rows": [{"task": {"task_id": "toy"}}]},
    )
    sealed_write(
        source / "results/toy/result.json",
        {"identity": plan["artifact_sha256"], "status": "complete"},
    )
    binary = tmp_path / "binary"
    binary.touch()
    for k, v in {
        "AF_PYTHON": str(binary),
        "AF_VLLM_IMAGE": str(binary),
        "AF_HF_HOME": str(tmp_path / "hf"),
        "AF_JUDGE_REVISION": "b" * 40,
        "AF_COMMIT": "a" * 40,
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(submitter, "source_commit", lambda p: "a" * 40)
    return source, root


def test_submission_phase_dependencies_and_duplicate_prevention(tmp_path, monkeypatch):
    source, root = submission_source(tmp_path, monkeypatch)
    calls = []

    def submit(*args):
        calls.append(args)
        return str(100 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", submit)
    result = submitter.submit(source, root)
    assert len(calls) == 6
    assert "--gres=gpu:h100:2" in calls[1][2] and "--gres=gpu:h100:1" in calls[2][2]
    for i in range(1, 5):
        assert f"--dependency=afterok:{100 + i}" in calls[i][2]
    assert "--dependency=afterany:105" in calls[-1][2]
    assert submitter.submit(source, root) == result and len(calls) == 6


def test_uncertain_scheduler_reply_is_not_resubmitted(tmp_path, monkeypatch):
    source, root = submission_source(tmp_path, monkeypatch)

    def uncertain(*args):
        raise ValueError("uncertain scheduler reply")

    monkeypatch.setattr(submitter, "submit_job", uncertain)
    with pytest.raises(ValueError, match="uncertain scheduler"):
        submitter.submit(source, root)
    with pytest.raises(ValueError, match="partial/uncertain submission"):
        submitter.submit(source, root)


def test_historical_pruning_receipts_checked_without_old_package(tmp_path, monkeypatch):
    from scripts import smoke_process_pruning

    old = smoke_process_pruning.fixture(tmp_path)
    monkeypatch.setattr(public, "_run_backend", backend([]))
    result = io.pruning.run_one(tmp_path, 0)
    cell, row = old["cells"]["toy"], old["rows"][0]
    request = PublicFitRequest.model_validate(result["choice"]["request"])
    directory = tmp_path / "results" / row["task"]["task_id"] / "pruned"
    # A historical source revision cannot be executed under new code. Import can
    # still check immutable hashes, inputs and result/backend receipts read-only.
    monkeypatch.setattr(public, "_source_identity", lambda: "f" * 64)
    io.historical_fit(directory, request, result["fits"]["pruned"], row["parent"], cell)
    bad = copy.deepcopy(cell)
    bad["validation"]["fingerprint"] = "changed"
    with pytest.raises(ValueError, match="inputs differ"):
        io.historical_fit(
            directory, request, result["fits"]["pruned"], row["parent"], bad
        )
    raw = public._read(directory / "backend_result.json")
    raw["parameters"]["a"] = 123
    public._write(directory / "backend_result.json", raw)
    with pytest.raises(ValueError, match="backend digest"):
        io.historical_fit(
            directory, request, result["fits"]["pruned"], row["parent"], cell
        )


def test_full_historical_handoff_keeps_actual_pruning_choice(tmp_path, monkeypatch):
    from autoformalism.rebuttal import review_deadline_pipeline as pipeline
    from scripts import smoke_shared_process_integration as integration

    source, prune, output = (tmp_path / s for s in ("integration", "pruning", "critic"))
    plan = integration.fixture(source)
    monkeypatch.setattr(public, "_run_backend", backend([]))
    monkeypatch.setattr(pipeline, "replay_packet", lambda *args: None)
    for task in plan["tasks"]:
        for index in (0, 1):
            client = pipeline._client(
                source,
                plan,
                task,
                index,
                "http://offline",
                lambda: True,
                integration.transport_for([]),
            )
            pipeline.propose_one(source, plan, task, index, client)
            pipeline.fit_one(source, plan, task, index)
    frozen = io.pruning.freeze(source, prune)
    for index in range(len(frozen["rows"])):
        io.pruning.run_one(prune, index)
    before = {str(p): p.read_bytes() for p in prune.rglob("*.json")}
    imported = io.freeze(prune, output, "a" * 40)
    assert len(imported["rows"]) == len(plan["tasks"])
    assert before == {str(p): p.read_bytes() for p in prune.rglob("*.json")}
    assert io.freeze(prune, output, "a" * 40) == imported
    for row in imported["rows"]:
        previous = sealed_read(
            prune / "results" / row["task"]["task_id"] / "result.json"
        )
        assert row["pruning_origin"] == previous["selection"]["selected"]
        assert row["fit"] == previous["selection"]["fit"]
        assert "scientific_context" in row["bundle"]["brief"]
    with pytest.raises(ValueError, match="separate"):
        io.freeze(prune, prune / "child", "a" * 40)
    with pytest.raises(ValueError, match="revision differs"):
        io.freeze(prune, output, "b" * 40)


def test_published_outcome_requires_original_fit_receipts(tmp_path, monkeypatch):
    plan, _ = ready(tmp_path, monkeypatch)
    monkeypatch.setattr(public, "_run_backend", backend([]))
    campaign.propose_one(
        tmp_path, 0, "http://offline", transport=smoke.transport_for([])
    )
    campaign.review_one(tmp_path, 0, "child", "http://offline")
    campaign.fit_one(tmp_path, 0)
    path = campaign.directory(tmp_path, plan["rows"][0]) / "fit/backend_result.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="backend digest"):
        campaign.report(tmp_path)
    with pytest.raises(ValueError, match="backend digest"):
        campaign.fit_one(tmp_path, 0)
