"""Treatment separation, complete denominators, concurrency and irreversible freeze."""

import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.rebuttal import component_campaign as campaign
from autoformalism.rebuttal import component_critic as critic
from autoformalism.rebuttal import component_workers as workers
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.search import scientific_verification as gates
from autoformalism.staged_topology import content_hash
from scripts import smoke_fresh_shared as smoke


def config(**updates):
    value = json.loads(
        (io.REPO / "configs/final_component_campaign_v1.json").read_text()
    )
    return io.DeadlineConfig.model_validate({**value, **updates})


def fixture(root):
    plan = smoke.fixture(root)
    cell = next(iter(plan["cells"].values()))
    old_name = next(iter(plan["cells"]))
    if old_name != campaign.CELLS[0]:
        shutil.copytree(
            root / "public/phase_b_v1" / old_name,
            root / "public/phase_b_v1" / campaign.CELLS[0],
        )
    cfg = config(campaign_blocks=[0])
    plan.pop("artifact_sha256")
    plan.pop("fresh_shared_policy")
    plan.update(
        protocol=io.ABLATION_PROTOCOL,
        config=cfg.model_dump(mode="json"),
        tasks=io.tasks(cfg),
        cells={campaign.CELLS[0]: cell},
        component_policy=campaign.policy(),
        launcher_sha256=io.launcher_hash(io.ABLATION_PROTOCOL),
    )
    (root / "plan.json").unlink()
    return sealed_write(root / "plan.json", plan)


def test_nine_cells_primary_and_non_cartesian_secondary():
    tasks = io.tasks(config())
    assert len(tasks) == 160
    assert len({t["task_id"] for t in tasks}) == 160
    assert sum(not t["secondary"] for t in tasks) == 144
    assert len(campaign.SECONDARY_CELLS) == 4
    for block in range(36):
        group = [t for t in tasks if t["block"] == block]
        assert {
            (t["critic"], t["scientific_verifier"]) for t in group if not t["secondary"]
        } == set(campaign.ARMS)
        assert len({(t["cell"], t["seed"], t["arm"]) for t in group}) == 1
    assert {t["cell"] for t in tasks} == set(campaign.CELLS)
    assert {t["cell"] for t in tasks if t["secondary"]} == set(campaign.SECONDARY_CELLS)
    assert len(io.tasks(config(secondary_shared_comparison=False))) == 144
    assert len(io.tasks(config(campaign_blocks=[0]))) == 5
    with pytest.raises(ValueError):
        config(campaign_blocks=[0, 0])
    with pytest.raises(ValueError):
        config(public_cells=campaign.CELLS[:6])
    with pytest.raises(ValueError):
        config(fit_profile="collocation-single-target-v2")


def test_scoped_verifier_never_leaks_to_other_worker_or_after_exception():
    assert gates.enabled()
    with gates.scope(False):
        assert not gates.enabled()
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(gates.enabled).result()
        with pytest.raises(RuntimeError), gates.scope(True):
            assert gates.enabled()
            raise RuntimeError("interrupt")
        assert not gates.enabled()
    assert gates.enabled()


def test_same_brief_even_with_verifier_disabled(tmp_path):
    plan = fixture(tmp_path)
    cell = plan["cells"][campaign.CELLS[0]]
    task = plan["tasks"][0]
    assert pipeline._brief(cell, task) == pipeline._brief(
        cell, {**task, "scientific_verifier": False}
    )
    assert (
        pipeline._certificate_feedback(
            {"anything": "private audit"}, {**task, "scientific_verifier": False}
        )
        == {}
    )
    campaign.verify(tmp_path)


def test_scientific_gate_off_preserves_runtime_and_full_offline_evidence(tmp_path):
    plan = fixture(tmp_path)
    task = plan["tasks"][0]
    cell = plan["cells"][task["cell"]]
    client = pipeline._client(
        tmp_path, plan, task, 0, "http://offline", lambda: True, smoke.transport_for([])
    )
    proposal = pipeline.propose_one(tmp_path, plan, task, 0, client)
    assert proposal["bundle"], proposal
    bundle = proposal["bundle"]
    # A missing required scientific pathway must remain visible in offline scoring.
    changed = copy.deepcopy(cell)
    changed["mechanism_spec"]["required_mechanisms"][0]["required_drivers"] = [
        "missing_public_driver"
    ]
    on = pipeline.certificates(bundle, changed, task)
    off = pipeline.certificates(bundle, changed, {**task, "scientific_verifier": False})
    assert not on["eligible_for_development_selection"]
    assert off["eligible_for_development_selection"]
    assert on["mechanisms"] == off["mechanisms"] and on["targets"] == off["targets"]
    broken = copy.deepcopy(bundle)
    broken["candidate"]["state_equations"][0]["rhs"] = "unknown_external(x)"
    with pytest.raises(ModelValidationError), gates.scope(False):
        pipeline.certificates(broken, changed, {**task, "scientific_verifier": False})


def test_worker_order_locks_resume_and_no_implicit_test(tmp_path, monkeypatch):
    plan = fixture(tmp_path)
    task = next(t for t in plan["tasks"] if not t["critic"])
    monkeypatch.setattr(campaign, "verify", lambda *a, **k: {**plan, "tasks": [task]})
    observed = []

    def perform(root, plan, task, index, stage, *_):
        observed.append((index, stage))
        return sealed_write(
            io.round_path(root, task, index) / "proposal.json",
            {"status": "constructed"},
        )

    result = workers.run(
        tmp_path, "propose", once=True, base_url="http://offline", performer=perform
    )
    assert result["completed_operations"] == 1
    workers.run(
        tmp_path, "propose", once=True, base_url="http://offline", performer=perform
    )
    assert observed == [(0, "propose")]
    path = tmp_path / "claims/test.lock"
    with workers.claim(path) as first, workers.claim(path) as second:
        assert first and not second
    assert not workers.ready(tmp_path, plan, task, 1, "propose")
    sealed_write(
        io.round_path(tmp_path, task, 0) / "result.json",
        {"selected": None, "proposal_status": "provider_request_failed"},
    )
    assert not workers.ready(tmp_path, plan, task, 1, "propose")
    (tmp_path / "evaluation_freeze.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed"):
        workers.run(
            tmp_path, "propose", once=True, base_url="http://offline", performer=perform
        )


def test_critic_prerequisite_and_compatibility_alone_is_not_gate(tmp_path):
    plan = fixture(tmp_path)
    task = plan["tasks"][0]
    assert task["critic"]
    assert not workers.ready(tmp_path, plan, task, 0, "propose")
    expected = critic.judge.judge_protocol(sign_policy=critic.SIGN_POLICY)
    expected.pop("distinct_seed_attempts")
    calibration = sealed_write(
        tmp_path / "calibration.json",
        {
            "scientific_protocol": expected,
            "transport_policy": critic.jetstream.POLICY,
            "output_contract": critic.jetstream.OUTPUT_CONTRACT,
            "endpoint": critic.jetstream.ENDPOINT,
        },
    )
    summary = {
        "protocol": critic.CALIBRATION_PROTOCOL,
        "identity": calibration["artifact_sha256"],
        "status": "complete",
        "known_case_gate_passed": False,
        "planned_orientations": 140,
        "test_data_opened": False,
    }
    summary_path = tmp_path / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="revalidation"):
        critic.authorize(tmp_path, tmp_path / "calibration.json", summary_path)
    summary["known_case_gate_passed"] = True
    summary_path.write_text(json.dumps(summary))
    critic.authorize(tmp_path, tmp_path / "calibration.json", summary_path)
    assert workers.ready(tmp_path, plan, task, 0, "propose")
    sealed_write(
        io.round_path(tmp_path, task, 0) / "proposal.json",
        {"status": "construction_failed", "bundle": None},
    )
    assert workers.ready(tmp_path, plan, task, 0, "critic")
    assert not workers.ready(tmp_path, plan, task, 0, "fit")
    assert (
        critic.review_one(tmp_path, plan, task, 0, lambda: "never-used")["status"]
        == "no_executable_candidate"
    )
    assert workers.ready(tmp_path, plan, task, 0, "fit")


def test_public_staging_preflights_every_cell_and_never_copies_test(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    for cell in campaign.CELLS:
        path = source / "phase_b_v1" / cell
        path.mkdir(parents=True)
        for name in io.FILES:
            (path / name).write_text(name)
        (path / "test.csv").write_text("must stay sealed")
    missing = source / "phase_b_v1" / campaign.CELLS[-1] / "validation.csv"
    missing.unlink()
    with pytest.raises(ValueError, match="missing public"):
        campaign.stage_public([source], target)
    assert not target.exists()
    missing.write_text("validation.csv")
    assert campaign.stage_public([source], target)["files"] == 36
    assert not list(target.rglob("test.csv"))


def test_partial_export_refused_and_empty_frozen_roster_complete(tmp_path):
    plan = fixture(tmp_path)
    with pytest.raises(ValueError, match="unfinished"):
        campaign.export(tmp_path)
    for task in plan["tasks"]:
        for i in range(15):
            sealed_write(
                io.round_path(tmp_path, task, i) / "result.json",
                {
                    "task": task,
                    "round": i,
                    "status": "construction_failed",
                    "proposal_status": "construction_failed",
                    "selected": None,
                    "trial": None,
                    "cost": {},
                    "test_data_opened": False,
                },
            )
        campaign.prune_one(tmp_path, plan, task)
    value = campaign.export(tmp_path)
    assert value["subject_count"] == 0
    assert len(value["roster"]) == 7  # five arms, two pruning comparators
    assert campaign.export(tmp_path) == value
    assert campaign.evaluation_report(tmp_path)["status_counts"] == {
        "model_unavailable": 7
    }
    with pytest.raises(ValueError):
        campaign.prune_one(tmp_path, plan, plan["tasks"][0])


def test_inventory_and_pathway_gate_off_still_requires_complete_targets():
    from autoformalism.staged_topology import freeze_inventory, public_structure_checks
    from tests.test_staged_topology_contract import brief, equation, var

    inventory = (var("x"),)
    with pytest.raises(ValueError, match="drivers"):
        freeze_inventory(brief(), inventory)
    with gates.scope(False):
        assert freeze_inventory(brief(), inventory) == inventory
        assert not public_structure_checks(brief(), (equation("x", ()),))
        with pytest.raises(ValueError, match="targets"):
            freeze_inventory(brief(), (var("z"),))
    assert not all(
        r["passed"] for r in public_structure_checks(brief(), (equation("x", ()),))
    )


def test_bounded_fresh_critic_revision_and_pruning_integration(tmp_path, monkeypatch):
    """Real toy multi-output fits; prescribed LLM replies, no benchmark reads."""
    from autoformalism.rebuttal.repair_evidence import model_hash
    from autoformalism.schemas import CandidateModel

    plan = fixture(tmp_path)
    task, calls = plan["tasks"][0], []
    monkeypatch.setattr(critic, "checked_authorization", lambda *a: {})
    reviewed = []

    def review(request, directory, base_url, **kwargs):
        assert set(request) == {
            "parent",
            "candidate",
            "context",
            "public_prompt",
            "seed",
            "model_revision",
            "protocol",
        }
        assert "training" not in request and "validation" not in request
        reviewed.append(request)
        return {
            "request_sha256": content_hash(request),
            "status": "reviewed",
            "findings": [],
            "cost": {"physical_requests": 4, "observed_total_tokens": 0},
        }

    for index in (0, 1):
        client = pipeline._client(
            tmp_path,
            plan,
            task,
            index,
            "http://offline",
            lambda: True,
            smoke.transport_for(calls),
        )
        client.token_transport = lambda *a: {"count": 4000, "max_model_len": 32768}
        proposal = pipeline.propose_one(tmp_path, plan, task, index, client)
        assert proposal["status"] == ("constructed" if index == 0 else "committed"), (
            proposal
        )
        receipt = critic.review_one(
            tmp_path, plan, task, index, lambda: "never-used", performer=review
        )
        assert receipt["candidate_sha256"] == model_hash(
            CandidateModel.model_validate(proposal["bundle"]["candidate"])
        )
        assert (
            critic.review_one(
                tmp_path, plan, task, index, lambda: "never-used", performer=review
            )
            == receipt
        )
        result = pipeline.fit_one(tmp_path, plan, task, index)
        assert result["selected"], result
        assert set(
            result["trial"]["fit"]["validation"]["per_target_normalized_mse"]
        ) == {"v01", "w"}
    assert len(reviewed) == 2
    assert any("scientific_advice" in p for p in calls)
    # Finish the toy's unexecuted slots as explicit unavailable outcomes, and
    # carry its known two-visit incumbent to exercise the final artifact join.
    for other in plan["tasks"]:
        for index in range(15):
            path = io.round_path(tmp_path, other, index) / "result.json"
            if path.exists():
                continue
            sealed_write(
                path,
                {
                    "task": other,
                    "round": index,
                    "status": "complete" if other == task else "construction_failed",
                    "selected": result["selected"] if other == task else None,
                    "trial": None,
                    "cost": {},
                    "test_data_opened": False,
                },
            )
        campaign.prune_one(tmp_path, plan, other)
    critic.review_final(tmp_path, plan, task, lambda: "never-used", performer=review)
    assert campaign.report(tmp_path)["endpoint_status_counts"].get("missing", 0) == 0
    frozen = campaign.export(tmp_path)
    assert frozen["subject_count"] >= 2
    assert len(frozen["roster"]) == 7
    assert campaign.export(tmp_path) == frozen
    assert not (tmp_path / "evaluation").exists()
    (tmp_path / "subjects.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="subjects changed"):
        campaign.export(tmp_path)


def test_submission_resume_and_uncertain_reply_do_not_duplicate_jobs(
    tmp_path, monkeypatch
):
    from scripts import submit_component_campaign as submitter

    plan = fixture(tmp_path)
    monkeypatch.setattr(submitter, "checked_authorization", lambda *a: {})
    monkeypatch.setattr(submitter, "account", lambda *a: "test-account")
    helper = submitter.support("submit_shared_process_pilot")
    monkeypatch.setattr(helper, "source_commit", lambda *a: "a" * 40)
    monkeypatch.setattr(submitter, "support", lambda *a: helper)
    for name in (
        "AF_PYTHON",
        "AF_VLLM_IMAGE",
        "AF_HF_HOME",
        "AF_COMPUTE_CACHE_ROOT",
        "AF_IPC_TMP_ROOT",
        "AF_JETSTREAM_API_KEY",
    ):
        monkeypatch.setenv(
            name, "never-written-secret" if name.endswith("KEY") else "test-path"
        )
    for name in ("AF_REPO_ROOT", "AF_OUTPUT_ROOT", "AF_SITE", "AF_WORKER_SECONDS"):
        monkeypatch.setenv(name, "test")
    monkeypatch.delenv("AF_COMMIT", raising=False)
    calls = []

    def queue(directory, key, opts, worker, stage, index):
        calls.append(stage)
        public = submitter.public
        public._write(
            directory / f"{key}.intent.json",
            {"argv": ["sbatch", "--parsable", *opts, str(worker), stage, str(index)]},
        )
        if stage == "critic" and len(calls) == 2:
            raise ValueError("uncertain response")
        (directory / f"{key}.id").write_text(str(100 + len(calls)))
        return str(100 + len(calls))

    monkeypatch.setattr(helper, "submit_job", queue)
    with pytest.raises(ValueError, match="uncertain"):
        submitter.submit(tmp_path, "wave1", 1, 1, 1, 1)
    with pytest.raises(ValueError, match="uncertain"):
        submitter.submit(tmp_path, "wave1", 1, 1, 1, 1)
    assert calls == ["propose", "critic"]
    assert all(
        "never-written-secret" not in p.read_text()
        for p in (tmp_path / "submissions").rglob("*.json")
    )
    assert plan["config"]["rounds"] == 15


def test_merge_refuses_duplicate_blocks_and_scientific_drift(tmp_path):
    cfg = config().model_dump(mode="json")

    def value(block):
        return {
            "identity": str(block),
            "source_sha256": "s",
            "public_assets": {},
            "config": {**cfg, "campaign_blocks": [block]},
            "frozen_for_evaluation": False,
            "rows": [
                {
                    "task": {"task_id": str(block), "block": block},
                    "endpoint": "final",
                    "status": "missing",
                }
            ],
        }

    paths = []
    for i in (0, 1):
        path = tmp_path / f"site{i}.json"
        path.write_text(json.dumps(value(i)))
        paths.append(path)
    result = campaign.merge(paths, tmp_path / "combined")
    assert result["present_blocks"] == [0, 1] and not result["all_sites_frozen"]
    with pytest.raises(ValueError, match="duplicate"):
        campaign.merge([paths[0], paths[0]], tmp_path / "duplicate")
    changed = value(1)
    changed["config"]["rounds"] = 2
    paths[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differ"):
        campaign.merge(paths, tmp_path / "different")


@pytest.mark.parametrize("unfitted", [False, True])
def test_whole_model_multi_revision_keeps_legacy_limits_and_causal_checks(unfitted):
    from pydantic import ValidationError

    from autoformalism.search import review_revision_multi as edits
    from scripts.smoke_review_multi import example

    bundle, packet, *_ = example()
    raw = {
        "hypothesis": "Revise a connected algebraic chain and all its consumers.",
        "equations": [
            {
                "component": f"shared{i}",
                "kind": "algebraic",
                "expression": "g" if i == 0 else f"shared{i - 1}",
            }
            for i in range(7)
        ]
        + [{"component": "disposal", "expression": "c*q*shared6+Uii"}],
    }
    evidence = None if unfitted else packet
    with pytest.raises(ValidationError):
        edits.apply_edits(bundle, evidence, raw)
    before = copy.deepcopy(bundle)
    result = edits.apply_edits(bundle, evidence, raw, whole_model=True)
    assert result["outcome"] == "committed"
    assert result["provenance"]["patch_count_limits"] is None
    assert len(result["bundle"]["candidate"]["observation_mappings"]) == 3
    assert bundle == before
    raw["equations"][0]["expression"] = "shared6"
    with pytest.raises((ValueError, ModelValidationError)):
        edits.apply_edits(bundle, evidence, raw, whole_model=True)


def test_whole_model_multi_reply_prompt_and_schema_agree():
    from autoformalism.search import response_revision
    from autoformalism.search import review_revision_multi as edits

    schema = edits.WholeModelRevision.model_json_schema()["properties"]
    for key in (
        "equations", "remove_variables", "initializers", "new_parameters",
        "output_mappings",
    ):
        assert "maxItems" not in schema[key]
    assert "six" not in edits.WHOLE_MODEL_SYSTEM_PROMPT
    assert "six" not in response_revision.WHOLE_MODEL_SYSTEM_PROMPT
    assert "public_target_contract" in edits.WHOLE_MODEL_SYSTEM_PROMPT
    assert "training_evidence" in response_revision.WHOLE_MODEL_SYSTEM_PROMPT
