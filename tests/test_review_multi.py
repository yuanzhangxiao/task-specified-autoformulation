"""Joint-output repair, public gates, unfitted recovery and bounded continuation."""

import copy
import json
import subprocess
import sys

import pytest

from autoformalism.expressions import ModelValidationError
from autoformalism.fitting import public_fitting as public
from autoformalism.llm.review_revision import RevisionClient
from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_comparison import BudgetedRepairClient
from autoformalism.schemas.public_fitting import PublicFitMetrics, PublicFitResult
from autoformalism.search import review_multi_construction as construction
from autoformalism.search import review_revision_multi as edits
from autoformalism.search import review_revision_v3 as old
from scripts.review_deadline import interrupted
from scripts.smoke_review_multi import CELL, example, fixture
from tests.test_review_continuation import _submit_module


def response(raw):
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}],
        "usage": {"total_tokens": 100},
    }


def patch(**kwargs):
    return {"hypothesis": "Synthetic conditional hypothesis", **kwargs}


def test_joint_mappings_preserve_unmentioned_output_and_resolve_new_parameters():
    bundle, packet, *_ = example()
    before = copy.deepcopy(bundle)
    result = edits.apply_edits(
        bundle,
        packet,
        patch(
            output_mappings=[
                {"channel": "I", "expression": "i+shift"},
                {"channel": "U", "expression": "disposal+shift"},
            ],
            new_parameters=[{"name": "shift"}],
            evidence_refs=["E001", "unavailable_samples"],
            initializers=[{"state": "i", "causal_map": {"expression": "I"}}],
        ),
    )
    model = result["bundle"]["initialization"]["base_candidate"]
    assert {m["channel"]: m["expression"] for m in model["observation_mappings"]} == {
        "Gp": "g",
        "I": "i+shift",
        "U": "disposal+shift",
    }
    assert len([p for p in model["parameters"] if p["name"] == "shift"]) == 1
    assert (
        model["state_equations"]
        == before["initialization"]["base_candidate"]["state_equations"]
    )
    audit = result["provenance"]["citation_audit"]
    assert audit["valid_evidence_ids"] == [old.references(packet)["E001"]]
    assert audit["unresolved_references"] == ["unavailable_samples"]
    assert audit["scientific_claims_verified"] is False
    assert bundle == before


@pytest.mark.parametrize("expression", ["unavailable*g", "__import__('os')", "I+g"])
def test_unresolved_citations_do_not_bypass_grammar_or_channel_checks(expression):
    bundle, packet, *_ = example()
    with pytest.raises((ValueError, ModelValidationError)):
        edits.apply_edits(
            bundle,
            packet,
            patch(
                evidence_refs=["fake"],
                equations=[{"component": "g", "expression": expression}],
            ),
        )


def test_output_schema_and_channel_contract():
    schema = edits.ScientificRevision.model_json_schema()
    assert "output_expression" not in schema["properties"]
    assert "output_mappings" in schema["properties"]
    assert "one PUBLIC target" not in edits.SYSTEM_PROMPT
    assert "output_expression" not in edits.SYSTEM_PROMPT
    bundle, packet, *_ = example()
    with pytest.raises(ValueError, match="declared public targets"):
        edits.apply_edits(
            bundle,
            packet,
            patch(output_mappings=[{"channel": "extra", "expression": "g"}]),
        )
    with pytest.raises(ValueError, match="exactly one"):
        old.apply_edits(bundle, packet, {"hypothesis": "Historical contract retained"})


def test_mapping_duplicates_and_parameter_only_in_output():
    bundle, packet, *_ = example()
    raw = patch(
        output_mappings=[{"channel": "U", "expression": "gain*disposal"}] * 2,
        new_parameters=[{"name": "gain"}, {"name": "unused"}],
    )
    result = edits.apply_edits(bundle, packet, raw)
    assert result["provenance"]["identical_output_mappings"] == ["U"]
    assert [
        p["parameter"] for p in result["provenance"]["unused_new_declarations_removed"]
    ] == ["unused"]
    raw["output_mappings"][1] = {"channel": "U", "expression": "disposal"}
    with pytest.raises(ValueError, match="Conflicting"):
        edits.apply_edits(bundle, packet, raw)


def test_unfitted_patch_has_no_numerical_evidence_or_verified_citations():
    bundle, *_ = example(draft=True)
    user = edits.payload(bundle, None, None)
    assert user["stage"] == "construction_repair"
    assert user["retained_fitted_parameters"] is None
    assert user["training_evidence"] is None and user["evidence_catalog"] == []
    result = edits.apply_edits(
        bundle,
        None,
        patch(
            evidence_refs=["E001"],
            equations=[
                {
                    "component": "disposal",
                    "kind": "algebraic",
                    "expression": "c*q*g+Uii",
                }
            ],
        ),
    )
    audit = result["provenance"]["citation_audit"]
    assert audit["valid_evidence_ids"] == [] and audit["unresolved_references"] == [
        "E001"
    ]
    assert (
        result["bundle"]["initialization"]["plan"] == bundle["initialization"]["plan"]
    )


def continued(tmp_path, visits=2):
    source, root = tmp_path / "source", tmp_path / "continued"
    fixture(source)
    plan = continuation.prepare(source, root, 2, visits, protocol=io.MULTI_PROTOCOL)
    return source, root, plan


def test_import_recovers_failed_draft_reopens_incumbent_and_preserves_source(tmp_path):
    source = tmp_path / "source"
    original = fixture(source)
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    root = tmp_path / "continued"
    plan = continuation.prepare(source, root, 2, 12, protocol=io.MULTI_PROTOCOL)
    assert plan["config"]["rounds"] == 13  # import plus twelve NEW visits
    assert plan["config"]["fit_profile"] == original["config"]["fit_profile"]
    assert plan["continuation"]["source_round"] + plan["config"]["rounds"] - 1 == 14
    for task in plan["tasks"]:
        anchor = io.read_round(root, task, 0)
        assert not anchor["closed"]
        assert anchor["selected"] == io.read_round(source, task, 2)["selected"]
        assert bool(anchor["construction_draft"]) == (task["seed"] == 1)
    assert continuation.prepare(source, root, 2, 12, protocol=io.MULTI_PROTOCOL) == plan
    assert {str(p): p.read_bytes() for p in source.rglob("*.json")} == before
    with pytest.raises(ValueError, match="between one and 12"):
        continuation.prepare(
            source, tmp_path / "too_long", 2, 13, protocol=io.MULTI_PROTOCOL
        )
    with pytest.raises(ValueError):
        continuation.prepare(source, tmp_path / "legacy", 2, 2)


def test_import_requires_complete_source_and_detects_draft_tampering(tmp_path):
    source, root, plan = continued(tmp_path)
    task = plan["tasks"][1]
    path = root / "imports" / task["task_id"] / "draft.json"
    raw = sealed_read(path)
    raw.pop("artifact_sha256")
    raw["bundle"]["candidate"]["change_summary"] = "tampered"
    path.unlink()
    sealed_write(path, raw)
    with pytest.raises(ValueError, match="draft differs"):
        io.verify(root)
    (io.round_path(source, task, 2) / "result.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        continuation.prepare(
            source, tmp_path / "missing", 2, 2, protocol=io.MULTI_PROTOCOL
        )


def test_representation_feedback_and_explicit_repair_are_cached(tmp_path):
    _, root, plan = continued(tmp_path)
    task = plan["tasks"][1]
    calls = []

    def transport(url, body, timeout):
        user = json.loads(body["messages"][1]["content"])
        calls.append(user)
        assert "validation" not in user and "test" not in user
        return response(
            patch()
            if len(calls) == 1
            else patch(
                equations=[
                    {
                        "component": "disposal",
                        "kind": "algebraic",
                        "expression": "c*q*g+Uii",
                    }
                ]
            )
        )

    client = pipeline._client(
        root, plan, task, 1, "http://offline", lambda: True, transport
    )
    assert isinstance(client, RevisionClient)
    proposal = pipeline.propose_one(root, plan, task, 1, client)
    assert proposal["status"] == "constructed", proposal
    assert len(calls) == 2
    failures = calls[0]["public_requirement_findings"]["failed_target_predicates"]
    assert [f["predicate"] for f in failures] == [
        "target_representation:instantaneous_process"
    ]
    assert calls[1]["retry_feedback"]["details"]["failed_target_predicates"]
    assert (
        calls[0]["public_target_contract"]["requirements"]
        == plan["cells"][CELL]["target_contract"]["targets"]
    )
    assert proposal["certificate"]["eligible_for_development_selection"]
    assert pipeline.propose_one(root, plan, task, 1, None) == proposal
    assert len(calls) == 2


def test_missing_required_dependency_stays_blocked_and_repair_retries_next_visit(
    tmp_path,
):
    _, root, plan = continued(tmp_path)
    task = plan["tasks"][1]

    def transport(*args):
        return response(
            patch(
                equations=[
                    {
                        "component": "disposal",
                        "kind": "algebraic",
                        "expression": "c*q*g",
                    }
                ]
            )
        )

    client = pipeline._client(
        root, plan, task, 1, "http://offline", lambda: True, transport
    )
    proposal = pipeline.propose_one(root, plan, task, 1, client)
    assert proposal["status"] == "construction_repair_failed"
    assert len(client.records) == 3
    assert "required_dependency:basal" in json.dumps(proposal["attempts"])
    result = pipeline.fit_one(root, plan, task, 1)
    assert result["selected"] is None and not result["closed"]
    assert result["construction_draft"] == proposal["construction_draft"]
    assert not (io.round_path(root, task, 1) / "fit").exists()
    next_client = pipeline._client(
        root, plan, task, 2, "http://offline", lambda: True, transport
    )
    assert isinstance(next_client, RevisionClient)
    following = pipeline.propose_one(root, plan, task, 2, next_client)
    assert following["status"] != "closed_lineage" and len(next_client.records) == 3


def test_interrupted_repair_reuses_physical_calls(tmp_path):
    _, root, plan = continued(tmp_path)
    task, calls = plan["tasks"][1], []

    def transport(*args):
        calls.append(1)
        return response(patch())

    client = pipeline._client(
        root, plan, task, 1, "http://offline", lambda: len(calls) < 1, transport
    )
    with pytest.raises(DeferredCall):
        pipeline.propose_one(root, plan, task, 1, client)
    resumed = pipeline._client(
        root, plan, task, 1, "http://offline", lambda: True, transport
    )
    proposal = pipeline.propose_one(root, plan, task, 1, resumed)
    assert proposal["status"] == "construction_repair_failed" and len(calls) == 3


def test_no_draft_gets_construction_budget_and_contract_before_inventory(tmp_path):
    source = tmp_path / "source"
    old_plan = fixture(source)
    task = old_plan["tasks"][1]
    (io.round_path(source, task, 0) / "proposal.json").unlink()
    root = tmp_path / "continued"
    plan = continuation.prepare(source, root, 2, 2, protocol=io.MULTI_PROTOCOL)
    client = pipeline._client(root, plan, task, 1, "http://offline", lambda: True)
    assert isinstance(client, BudgetedRepairClient)
    brief = construction.construction_brief(plan["cells"][CELL], task)
    assert "instantaneous_process" in brief.scientific_context
    assert "public_prompt_sha256" in brief.scientific_context


def test_changed_observation_requires_explicit_new_latent_boundary():
    bundle, packet, *_ = example()
    with pytest.raises(ValueError, match="explicit initializer"):
        edits.apply_edits(
            bundle,
            packet,
            patch(
                output_mappings=[{"channel": "I", "expression": "i+offset"}],
                new_parameters=[{"name": "offset"}],
            ),
        )


def test_refit_only_without_incumbent_never_constructs(tmp_path, monkeypatch):
    _, root, plan = continued(tmp_path)
    task = {**plan["tasks"][1], "arm": "refit_only"}

    def forbidden(*args, **kwargs):
        raise AssertionError("refit-only control must not construct")

    monkeypatch.setattr(construction, "propose", forbidden)
    client = pipeline._client(root, plan, task, 1, "http://offline", lambda: True)
    assert isinstance(client, RevisionClient)
    proposal = pipeline.propose_one(root, plan, task, 1, client)
    assert proposal["status"] == "closed_lineage" and not client.records


def test_unexecutable_fresh_construction_is_recorded_without_aborting_batch(
    tmp_path, monkeypatch
):
    _, _, plan = continued(tmp_path)
    task = plan["tasks"][1]
    monkeypatch.setattr(
        construction,
        "run_staged_topology",
        lambda *a, **kw: {
            "complete_topology": True,
            "public_structure_checks_passed": True,
        },
    )
    monkeypatch.setattr(
        construction, "run_staged_functions", lambda *a, **kw: {"complete_model": True}
    )

    def rejected(*args):
        raise ValueError("construction reconstruction did not pass")

    monkeypatch.setattr(pipeline, "_bundle", rejected)
    result = construction.propose(plan, task, {}, None, tmp_path)
    assert result["status"] == "contract_failed"
    assert result["bundle"] is None and result["construction_draft"] is None
    assert "reconstruction did not pass" in result["error"]


def test_new_scheduler_requires_proposer_success_and_preserves_submission_receipts(
    tmp_path, monkeypatch
):
    _, root, plan = continued(tmp_path)
    module = _submit_module()
    for name in ("AF_PYTHON", "AF_VLLM_IMAGE"):
        path = tmp_path / name
        path.touch()
        monkeypatch.setenv(name, str(path))
    for name in ("AF_HF_HOME", "AF_COMPUTE_CACHE_ROOT", "AF_IPC_TMP_ROOT"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.delenv("AF_COMMIT", raising=False)
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda argv, **kw: "" if "status" in argv else "a" * 40,
    )
    jobs = []

    def submit(argv, **kw):
        jobs.append(argv)
        return subprocess.CompletedProcess(argv, 0, str(100 + len(jobs)), "")

    monkeypatch.setattr(module.subprocess, "run", submit)
    receipt = module.submit(root, 1)
    assert "--dependency=afterok:102" in jobs[2]
    assert "--dependency=afterok:104" in jobs[4]
    assert module.submit(root, 1) == receipt and len(jobs) == 5
    assert receipt["plan_sha256"] == plan["artifact_sha256"]


def test_finish_job_fails_when_round_results_are_missing(tmp_path):
    _, root, _ = continued(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(io.REPO / "scripts/review_deadline.py"),
            "finish-round",
            "--root",
            str(root),
            "--round",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "round incomplete" in result.stderr
    assert (root / "summary.json").exists()


def test_interrupted_fit_retains_unfitted_draft_without_replenishing_visit(tmp_path):
    _, root, plan = continued(tmp_path)
    task = plan["tasks"][1]
    draft = io.read_round(root, task, 0)["construction_draft"]
    result = interrupted(root, plan, task, 1, "synthetic interruption")
    assert result["construction_draft"] == draft and not result["closed"]
    assert result["fresh_budget_on_resume"] is False
    assert interrupted(root, plan, task, 1, "another call") == result


@pytest.mark.parametrize(
    "status", ["revision_failed", "no_change", "residual_evidence_unavailable"]
)
def test_incumbent_survives_unsuccessful_revision_and_refit(
    tmp_path, monkeypatch, status
):
    _, root, plan = continued(tmp_path)
    task = plan["tasks"][0]
    parent = io.read_round(root, task, 0)
    directory = io.round_path(root, task, 1)
    sealed_write(
        directory / "proposal.json",
        {
            "parent_sha256": parent["artifact_sha256"],
            "status": status,
            "bundle": None,
            "cost": {"physical_requests": 0},
        },
    )
    fit = PublicFitResult.model_validate(parent["selected"]["fit"]).model_copy(
        update={
            "status": "fit_failed",
            "parameters": None,
            "training": PublicFitMetrics(available=False),
            "validation": PublicFitMetrics(available=False),
        }
    )
    calls = []
    monkeypatch.setattr(
        pipeline.sibling_fit, "prepare_child_fit", lambda *a, **kw: calls.append(a)
    )

    def execute(path):
        path.mkdir(parents=True)
        public._write(path / "result.json", fit.model_dump(mode="json"))
        return fit

    monkeypatch.setattr(pipeline.sibling_fit, "execute_child_fit", execute)
    result = pipeline.fit_one(root, plan, task, 1)
    assert result["selected"] == parent["selected"] and not result["closed"]
    assert result["fit_trigger"] == "incumbent_fallback" and len(calls) == 1
    assert pipeline.fit_one(root, plan, task, 1) == result and len(calls) == 1
