"""Public-only plans, exact traces, bounded context and the restored repair path."""

import json

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.response_revision import PromptPreflightError
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_contract as contract
from autoformalism.research import phase_c_construction as adapter
from autoformalism.research import variable_confirmation as v
from autoformalism.schemas.staged_topology import VariableReply
from autoformalism.search import review_multi_construction as repair
from tests.test_phase_c_construction_baseline import (
    construction_transport,
    fixture,
    tokenize,
)


def test_only_reviewed_legacy_inferences_removed_and_dependencies_retained():
    path = (
        v.REPO
        / "configs/target_eval/phase_b_v2/specs"
        / "phase_b_dalla_man_t2_canonical_named_easy.json"
    )
    raw = json.loads(path.read_text())
    raw["benchmark_id"] = "phase_c_dalla_man_t2_canonical_named_easy_rates_v1"
    original = json.dumps(raw, sort_keys=True)
    cell = contract.correct_roles({"target_contract": raw})
    assert json.dumps(raw, sort_keys=True) == original
    assert contract.target_definitions(cell) == {}
    assert len(cell["construction_contract_policy"]["changes"]) == 2
    assert (
        cell["target_contract"]["targets"][2]["required_dependencies"]
        == raw["targets"][2]["required_dependencies"]
    )
    raw["targets"][2]["representation_requirement"] = (
        "Explicit public requirement: U must be instantaneous."
    )
    cell = contract.correct_roles({"target_contract": raw})
    assert contract.target_definitions(cell) == {"U": "algebraic"}
    assert (
        "U must be instantaneous"
        in contract.visible_brief(
            {**cell, "brief": fixture_brief()}, {"arm": "full"}
        ).scientific_context
    )


def fixture_brief():
    return {
        "scientific_context": "Public context",
        "requirements": [],
        "public_variables": [{"name": "U", "data_role": "target"}],
    }


def test_review_plan_reports_agenda_and_review_separately(tmp_path):
    source = tmp_path / "old"
    old = fixture(source)
    root = tmp_path / "new"
    plan = v.freeze(source / "plan.json", root, review_inventory=True)
    assert plan == v.verify(root)
    assert plan["config"]["model_settings"] == old["config"]["model_settings"]
    assert plan["inventory_review_policy"] == v.REVIEW_POLICY
    with pytest.raises(ValueError):
        v.freeze(source / "plan.json", root, review_inventory=False)
    calls = []
    replies = []
    agenda_transport = construction_transport(calls)

    def transport(url, body, timeout):
        if not replies:
            result = agenda_transport(url, body, timeout)
            replies.append(result)
            return result
        # The fixture has no optional observations. Repeat the complete draft.
        assert json.loads(body["messages"][1]["content"])["policy"] == v.REVIEW_POLICY
        calls.append(body)
        return replies[0]

    kwargs = {"transport": transport, "token_transport": tokenize}
    task = plan["tasks"][0]
    outcome = v.propose(root, plan, task, "http://offline", **kwargs)
    assert outcome["status"] == "variables_complete"
    assert outcome["inventory_review"]["status"] == "accepted"
    assert v.propose(root, plan, task, "http://offline", **kwargs) == outcome
    assert len(calls) == 2
    report = v.report(root, plan)
    assert report["first_reply_total"] == report["first_reply_accepted"] == 1
    assert report["inventory_review_calls"] == 1
    assert report["inventory_review_repairs"] == 0
    assert report["inventory_review_status_counts"] == {"accepted": 1}
    assert report["rows"][0]["cost"]["physical_requests"] == 2
    assert report["optimizer_calls"] == report["solver_rollouts"] == 0
    trace = json.loads(next(root.rglob("trace.json")).read_text())
    assert {r["step"] for r in trace["calls"]} == {"variables_0", "inventory_review"}
    assert all(r["runtime_events"] for r in trace["calls"])


def test_freeze_run_resume_report_without_topology_or_data(tmp_path):
    source = tmp_path / "old"
    old = fixture(source)
    root = tmp_path / "new"
    plan = v.freeze(source / "plan.json", root)
    assert plan == v.verify(root) == v.freeze(source / "plan.json", root)
    assert plan["source_plan_sha256"] == old["artifact_sha256"]
    assert all(
        set(c)
        == {
            "brief",
            "context",
            "target_contract",
            "evidence",
            "construction_contract_policy",
        }
        for c in plan["cells"].values()
    )
    assert all(
        k not in c
        for c in plan["cells"].values()
        for k in ("training", "validation", "independent_rules", "mechanism_spec")
    )
    calls = []
    task = plan["tasks"][0]
    kwargs = {"transport": construction_transport(calls), "token_transport": tokenize}
    result = v.propose(root, plan, task, "http://offline", **kwargs)
    assert result["status"] == "variables_complete", result
    assert result["memory_candidates"] == {"memory": ["m"]}
    assert len(calls) == 1
    assert v.propose(root, plan, task, "http://offline", **kwargs) == result
    assert len(calls) == 1
    summary = v.report(root, plan)
    assert summary["status_counts"] == {"variables_complete": 1}
    assert summary["optimizer_calls"] == 0 and summary["solver_rollouts"] == 0
    assert summary["first_reply_total"] == summary["first_reply_accepted"] == 1
    assert (root / "VARIABLES.html").exists()
    assert not list(root.rglob("fit"))
    trace = json.loads(next(root.rglob("trace.json")).read_text())
    assert all(c["step"].startswith("variables_") for c in trace["calls"])
    assert all(
        "Runtime public target contract" in json.dumps(c["request"])
        for c in trace["calls"]
    )
    assert not list(root.rglob("topology_stage.json"))
    assert sealed_read(source / "plan.json") == old


def test_drain_does_not_create_terminal_result_or_reset_cache(tmp_path):
    fixture(tmp_path / "old")
    plan = v.freeze(tmp_path / "old/plan.json", tmp_path / "new")
    with pytest.raises(DeferredCall):
        v.propose(
            tmp_path / "new",
            plan,
            plan["tasks"][0],
            "http://unused",
            can_start=lambda: False,
        )
    assert not list((tmp_path / "new").rglob("proposal.json"))


def test_context_preflight_refuses_without_truncation_or_generation(tmp_path):
    calls = []
    client = ConstructionClient(
        settings=StagedModelSettings(),
        directory=tmp_path,
        namespace="test",
        seed=0,
        base_url="http://offline",
        token_transport=lambda *args: {"count": 25000, "max_model_len": 32768},
        transport=lambda *args: calls.append(args),
    )
    with pytest.raises(PromptPreflightError, match="no generation"):
        client.call(
            system="Scientific rules",
            user="all context",
            response_model=VariableReply,
            step="variables_0",
            attempt=0,
        )
    assert not calls and not client.records
    assert (
        json.loads(next((tmp_path / "preflight").glob("*.json")).read_text())["status"]
        == "context_limit_exceeded"
    )


def test_adapter_repairs_executable_draft_and_keeps_route_evidence(
    tmp_path, monkeypatch
):
    cell = {
        "brief": fixture_brief(),
        "target_contract": {"public_prompt_sha256": "0" * 64, "targets": []},
        "evidence": {},
    }
    bundle = {"candidate": {"state_equations": [], "processes": []}}
    checks = []

    def certificates(candidate, *_):
        checks.append(candidate)
        return {"eligible_for_development_selection": candidate.get("repaired", False)}

    def construct(visible, task, client, directory, **kwargs):
        assert (
            "Runtime public target contract" in visible["brief"]["scientific_context"]
        )
        assert kwargs["retain_failed_draft"] and kwargs["explicit_mechanism_bindings"]
        return {
            "status": "construction_failed",
            "bundle": None,
            "construction_draft": bundle,
            "attempts": [
                {"route": "process", "error": "public requirement check failed"}
            ],
        }

    monkeypatch.setattr(adapter.shared_construction, "construct", construct)
    monkeypatch.setattr(contract, "target_definitions", lambda cell: {})
    monkeypatch.setattr(repair.pipeline, "certificates", certificates)
    monkeypatch.setattr(
        repair.pipeline, "_certificate_feedback", lambda *_: {"requires_repair": True}
    )
    monkeypatch.setattr(
        repair.edits, "payload", lambda *_: {"model": {"equations": []}}
    )
    monkeypatch.setattr(
        repair.edits,
        "apply_edits",
        lambda *_, **kw: {"bundle": {**bundle, "repaired": True}, "provenance": {}},
    )

    class Client:
        def call(self, **kwargs):
            assert kwargs["step"] == "repair_public_construction"
            assert "public_target_contract" in json.loads(kwargs["user"])
            return {
                "request_hash": "example",
                "status": "responded",
                "raw_response": {
                    "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]
                },
            }

    result = adapter.propose(cell, {"arm": "full"}, Client(), tmp_path)
    assert result["status"] == "constructed" and len(checks) == 2
    assert len(result["public_contract_repair_attempts"]) == 1
    assert result["construction_attempts"][0]["route"] == "process"


def test_variable_submission_has_no_fit_jobs_and_preserves_receipts(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from scripts import submit_phase_c_baseline as submitter

    fixture(tmp_path / "old")
    root = tmp_path / "new"
    v.freeze(tmp_path / "old/plan.json", root)
    calls = []

    def submit_job(directory, stage, opts, worker, action, index):
        calls.append((stage, opts))
        job = str(1000 + len(calls))
        (directory / f"{stage}.intent.json").write_text(
            json.dumps(
                {
                    "argv": [
                        "sbatch",
                        "--parsable",
                        *opts,
                        str(worker),
                        action,
                        str(index),
                    ]
                }
            )
        )
        (directory / f"{stage}.id").write_text(job)
        return job

    monkeypatch.setattr(
        submitter,
        "support",
        lambda _: SimpleNamespace(
            source_commit=lambda _: "0" * 40, submit_job=submit_job
        ),
    )
    for key in (
        "AF_PYTHON",
        "AF_VLLM_IMAGE",
        "AF_HF_HOME",
        "AF_COMPUTE_CACHE_ROOT",
        "AF_IPC_TMP_ROOT",
    ):
        monkeypatch.setenv(key, "/test")
    result = submitter.submit(root, "variables-1", variables_only=True)
    assert set(result["jobs"]) == {"propose", "report"}
    assert "--dependency=afterany:1001" in calls[1][1]
    assert not any("--array" in option for _, opts in calls for option in opts)
    assert (
        submitter.submit(root, "variables-1", variables_only=True) == result
        and len(calls) == 2
    )
    (root / "submissions/variables-1/report.id").unlink()
    with pytest.raises(ValueError, match="uncertain"):
        submitter.submit(root, "variables-1", variables_only=True)
