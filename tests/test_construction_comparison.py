"""Matched schedules, real caching, deferred work and before/after evidence."""

import json
import subprocess
from collections import Counter
from types import SimpleNamespace

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_comparison as campaign
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_schedules as schedules
from scripts import submit_phase_c_baseline as submitter
from tests.test_construction_ledger import brief, context, equation, patch, variable
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize
from tests.test_phase_c_topology import sources


def source_fixture(tmp_path):
    plan = sources(tmp_path)
    plan.pop("artifact_sha256")
    plan["config"]["limits"] = next(iter(plan["cells"].values()))["brief"]["limits"]
    path = tmp_path / "new/plan.json"
    path.unlink()  # Normalize limits in the synthetic fixture only.
    return sealed_write(path, plan)


def test_comparison_submission_has_no_fit_and_resumes_receipts(tmp_path, monkeypatch):
    source_fixture(tmp_path)
    root = tmp_path / "comparison"
    campaign.freeze(tmp_path / "new", root)
    calls = []

    def submit_job(directory, stage, opts, worker, action, index):
        calls.append((stage, opts))
        job = str(100 + len(calls))
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
    result = submitter.submit(root, "comparison-1", compare_construction=True)
    assert set(result["jobs"]) == {"propose", "report"}
    assert result["resources"] == "aces-1h100-comparison-only-1"
    assert "--dependency=afterany:101" in calls[1][1]
    assert submitter.submit(root, "comparison-1", compare_construction=True) == result
    assert len(calls) == 2
    with pytest.raises(ValueError, match="choose one"):
        submitter.submit(root, "bad", topology_only=True, compare_construction=True)
    (root / "submissions/comparison-1/report.id").unlink()
    with pytest.raises(ValueError, match="uncertain"):
        submitter.submit(root, "comparison-1", compare_construction=True)


def transport(calls, *, bad_path=False, invalid_first=False):
    """Honor the policy scope while exercising the actual common edit schema."""

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        assert "supplied/unused declarations are needed" not in body["messages"][0][
            "content"
        ].replace("no supplied/unused declarations are needed", "")
        stage, policy = p["stage"], p["policy"]
        targets = [
            v["name"] for v in p["public_source_catalog"] if v["data_role"] == "target"
        ]
        driver = next(
            v["name"]
            for v in p["public_source_catalog"]
            if v["data_role"] == "external_input"
        )
        variables = [variable(n) for n in [*targets, "m"]]
        bindings = [
            {"requirement_id": r["id"], "memory_states": ["m"]}
            for r in p["memory_requirements"]
        ]
        if invalid_first and len(calls) == 1:
            return response({"unrecognized_field": True})
        if stage == "variables":
            edits = patch(
                variables=variables, mechanism_bindings=bindings, stage_complete=True
            )
        elif stage == "relationships":
            edits = patch(
                variables=[] if policy == "separate" else variables,
                mechanism_bindings=bindings,
                stage_complete=True,
            )
        else:
            names = [p["selected_lhs"]] if p["selected_lhs"] else [*targets, "m"]
            eqs = [
                equation(
                    n,
                    *(
                        ["m"]
                        if n != "m" or (bad_path and stage != "repair")
                        else [driver, "m"]
                    ),
                )
                for n in names
            ]
            edits = patch(equations=eqs, stage_complete=p["selected_lhs"] is None)
        return response(edits.model_dump(mode="json"))

    return send


def run(tmp_path, policy, calls, **transport_options):
    settings = StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288)
    client = ConstructionClient(
        settings=settings,
        directory=tmp_path / "calls",
        namespace="toy",
        seed=0,
        base_url="http://offline",
        transport=transport(calls, **transport_options),
        token_transport=tokenize,
    )
    return schedules.run(
        brief(True),
        context(),
        {},
        brief(True).model_dump(mode="json"),
        client,
        tmp_path / "construction",
        policy,
    )


@pytest.mark.parametrize("policy", schedules.POLICIES)
def test_schedules_first_draft_and_exact_cache_replay(tmp_path, policy):
    calls = []
    result = run(tmp_path, policy, calls)
    assert result["status"] == "topology_complete", result
    assert result["before_repair"]["assessment"]["eligible"]
    assert not any(p["stage"] == "repair" for p in calls)
    assert calls[0]["current_draft"]["declarations"] == ledger.Draft().model_dump(
        mode="json"
    )
    if policy != "joint_adaptive":
        assert [p["selected_lhs"] for p in calls if p["stage"] == "equations"] == [
            "y",
            "m",
        ]
    n = len(calls)
    assert run(tmp_path, policy, calls) == result
    assert len(calls) == n


@pytest.mark.parametrize("policy", schedules.POLICIES)
def test_global_repair_after_all_equations_retains_initial_failure(tmp_path, policy):
    calls = []
    result = run(tmp_path, policy, calls, bad_path=True, invalid_first=True)
    assert result["status"] == "topology_complete", result
    assert not result["before_repair"]["assessment"]["eligible"]
    assert result["before_repair"]["draft"] != result["draft"]
    repairs = [p for p in calls if p["stage"] == "repair"]
    assert len(repairs) == 1
    assert {
        e["name"] for e in repairs[0]["current_draft"]["declarations"]["equations"]
    } == {"y", "m"}
    assert repairs[0]["runtime_diagnostics"]["structural_failures"]
    # Both initial/local-format feedback and whole-model repair retain all context.
    for p in calls:
        assert p["public_brief"] == brief(True).model_dump(mode="json")
        assert p["public_source_catalog"]
    assert calls[1]["runtime_diagnostics"]["rejected_reply"] == {
        "unrecognized_field": True
    }


def test_policy_scope_does_not_change_structural_rules():
    edits = patch(
        variables=[variable("x")], equations=[equation("y", "x"), equation("x", "u")]
    )
    schedules.validate_scope("joint_adaptive", "equations", None, edits)
    with pytest.raises(ValueError, match="only equation"):
        schedules.validate_scope("joint_fixed", "equations", "y", edits)
    with pytest.raises(ValueError, match="fixes the variable"):
        schedules.validate_scope("separate", "equations", "y", edits)
    for policy in schedules.POLICIES:
        schedules.validate_scope(policy, "repair", None, edits)


def test_resume_after_deferred_request_has_no_new_first_calls(tmp_path):
    calls = []
    settings = StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288)

    def client(allowed):
        return ConstructionClient(
            settings=settings,
            directory=tmp_path / "calls",
            namespace="toy",
            seed=0,
            base_url="http://offline",
            transport=transport(calls),
            token_transport=tokenize,
            can_start=allowed,
        )

    with pytest.raises(DeferredCall):
        schedules.run(
            brief(True),
            context(),
            {},
            brief(True).model_dump(mode="json"),
            client(lambda: len(calls) < 2),
            tmp_path / "construction",
            "joint_fixed",
        )
    result = schedules.run(
        brief(True),
        context(),
        {},
        brief(True).model_dump(mode="json"),
        client(lambda: True),
        tmp_path / "construction",
        "joint_fixed",
    )
    assert result["status"] == "topology_complete"
    assert len(calls) == 3


def test_final_reference_failure_stops_with_equal_bounded_repair(tmp_path):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        return response(
            patch(
                variables=[variable("y")],
                equations=[]
                if p["stage"] == "relationships"
                else [equation("y", "missing")],
                stage_complete=True,
            ).model_dump(mode="json")
        )

    client = ConstructionClient(
        settings=StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288),
        directory=tmp_path / "calls",
        namespace="toy",
        seed=0,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
    )
    result = schedules.run(
        brief(),
        context(),
        {},
        brief().model_dump(mode="json"),
        client,
        tmp_path / "construction",
        "joint_adaptive",
    )
    assert result["status"] == "topology_incomplete"
    assert len([p for p in calls if p["stage"] == "repair"]) == 3
    assert "missing_declarations" in str(result["assessment"]["errors"])


def test_freeze_has_96_fresh_matched_tasks_and_no_old_models(tmp_path):
    source_fixture(tmp_path)
    plan = campaign.freeze(tmp_path / "new", tmp_path / "comparison")
    assert campaign.verify(tmp_path / "comparison") == plan
    assert Counter(t["policy"] for t in plan["tasks"]) == dict.fromkeys(
        schedules.POLICIES, 32
    )
    assert sum("detention" in t["benchmark_id"] for t in plan["tasks"]) == 24
    assert "inputs" not in plan
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
    result = campaign.report(tmp_path / "comparison", plan)
    assert result["status_counts"] == {"pending": 96}
    assert all(g["tokens"]["median"] is None for g in result["groups"].values())
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(plan["config"]))
    dispatch = subprocess.run(
        [
            "bash",
            "scripts/hpc/run_staged_topology_server.sh",
            "--check-config",
            str(cfg),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert dispatch.stdout.strip() == "phase_c_construction_comparison.py"


def test_campaign_trace_cost_reporting_and_corrupt_cache_refusal(tmp_path):
    source_fixture(tmp_path)
    root = tmp_path / "comparison"
    plan = campaign.freeze(tmp_path / "new", root)
    task, calls = plan["tasks"][0], []
    args = {"transport": transport(calls), "token_transport": tokenize}
    result = campaign.propose(root, plan, task, "http://offline", **args)
    assert result["status"] == "topology_complete", result
    assert result["cost"]["physical_requests"] == len(calls)
    assert not result["parameter_fitting_performed"]
    assert not any(
        "validation" in p or "independent_rules" in json.dumps(p) for p in calls
    )
    assert campaign.propose(root, plan, task, "http://offline", **args) == result
    summary = campaign.report(root, plan)
    assert summary["status_counts"] == {"topology_complete": 1, "pending": 95}
    assert "Before overall repair" in (root / "TOPOLOGY.html").read_text()
    directory = campaign.baseline.location(root, task)
    assert len(json.loads((directory / "trace.json").read_text())["calls"]) == len(
        calls
    )
    assert sealed_read(directory / "construction/before_repair.json")["assessment"][
        "eligible"
    ]
    next((directory / "calls").glob("*.json")).unlink()
    with pytest.raises(ValueError, match="call records"):
        campaign.propose(root, plan, task, "http://offline")


def test_initial_request_budget_preserves_repair_and_deterministic_resume(tmp_path):
    calls = []

    def attempt():
        client = ConstructionClient(
            settings=StagedModelSettings(
                maximum_requests=5, maximum_total_tokens=524288
            ),
            directory=tmp_path / "calls",
            namespace="limited",
            seed=0,
            base_url="http://offline",
            transport=transport(calls),
            token_transport=tokenize,
        )
        return schedules.run(
            brief(True),
            context(),
            {},
            brief(True).model_dump(mode="json"),
            client,
            tmp_path / "construction",
            "separate",
        )

    result = attempt()
    assert result["status"] == "topology_complete"
    assert not result["before_repair"]["ready_requested"]
    assert result["before_repair"]["cost"]["physical_requests"] == 2
    assert len(calls) == 3
    assert attempt() == result
    assert len(calls) == 3


def test_deferred_global_repair_reuses_initial_draft(tmp_path):
    calls = []

    def attempt(limit):
        client = ConstructionClient(
            settings=StagedModelSettings(
                maximum_requests=128, maximum_total_tokens=524288
            ),
            directory=tmp_path / "calls",
            namespace="repair-resume",
            seed=0,
            base_url="http://offline",
            transport=transport(calls, bad_path=True),
            token_transport=tokenize,
            can_start=lambda: len(calls) < limit,
        )
        return schedules.run(
            brief(True),
            context(),
            {},
            brief(True).model_dump(mode="json"),
            client,
            tmp_path / "construction",
            "joint_adaptive",
        )

    with pytest.raises(DeferredCall):
        attempt(2)
    first = (tmp_path / "construction/before_repair.json").read_bytes()
    result = attempt(3)
    assert result["status"] == "topology_complete"
    assert len(calls) == 3
    assert (tmp_path / "construction/before_repair.json").read_bytes() == first


def test_missing_transaction_rejected_even_when_all_calls_survive(tmp_path):
    source_fixture(tmp_path)
    root = tmp_path / "comparison"
    plan = campaign.freeze(tmp_path / "new", root)
    task = plan["tasks"][0]
    campaign.propose(
        root,
        plan,
        task,
        "http://offline",
        transport=transport([]),
        token_transport=tokenize,
    )
    directory = campaign.baseline.location(root, task)
    sorted((directory / "construction/events").glob("*.json"))[-1].unlink()
    with pytest.raises(ValueError, match="transaction"):
        campaign.propose(root, plan, task, "http://offline")
