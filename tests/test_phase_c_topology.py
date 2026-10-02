"""Saved-inventory provenance, real topology stages, bounded repair and resume."""

import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_baseline as baseline
from autoformalism.research import topology_confirmation as campaign
from autoformalism.research import topology_inspection as inspection
from autoformalism.research import variable_confirmation as variables
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search.staged_topology_runner import runtime_seeded_inventory
from autoformalism.search.variable_equation_usage import POLICY
from scripts import submit_phase_c_baseline as submitter
from tests.test_explicit_variable_bindings import response, variable
from tests.test_phase_c_construction_baseline import fixture, tokenize


def sources(tmp_path):
    """Synthetic public contexts under the roster names, not benchmark answers."""
    original = fixture(tmp_path / "original")
    cell = next(iter(original["cells"].values()))
    public_cell = {
        k: cell[k] for k in ("brief", "context", "target_contract", "evidence")
    }
    tasks = [
        {
            **original["tasks"][0],
            "task_id": f"case{i}_seed{seed}_{arm}",
            "benchmark_id": name,
            "seed": seed,
            "arm": arm,
        }
        for i, name in enumerate(campaign.phase_c_inputs.ROSTER)
        for seed in (0, 1)
        for arm in ("full", "brief_only")
    ]
    for label, targeted in (("base", False), ("usage", True)):
        selected = [
            t
            for t in tasks
            if not targeted or t["benchmark_id"] in variables.TARGETED_CASES
        ]
        plan = sealed_write(
            tmp_path / label / "plan.json",
            {
                "protocol": variables.PROTOCOL,
                "config": {
                    **original["config"],
                    "protocol": variables.PROTOCOL,
                    "review_inventory": not targeted,
                    "targeted_usage": targeted,
                },
                "cells": {
                    t["benchmark_id"]: copy.deepcopy(public_cell) for t in selected
                },
                "tasks": selected,
                "source_identity": {"historical": label},
                "source_plan_sha256": original["artifact_sha256"],
                "test_data_opened": False,
                "equation_usage_policy": POLICY if targeted else None,
            },
        )
        inventory = [
            v.model_dump(mode="json")
            for v in runtime_seeded_inventory(
                PublicScientificBrief.model_validate(cell["brief"])
            )
        ]
        inventory.extend([variable("m"), variable("v01"), variable("w", "algebraic")])
        for task in selected:
            sealed_write(
                baseline.location(tmp_path / label, task) / "proposal.json",
                {
                    "identity": baseline.namespace(plan, task),
                    "status": "variables_complete",
                    "inventory": inventory,
                    "memory_candidates": {"memory": ["m"]},
                    "cost": _cost([]),
                },
            )
    return campaign.freeze(tmp_path / "base", tmp_path / "usage", tmp_path / "new")


def transport(
    calls,
    *,
    process=False,
    invalid_process=False,
    repair=False,
    revision=False,
    fail_process_route=False,
):
    """Return deterministic schemas through the real provider/caching interface."""

    def respond(url, body, timeout):
        text = body["messages"][1]["content"]
        p = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        calls.append(p)
        assert "variable_usage_context" in p
        if "eligible_equation_targets" in p:
            return response(
                {
                    "processes": [
                        {
                            "name": "q",
                            "depends_on": ["m"],
                            "kind": "transfer",
                            "scientific_meaning": "shared release",
                            "uses": [
                                {"target": "m", "sign": "negative", "conversion": "1"},
                                {
                                    "target": "m" if invalid_process else "v01",
                                    "sign": "positive",
                                    "conversion": "1",
                                },
                            ],
                        }
                    ]
                    if process or invalid_process
                    else []
                }
            )
        assert "selected_lhs" in p, "no variable/function/initializer call permitted"
        name = p["selected_lhs"]["name"]
        assert p["selected_lhs"]["scientific_role"]
        assert p["variable_usage_context"]["memory_bindings"] == {"memory": ["m"]}
        if revision:
            return response(
                {
                    "terms": [],
                    "inventory_revision": {
                        "variable": variable("other"),
                        "reason": "another memory coordinate required",
                    },
                }
            )
        in_process = any(v["name"] == "q" for v in p["frozen_inventory"])
        bad = (repair and name == "v01" and "runtime_diagnostics" not in p) or (
            fail_process_route and in_process
        )
        names = ["unknown"] if bad else ["u01", "m"] if name == "m" else ["m"]
        terms = [
            {
                "sources": names,
                "outer_weight_sign": "positive",
                "scientific_role": "scientific contribution",
            }
        ]
        if in_process and name == "v01" and not bad:
            terms = []  # Runtime supplies the single process contribution.
        return response({"terms": terms, "inventory_revision": None})

    return respond


def propose(root, plan, task, calls, **kwargs):
    return campaign.propose(
        root,
        plan,
        task,
        "http://offline",
        token_transport=tokenize,
        transport=transport(calls, **kwargs),
    )


def test_import_exact_32_with_basins_and_source_provenance(tmp_path):
    plan = sources(tmp_path)
    assert campaign.verify(tmp_path / "new") == plan
    assert Counter(v["origin"]["source"] for v in plan["inputs"].values()) == {
        "variables_v2": 20,
        "usage_diagnostic": 12,
    }
    assert len(plan["tasks"]) == 32
    assert sum("detention" in t["benchmark_id"] for t in plan["tasks"]) == 8
    assert all(
        set(c) == {"brief", "context", "target_contract", "evidence"}
        for c in plan["cells"].values()
    )
    assert (
        campaign.freeze(tmp_path / "base", tmp_path / "usage", tmp_path / "new") == plan
    )
    assert campaign.report(tmp_path / "new", plan)["status_counts"] == {"pending": 32}
    assert (
        campaign.report(tmp_path / "new", plan)["new_tokens_per_construction_median"]
        is None
    )


@pytest.mark.parametrize(
    "mutation", ["failed", "namespace", "missing", "cost", "binding"]
)
def test_bad_source_inventory_is_never_replaced_by_another_run(tmp_path, mutation):
    plan = sources(tmp_path)
    task = next(
        t for t in plan["tasks"] if t["benchmark_id"] in variables.TARGETED_CASES
    )
    path = baseline.location(tmp_path / "usage", task) / "proposal.json"
    value = sealed_read(path)
    path.unlink()  # Only a temporary fixture, never a campaign repair.
    value.pop("artifact_sha256")
    if mutation == "failed":
        value["status"] = "failed"
    elif mutation == "namespace":
        value["identity"] = "wrong"
    elif mutation == "cost":
        value["cost"]["physical_requests"] = 1
    elif mutation == "binding":
        value["memory_candidates"] = {"memory": ["v01"]}
    if mutation != "missing":
        sealed_write(path, value)
    with pytest.raises(ValueError):
        campaign.freeze(tmp_path / "base", tmp_path / "usage", tmp_path / "bad")


def test_process_and_topology_only_repair_context_usage_and_resume(tmp_path):
    plan = sources(tmp_path)
    root, task, calls = tmp_path / "new", plan["tasks"][0], []
    result = propose(root, plan, task, calls, process=True, repair=True)
    assert result["status"] == "topology_complete", result
    assert not result["fallback_used"]
    assert (
        not result["function_generation_performed"]
        and not result["parameter_fitting_performed"]
    )
    v = [p for p in calls if p.get("selected_lhs", {}).get("name") == "v01"]
    assert len(v) == 2
    for key in (
        "public_brief",
        "frozen_inventory",
        "selected_lhs",
        "current_equation_sketch",
        "committed_process_bindings",
        "variable_usage_context",
    ):
        assert v[0][key] == v[1][key]
    assert "runtime_diagnostics" in v[1]
    assert all("training_observations" in p["public_brief"] for p in calls)
    n = len(calls)
    assert propose(root, plan, task, calls) == result and len(calls) == n
    report = campaign.report(root, plan)
    row = report["rows"][0]
    assert row["equation_repairs"] == 1 and report["new_physical_calls"] == n
    assert all(
        c["status"] == "pass"
        for c in row["routes"][0]["structural_evidence"]["path_checks"]
    )
    assert not list(root.rglob("function_stage.json"))
    assert "q" in " ".join(row["routes"][0]["skeletons"])
    # Successful artifacts cannot hide missing provider records.
    next((baseline.location(root, task) / "calls").glob("*.json")).unlink()
    with pytest.raises(ValueError, match=r"missing call|accounting differs"):
        propose(root, plan, task, calls)


@pytest.mark.parametrize("invalid", [False, True])
def test_optional_empty_or_invalid_process_does_not_stop_topology(tmp_path, invalid):
    plan, calls = sources(tmp_path), []
    task = next(t for t in plan["tasks"] if t["arm"] == "brief_only")
    result = propose(tmp_path / "new", plan, task, calls, invalid_process=invalid)
    assert result["status"] == "topology_complete"
    assert not result["fallback_used"]
    assert all("training_observations" not in p["public_brief"] for p in calls)


def test_process_failure_uses_same_inventory_and_one_budget(tmp_path):
    plan, calls = sources(tmp_path), []
    result = propose(
        tmp_path / "new",
        plan,
        plan["tasks"][0],
        calls,
        process=True,
        fail_process_route=True,
    )
    assert result["status"] == "topology_complete" and result["fallback_used"]
    assert [r["route"] for r in result["attempts"]] == ["process", "ordinary_fallback"]
    assert result["cost"]["physical_requests"] == len(calls)
    assert "q" not in {v["name"] for v in result["topology_result"]["inventory"]}


def test_inventory_revision_is_reported_without_silent_mutation_or_fallback(tmp_path):
    plan, calls = sources(tmp_path), []
    result = propose(
        tmp_path / "new", plan, plan["tasks"][0], calls, process=True, revision=True
    )
    assert result["status"] == "inventory_revision_requested"
    assert len(result["attempts"]) == 1
    assert (
        result["topology_result"]["inventory_revision"]["variable"]["name"] == "other"
    )
    assert "other" not in {v["name"] for v in result["topology_result"]["inventory"]}


def test_interruption_resumes_cached_process_and_prior_equations(tmp_path):
    plan, calls = sources(tmp_path), []
    root, task = tmp_path / "new", plan["tasks"][0]
    with pytest.raises(DeferredCall):
        campaign.propose(
            root,
            plan,
            task,
            "http://offline",
            transport=transport(calls, process=True),
            token_transport=tokenize,
            can_start=lambda: len(calls) < 2,
        )
    assert not (baseline.location(root, task) / "proposal.json").exists()
    assert campaign.report(root, plan)["status_counts"]["in_progress"] == 1
    partial = len(calls)
    result = propose(root, plan, task, calls, process=True)
    assert result["status"] == "topology_complete"
    assert result["cost"]["physical_requests"] == len(calls) > partial
    assert sum("eligible_equation_targets" in p for p in calls) == 1


def test_graph_inspection_does_not_treat_unknown_as_success_or_infer_prose():
    brief = {
        "requirements": [
            {
                "id": "memory",
                "drivers": ["u"],
                "targets": ["x"],
                "requires_dynamic_memory": True,
            }
        ],
        "target_dependencies": [],
    }

    def eq(n, s):
        return {"name": n, "definition": "differential", "terms": [{"sources": s}]}

    top = {
        "equations": [eq("m", ["u"]), eq("p", ["m"]), eq("x", ["p"])],
        "complete_topology": True,
        "memory_candidates": {"memory": ["m"]},
        "inventory": [
            variable("x"),
            variable("u", "supplied"),
            variable("z", "unused"),
        ],
    }
    evidence = inspection.structural_evidence(brief, top)
    assert evidence["path_checks"][0]["memory_witnesses"][0]["memory_to_target"] == [
        "m",
        "p",
        "x",
    ]
    assert evidence["channel_usage"][1]["declared_rhs_in"] == ["m"]
    assert evidence["channel_usage"][2]["declared_rhs_in"] == []
    top["equations"][-1] = eq("x", ["u"])
    assert (
        inspection.structural_evidence(brief, top)["path_checks"][0]["status"] == "fail"
    )
    top["complete_topology"] = False
    assert (
        inspection.structural_evidence(brief, top)["path_checks"][0]["status"]
        == "unresolved"
    )


def test_submission_only_two_jobs_idempotent_and_ambiguous_reply_blocked(
    tmp_path, monkeypatch
):
    sources(tmp_path)
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
    root = tmp_path / "new"
    result = submitter.submit(root, "topology-1", topology_only=True)
    assert set(result["jobs"]) == {"propose", "report"}
    assert result["resources"] == "aces-1h100-topology-only-1"
    assert "--dependency=afterany:101" in calls[1][1]
    assert submitter.submit(root, "topology-1", topology_only=True) == result
    assert len(calls) == 2
    (root / "submissions/topology-1/report.id").unlink()
    with pytest.raises(ValueError, match="uncertain"):
        submitter.submit(root, "topology-1", topology_only=True)
