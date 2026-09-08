"""Source-bound pre-fitting fact audit and localized revision tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import StagedTopologyClient
from autoformalism.rebuttal.staged_function_campaign import function_diagnostic
from autoformalism.rebuttal.staged_function_hybrid_campaign import summarize_hybrid
from autoformalism.rebuttal.staged_prefit_feedback_campaign import (
    audit_from_plan,
    freeze_prefit_campaign,
    run_prefit_campaign,
    select_prefit_finding,
)
from autoformalism.schemas.staged_functions import PrefitScientificReviewReply
from autoformalism.schemas.staged_topology import ModelingLimits, PublicScientificBrief
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.staged_topology import content_hash

REPOSITORY = Path(__file__).resolve().parents[1]


def _provider_response(payload: dict) -> dict:
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(payload)}}
        ],
        "usage": {"total_tokens": 100},
    }


def _source_transport(url, body, timeout):
    del url, timeout
    payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
    if "selected_state" in payload:
        return _provider_response({"initial": {"fixed_value": 0.0}})
    lhs = payload["selected_equation"]["lhs"]
    if lhs == "x":
        expression = "gain * z - decay * x"
        names = ["decay", "gain"]
    else:
        expression = "input_gain * u - memory_decay * z"
        names = ["input_gain", "memory_decay"]
    return _provider_response(
        {
            "functions": [
                {
                    "expression": expression,
                    "parameters": [
                        {"name": name, "role": "nonnegative_coefficient"}
                        for name in names
                    ],
                }
            ]
        }
    )


def _review(status: str, *, fail_category: str = "functional_semantics") -> dict:
    categories = (
        "mechanism_topology",
        "dimensional_consistency",
        "dynamic_plausibility",
        "functional_semantics",
        "parameter_parsimony",
        "latent_initialization",
    )
    findings = []
    for category in categories:
        selected = status if category == fail_category else "pass"
        findings.append(
            {
                "category": category,
                "status": selected,
                "interaction_ids": (
                    ["term_0_0"]
                    if selected == "fail"
                    and category != "mechanism_topology"
                    and category != "latent_initialization"
                    else []
                ),
                "latent_states": (
                    ["z"]
                    if selected == "fail" and category == "latent_initialization"
                    else []
                ),
                "finding": "Public evidence for this category was reviewed.",
                "suggested_change": "Revise only the displayed runtime anchor.",
            }
        )
    return {"schema_version": "prefit-scientific-review-1", "findings": findings}


def _write_source(root: Path) -> tuple[Path, Path, dict]:
    fixture = function_diagnostic("driven_memory", ModelingLimits())
    source = fixture["source"]
    source.update(
        public_structure_checks_passed=True,
        test_data_opened=False,
        private_reference_opened=False,
    )
    config = json.loads(
        (REPOSITORY / "configs/staged_function_hybrid_repair_v2.json").read_text()
    )
    config.update(
        public_cells=["synthetic"],
        seeds=[0],
        source_function_plan_sha256="a" * 64,
        selected_topology={
            "task_id": "synthetic_seed0",
            "result_sha256": content_hash(source),
        },
    )
    from autoformalism.rebuttal.staged_function_hybrid_campaign import (
        HybridFunctionCampaignConfig,
    )

    hybrid_config = HybridFunctionCampaignConfig.model_validate(config)
    source_output = root / "source-run"
    source_result = run_staged_functions(
        PublicScientificBrief.model_validate(fixture["brief"]),
        ValidationContext.model_validate(fixture["context"]),
        source,
        StagedTopologyClient(
            settings=hybrid_config.model_settings,
            base_url="http://local",
            directory=source_output / "calls",
            namespace="source",
            seed=0,
            transport=_source_transport,
        ),
        source_output,
        generation_granularity="equation_batch_atomic_repair",
        function_repair_policy="certified_outer_gain",
    )
    assert source_result["complete_model"]
    source_task = {
        "task_id": "synthetic_seed0_hybrid_seed0",
        "kind": "benchmark",
        "seed": 0,
        "generation_granularity": "equation_batch_atomic_repair",
        "function_repair_policy": "certified_outer_gain",
        "brief": fixture["brief"],
        "context": fixture["context"],
        "source": source,
        "reviewed_source_task": "synthetic_seed0",
    }
    source_plan = {
        "schema_version": "scientific-staged-function-hybrid-repair-plan-2",
        "config": hybrid_config.model_dump(mode="json"),
        "source_function_plan_sha256": "a" * 64,
        "runtime_source_sha256": "b" * 64,
        "launcher_sha256": "c" * 64,
        "tasks": [source_task],
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    source_plan["plan_sha256"] = content_hash(source_plan)
    source_plan_path = root / "source-plan.json"
    source_plan_path.write_text(json.dumps(source_plan))
    source_results = root / "source-results"
    terminal = {
        "identity": content_hash([source_plan["plan_sha256"], source_task]),
        "task_id": source_task["task_id"],
        "seed": 0,
        "result": source_result,
    }
    terminal_path = source_results / source_task["task_id"] / "terminal.json"
    terminal_path.parent.mkdir(parents=True)
    terminal_path.write_text(json.dumps(terminal))
    (source_results / "summary.json").write_text(
        json.dumps(summarize_hybrid(source_plan, [terminal]))
    )
    return source_plan_path, source_results, source_plan


def _write_prefit_config(root: Path, source_plan: dict) -> Path:
    config = json.loads(
        (REPOSITORY / "configs/staged_prefit_feedback_v1.json").read_text()
    )
    config.update(
        public_cells=["synthetic"],
        seeds=[0],
        source_hybrid_plan_sha256=source_plan["plan_sha256"],
    )
    path = root / "prefit-config.json"
    path.write_text(json.dumps(config))
    return path


def test_freeze_audit_and_localized_function_revision(
    tmp_path: Path, monkeypatch
) -> None:
    import autoformalism.rebuttal.staged_prefit_feedback_campaign as campaign

    source_plan_path, source_results, source_plan = _write_source(tmp_path)
    config = _write_prefit_config(tmp_path, source_plan)
    plan = freeze_prefit_campaign(
        config,
        source_plan_path,
        source_results,
        tmp_path / "plan.json",
    )
    audit = audit_from_plan(plan)
    assert audit["deterministic_prefit_pass_rate"] == 1.0
    calls = []
    review_count = 0

    def transport(url, body, timeout):
        nonlocal review_count
        del url, timeout
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        if payload["schema_version"] == "prefit-scientific-review-request-1":
            review_count += 1
            return _provider_response(_review("fail" if review_count == 1 else "pass"))
        assert payload["schema_version"] == "prefit-function-revision-request-1"
        return _provider_response(
            {
                "expression": "new_gain * z - new_decay * x",
                "parameters": [
                    {"name": "new_gain", "role": "nonnegative_coefficient"},
                    {"name": "new_decay", "role": "nonnegative_coefficient"},
                ],
            }
        )

    monkeypatch.setattr(
        campaign,
        "StagedTopologyClient",
        lambda **kwargs: StagedTopologyClient(**kwargs, transport=transport),
    )
    summary = run_prefit_campaign(
        tmp_path / "plan.json", tmp_path / "results", "http://local"
    )
    assert summary["status"] == "complete"
    assert summary["function_revision_count"] == 1
    assert summary["revised_candidate_count"] == 1
    assert summary["revised_deterministic_prefit_pass_rate"] == 1.0
    assert summary["selected_category_postreview_pass_rate"] == 1.0
    row = summary["rows"][0]
    assert row["changed_interaction_ids"] == ["term_0_0"]
    assert row["changed_initial_states"] == []
    assert len(calls) == 3


def test_topology_finding_routes_backward_without_function_edit(
    tmp_path: Path, monkeypatch
) -> None:
    import autoformalism.rebuttal.staged_prefit_feedback_campaign as campaign

    source_plan_path, source_results, source_plan = _write_source(tmp_path)
    config = _write_prefit_config(tmp_path, source_plan)
    freeze_prefit_campaign(
        config, source_plan_path, source_results, tmp_path / "plan.json"
    )
    calls = []

    def transport(url, body, timeout):
        del url, timeout
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        return _provider_response(_review("fail", fail_category="mechanism_topology"))

    monkeypatch.setattr(
        campaign,
        "StagedTopologyClient",
        lambda **kwargs: StagedTopologyClient(**kwargs, transport=transport),
    )
    summary = run_prefit_campaign(
        tmp_path / "plan.json", tmp_path / "results", "http://local"
    )
    assert summary["topology_backtrack_count"] == 1
    assert summary["revised_candidate_count"] == 0
    assert summary["physical_requests"] == 1
    assert len(calls) == 1


def test_latent_initializer_finding_changes_only_selected_state(
    tmp_path: Path, monkeypatch
) -> None:
    import autoformalism.rebuttal.staged_prefit_feedback_campaign as campaign

    source_plan_path, source_results, source_plan = _write_source(tmp_path)
    config = _write_prefit_config(tmp_path, source_plan)
    freeze_prefit_campaign(
        config, source_plan_path, source_results, tmp_path / "plan.json"
    )
    calls = []
    review_count = 0

    def transport(url, body, timeout):
        nonlocal review_count
        del url, timeout
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        if payload["schema_version"] == "prefit-scientific-review-request-1":
            review_count += 1
            return _provider_response(
                _review(
                    "fail" if review_count == 1 else "pass",
                    fail_category="latent_initialization",
                )
            )
        assert payload["schema_version"] == "prefit-initial-revision-request-1"
        return _provider_response({"initial": {"fixed_value": 1.0}})

    monkeypatch.setattr(
        campaign,
        "StagedTopologyClient",
        lambda **kwargs: StagedTopologyClient(**kwargs, transport=transport),
    )
    summary = run_prefit_campaign(
        tmp_path / "plan.json", tmp_path / "results", "http://local"
    )
    assert summary["initializer_revision_count"] == 1
    assert summary["function_revision_count"] == 0
    assert summary["revised_candidate_count"] == 1
    assert summary["rows"][0]["changed_initial_states"] == ["z"]
    assert summary["rows"][0]["changed_interaction_ids"] == []
    assert len(calls) == 3


def test_review_retry_receives_the_runtime_contract_diagnostic(
    tmp_path: Path, monkeypatch
) -> None:
    import autoformalism.rebuttal.staged_prefit_feedback_campaign as campaign

    source_plan_path, source_results, source_plan = _write_source(tmp_path)
    config = _write_prefit_config(tmp_path, source_plan)
    plan_path = tmp_path / "plan.json"
    freeze_prefit_campaign(config, source_plan_path, source_results, plan_path)
    review_count = 0
    retry_received_diagnostic = False

    def transport(url, body, timeout):
        nonlocal review_count, retry_received_diagnostic
        del url, timeout
        raw = body["messages"][1]["content"].split("\n", 1)[1]
        payload, _ = json.JSONDecoder().raw_decode(raw)
        if payload["schema_version"] == "prefit-scientific-review-request-1":
            review_count += 1
            if review_count == 1:
                invalid = _review("fail")
                invalid["findings"][3]["interaction_ids"] = ["invented_term"]
                return _provider_response(invalid)
            if review_count == 2:
                retry_received_diagnostic = "RUNTIME_CONTRACT_DIAGNOSTIC" in raw
                return _provider_response(_review("fail"))
            return _provider_response(_review("pass"))
        return _provider_response(
            {
                "expression": "new_gain * z - new_decay * x",
                "parameters": [
                    {"name": "new_gain", "role": "nonnegative_coefficient"},
                    {"name": "new_decay", "role": "nonnegative_coefficient"},
                ],
            }
        )

    monkeypatch.setattr(
        campaign,
        "StagedTopologyClient",
        lambda **kwargs: StagedTopologyClient(**kwargs, transport=transport),
    )
    summary = run_prefit_campaign(plan_path, tmp_path / "results", "http://local")
    assert summary["status"] == "complete"
    assert summary["physical_requests"] == 4
    assert retry_received_diagnostic


def test_review_schema_and_runtime_priority() -> None:
    review = PrefitScientificReviewReply.model_validate(_review("fail"))
    selected = select_prefit_finding(review)
    assert selected is not None
    assert selected.category.value == "functional_semantics"
    payload = _review("pass")
    payload["findings"].pop()
    with pytest.raises(ValueError, match="at least 6 items"):
        PrefitScientificReviewReply.model_validate(payload)
