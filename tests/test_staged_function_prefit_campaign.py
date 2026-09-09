"""Six-topology handoff, deterministic pre-fit audit, and resume tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import StagedTopologyClient
from autoformalism.rebuttal.staged_function_campaign import function_diagnostic
from autoformalism.rebuttal.staged_function_prefit_campaign import (
    deterministic_prefit_audit,
    freeze_function_prefit_campaign,
    run_function_prefit_campaign,
    summarize_function_prefit,
)
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    ModelingLimits,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.staged_topology import content_hash, lower_topology


def _response(payload: dict) -> dict:
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(payload)}}
        ],
        "usage": {"total_tokens": 100},
    }


def _transport(calls: list[dict]):
    def transport(url, body, timeout):
        del url, timeout
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        if "selected_state" in payload:
            return _response({"initial": {"fixed_value": 0.0}})
        return _response(
            {
                "functions": [
                    {
                        "expression": "-".join(term["sources"]) or "1",
                        "parameters": [],
                    }
                    for term in payload["selected_equation"]["terms"]
                ]
            }
        )

    return transport


def _write_source(root: Path) -> tuple[Path, Path]:
    config = json.loads(
        Path("configs/staged_function_prefit_handoff_v1.json").read_text()
    )
    source_config = json.loads(
        Path("configs/staged_prefunction_hybrid_v2.json").read_text()
    )
    fixture = function_diagnostic("driven_memory", ModelingLimits())
    source = fixture["source"]
    for equation in source["equations"]:
        for term in equation["terms"]:
            term["outer_weight_sign"] = (
                "positive" if term.pop("outer_sign") == "add" else "negative"
            )
    brief = PublicScientificBrief.model_validate(fixture["brief"])
    context = ValidationContext.model_validate(fixture["context"])
    topology, aliases = lower_topology(
        brief,
        tuple(ScientificVariable.model_validate(item) for item in source["inventory"]),
        tuple(EquationDefinition.model_validate(item) for item in source["equations"]),
        context,
    )
    source.update(
        status="complete",
        complete_topology=True,
        public_structure_checks_passed=True,
        public_target_coverage_passed=True,
        public_source_coverage_passed=True,
        public_mechanism_coverage_passed=True,
        public_polarity_consistency_passed=True,
        topology=topology.model_dump(mode="json"),
        generated_auxiliary_aliases=aliases,
        test_data_opened=False,
        private_reference_opened=False,
        function_generation_performed=False,
        parameter_fitting_performed=False,
    )
    tasks = []
    for cell in config["public_cells"]:
        for seed in config["seeds"]:
            tasks.append(
                {
                    "task_id": f"{cell}_seed{seed}",
                    "benchmark_id": cell,
                    "seed": seed,
                    "source": {
                        "brief": fixture["brief"],
                        "context": fixture["context"],
                    },
                }
            )
    source_plan = {
        "config": source_config,
        "tasks": tasks,
        "parameter_fitting_performed": False,
        "function_generation_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    source_plan["plan_sha256"] = content_hash(source_plan)
    source_root = root / "source"
    (source_root / "results").mkdir(parents=True)
    (source_root / "plan.json").write_text(json.dumps(source_plan))
    summary = {
        "plan_sha256": source_plan["plan_sha256"],
        "status": "complete",
        "overall_result": "pass",
        "terminal_results": 6,
        "checks": {
            "minimum_topology_completion": True,
            "minimum_conditional_target_coverage": True,
            "minimum_conditional_source_coverage": True,
            "minimum_conditional_topology_obligation_coverage": True,
        },
    }
    (source_root / "results" / "summary.json").write_text(json.dumps(summary))
    for task in tasks:
        task_root = source_root / "results" / task["task_id"]
        task_root.mkdir()
        (task_root / "result.json").write_text(json.dumps(source))
        terminal = {
            "identity": content_hash([source_plan["plan_sha256"], task]),
            "task_id": task["task_id"],
            "result": source,
        }
        (task_root / "terminal.json").write_text(json.dumps(terminal))
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, source_root


def test_freeze_run_audit_and_resume(tmp_path: Path, monkeypatch) -> None:
    import autoformalism.rebuttal.staged_function_prefit_campaign as campaign

    config_path, source_root = _write_source(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = freeze_function_prefit_campaign(config_path, source_root, plan_path)
    assert len(plan["tasks"]) == 6
    assert not plan["source_topologies_regenerated"]
    calls: list[dict] = []
    monkeypatch.setattr(
        campaign,
        "StagedTopologyClient",
        lambda **kwargs: StagedTopologyClient(
            **kwargs,
            transport=_transport(calls),
        ),
    )
    results = tmp_path / "results"
    summary = run_function_prefit_campaign(
        plan_path, results, "http://localhost:8000"
    )
    assert summary["overall_result"] == "pass", summary
    assert summary["complete_model_rate"] == 1.0
    assert summary["deterministic_prefit_pass_rate"] == 1.0
    assert summary["latent_initializer_coverage_rate"] == 1.0
    assert summary["source_topologies_regenerated"] is False
    assert len(calls) == 18
    assert (
        run_function_prefit_campaign(plan_path, results, "http://localhost:8000")
        == summary
    )
    assert len(calls) == 18
    partial = summarize_function_prefit(plan, [])
    assert partial["status"] == "incomplete"
    assert partial["terminal_results"] == 0

    task = plan["tasks"][0]
    result = json.loads(
        (results / task["task_id"] / "result.json").read_text()
    )
    audit = deterministic_prefit_audit(
        PublicScientificBrief.model_validate(task["brief"]),
        task["source"],
        result,
    )
    assert audit["passed"]
    result["accepted_functions"][0]["selected_term"]["outer_weight_sign"] = (
        "unrestricted"
    )
    assert not deterministic_prefit_audit(
        PublicScientificBrief.model_validate(task["brief"]), task["source"], result
    )["passed"]


def test_freeze_rejects_nonpassing_source_or_mutated_terminal(tmp_path: Path) -> None:
    config_path, source_root = _write_source(tmp_path)
    summary_path = source_root / "results" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["overall_result"] = "fail"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="did not pass"):
        freeze_function_prefit_campaign(
            config_path, source_root, tmp_path / "plan.json"
        )

    _, source_root = _write_source(tmp_path / "second")
    terminal_path = next((source_root / "results").glob("*/terminal.json"))
    terminal = json.loads(terminal_path.read_text())
    terminal["identity"] = "0" * 64
    terminal_path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="source terminal mismatch"):
        freeze_function_prefit_campaign(
            tmp_path / "second" / "config.json",
            source_root,
            tmp_path / "second" / "plan.json",
        )
