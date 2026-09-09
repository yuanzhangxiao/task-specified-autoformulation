"""Public pre-function topology, source, mechanism, and polarity integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import StagedModelSettings, StagedTopologyClient
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    MechanismRequirement,
)
from autoformalism.rebuttal.staged_prefunction_campaign import (
    freeze_prefunction_campaign,
    run_prefunction_campaign,
)
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    PublicVariable,
    ScientificRequirement,
    ScientificVariable,
)
from autoformalism.search.staged_topology_runner import run_staged_topology
from autoformalism.staged_topology import (
    audit_equation_polarity_policy,
    compile_equation_polarity_policy,
)
from autoformalism.targets import PublicTargetContract, PublicTargetRequirement


def _response(payload: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(payload)}}
        ],
        "usage": {"total_tokens": 100},
    }


def _brief() -> PublicScientificBrief:
    return PublicScientificBrief(
        scientific_context="Input u activates x; q has unspecified direction.",
        public_variables=(
            PublicVariable(name="x", data_role="target"),
            PublicVariable(name="u", data_role="external_input"),
            PublicVariable(name="q", data_role="auxiliary"),
        ),
        requirements=(
            ScientificRequirement(
                id="drive",
                public_requirement="u activates x",
                targets=("x",),
                drivers=("u",),
                public_pathway_sign="positive",
            ),
            ScientificRequirement(
                id="context",
                public_requirement="q contributes with unspecified direction",
                targets=("x",),
                drivers=("q",),
                public_pathway_sign="unspecified",
            ),
        ),
    )


def test_public_polarity_policy_fixes_only_exact_supported_source_sets() -> None:
    policy = compile_equation_polarity_policy(_brief(), "x")
    assert policy.model_dump(mode="json") == {
        "schema_version": "equation-polarity-policy-1",
        "selected_lhs": "x",
        "default_outer_weight_sign": "unrestricted",
        "unfixed_source_ownership": "runtime_requires_unrestricted",
        "fixed_exact_source_sets": [
            {
                "sources": ["u"],
                "outer_weight_sign": "positive",
                "requirement_ids": ["drive"],
            }
        ],
    }
    equation = EquationDefinition(
        name="x",
        definition="differential",
        terms=(
            {
                "sources": ["u"],
                "outer_weight_sign": "positive",
                "scientific_role": "drive",
            },
            {
                "sources": ["q"],
                "outer_weight_sign": "unrestricted",
                "scientific_role": "context",
            },
            {
                "sources": ["x"],
                "outer_weight_sign": "unrestricted",
                "scientific_role": "uncommitted self response",
            },
        ),
    )
    audit = audit_equation_polarity_policy(equation, policy)
    assert audit["passed"]
    assert audit["fixed_evidence_term_count"] == 1
    assert audit["unrestricted_term_count"] == 2


def test_schema_valid_unsupported_fixed_sign_is_scored_without_retry(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def transport(
        url: str, body: dict[str, object], timeout: float
    ) -> dict[str, object]:
        del url, timeout
        request = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(request)
        assert request["interaction_polarity_policy"]["fixed_exact_source_sets"][0][
            "sources"
        ] == ["u"]
        return _response(
            {
                "terms": [
                    {
                        "sources": ["u"],
                        "outer_weight_sign": "positive",
                        "scientific_role": "drive",
                    },
                    {
                        "sources": ["q"],
                        "outer_weight_sign": "positive",
                        "scientific_role": "unsupported fixed context sign",
                    },
                ],
                "inventory_revision": None,
            }
        )

    client = StagedTopologyClient(
        settings=StagedModelSettings(attempts_per_step=3),
        base_url="http://localhost:8000",
        directory=tmp_path / "calls",
        namespace="prefunction",
        seed=0,
        transport=transport,
    )
    result = run_staged_topology(
        _brief(),
        ValidationContext(targets=("x",), auxiliaries=("q",), external_inputs=("u",)),
        client,
        tmp_path / "result",
        initial_inventory=(
            ScientificVariable(
                name="x", definition="differential", scientific_role="response"
            ),
            ScientificVariable(
                name="u", definition="supplied", scientific_role="input"
            ),
            ScientificVariable(
                name="q", definition="supplied", scientific_role="context"
            ),
        ),
        audit_public_polarity_policy=True,
    )
    assert result["complete_topology"]
    assert result["public_target_coverage_passed"]
    assert result["public_source_coverage_passed"]
    assert result["public_mechanism_coverage_passed"]
    assert not result["public_polarity_consistency_passed"]
    assert len(calls) == 1


def test_hybrid_variables_retain_valid_entries_skip_resolved_target_and_repair_memory(
    tmp_path: Path,
) -> None:
    requests: list[tuple[str, int, dict[str, object]]] = []

    def transport(
        url: str, body: dict[str, object], timeout: float
    ) -> dict[str, object]:
        del url, timeout
        step = str(body["response_format"]["json_schema"]["name"])
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        attempt = len([item for item in requests if item[0] == step])
        requests.append((step, attempt, payload))
        if step == "VariableReply" and attempt == 0:
            assert {item["name"] for item in payload["current_inventory"]} == {"u"}
            return _response(
                {
                    "variables": [
                        {
                            "name": "y",
                            "definition": "algebraic",
                            "scientific_role": "generated output",
                        },
                        {
                            "name": "u",
                            "definition": "differential",
                            "scientific_role": "invalid redefinition of supplied input",
                        },
                    ]
                }
            )
        if step == "VariableReply":
            assert {item["name"] for item in payload["current_inventory"]} == {
                "u",
                "y",
            }
            assert payload["runtime_diagnostics"]["unresolved_obligations"] == [
                "dynamic_memory_mediator:memory_path"
            ]
            return _response(
                {
                    "variables": [
                        {
                            "name": "z",
                            "definition": "differential",
                            "scientific_role": "input-driven memory",
                        }
                    ]
                }
            )
        selected = payload["selected_lhs"]["name"]
        policy = payload["interaction_polarity_policy"]
        assert policy["unfixed_source_ownership"] == "proposer"
        if selected == "z" and attempt == 0:
            return _response(
                {
                    "terms": [
                        {
                            "sources": ["z"],
                            "outer_weight_sign": "negative",
                            "scientific_role": "memory relaxation",
                        }
                    ],
                    "inventory_revision": None,
                }
            )
        if selected == "z":
            assert "dynamic-memory mediator" in payload["runtime_diagnostics"]["error"]
            return _response(
                {
                    "terms": [
                        {
                            "sources": ["z"],
                            "outer_weight_sign": "negative",
                            "scientific_role": "memory relaxation",
                        },
                        {
                            "sources": ["u"],
                            "outer_weight_sign": "positive",
                            "scientific_role": "input drive",
                        },
                    ],
                    "inventory_revision": None,
                }
            )
        assert selected == "y"
        return _response(
            {
                "terms": [
                    {
                        "sources": ["z"],
                        "outer_weight_sign": "positive",
                        "scientific_role": "memory readout",
                    }
                ],
                "inventory_revision": None,
            }
        )

    brief = PublicScientificBrief(
        scientific_context="Input u drives output y through dynamic memory.",
        public_variables=(
            PublicVariable(name="y", data_role="target"),
            PublicVariable(name="u", data_role="external_input"),
        ),
        requirements=(
            ScientificRequirement(
                id="memory_path",
                public_requirement="u reaches y through dynamic memory",
                targets=("y",),
                drivers=("u",),
                positive_requirements=("Represent dynamic memory.",),
                requires_dynamic_memory=True,
                public_pathway_sign="unspecified",
            ),
        ),
    )
    client = StagedTopologyClient(
        settings=StagedModelSettings(attempts_per_step=3),
        base_url="http://localhost:8000",
        directory=tmp_path / "calls",
        namespace="hybrid",
        seed=0,
        transport=transport,
    )
    result = run_staged_topology(
        brief,
        ValidationContext(targets=("y",), external_inputs=("u",)),
        client,
        tmp_path / "result",
        audit_public_polarity_policy=True,
        hybrid_variable_construction=True,
        proposer_owns_unfixed_signs=True,
    )
    assert result["complete_topology"]
    assert result["public_target_coverage_passed"]
    assert result["public_source_coverage_passed"]
    assert result["public_mechanism_coverage_passed"]
    assert result["public_polarity_consistency_passed"]
    assert result["memory_candidates"] == {"memory_path": ["z"]}
    assert len(requests) == 5
    variable_events = [
        event for event in result["events"] if event["step"].startswith("variables_")
    ]
    assert variable_events[0]["accepted_variable_names"] == ["y"]
    assert variable_events[0]["partial_acceptance"]
    assert variable_events[0]["rejected_variables"][0]["name"] == "u"
    assert variable_events[1]["accepted_variable_names"] == ["z"]
    assert variable_events[2]["skipped_as_resolved"]
    equation_z = [event for event in result["events"] if event["step"] == "equation_z"]
    assert [event["accepted"] for event in equation_z] == [False, True]


def _freeze_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repository"
    public_root = tmp_path / "public"
    target_root = repository / "configs/target_eval/phase_b_v2/specs"
    mechanism_root = repository / "configs/mechanism_eval/phase_b_v1/specs"
    target_root.mkdir(parents=True)
    mechanism_root.mkdir(parents=True)
    cells = ("cell_a", "cell_b")
    for index, cell in enumerate(cells):
        prompt = "A. Public task\nE. Evidence\nF. Required response\nReturn JSON.\n"
        prompt_path = public_root / "phase_b_v1" / cell / "proposer_prompt.txt"
        prompt_path.parent.mkdir(parents=True)
        prompt_path.write_text(prompt)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        (target_root / f"{cell}.json").write_text(
            PublicTargetContract(
                benchmark_id=cell,
                tier="easy",
                public_prompt_sha256=digest,
                targets=(
                    PublicTargetRequirement(
                        target_channel="x", public_requirement="generate x"
                    ),
                ),
            ).model_dump_json()
        )
        (mechanism_root / f"{cell}.json").write_text(
            MechanismEvaluationSpec(
                source="public_prompt",
                benchmark_id=cell,
                tier="easy",
                public_prompt_sha256=digest,
                required_mechanisms=(
                    MechanismRequirement(
                        id="drive",
                        public_requirement="input u drives x",
                        required_drivers=("u",),
                        required_targets=("x",),
                        required_sign="positive" if index == 0 else "unspecified",
                    ),
                ),
            ).model_dump_json()
        )
    config = {
        "protocol": "scientific-staged-prefunction-integration-1",
        "purpose": "test",
        "platform": "aces-h100x1",
        "serving_image_sha256": "a" * 64,
        "model_settings": {
            "model": "openai/gpt-oss-20b",
            "model_revision": "revision",
            "reasoning_effort": "low",
            "temperature": 0.2,
            "max_output_tokens": 1024,
            "timeout_seconds": 1,
            "maximum_requests": 4,
            "maximum_total_tokens": 65536,
            "attempts_per_step": 2,
        },
        "served_context_tokens": 16384,
        "public_cells": list(cells),
        "seeds": [0, 1, 2],
        "limits": {
            "generated_variables": 2,
            "terms_per_equation": 2,
            "total_terms": 2,
        },
        "gates": {
            "minimum_topology_completion": 1.0,
            "minimum_target_coverage": 1.0,
            "minimum_source_coverage": 1.0,
            "minimum_mechanism_coverage": 1.0,
            "minimum_polarity_consistency": 1.0,
        },
        "wall_seconds": 60,
        "shutdown_margin_seconds": 30,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    import autoformalism.rebuttal.staged_prefunction_campaign as campaign

    monkeypatch.setattr(
        campaign,
        "public_validation_context",
        lambda cell: ValidationContext(targets=("x",), external_inputs=("u",)),
    )
    return config_path, public_root, repository


def test_six_task_freeze_and_terminal_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, public_root, repository = _freeze_fixture(tmp_path, monkeypatch)
    plan_path = tmp_path / "plan.json"
    plan = freeze_prefunction_campaign(config, public_root, repository, plan_path)
    assert len(plan["tasks"]) == 6
    assert not plan["function_generation_performed"]
    assert not plan["parameter_fitting_performed"]
    first = plan["tasks"][0]["source"]
    fourth = plan["tasks"][3]["source"]
    assert (
        first["level_zero_polarity_contract"][0][
            "direct_exact_source_outer_weight_sign"
        ]
        == "positive"
    )
    assert (
        fourth["level_zero_polarity_contract"][0][
            "direct_exact_source_outer_weight_sign"
        ]
        == "unrestricted"
    )
    calls: list[str] = []

    def task_result(task, config, base_url, root, identity, can_start):
        del config, base_url, root, identity
        assert can_start()
        calls.append(task["task_id"])
        return {
            "status": "complete",
            "error": None,
            "complete_topology": True,
            "public_target_coverage_passed": True,
            "public_source_coverage_passed": True,
            "public_mechanism_coverage_passed": True,
            "public_polarity_consistency_passed": True,
            "polarity_policy_audits": [
                {
                    "term_count": 1,
                    "fixed_evidence_term_count": 0,
                    "unrestricted_term_count": 1,
                    "correct_term_count": 1,
                }
            ],
            "physical_requests": 1,
            "observed_total_tokens": 10,
            "provider_seconds": 0.1,
        }

    import autoformalism.rebuttal.staged_prefunction_campaign as campaign

    monkeypatch.setattr(campaign, "_task_result", task_result)
    output = tmp_path / "results"
    summary = run_prefunction_campaign(plan_path, output, "http://localhost:8000")
    assert summary["status"] == "complete"
    assert summary["overall_result"] == "pass"
    assert summary["topology_completion_rate"] == 1.0
    assert len(calls) == 6
    assert (
        run_prefunction_campaign(plan_path, output, "http://localhost:8000") == summary
    )
    assert len(calls) == 6


def test_hybrid_freeze_reports_conditional_gates_and_repair_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, public_root, repository = _freeze_fixture(tmp_path, monkeypatch)
    payload = json.loads(config.read_text())
    payload["protocol"] = "scientific-staged-prefunction-hybrid-2"
    payload["gates"] = {
        "minimum_topology_completion": 1.0,
        "minimum_conditional_target_coverage": 1.0,
        "minimum_conditional_source_coverage": 1.0,
        "minimum_conditional_topology_obligation_coverage": 1.0,
    }
    config.write_text(json.dumps(payload))
    plan_path = tmp_path / "hybrid-plan.json"
    plan = freeze_prefunction_campaign(config, public_root, repository, plan_path)
    assert plan["config"]["protocol"] == "scientific-staged-prefunction-hybrid-2"

    def task_result(task, config, base_url, root, identity, can_start):
        del task, config, base_url, root, identity
        assert can_start()
        return {
            "status": "complete",
            "error": None,
            "complete_topology": True,
            "public_target_coverage_passed": True,
            "public_source_coverage_passed": True,
            "public_mechanism_coverage_passed": True,
            "public_polarity_consistency_passed": True,
            "polarity_policy_audits": [
                {
                    "rows": [
                        {
                            "observed_outer_weight_sign": "negative",
                            "public_fixed_evidence": False,
                            "correct": True,
                        }
                    ]
                }
            ],
            "events": [
                {
                    "step": "variables_0",
                    "accepted": False,
                    "partial_acceptance": True,
                    "accepted_variable_names": ["x"],
                    "rejected_variables": [{"name": "u"}],
                },
                {
                    "step": "variables_1",
                    "accepted": True,
                    "skipped_as_resolved": True,
                },
                {
                    "step": "equation_z",
                    "accepted": False,
                    "error": "dynamic-memory mediator missing driver",
                },
                {"step": "equation_z", "accepted": True},
            ],
            "physical_requests": 3,
            "observed_total_tokens": 30,
            "provider_seconds": 0.3,
        }

    import autoformalism.rebuttal.staged_prefunction_campaign as campaign

    monkeypatch.setattr(campaign, "_task_result", task_result)
    summary = run_prefunction_campaign(
        plan_path, tmp_path / "hybrid-results", "http://localhost:8000"
    )
    assert summary["overall_result"] == "pass"
    assert summary["conditional_topology_obligation_coverage_rate"] == 1.0
    assert summary["explicit_sign_evidence_term_count"] == 0
    assert summary["proposer_owned_sign_counts"]["negative"] == 6
    assert summary["partial_variable_acceptance_count"] == 6
    assert summary["resolved_variable_agenda_skip_count"] == 6
    assert summary["memory_equation_repair_activation_count"] == 6
    assert summary["memory_equation_repair_recovered_step_count"] == 6
