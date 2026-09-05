"""Public-only end-to-end tests for incremental candidate construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.exceptions import LLMProviderError
from autoformalism.llm.models import LLMCallResult
from autoformalism.rebuttal.incremental_construction_pilot import (
    IncrementalConstructionPilotConfig,
    build_public_construction_problem,
    construct_public_candidates,
)
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.schemas import (
    ProposedConstructionFocus,
    ProposedConstructionIntent,
    ProposedFunctionalActionTransaction,
    ProposedTopologyActionTransaction,
    TopologyConstructionPhase,
)
from autoformalism.search.incremental_proposer import IncrementalProposerConfig
from autoformalism.targets import PublicTargetContract


def _result(request_hash: str, parsed) -> LLMCallResult:
    return LLMCallResult(
        request_hash=request_hash,
        parsed=parsed,
        raw_response={},
        cache_hit=False,
        attempts=1,
        latency_ms=1.0,
        usage=None,
    )


class _IncrementalPilotClient:
    """Return partial topology edits and one locally invalid function edit."""

    def __init__(self) -> None:
        self.intent_calls = 0
        self.topology_calls = 0
        self.functional_calls = 0

    def propose_construction_intent(self, **_kwargs) -> LLMCallResult:
        self.intent_calls += 1
        return _result(
            f"{self.intent_calls:064x}",
            ProposedConstructionIntent(
                objective="initial_construction",
                requirement_ids=("input_response",),
                target_channels=("target",),
            ),
        )

    def propose_topology_actions(self, **kwargs) -> LLMCallResult:
        self.topology_calls += 1
        parent = kwargs["parent"]
        if not parent.states:
            payload = {"actions": [{"action": "add_state", "name": "x"}]}
        else:
            payload = {
                "actions": [
                    {
                        "action": "add_interaction",
                        "interaction_id": "input_drive",
                        "target": "x",
                        "sources": ["input_u"],
                    },
                    {
                        "action": "set_target_mapping",
                        "channel": "target",
                        "source": "x",
                    },
                ]
            }
        return _result(
            f"{100 + self.topology_calls:064x}",
            ProposedTopologyActionTransaction.model_validate(payload),
        )

    def propose_functional_actions(self, **_kwargs) -> LLMCallResult:
        self.functional_calls += 1
        interaction_id = (
            "not_in_topology" if self.functional_calls == 1 else "input_drive"
        )
        return _result(
            f"{200 + self.functional_calls:064x}",
            ProposedFunctionalActionTransaction.model_validate(
                {
                    "actions": [
                        {
                            "action": "set_interaction_function",
                            "interaction_id": interaction_id,
                            "expression": "input_u",
                        }
                    ]
                }
            ),
        )


class _PhasedPilotClient:
    """Build one candidate in the runtime-required topology phase order."""

    def __init__(self) -> None:
        self.focus_calls = 0

    def propose_construction_focus(self, **kwargs) -> LLMCallResult:
        self.focus_calls += 1
        return _result(
            f"{300 + self.focus_calls:064x}",
            ProposedConstructionFocus(
                feedback_item_indices=(
                    (0,) if kwargs["allowed_feedback_item_indices"] else ()
                ),
                requirement_ids=kwargs["allowed_requirement_ids"],
                target_channels=("target",),
            ),
        )

    def propose_topology_actions(self, **kwargs) -> LLMCallResult:
        phase = kwargs["topology_phase"]
        if phase == TopologyConstructionPhase.COMPONENT_SPECIFICATION:
            payload = {"actions": [{"action": "add_state", "name": "x"}]}
        elif phase == TopologyConstructionPhase.DYNAMIC_TOPOLOGY:
            payload = {
                "actions": [
                    {
                        "action": "add_interaction",
                        "interaction_id": "input_drive",
                        "target": "x",
                        "sources": ["input_u"],
                    }
                ]
            }
        else:
            payload = {
                "actions": [
                    {
                        "action": "set_target_mapping",
                        "channel": "target",
                        "source": "x",
                    }
                ]
            }
        return _result(
            f"{400 + self.focus_calls:064x}",
            ProposedTopologyActionTransaction.model_validate(payload),
        )

    def propose_functional_actions(self, **_kwargs) -> LLMCallResult:
        return _result(
            f"{500 + self.focus_calls:064x}",
            ProposedFunctionalActionTransaction.model_validate(
                {
                    "actions": [
                        {
                            "action": "set_interaction_function",
                            "interaction_id": "input_drive",
                            "expression": "input_u",
                        }
                    ]
                }
            ),
        )


class _DeadProviderClient:
    """Represent a local model server that died after its internal retries."""

    def propose_construction_intent(self, **_kwargs) -> LLMCallResult:
        raise LLMProviderError(
            "vLLM connection failed: [Errno 111] Connection refused",
            retryable=True,
        )


def _target_contract() -> PublicTargetContract:
    return PublicTargetContract.model_validate(
        {
            "benchmark_id": "synthetic",
            "tier": "easy",
            "public_prompt_sha256": "0" * 64,
            "targets": [
                {
                    "target_channel": "target",
                    "public_requirement": "Generate the target response.",
                }
            ],
        }
    )


def _mechanism_spec() -> MechanismEvaluationSpec:
    return MechanismEvaluationSpec.model_validate(
        {
            "benchmark_id": "synthetic",
            "tier": "easy",
            "required_mechanisms": [
                {
                    "id": "input_response",
                    "required_drivers": ["input_u"],
                    "required_targets": ["target"],
                }
            ],
        }
    )


def test_phased_public_problem_excludes_conflicting_response_instructions() -> None:
    public_problem = build_public_construction_problem(
        public_prompt=(
            "A. Task specification\nInfer a causal model.\n\n"
            "F. Required response\nReturn one complete model with prose.\n"
        ),
        context=ValidationContext(
            targets=("target",), external_inputs=("input_u",)
        ),
        target_contract=_target_contract(),
        mechanism_spec=_mechanism_spec(),
        construction_protocol="phased_runtime_agenda_v2",
    )
    payload = json.loads(public_problem)

    assert payload["schema_version"] == "public-construction-problem-2"
    assert payload["benchmark_response_instructions_removed"] is True
    assert "Infer a causal model" in payload["benchmark_scientific_context"]
    assert "Return one complete model" not in public_problem
    assert "benchmark_prompt" not in payload


def test_dead_local_provider_stops_after_one_recorded_attempt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "construction_result.json"
    with pytest.raises(LLMProviderError, match="Connection refused"):
        construct_public_candidates(
            client=_DeadProviderClient(),
            proposer_config=IncrementalProposerConfig(
                checkpoint_directory=tmp_path / "checkpoints",
                run_fingerprint="c" * 64,
                intent_system_prompt="Select one objective.",
                topology_action_system_prompt="Return topology actions.",
                functional_action_system_prompt="Return functions.",
            ),
            pilot_config=IncrementalConstructionPilotConfig(
                topology_branch_count=2,
                function_children_per_topology=1,
                maximum_topology_action_steps=4,
                maximum_functional_action_steps=1,
            ),
            public_problem="Infer a causal input-response model.",
            context=ValidationContext(
                targets=("target",), external_inputs=("input_u",)
            ),
            target_contract=_target_contract(),
            mechanism_spec=_mechanism_spec(),
            output_path=output,
        )

    partial = json.loads(output.read_text(encoding="utf-8"))
    assert partial["status"] == "incomplete"
    assert len(partial["attempts"]) == 1
    assert partial["attempts"][0]["failure_class"] == "provider_transport"


def test_incremental_pilot_preserves_partial_edits_and_localizes_failure(
    tmp_path: Path,
) -> None:
    client = _IncrementalPilotClient()
    output = tmp_path / "result.json"

    result = construct_public_candidates(
        client=client,
        proposer_config=IncrementalProposerConfig(
            checkpoint_directory=tmp_path / "checkpoints",
            run_fingerprint="a" * 64,
            intent_system_prompt="Select one public scientific objective.",
            topology_action_system_prompt="Return one topology edit.",
            functional_action_system_prompt="Return one function edit.",
        ),
        pilot_config=IncrementalConstructionPilotConfig(
            topology_branch_count=1,
            function_children_per_topology=1,
            maximum_topology_action_steps=2,
            maximum_functional_action_steps=2,
        ),
        public_problem="Infer a causal input-response model.",
        context=ValidationContext(targets=("target",), external_inputs=("input_u",)),
        target_contract=_target_contract(),
        mechanism_spec=_mechanism_spec(),
        output_path=output,
    )

    assert result.status == "complete"
    assert result.complete_topology_count == 1
    assert result.complete_candidate_count == 1
    assert [item.status for item in result.attempts] == [
        "applied_incomplete",
        "complete",
        "rejected",
        "complete",
    ]
    first = result.attempts[0]
    assert first.failure_class == "deterministic_topology_completeness"
    assert first.result is not None
    rejected = result.attempts[2]
    assert rejected.failure_class == "deterministic_action_contract"
    assert "not_in_topology" in (rejected.error or "")
    candidate = result.candidates[0]
    assert candidate.public_target_evaluation.passed is True
    assert candidate.public_mechanism_evaluation.graph_mechanism_compliance == 1.0
    assert output.exists()


def test_phased_pilot_orders_topology_work_and_uses_runtime_agenda(
    tmp_path: Path,
) -> None:
    client = _PhasedPilotClient()
    result = construct_public_candidates(
        client=client,
        proposer_config=IncrementalProposerConfig(
            checkpoint_directory=tmp_path / "checkpoints",
            run_fingerprint="b" * 64,
            intent_system_prompt="Select anchors inside the runtime agenda.",
            topology_action_system_prompt="Return phase-compatible topology edits.",
            functional_action_system_prompt="Return one function edit.",
            decision_policy="runtime_priority_v2",
        ),
        pilot_config=IncrementalConstructionPilotConfig(
            topology_branch_count=1,
            function_children_per_topology=1,
            maximum_topology_action_steps=3,
            maximum_functional_action_steps=1,
            construction_protocol="phased_runtime_agenda_v2",
        ),
        public_problem="Infer a causal input-response model.",
        context=ValidationContext(targets=("target",), external_inputs=("input_u",)),
        target_contract=_target_contract(),
        mechanism_spec=_mechanism_spec(),
    )

    assert result.status == "complete"
    assert result.construction_protocol == "phased_runtime_agenda_v2"
    assert [item.topology_phase for item in result.attempts[:3]] == [
        TopologyConstructionPhase.COMPONENT_SPECIFICATION,
        TopologyConstructionPhase.DYNAMIC_TOPOLOGY,
        TopologyConstructionPhase.ALGEBRAIC_READOUT_TOPOLOGY,
    ]
    assert all(item.failure_class is None for item in result.attempts[:3])
    topology = result.candidates[0].topology_draft
    assert all(not item.mechanisms for item in topology.states)
    assert all(not item.mechanisms for item in topology.interactions)
    assert client.focus_calls == 4
