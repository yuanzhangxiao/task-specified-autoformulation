"""Checkpointed intent and action proposer orchestration tests."""

from __future__ import annotations

import json
from pathlib import Path

from autoformalism.construction import finalize_topology_draft
from autoformalism.expressions import ValidationContext
from autoformalism.llm import LLMCallResult
from autoformalism.schemas import (
    FunctionalDraft,
    ProposedConstructionIntent,
    ProposedFunctionalActionTransaction,
    ProposedTopologyActionTransaction,
    TopologyDraft,
)
from autoformalism.search import CandidateFeedbackEvidence, route_proposer_feedback
from autoformalism.search.incremental_proposer import (
    IncrementalProposer,
    IncrementalProposerConfig,
)
from autoformalism.staging import topology_commitment_sha256


def _context() -> ValidationContext:
    return ValidationContext(targets=("target",), external_inputs=("input_u",))


def _feedback():
    return route_proposer_feedback(
        CandidateFeedbackEvidence(
            graph_mechanism_failures=("Input response is not connected.",),
            fit_failures=("The target response is too slow.",),
        )
    )


def _result(request_hash: str, parsed) -> LLMCallResult:
    return LLMCallResult(
        request_hash=request_hash,
        parsed=parsed,
        raw_response={},
        cache_hit=False,
        attempts=1,
        latency_ms=5.0,
        usage=None,
    )


class _FakeIncrementalClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def propose_construction_intent(self, **kwargs) -> LLMCallResult:
        self.calls.append(("intent", kwargs["user_prompt"]))
        parsed = ProposedConstructionIntent(
            objective="initial_construction",
            requirement_ids=("input_response",),
            target_channels=("target",),
        )
        return _result("1" * 64, parsed)

    def propose_topology_actions(self, **kwargs) -> LLMCallResult:
        self.calls.append(("topology", kwargs["user_prompt"]))
        parsed = ProposedTopologyActionTransaction.model_validate(
            {
                "actions": [
                    {"action": "add_state", "name": "x"},
                    {"action": "add_state", "name": "z"},
                    {
                        "action": "add_interaction",
                        "interaction_id": "z_drive",
                        "target": "z",
                        "sources": ["input_u"],
                    },
                    {
                        "action": "add_interaction",
                        "interaction_id": "z_decay",
                        "target": "z",
                        "sources": ["z"],
                        "polarity": "subtractive",
                    },
                    {
                        "action": "add_interaction",
                        "interaction_id": "x_drive",
                        "target": "x",
                        "sources": ["z"],
                    },
                    {
                        "action": "set_target_mapping",
                        "channel": "target",
                        "source": "x",
                    },
                ]
            }
        )
        return _result("2" * 64, parsed)

    def propose_functional_actions(self, **kwargs) -> LLMCallResult:
        self.calls.append(("functional", kwargs["user_prompt"]))
        parsed = ProposedFunctionalActionTransaction.model_validate(
            {
                "actions": [
                    {
                        "action": "set_interaction_function",
                        "interaction_id": "z_drive",
                        "expression": "input_u",
                    },
                    {
                        "action": "set_interaction_function",
                        "interaction_id": "z_decay",
                        "expression": "k_z * z",
                        "parameters": [{"name": "k_z", "role": "rate"}],
                    },
                    {
                        "action": "set_interaction_function",
                        "interaction_id": "x_drive",
                        "expression": "gain * z",
                        "parameters": [{"name": "gain", "role": "scale"}],
                    },
                    {
                        "action": "set_latent_initial",
                        "state": "z",
                        "initial": {"fixed_value": 0.0},
                    },
                ]
            }
        )
        return _result("3" * 64, parsed)


def _config(path: Path) -> IncrementalProposerConfig:
    return IncrementalProposerConfig(
        checkpoint_directory=path,
        run_fingerprint="a" * 64,
        intent_system_prompt="Choose a scientific focus.",
        topology_action_system_prompt="Return topology actions.",
        functional_action_system_prompt="Return functional actions.",
    )


def test_incremental_proposer_checkpoints_decision_action_and_application(
    tmp_path: Path,
) -> None:
    client = _FakeIncrementalClient()
    proposer = IncrementalProposer(client=client, config=_config(tmp_path))
    arguments = {
        "public_problem": "Infer a driven response model from public data.",
        "context": _context(),
        "allowed_requirement_ids": ("input_response",),
        "feedback": _feedback(),
        "parent": TopologyDraft(),
    }

    first = proposer.revise_topology(**arguments)
    resumed = proposer.revise_topology(**arguments)

    assert [role for role, _ in client.calls] == ["intent", "topology"]
    assert first.application == resumed.application
    assert first.transposition_new is True
    assert resumed.transposition_new is True
    assert resumed.intent_call.checkpoint_hit is True
    assert resumed.action_call.checkpoint_hit is True
    assert (tmp_path / "transpositions.json").exists()
    checkpoint = next(tmp_path.glob("topology-action-*.json"))
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["result"]["transaction"] == first.transaction.model_dump(
        mode="json"
    )
    assert payload["result"]["application"] == first.application.model_dump(
        mode="json"
    )


def test_function_actions_are_conditioned_on_exact_topology_and_resume(
    tmp_path: Path,
) -> None:
    client = _FakeIncrementalClient()
    proposer = IncrementalProposer(client=client, config=_config(tmp_path))
    topology_result = proposer.revise_topology(
        public_problem="Infer a driven response model from public data.",
        context=_context(),
        allowed_requirement_ids=("input_response",),
        feedback=_feedback(),
        parent=TopologyDraft(),
    )
    topology = finalize_topology_draft(
        topology_result.application.draft, _context()
    )
    parent = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    arguments = {
        "public_problem": "Infer a driven response model from public data.",
        "context": _context(),
        "allowed_requirement_ids": ("input_response",),
        "feedback": _feedback(),
        "topology": topology,
        "parent": parent,
    }

    first = proposer.revise_functions(**arguments)
    resumed = proposer.revise_functions(**arguments)

    assert [role for role, _ in client.calls] == [
        "intent",
        "topology",
        "intent",
        "functional",
    ]
    assert first.compatibility.status == "compatible"
    assert first.expansion is not None
    assert first.application == resumed.application
    assert resumed.intent_call.checkpoint_hit is True
    assert resumed.action_call.checkpoint_hit is True
    functional_prompt = json.loads(client.calls[-1][1])
    assert functional_prompt["topology_commitment_sha256"] == (
        topology_commitment_sha256(topology)
    )
    assert "test" not in functional_prompt
    assert "private" not in functional_prompt


def test_incompatible_function_retry_cannot_mutate_parent(tmp_path: Path) -> None:
    client = _FakeIncrementalClient()
    proposer = IncrementalProposer(client=client, config=_config(tmp_path))
    topology_result = proposer.revise_topology(
        public_problem="Infer a driven response model from public data.",
        context=_context(),
        allowed_requirement_ids=("input_response",),
        feedback=_feedback(),
        parent=TopologyDraft(),
    )
    topology = finalize_topology_draft(
        topology_result.application.draft, _context()
    )
    parent = FunctionalDraft(
        topology_commitment_sha256=topology_commitment_sha256(topology)
    )
    before = parent.model_dump_json()

    proposer.revise_functions(
        public_problem="Infer a driven response model from public data.",
        context=_context(),
        allowed_requirement_ids=("input_response",),
        feedback=_feedback(),
        topology=topology,
        parent=parent,
    )

    assert parent.model_dump_json() == before
