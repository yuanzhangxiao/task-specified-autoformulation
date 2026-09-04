"""Checkpointed staged proposer orchestration tests."""

from __future__ import annotations

import json
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.llm import LLMCallResult
from autoformalism.schemas import (
    ProposedFunctionalCandidate,
    ProposedTopologyCandidate,
)
from autoformalism.search import (
    CandidateFeedbackEvidence,
    route_proposer_feedback,
)
from autoformalism.search.staged_proposer import StagedProposer, StagedProposerConfig
from autoformalism.staging import (
    enrich_functional_proposal,
    enrich_topology_proposal,
)


def _context() -> ValidationContext:
    return ValidationContext(targets=("target",), external_inputs=("input_u",))


def _topology_proposal() -> ProposedTopologyCandidate:
    return ProposedTopologyCandidate.model_validate(
        {
            "candidate_id": "graph_0",
            "change_summary": "One latent response drives the output.",
            "states": [{"name": "x"}, {"name": "z"}],
            "interactions": [
                {
                    "interaction_id": "z_drive",
                    "target": "z",
                    "target_kind": "state_derivative",
                    "sources": ["input_u"],
                },
                {
                    "interaction_id": "z_decay",
                    "target": "z",
                    "target_kind": "state_derivative",
                    "sources": ["z"],
                    "polarity": "subtractive",
                },
                {
                    "interaction_id": "x_drive",
                    "target": "x",
                    "target_kind": "state_derivative",
                    "sources": ["z"],
                },
                {
                    "interaction_id": "x_decay",
                    "target": "x",
                    "target_kind": "state_derivative",
                    "sources": ["x"],
                    "polarity": "subtractive",
                },
            ],
            "target_mappings": [{"channel": "target", "source": "x"}],
        }
    )


def _functional_proposal() -> ProposedFunctionalCandidate:
    return ProposedFunctionalCandidate.model_validate(
        {
            "candidate_id": "functions_0",
            "change_summary": "Assign stable linear dynamics.",
            "interaction_functions": [
                {"interaction_id": "z_drive", "expression": "input_u"},
                {"interaction_id": "z_decay", "expression": "k_z * z"},
                {"interaction_id": "x_drive", "expression": "gain * z"},
                {"interaction_id": "x_decay", "expression": "k_x * x"},
            ],
            "parameters": [
                {"name": "k_z", "role": "rate"},
                {"name": "gain", "role": "scale"},
                {"name": "k_x", "role": "rate"},
            ],
            "latent_initials": [
                {"state": "z", "initial": {"fixed_value": 0.0}}
            ],
        }
    )


class _FakeStagedClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def propose_topology(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        context: ValidationContext,
        cache_only: bool = False,
    ) -> LLMCallResult:
        del system_prompt, cache_only
        self.calls.append(("topology", user_prompt))
        topology = enrich_topology_proposal(_topology_proposal(), context)
        return _result("1" * 64, topology)

    def propose_functions(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        topology,
        context: ValidationContext,
        cache_only: bool = False,
    ) -> LLMCallResult:
        del system_prompt, context, cache_only
        self.calls.append(("functional", user_prompt))
        functional = enrich_functional_proposal(_functional_proposal(), topology)
        return _result("2" * 64, functional)


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


def _config(path: Path) -> StagedProposerConfig:
    return StagedProposerConfig(
        checkpoint_directory=path,
        run_fingerprint="a" * 64,
        topology_system_prompt="Return a topology.",
        functional_system_prompt="Return interaction functions.",
    )


def _feedback():
    return route_proposer_feedback(
        CandidateFeedbackEvidence(
            target_contract_failures=("Connect the delayed mechanism to target.",),
            fit_failures=("The current expression integrates poorly.",),
        )
    )


def test_staged_proposer_routes_prompts_and_resumes_checkpoints(
    tmp_path: Path,
) -> None:
    client = _FakeStagedClient()
    proposer = StagedProposer(client=client, config=_config(tmp_path))

    first = proposer.construct(
        public_problem="Infer a driven response model.",
        context=_context(),
        feedback=_feedback(),
    )
    second = proposer.construct(
        public_problem="Infer a driven response model.",
        context=_context(),
        feedback=_feedback(),
    )

    assert [role for role, _ in client.calls] == ["topology", "functional"]
    topology_prompt = json.loads(client.calls[0][1])
    functional_prompt = json.loads(client.calls[1][1])
    assert {
        item["source"] for item in topology_prompt["routed_feedback"]["items"]
    } == {"public_target_contract"}
    assert {
        item["source"] for item in functional_prompt["routed_feedback"]["items"]
    } == {"numerical_fitter"}
    assert "topology_commitment_sha256" in functional_prompt
    assert first.expansion.candidate == second.expansion.candidate
    assert second.topology_call is not None
    assert second.topology_call.checkpoint_hit is True
    assert second.functional_call.checkpoint_hit is True


def test_fixed_topology_refinement_skips_topology_call(tmp_path: Path) -> None:
    client = _FakeStagedClient()
    topology = enrich_topology_proposal(_topology_proposal(), _context())
    proposer = StagedProposer(client=client, config=_config(tmp_path))

    result = proposer.construct(
        public_problem="Infer a driven response model.",
        context=_context(),
        feedback=_feedback(),
        fixed_topology=topology,
    )

    assert [role for role, _ in client.calls] == ["functional"]
    assert result.topology_reused is True
    assert result.topology_call is None


def test_topology_refinement_receives_incumbent_and_still_rebuilds_functions(
    tmp_path: Path,
) -> None:
    client = _FakeStagedClient()
    incumbent = enrich_topology_proposal(_topology_proposal(), _context())
    proposer = StagedProposer(client=client, config=_config(tmp_path))

    proposer.construct(
        public_problem="Infer a driven response model.",
        context=_context(),
        feedback=_feedback(),
        incumbent_topology=incumbent,
    )

    assert [role for role, _ in client.calls] == ["topology", "functional"]
    topology_prompt = json.loads(client.calls[0][1])
    assert topology_prompt["incumbent_topology"]["candidate_id"] == "graph_0"
    assert "preserve unaffected" in topology_prompt["revision_rule"]
