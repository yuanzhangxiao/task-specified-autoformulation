"""End-to-end integration of staged proposal with checkpointed search."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import (
    ModelValidationError,
    ValidationContext,
    ValidationDiagnostic,
)
from autoformalism.fitting import FitConfig
from autoformalism.llm import LLMCallResult
from autoformalism.pruning import PruningConfig
from autoformalism.schemas import (
    ProposedFunctionalCandidate,
    ProposedTopologyCandidate,
)
from autoformalism.search import SearchConfig, SearchController
from autoformalism.staging import (
    enrich_functional_proposal,
    enrich_topology_proposal,
)


def _split(name: SplitName) -> DatasetSplit:
    time = np.linspace(0.0, 2.0, 21)
    target = np.exp(-0.6 * time)
    return DatasetSplit(
        name,
        (
            Trajectory(
                name.value,
                time,
                {"target": target},
                {},
                {},
                {},
                {},
            ),
        ),
        f"{name.value}-fingerprint",
    )


def _topology() -> ProposedTopologyCandidate:
    return ProposedTopologyCandidate.model_validate(
        {
            "candidate_id": "topology_0",
            "states": [{"name": "x"}],
            "interactions": [
                {
                    "interaction_id": "decay",
                    "target": "x",
                    "target_kind": "state_derivative",
                    "sources": ["x"],
                    "polarity": "subtractive",
                }
            ],
            "target_mappings": [{"channel": "target", "source": "x"}],
        }
    )


def _functions(identifier: str, expression: str) -> ProposedFunctionalCandidate:
    return ProposedFunctionalCandidate.model_validate(
        {
            "candidate_id": identifier,
            "interaction_functions": [
                {"interaction_id": "decay", "expression": expression}
            ],
            "parameters": [{"name": "rate", "role": "rate"}],
        }
    )


class _StagedSearchClient:
    def __init__(self) -> None:
        self.functions = deque(
            (
                _functions("candidate_0", "rate * x"),
                _functions("candidate_1", "rate * x ** 2"),
            )
        )
        self.calls: list[tuple[str, str]] = []

    def propose_topology(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        context: ValidationContext,
        cache_only: bool = False,
    ) -> LLMCallResult:
        assert cache_only is False
        self.calls.append(("topology", user_prompt))
        return _result(
            "1" * 64,
            enrich_topology_proposal(_topology(), context),
        )

    def propose_functions(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        topology,
        context: ValidationContext,
        cache_only: bool = False,
    ) -> LLMCallResult:
        assert cache_only is False
        self.calls.append(("functional", user_prompt))
        return _result(
            str(len(self.calls)) * 64,
            enrich_functional_proposal(self.functions.popleft(), topology),
        )

    def judge(self, *, system_prompt: str, user_prompt: str) -> LLMCallResult:
        raise AssertionError("judge must remain disabled in this fixture")


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


def _config(path: Path) -> SearchConfig:
    fit = FitConfig(
        number_of_starts=1,
        random_seed=0,
        maximum_function_evaluations=200,
        maximum_wall_time_seconds=30.0,
    )
    return SearchConfig(
        checkpoint_directory=path,
        maximum_iterations=2,
        beam_size=1,
        stagnation_iterations=2,
        use_judge=False,
        proposer_construction_mode="staged_v2",
        proposer_feedback_mode="rich_v1",
        proposal_policy="incumbent_refinement_v1",
        evaluate_test=False,
        proposer_system_prompt="Unused complete-candidate prompt.",
        staged_public_problem="Infer one decaying public target.",
        staged_topology_system_prompt="Return the graph only.",
        staged_functional_system_prompt="Return functions only.",
        judge_system_prompt="Unused judge prompt.",
        fit_config=fit,
        final_fit_config=fit,
        pruning_config=PruningConfig(),
        apply_postfit_pruning=False,
    )


def test_staged_search_routes_fit_feedback_and_reuses_topology(
    tmp_path: Path,
) -> None:
    client = _StagedSearchClient()
    context = ValidationContext(targets=("target",))
    controller = SearchController(
        llm_client=client,
        context=context,
        training=_split(SplitName.TRAIN),
        validation=_split(SplitName.VALIDATION),
        test_loader=lambda _selection: _split(SplitName.TEST),
        config=_config(tmp_path / "checkpoints"),
    )

    result = controller.run()

    assert result.completed_iterations == 2
    assert [role for role, _ in client.calls] == [
        "topology",
        "functional",
        "functional",
    ]
    second_function_prompt = json.loads(client.calls[-1][1])
    assert any(
        item["source"] == "validation_metric"
        for item in second_function_prompt["routed_feedback"]["items"]
    )
    round_one = json.loads(
        (tmp_path / "checkpoints" / "round_0001.json").read_text()
    )
    assert (
        round_one["staged_proposal"]["revision_decision"]
        == "function_only_revision"
    )
    assert round_one["staged_proposal"]["runtime_parent_candidate_id"] == (
        "candidate_0"
    )
    assert round_one["pruning"]["applied"] is False

    resumed_client = _StagedSearchClient()
    resumed_client.functions.clear()
    resumed = SearchController(
        llm_client=resumed_client,
        context=context,
        training=_split(SplitName.TRAIN),
        validation=_split(SplitName.VALIDATION),
        test_loader=lambda _selection: _split(SplitName.TEST),
        config=_config(tmp_path / "checkpoints"),
    ).run()
    assert resumed.frozen_selection == result.frozen_selection
    assert resumed_client.calls == []


def test_staged_validation_exception_invalidates_round_without_losing_incumbent(
    tmp_path: Path,
) -> None:
    class _EscapingValidationClient(_StagedSearchClient):
        def propose_functions(self, **kwargs):
            if len([item for item in self.calls if item[0] == "functional"]) == 1:
                raise ModelValidationError(
                    (
                        ValidationDiagnostic(
                            code="SYNTAX_ERROR",
                            location="interaction:decay",
                            message="expression cannot be parsed",
                        ),
                    )
                )
            return super().propose_functions(**kwargs)

    client = _EscapingValidationClient()
    controller = SearchController(
        llm_client=client,
        context=ValidationContext(targets=("target",)),
        training=_split(SplitName.TRAIN),
        validation=_split(SplitName.VALIDATION),
        test_loader=lambda _selection: _split(SplitName.TEST),
        config=_config(tmp_path / "checkpoints"),
    )

    result = controller.run()

    assert result.completed_iterations == 1
    failed_round = json.loads(
        (tmp_path / "checkpoints" / "round_0001.json").read_text()
    )
    assert failed_round["valid"] is False
    assert failed_round["failure_class"] == "staged_proposal_contract"
    assert "SYNTAX_ERROR" in failed_round["error"]


def test_staged_search_rejects_legacy_pruning_contract(tmp_path: Path) -> None:
    payload = _config(tmp_path).model_dump()
    payload["apply_postfit_pruning"] = True
    with pytest.raises(ValidationError, match="apply_postfit_pruning=False"):
        SearchConfig.model_validate(payload)
