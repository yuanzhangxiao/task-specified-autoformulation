"""Offline end-to-end judge, beam, checkpoint, and final-test integration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import FitConfig
from autoformalism.llm import MockLLMClient
from autoformalism.llm.exceptions import LLMCacheMissError, LLMResponseError
from autoformalism.pruning import PruningConfig
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    MechanismRequirement,
)
from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    CandidateAbsoluteAssessment,
    CandidateModel,
    HybridJudgeResult,
    PairedAbsoluteAssessment,
    RelativeAssessment,
    RelativeCriterion,
    RelativeVerdict,
    ScientificJudgeResult,
)
from autoformalism.search import CheckpointError, SearchConfig, SearchController
from autoformalism.search.controller import _beam, _hybrid_incumbent
from autoformalism.search.hybrid_pair import HybridPairJudgment
from autoformalism.targets import PublicTargetContract


def _candidate(
    identifier: str,
    rhs: str,
    parameters: tuple[tuple[str, float, float], ...],
    parent: str | None = None,
) -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": identifier,
            "parent_candidate_id": parent,
            "change_summary": "Exploratory dynamical structure.",
            "states": [
                {
                    "name": "x",
                    "kind": "observed",
                    "unit": "unit",
                    "description": "Observed state.",
                }
            ],
            "state_equations": [{"state": "x", "rhs": rhs}],
            "observation_mappings": [
                {"channel": "target", "expression": "x", "unit": "unit"}
            ],
            "parameters": [
                {
                    "name": name,
                    "scope": "global",
                    "bounds": {"lower": lower, "upper": upper},
                    "initialization_range": {"lower": lower, "upper": upper},
                    "unit": "unit",
                    "description": f"Parameter {name}.",
                }
                for name, lower, upper in parameters
            ],
            "initial_conditions": [
                {
                    "state": "x",
                    "scope": "global",
                    "initialization_range": {"lower": 0.5, "upper": 1.5},
                }
            ],
        }
    )


def _judge(score: float = 1.0) -> ScientificJudgeResult:
    return ScientificJudgeResult.model_validate(
        {
            "schema_version": "2",
            "hard_red_flags": [],
            "category_scores": {
                "mechanistic_coherence": score,
                "source_sink_balance_semantics": score,
                "dynamic_plausibility": score,
                "mechanism_coupling_task_sufficiency": score,
                "nonredundancy_accounting": score,
                "latent_state_complexity_justification": score,
            },
            "missing_requirements": [],
            "actionable_edits": [],
        }
    )


def _split(name: SplitName, identifier: str) -> DatasetSplit:
    time = np.linspace(0.0, 2.0, 21)
    target = np.exp(-0.6 * time)
    trajectory = Trajectory(
        identifier,
        time,
        {"target": target},
        {},
        {},
        {},
        {},
    )
    return DatasetSplit(name, (trajectory,), f"{name.value}-fingerprint")


def _config(path: Path, iterations: int) -> SearchConfig:
    return SearchConfig(
        checkpoint_directory=path,
        maximum_iterations=iterations,
        beam_size=2,
        stagnation_iterations=3,
        validation_mse_target=0.0,
        cheap_prefit_judge=True,
        proposer_system_prompt="Propose one valid model.",
        judge_system_prompt="Judge task compliance.",
        fit_config=FitConfig(
            number_of_starts=2,
            random_seed=19,
            maximum_function_evaluations=600,
        ),
        pruning_config=PruningConfig(validation_mse_tolerance=1e-7),
    )


def _controller(
    client: MockLLMClient,
    config: SearchConfig,
    callback=None,
    test_loader=None,
    pairwise_judge=None,
    public_mechanism_spec=None,
) -> SearchController:
    return SearchController(
        llm_client=client,
        context=ValidationContext(targets=("target",)),
        training=_split(SplitName.TRAIN, "train"),
        validation=_split(SplitName.VALIDATION, "validation"),
        test_loader=(
            test_loader
            if test_loader is not None
            else lambda _frozen: _split(SplitName.TEST, "test")
        ),
        config=config,
        pairwise_judge=pairwise_judge,
        public_mechanism_spec=public_mechanism_spec,
        stage_callback=callback,
    )


def _mechanism_spec() -> MechanismEvaluationSpec:
    return MechanismEvaluationSpec(
        source="public_prompt",
        benchmark_id="synthetic",
        tier="easy",
        public_prompt_sha256="0" * 64,
        required_mechanisms=(
            MechanismRequirement(
                id="decay",
                public_requirement="a generated decay pathway",
                tag_aliases=("decay",),
                required_targets=("target",),
            ),
        ),
    )


class _FixedPairwiseJudge:
    fingerprint = "fixed-pairwise-judge"

    def __init__(self, decision_for_incumbent: float | None) -> None:
        self._decision = decision_for_incumbent
        self.calls = 0

    def compare(
        self, incumbent: CandidateModel, challenger: CandidateModel
    ) -> HybridPairJudgment:
        self.calls += 1
        absolute = PairedAbsoluteAssessment(
            criterion=AbsoluteCriterion.SOURCE_ROLES_CONSISTENT,
            subject_id="candidate",
            candidate_a=CandidateAbsoluteAssessment(
                verdict=AbsoluteVerdict.PASS,
                evidence="Incumbent evidence.",
            ),
            candidate_b=CandidateAbsoluteAssessment(
                verdict=AbsoluteVerdict.PASS,
                evidence="Challenger evidence.",
            ),
        )
        consensus = HybridJudgeResult(
            absolute_assessments=(absolute,),
            comparative_assessments=tuple(
                RelativeAssessment(
                    criterion=criterion,
                    verdict=RelativeVerdict.TIE,
                    evidence="Fixed comparison evidence.",
                )
                for criterion in RelativeCriterion
            ),
        )
        return HybridPairJudgment(
            incumbent_candidate_id=incumbent.candidate_id,
            challenger_candidate_id=challenger.candidate_id,
            seed_attempt_index=0,
            seed=12000,
            decision_value_for_incumbent=self._decision,
            decision_scale=1.25,
            tie_threshold=0.05,
            preferred="candidate_b",
            orientation_values_for_incumbent=(self._decision, self._decision),
            orientation_half_gap=0.0,
            consensus_result=consensus,
            deterministic_assessments=(),
            request_hashes=("a", "b", "c", "d"),
        )


def test_search_rejects_public_target_contract_violation_before_fitting(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        "missing_required_input",
        "-decay * x",
        (("decay", 0.2, 1.0),),
    )
    client = MockLLMClient(proposer_responses=[candidate])
    contract = PublicTargetContract.model_validate(
        {
            "benchmark_id": "synthetic_contract_test",
            "tier": "easy",
            "public_prompt_sha256": "0" * 64,
            "targets": [
                {
                    "target_channel": "target",
                    "public_requirement": "generate target",
                    "required_dependencies": [
                        {
                            "dependency_id": "forcing_contribution",
                            "acceptable_symbols": ["forcing"],
                            "public_requirement": "forcing contributes to target",
                        }
                    ],
                }
            ],
        }
    )
    checkpoint_root = tmp_path / "checkpoints"
    controller = SearchController(
        llm_client=client,
        context=ValidationContext(
            targets=("target",), external_inputs=("forcing",)
        ),
        training=_split(SplitName.TRAIN, "train"),
        validation=_split(SplitName.VALIDATION, "validation"),
        test_loader=lambda _frozen: _split(SplitName.TEST, "test"),
        config=_config(checkpoint_root, 1),
        public_target_contract=contract,
    )

    with pytest.raises(RuntimeError, match="no valid fitted candidates"):
        controller.run()

    checkpoint = json.loads(
        (checkpoint_root / "round_0000.json").read_text(encoding="utf-8")
    )
    assert checkpoint["valid"] is False
    assert checkpoint["error"].startswith("PUBLIC_TARGET_CONTRACT_FAILED")
    assert checkpoint["public_target_evaluation"]["passed"] is False
    assert not any(call["role"] == "judge" for call in client.calls)


def test_end_to_end_mock_search_feedback_lineage_and_one_time_test(
    tmp_path: Path,
) -> None:
    first = _candidate("candidate_one", "-decay * x", (("decay", 0.2, 1.0),))
    second = _candidate(
        "candidate_two",
        "-decay * x + offset",
        (("decay", 0.2, 1.0), ("offset", -0.1, 0.1)),
        parent="candidate_one",
    )
    client = MockLLMClient(
        proposer_responses=[first, second],
        judge_responses=[_judge()] * 6,
    )
    controller = _controller(client, _config(tmp_path / "checkpoints", 2))

    result = controller.run()
    calls_after_first_run = len(client.calls)
    resumed = controller.run()

    assert result.completed_iterations == 2
    assert result.frozen_selection.validation_mse >= 0.0
    assert result.final_fit.success
    assert not result.test_metrics.failed_trajectories
    assert resumed.frozen_selection.selection_hash == (
        result.frozen_selection.selection_hash
    )
    assert len(client.calls) == calls_after_first_run
    assert [call["role"] for call in client.calls].count("proposer") == 2
    assert [call["role"] for call in client.calls].count("judge") == 6

    judge_request = json.loads(
        next(
            call["user_prompt"]
            for call in client.calls
            if call["role"] == "judge"
        )
    )
    assert judge_request["deterministic_certifications"] == [
        "response schema is valid",
        "every state has exactly one governing equation",
        "every expression symbol is declared or supplied",
        "target mappings exist",
        "algebraic definitions are acyclic",
        "only causally available public channels are used",
        "parameter declarations and bounds are valid",
        "restricted expressions are executable",
    ]
    assert judge_request["candidate"]["candidate_id"] == "candidate_one"
    assert "fit" not in judge_request
    assert "validation" not in judge_request
    assert "test" not in judge_request

    second_proposal_prompt = [
        call["user_prompt"]
        for call in client.calls
        if call["role"] == "proposer"
    ][1]
    for required in (
        "equations",
        "fitted_parameters",
        "training_normalized_mse",
        "validation_normalized_mse",
        "judge_category_scores",
        "numerical_failures",
        "pruning_diagnostics",
    ):
        assert required in second_proposal_prompt
    assert "deterministic_runtime" not in second_proposal_prompt
    assert "test_metrics" not in second_proposal_prompt
    assert "test_normalized_mse" not in second_proposal_prompt
    assert second.parent_candidate_id == first.candidate_id


def test_weighted_selection_can_trade_validation_fit_for_scientific_score(
    tmp_path: Path,
) -> None:
    best_fit = _candidate("best_fit", "-0.6 * x", ())
    better_science = _candidate(
        "better_science", "-0.55 * x", (), parent="best_fit"
    )
    client = MockLLMClient(
        proposer_responses=[best_fit, better_science],
        judge_responses=[_judge(0.1)] * 3 + [_judge(0.9)] * 3,
    )
    validation_config = _config(tmp_path / "weighted", 2).model_copy(
        update={"stagnation_iterations": 3}
    )
    controller = _controller(client, validation_config)
    controller.run()
    records = controller._completed_records()

    validation_selected = _beam(records, 1, validation_config)[0]
    weighted_config = validation_config.model_copy(
        update={
            "selection_policy": "normalized_weighted_sum",
            "judge_weight": 2.0,
        }
    )
    weighted_selected = _beam(records, 1, weighted_config)[0]

    assert validation_selected.candidate.candidate_id == "best_fit"
    assert weighted_selected.candidate.candidate_id == "better_science"


def test_incumbent_relative_hybrid_uses_one_common_challenge_and_resumes(
    tmp_path: Path,
) -> None:
    incumbent = _candidate("incumbent", "-0.6 * x", ())
    challenger = _candidate(
        "challenger",
        "-0.55 * x",
        (),
        parent="incumbent",
    )
    client = MockLLMClient(proposer_responses=[incumbent, challenger])
    pairwise = _FixedPairwiseJudge(decision_for_incumbent=-1.0)
    config = _config(tmp_path / "hybrid", 2).model_copy(
        update={
            "beam_size": 1,
            "cheap_prefit_judge": False,
            "evaluate_test": False,
            "selection_policy": "incumbent_relative_hybrid",
            "hybrid_science_weight": 0.75,
        }
    )

    controller = _controller(client, config, pairwise_judge=pairwise)
    result = controller.run()
    records = controller._completed_records()
    selected, path_score = _hybrid_incumbent(records)

    assert pairwise.calls == 1
    assert selected.pruned_candidate.candidate_id == "challenger"
    assert path_score > 0.0
    assert result.frozen_selection.selection_hash == selected.structural_hash
    assert records[0].incumbent_challenge is None
    challenge = records[1].incumbent_challenge
    assert challenge is not None
    assert challenge.science_preference_for_challenger > 0.0
    assert challenge.selected_hash == records[1].structural_hash
    assert not [call for call in client.calls if call["role"] == "judge"]

    resumed = _controller(client, config, pairwise_judge=pairwise).run()
    assert (
        resumed.frozen_selection.selection_hash
        == result.frozen_selection.selection_hash
    )
    assert pairwise.calls == 1


def test_matched_no_judge_arm_uses_same_candidates_without_pairwise_calls(
    tmp_path: Path,
) -> None:
    incumbent = _candidate("incumbent", "-0.6 * x", ())
    challenger = _candidate(
        "challenger",
        "-0.55 * x",
        (),
        parent="incumbent",
    )
    hybrid_client = MockLLMClient(
        proposer_responses=[incumbent, challenger]
    )
    no_judge_client = MockLLMClient(
        proposer_responses=[incumbent, challenger]
    )
    pairwise = _FixedPairwiseJudge(decision_for_incumbent=-1.0)
    base = {
        "beam_size": 1,
        "cheap_prefit_judge": False,
        "evaluate_test": False,
        "stagnation_iterations": 3,
    }
    hybrid_config = _config(tmp_path / "hybrid-arm", 2).model_copy(
        update={
            **base,
            "selection_policy": "incumbent_relative_hybrid",
            "hybrid_science_weight": 0.75,
        }
    )
    no_judge_config = _config(tmp_path / "no-judge-arm", 2).model_copy(
        update={
            **base,
            "selection_policy": "validation_only",
            "use_judge": False,
        }
    )

    hybrid_result = _controller(
        hybrid_client,
        hybrid_config,
        pairwise_judge=pairwise,
    ).run()
    no_judge_result = _controller(no_judge_client, no_judge_config).run()

    assert pairwise.calls == 1
    assert hybrid_result.frozen_selection.candidate.candidate_id == "challenger"
    assert no_judge_result.frozen_selection.candidate.candidate_id == "incumbent"
    assert [call["role"] for call in no_judge_client.calls] == [
        "proposer",
        "proposer",
    ]


def test_losing_hybrid_challenge_is_feedback_but_not_an_eligible_parent(
    tmp_path: Path,
) -> None:
    incumbent = _candidate("incumbent", "-0.6 * x", ())
    rejected = _candidate(
        "rejected_challenger",
        "-0.55 * x",
        (),
        parent="incumbent",
    )
    next_candidate = _candidate(
        "next_candidate",
        "-0.65 * x",
        (),
        parent="incumbent",
    )
    client = MockLLMClient(
        proposer_responses=[incumbent, rejected, next_candidate]
    )
    pairwise = _FixedPairwiseJudge(decision_for_incumbent=1.0)
    config = _config(tmp_path / "hybrid-rejected-feedback", 3).model_copy(
        update={
            "beam_size": 1,
            "cheap_prefit_judge": False,
            "evaluate_test": False,
            "selection_policy": "incumbent_relative_hybrid",
            "hybrid_science_weight": 1.0,
        }
    )

    controller = _controller(client, config, pairwise_judge=pairwise)
    controller.run()

    third_prompt = json.loads(
        [call for call in client.calls if call["role"] == "proposer"][2][
            "user_prompt"
        ]
    )
    incumbent_feedback = third_prompt["beam_feedback"][0]
    rejected_feedback = incumbent_feedback["recent_rejected_challenger"]
    assert incumbent_feedback["candidate_id"] == "incumbent"
    assert rejected_feedback["candidate_id"] == "rejected_challenger"
    assert rejected_feedback["eligible_parent"] is False
    assert rejected_feedback["comparison"]["selected_candidate_hash"] == (
        controller._completed_records()[0].structural_hash
    )
    assert "request_hashes" not in json.dumps(rejected_feedback)
    assert "transport" not in json.dumps(rejected_feedback)


def test_development_only_search_never_calls_test_loader(tmp_path: Path) -> None:
    candidate = _candidate(
        "development_candidate", "-decay * x", (("decay", 0.2, 1.0),)
    )
    client = MockLLMClient(
        proposer_responses=[candidate],
        judge_responses=[_judge()] * 3,
    )
    config = _config(tmp_path / "development", 1).model_copy(
        update={"evaluate_test": False}
    )

    def forbidden_test_loader(_frozen):
        raise AssertionError("development-only search opened test data")

    result = _controller(
        client, config, test_loader=forbidden_test_loader
    ).run()

    assert result.test_metrics is None
    assert result.test_trajectory_initial_conditions is None
    final = json.loads((tmp_path / "development" / "final.json").read_text())
    assert final["stage"] == "development_complete"
    assert "test_metrics" not in final


def test_prefit_rejection_is_feedback_for_next_proposal(tmp_path: Path) -> None:
    invalid = _candidate(
        "invalid_one",
        "unknown_input - decay * x",
        (("decay", 0.2, 1.0),),
    )
    valid = _candidate(
        "valid_two",
        "-decay * x",
        (("decay", 0.2, 1.0),),
        parent="invalid_one",
    )
    client = MockLLMClient(
        proposer_responses=[invalid, valid],
        judge_responses=[_judge()] * 3,
    )

    config = _config(tmp_path / "recovery", 2).model_copy(
        update={"proposer_feedback_mode": "structured"}
    )
    result = _controller(client, config).run()

    proposer_prompts = [
        json.loads(call["user_prompt"])
        for call in client.calls
        if call["role"] == "proposer"
    ]
    rejection = proposer_prompts[1]["beam_feedback"][0]
    assert result.completed_iterations == 1
    assert rejection["candidate_id"] == "invalid_one"
    assert rejection["eligible_parent"] is True
    assert rejection["rejected_before_fit"] is True
    assert rejection["deterministic_runtime"]["candidate_validation"] == "failed"
    assert rejection["deterministic_runtime"]["validation_diagnostics"][0][
        "code"
    ] == "UNDEFINED_SYMBOL"
    assert "UNDEFINED_SYMBOL" in rejection["numerical_failures"][
        "deterministic_validation"
    ][0]
    assert "test" not in json.dumps(rejection).lower()


def test_rich_incumbent_refinement_binds_lineage_and_exposes_actionable_feedback(
    tmp_path: Path,
) -> None:
    incumbent = _candidate(
        "incumbent",
        "-decay * x",
        (("decay", 0.2, 1.0),),
    )
    proposed_without_parent = _candidate(
        "refined",
        "-decay * x + offset",
        (("decay", 0.2, 1.0), ("offset", -0.1, 0.1)),
    )
    client = MockLLMClient(
        proposer_responses=[incumbent, proposed_without_parent],
        judge_responses=[],
    )
    config = _config(tmp_path / "rich-refinement", 2).model_copy(
        update={
            "beam_size": 1,
            "cheap_prefit_judge": False,
            "use_judge": False,
            "evaluate_test": False,
            "proposer_feedback_mode": "rich_v1",
            "proposal_policy": "incumbent_refinement_v1",
        }
    )

    result = _controller(
        client,
        config,
        public_mechanism_spec=_mechanism_spec(),
    ).run()

    prompts = [
        json.loads(call["user_prompt"])
        for call in client.calls
        if call["role"] == "proposer"
    ]
    assert prompts[0]["proposal_mode"] == "exploratory"
    assert prompts[1]["proposal_mode"] == "incumbent_refinement"
    assert prompts[1]["feedback_schema_version"] == "proposer-feedback-rich-1"
    assert prompts[1]["refinement_contract"][
        "required_parent_candidate_id"
    ] == "incumbent"
    incumbent_feedback = prompts[1]["beam_feedback"][0]
    assert incumbent_feedback["incumbent_snapshot"]["states"][0]["rhs"] == (
        "-decay * x"
    )
    assert incumbent_feedback["per_target_error"][
        "validation_normalized_mse"
    ]["target"] >= 0.0
    mechanism = incumbent_feedback["deterministic_runtime"][
        "public_mechanism_evaluation"
    ]
    assert mechanism["mechanism_results"][0]["status"] == "ambiguous"
    assert mechanism["annotation_results"][0]["status"] == "failed"
    assert "test" not in json.dumps(prompts[1]).lower()
    second_checkpoint = json.loads(
        (tmp_path / "rich-refinement" / "round_0001.json").read_text()
    )
    assert second_checkpoint["candidate"]["parent_candidate_id"] == "incumbent"
    assert second_checkpoint["raw_candidate"]["parent_candidate_id"] is None
    assert any(
        "bound refinement lineage" in item
        for item in second_checkpoint["deterministic_repairs"]
    )
    assert result.completed_iterations == 2


def test_refinement_policy_preserves_identical_round_zero_request(
    tmp_path: Path,
) -> None:
    candidate = _candidate("initial", "-0.6 * x", ())
    prompts = []
    for policy in ("exploratory", "incumbent_refinement_v1"):
        client = MockLLMClient(proposer_responses=[candidate], judge_responses=[])
        config = _config(tmp_path / policy, 1).model_copy(
            update={
                "beam_size": 1,
                "cheap_prefit_judge": False,
                "use_judge": False,
                "evaluate_test": False,
                "proposer_feedback_mode": "rich_v1",
                "proposal_policy": policy,
            }
        )
        _controller(client, config).run()
        prompts.append(
            next(
                call["user_prompt"]
                for call in client.calls
                if call["role"] == "proposer"
            )
        )

    assert prompts[0] == prompts[1]


def test_no_judge_ablation_makes_no_judge_calls(tmp_path: Path) -> None:
    candidate = _candidate(
        "no_judge_candidate", "-decay * x", (("decay", 0.2, 1.0),)
    )
    client = MockLLMClient(proposer_responses=[candidate], judge_responses=[])
    config = _config(tmp_path / "no_judge", 1).model_copy(
        update={"use_judge": False}
    )

    result = _controller(client, config).run()

    assert result.completed_iterations == 1
    assert [call["role"] for call in client.calls] == ["proposer"]


def test_llm_red_flags_are_advisory_and_do_not_block_fitted_candidate(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        "advisory_candidate",
        "-decay * x",
        (("decay", 0.2, 1.0),),
    )
    flagged = ScientificJudgeResult.model_validate(
        {
            "schema_version": "2",
            "hard_red_flags": [
                {
                    "code": "subjective_mechanism_concern",
                    "evidence": "The mechanism may be too simple.",
                }
            ],
            "category_scores": {
                "mechanistic_coherence": 0.2,
                "source_sink_balance_semantics": 0.2,
                "dynamic_plausibility": 0.2,
                "mechanism_coupling_task_sufficiency": 0.2,
                "nonredundancy_accounting": 0.2,
                "latent_state_complexity_justification": 0.2,
            },
            "missing_requirements": ["More mechanistic detail may help."],
            "actionable_edits": [],
        }
    )
    client = MockLLMClient(
        proposer_responses=[candidate],
        judge_responses=[flagged, flagged, flagged],
    )

    result = _controller(client, _config(tmp_path / "advisory", 1)).run()

    assert result.completed_iterations == 1
    assert result.final_fit.success


def test_judge_response_failure_falls_back_and_search_continues(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        "judge_failure_candidate",
        "-decay * x",
        (("decay", 0.2, 1.0),),
    )

    class FailingJudgeClient(MockLLMClient):
        def judge(self, *, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            raise LLMResponseError("malformed judge JSON")

    client = FailingJudgeClient(proposer_responses=[candidate])

    result = _controller(client, _config(tmp_path / "judge_failure", 1)).run()

    assert result.completed_iterations == 1
    assert result.final_fit.success


def test_proposer_response_failure_checkpoints_round_and_search_continues(
    tmp_path: Path,
) -> None:
    first = _candidate(
        "first_valid", "-decay * x", (("decay", 0.2, 1.0),)
    )
    third = _candidate(
        "third_valid",
        "-decay * x + offset",
        (("decay", 0.2, 1.0), ("offset", -0.1, 0.1)),
        parent="first_valid",
    )

    class FailingSecondProposalClient(MockLLMClient):
        proposal_calls = 0

        def propose(self, *, system_prompt: str, user_prompt: str):
            self.proposal_calls += 1
            if self.proposal_calls == 2:
                raise LLMResponseError("missing required target mapping")
            return super().propose(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

    client = FailingSecondProposalClient(
        proposer_responses=[first, third],
        judge_responses=[_judge()] * 6,
    )
    checkpoint_directory = tmp_path / "proposer_failure"

    result = _controller(client, _config(checkpoint_directory, 3)).run()

    assert result.completed_iterations == 2
    assert result.final_fit.success
    failed_round = json.loads(
        (checkpoint_directory / "round_0001.json").read_text()
    )
    assert failed_round["stage"] == "complete"
    assert failed_round["valid"] is False
    assert failed_round["error"] == (
        "LLMResponseError: missing required target mapping"
    )


def test_round_zero_cache_precondition_fails_before_independent_generation(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        "unused_candidate", "-decay * x", (("decay", 0.2, 1.0),)
    )
    client = MockLLMClient(proposer_responses=[candidate])
    config = SearchConfig(
        checkpoint_directory=tmp_path / "cache_precondition",
        maximum_iterations=1,
        beam_size=1,
        stagnation_iterations=1,
        validation_mse_target=0.0,
        cheap_prefit_judge=False,
        use_judge=False,
        require_initial_proposer_cache_hit=True,
        evaluate_test=False,
        proposer_system_prompt="Propose one valid model.",
        judge_system_prompt="Unused.",
        fit_config=FitConfig(number_of_starts=1),
        pruning_config=PruningConfig(validation_mse_tolerance=1e-7),
    )

    with pytest.raises(LLMCacheMissError, match="no persistent cache"):
        _controller(client, config).run()

    assert not client.calls


def test_resume_continues_after_exact_completed_stage_without_repeat(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        "resume_candidate", "-decay * x", (("decay", 0.2, 1.0),)
    )
    client = MockLLMClient(
        proposer_responses=[candidate],
        judge_responses=[_judge()] * 3,
    )
    config = _config(tmp_path / "resume", 1)

    def interrupt(stage: str, _round: int | None) -> None:
        if stage == "fitted":
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _controller(client, config, interrupt).run()
    calls_at_interruption = list(client.calls)

    result = _controller(client, config).run()

    assert result.final_fit.success
    assert [call["role"] for call in calls_at_interruption] == [
        "proposer",
        "judge",
    ]
    assert [call["role"] for call in client.calls].count("proposer") == 1
    assert [call["role"] for call in client.calls].count("judge") == 3


def test_test_loader_is_deferred_and_resume_accesses_it_once(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        "deferred_test", "-decay * x", (("decay", 0.2, 1.0),)
    )
    client = MockLLMClient(
        proposer_responses=[candidate],
        judge_responses=[_judge()] * 3,
    )
    calls = 0

    def test_loader(_frozen):
        nonlocal calls
        calls += 1
        return _split(SplitName.TEST, "test")

    def interrupt(stage: str, _round: int | None) -> None:
        if stage == "test_started":
            assert calls == 0
            raise RuntimeError("stop before test access")

    config = _config(tmp_path / "deferred", 1)
    with pytest.raises(RuntimeError, match="stop before test access"):
        _controller(client, config, interrupt, test_loader).run()
    assert calls == 0

    result = _controller(client, config, test_loader=test_loader).run()
    assert not result.test_metrics.failed_trajectories
    assert calls == 1

    _controller(client, config, test_loader=test_loader).run()
    assert calls == 1


def test_failure_after_test_access_fails_closed_without_second_access(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        "test_failure", "-decay * x", (("decay", 0.2, 1.0),)
    )
    client = MockLLMClient(
        proposer_responses=[candidate],
        judge_responses=[_judge()] * 3,
    )
    calls = 0

    def failing_loader(_frozen):
        nonlocal calls
        calls += 1
        raise RuntimeError("failure after opening test")

    config = _config(tmp_path / "test-failure", 1)
    with pytest.raises(RuntimeError, match="failure after opening test"):
        _controller(client, config, test_loader=failing_loader).run()
    assert calls == 1

    with pytest.raises(CheckpointError, match="refusing to access test again"):
        _controller(client, config, test_loader=failing_loader).run()
    assert calls == 1


def test_alpha_renamed_structural_duplicate_is_rejected(
    tmp_path: Path,
) -> None:
    first = _candidate("first", "-decay * x", (("decay", 0.2, 1.0),))
    renamed = first.model_dump(mode="json")
    renamed.update(
        candidate_id="renamed",
        parent_candidate_id="first",
        states=[
            {
                "name": "renamed_state",
                "kind": "observed",
                "unit": "unit",
                "description": "Renamed state.",
            }
        ],
        state_equations=[
            {"state": "renamed_state", "rhs": "-renamed_rate * renamed_state"}
        ],
        observation_mappings=[
            {
                "channel": "target",
                "expression": "renamed_state",
                "unit": "unit",
            }
        ],
        parameters=[
            {
                "name": "renamed_rate",
                "scope": "global",
                "bounds": {"lower": 0.2, "upper": 1.0},
                "initialization_range": {"lower": 0.2, "upper": 1.0},
                "unit": "unit",
                "description": "Renamed rate.",
            }
        ],
        initial_conditions=[
            {
                "state": "renamed_state",
                "scope": "global",
                "initialization_range": {"lower": 0.5, "upper": 1.5},
            }
        ],
    )
    client = MockLLMClient(
        proposer_responses=[first, CandidateModel.model_validate(renamed)],
        judge_responses=[_judge()] * 3,
    )
    config = _config(tmp_path / "duplicates", 2)

    result = _controller(client, config).run()
    rejected = json.loads(
        (config.checkpoint_directory / "round_0001.json").read_text(
            encoding="utf-8"
        )
    )

    assert result.completed_iterations == 1
    assert rejected["error"] == "structural duplicate"
    assert [call["role"] for call in client.calls].count("judge") == 3


def test_structural_duplicate_is_included_in_next_bounded_feedback(
    tmp_path: Path,
) -> None:
    first = _candidate("first", "-decay * x", (("decay", 0.2, 1.0),))
    duplicate = first.model_copy(
        update={"candidate_id": "duplicate", "parent_candidate_id": "first"}
    )
    distinct = _candidate(
        "distinct",
        "-decay * x + offset",
        (("decay", 0.2, 1.0), ("offset", -0.1, 0.1)),
        parent="first",
    )
    client = MockLLMClient(
        proposer_responses=[first, duplicate, distinct],
        judge_responses=[_judge()] * 6,
    )
    config = _config(tmp_path / "duplicate-feedback", 3).model_copy(
        update={"beam_size": 1}
    )

    result = _controller(client, config).run()
    proposer_prompts = [
        json.loads(call["user_prompt"])
        for call in client.calls
        if call["role"] == "proposer"
    ]
    feedback = proposer_prompts[2]["beam_feedback"]

    assert result.completed_iterations == 2
    assert len(feedback) == 1
    assert feedback[0]["recent_rejected_candidate"]["candidate_id"] == (
        "duplicate"
    )
    assert "renaming" in feedback[0]["recent_rejected_candidate"][
        "required_edit"
    ]


def test_previously_failed_structure_is_not_fit_again_and_is_explained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _candidate("failed_first", "-decay * x", (("decay", 0.2, 1.0),))
    duplicate = first.model_copy(
        update={
            "candidate_id": "failed_duplicate",
            "parent_candidate_id": "failed_first",
        }
    )
    distinct = _candidate(
        "revised",
        "-decay * x + offset",
        (("decay", 0.2, 1.0), ("offset", -0.1, 0.1)),
        parent="failed_first",
    )
    client = MockLLMClient(
        proposer_responses=[first, duplicate, distinct],
        judge_responses=[_judge()] * 4,
    )
    config = _config(tmp_path / "failed-duplicate", 3).model_copy(
        update={"beam_size": 1, "proposer_feedback_mode": "rich_v1"}
    )

    from autoformalism.search import controller as controller_module

    real_fit = controller_module.fit_candidate
    failed_structure_fit_count = 0

    def controlled_fit(model, *args, **kwargs):
        nonlocal failed_structure_fit_count
        fitted = real_fit(model, *args, **kwargs)
        if model.validated.candidate.candidate_id == "failed_first":
            failed_structure_fit_count += 1
            return replace(fitted, success=False, message="synthetic fit failure")
        return fitted

    monkeypatch.setattr(controller_module, "fit_candidate", controlled_fit)
    result = _controller(client, config).run()

    duplicate_record = json.loads(
        (config.checkpoint_directory / "round_0001.json").read_text(
            encoding="utf-8"
        )
    )
    prompts = [
        json.loads(call["user_prompt"])
        for call in client.calls
        if call["role"] == "proposer"
    ]
    remembered = prompts[2]["beam_feedback"][0]
    failed_memory = prompts[2]["failed_structure_memory"]

    assert result.completed_iterations == 1
    assert failed_structure_fit_count == 1
    assert duplicate_record["error"] == "previously failed structural duplicate"
    assert remembered["candidate_id"] == "failed_duplicate"
    assert remembered["prior_structural_failure"]["candidate_id"] == "failed_first"
    assert "will not be fit again" in remembered["prior_structural_failure"][
        "instruction"
    ]
    assert len(failed_memory) == 1
    assert failed_memory[0]["candidate_id"] == "failed_first"
    assert failed_memory[0]["fit"]["message"] == "synthetic fit failure"


def test_nonexistent_lineage_parent_is_rejected(tmp_path: Path) -> None:
    candidate = _candidate(
        "bad_lineage",
        "-decay * x",
        (("decay", 0.2, 1.0),),
        parent="does_not_exist",
    )
    client = MockLLMClient(proposer_responses=[candidate])
    config = _config(tmp_path / "lineage", 1)

    with pytest.raises(RuntimeError, match="no valid fitted candidates"):
        _controller(client, config).run()

    rejected = json.loads(
        (config.checkpoint_directory / "round_0000.json").read_text(
            encoding="utf-8"
        )
    )
    assert "parent" in rejected["error"]


def test_validation_target_stops_before_remaining_budget(tmp_path: Path) -> None:
    first = _candidate("target_stop", "-decay * x", (("decay", 0.2, 1.0),))
    unused = _candidate(
        "unused",
        "-decay * x + offset",
        (("decay", 0.2, 1.0), ("offset", -0.1, 0.1)),
        parent="target_stop",
    )
    client = MockLLMClient(
        proposer_responses=[first, unused],
        judge_responses=[_judge()] * 3,
    )
    config = _config(tmp_path / "target-stop", 2).model_copy(
        update={"validation_mse_target": 1.0}
    )

    result = _controller(client, config).run()

    assert result.stopping_reason == "validation_target"
    assert result.completed_iterations == 1
    assert [call["role"] for call in client.calls].count("proposer") == 1


def test_incompatible_resume_configuration_is_rejected(tmp_path: Path) -> None:
    candidate = _candidate(
        "fingerprint", "-decay * x", (("decay", 0.2, 1.0),)
    )
    client = MockLLMClient(
        proposer_responses=[candidate],
        judge_responses=[_judge()] * 3,
    )
    config = _config(tmp_path / "fingerprint", 1)

    def interrupt(stage: str, _round: int | None) -> None:
        if stage == "fitted":
            raise RuntimeError("interrupt")

    with pytest.raises(RuntimeError, match="interrupt"):
        _controller(client, config, interrupt).run()

    changed = config.model_copy(update={"beam_size": config.beam_size + 1})
    with pytest.raises(CheckpointError, match="fingerprint"):
        _controller(client, changed)


def test_feedback_respects_configured_beam_size(tmp_path: Path) -> None:
    first = _candidate("beam_one", "-decay * x", (("decay", 0.2, 1.0),))
    second = _candidate(
        "beam_two",
        "-decay * x + offset",
        (("decay", 0.2, 1.0), ("offset", -0.1, 0.1)),
        parent="beam_one",
    )
    third = _candidate(
        "beam_three",
        "-decay * x + quadratic * x * x",
        (("decay", 0.2, 1.0), ("quadratic", -0.1, 0.1)),
        parent="beam_one",
    )
    client = MockLLMClient(
        proposer_responses=[first, second, third],
        judge_responses=[_judge()] * 9,
    )
    config = _config(tmp_path / "beam", 3).model_copy(
        update={"beam_size": 1}
    )

    _controller(client, config).run()
    proposer_prompts = [
        json.loads(call["user_prompt"])
        for call in client.calls
        if call["role"] == "proposer"
    ]

    assert len(proposer_prompts) == 3
    assert all(
        len(prompt["beam_feedback"]) <= 1 for prompt in proposer_prompts
    )
