"""Offline end-to-end judge, beam, checkpoint, and final-test integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import FitConfig
from autoformalism.llm import MockLLMClient
from autoformalism.llm.exceptions import LLMResponseError
from autoformalism.pruning import PruningConfig
from autoformalism.schemas import CandidateModel, JudgeResult
from autoformalism.search import CheckpointError, SearchConfig, SearchController


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


def _judge(score: float = 1.0) -> JudgeResult:
    return JudgeResult.model_validate(
        {
            "hard_red_flags": [],
            "category_scores": {
                "task_output_coverage": score,
                "mechanism_state_adequacy": score,
                "mathematical_completeness": score,
                "data_causal_consistency": score,
                "constraint_compliance": score,
                "parsimony_interpretability": score,
            },
            "aggregate_score": score,
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
        stage_callback=callback,
    )


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
    assert "test_metrics" not in second_proposal_prompt
    assert "test_normalized_mse" not in second_proposal_prompt
    assert second.parent_candidate_id == first.candidate_id


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

    result = _controller(client, _config(tmp_path / "recovery", 2)).run()

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
    assert "UNDEFINED_SYMBOL" in rejection["numerical_failures"][
        "deterministic_validation"
    ][0]
    assert "test" not in json.dumps(rejection).lower()


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
    flagged = JudgeResult.model_validate(
        {
            "hard_red_flags": [
                {
                    "code": "subjective_mechanism_concern",
                    "evidence": "The mechanism may be too simple.",
                }
            ],
            "category_scores": {
                "task_output_coverage": 0.2,
                "mechanism_state_adequacy": 0.2,
                "mathematical_completeness": 0.2,
                "data_causal_consistency": 0.2,
                "constraint_compliance": 0.2,
                "parsimony_interpretability": 0.2,
            },
            "aggregate_score": 0.2,
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
