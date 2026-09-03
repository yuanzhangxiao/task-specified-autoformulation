"""Offline tests for the bounded raw-data frontier-agent baseline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from autoformalism.baselines.raw_data_agent import (
    GeminiRawDataAgent,
    OpenAIRawDataAgent,
    RawAgentArtifact,
    RawAgentConfig,
    RawAgentFittedModel,
    RawAgentFittedParameter,
    RawAgentInputs,
    RawAgentOutputContract,
    RawAgentProvider,
    evaluate_raw_agent_fitted_model,
    raw_agent_request_hash,
    raw_agent_system_prompt,
    raw_agent_user_prompt,
    repair_raw_data_agent_candidate,
    run_raw_data_agent,
)
from autoformalism.fitting import EvaluationMetrics
from autoformalism.schemas import ProposerCandidateV2, enrich_proposal_v2


def _proposal() -> ProposerCandidateV2:
    return ProposerCandidateV2.model_validate(
        {
            "schema_version": "2",
            "candidate_id": "raw_agent_candidate",
            "change_summary": "One-state decay.",
            "states": [
                {
                    "name": "x",
                    "kind": "observed",
                    "observed_channel": "x",
                    "rhs": "-k * x + u",
                }
            ],
            "algebraics": [],
            "parameters": [
                {"name": "k", "bounds": {"lower": 0.0, "upper": 2.0}}
            ],
        }
    )


def _inputs(tmp_path: Path) -> RawAgentInputs:
    train = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"
    train.write_text("trajectory_id,time,x,u\na,0,1,0\n", encoding="utf-8")
    validation.write_text(
        "trajectory_id,time,x,u\nb,0,2,0\n", encoding="utf-8"
    )
    return RawAgentInputs(
        benchmark_id="public_cell",
        tier="easy",
        public_prompt="Discover a causal model for x.",
        train_path=train,
        validation_path=validation,
        targets=("x",),
        auxiliaries=(),
        external_inputs=("u",),
        fixed_covariates=(),
        lagged_targets=("x",),
    )


def _config(provider: RawAgentProvider) -> RawAgentConfig:
    return RawAgentConfig(
        provider=provider,
        model="frontier-test-model",
        repetition=2,
        reasoning_effort="high",
        max_tool_calls=7,
        max_output_tokens=4096,
        max_attempts=1,
    )


def _fitted_model() -> RawAgentFittedModel:
    return RawAgentFittedModel(
        candidate=_proposal(),
        fitted_parameters=(RawAgentFittedParameter(name="k", value=0.25),),
        fit_method_summary="Bounded least squares on train trajectories.",
    )


def test_fitted_model_contract_requires_exact_finite_values() -> None:
    assert _fitted_model().fitted_parameter_values == {"k": 0.25}
    with pytest.raises(ValidationError, match=r"missing=\['k'\]"):
        RawAgentFittedModel(
            candidate=_proposal(),
            fitted_parameters=(),
            fit_method_summary="Train fit.",
        )
    result = RawAgentFittedModel(
        candidate=_proposal(),
        fitted_parameters=(RawAgentFittedParameter(name="k", value=3.0),),
        fit_method_summary="Train fit.",
    )
    assert result.fitted_parameter_values == {"k": 3.0}


class _Adapter:
    def __init__(self) -> None:
        self.calls = 0
        self.repairs = 0

    def call(self, **_: Any) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(
            parsed=_proposal(),
            raw_response={"provider": "fake"},
            response_id="response-1",
            usage=None,
            tool_call_count=3,
        )

    def repair(self, **_: Any) -> SimpleNamespace:
        self.repairs += 1
        return SimpleNamespace(
            parsed=_proposal(),
            raw_response={"provider": "fake-repair"},
            response_id="repair-1",
            usage=None,
            tool_call_count=0,
        )


def test_agent_result_is_checkpointed_and_cacheable(tmp_path: Path) -> None:
    adapter = _Adapter()
    inputs = _inputs(tmp_path)
    output = tmp_path / "run"

    first = run_raw_data_agent(
        config=_config(RawAgentProvider.OPENAI),
        inputs=inputs,
        output_directory=output,
        adapter=adapter,
    )
    second = run_raw_data_agent(
        config=_config(RawAgentProvider.OPENAI),
        inputs=inputs,
        output_directory=output,
        adapter=adapter,
    )

    assert adapter.calls == 1
    assert first == second
    assert first.candidate.observation_mappings[0].channel == "x"
    assert first.requested_max_tool_calls == 7
    assert first.provider_reported_max_tool_calls is None
    assert first.tool_call_limit_exceeded is False
    assert (output / "agent_result.json").is_file()
    assert len(list((output / "cache").glob("*.json"))) == 1


def test_contract_repair_is_diagnostics_only_and_checkpointed(tmp_path: Path) -> None:
    adapter = _Adapter()
    inputs = _inputs(tmp_path)
    config = _config(RawAgentProvider.OPENAI)
    output = tmp_path / "run"
    original = run_raw_data_agent(
        config=config,
        inputs=inputs,
        output_directory=output,
        adapter=adapter,
    )

    first = repair_raw_data_agent_candidate(
        config=config,
        inputs=inputs,
        original=original,
        diagnostics="INVALID_INITIALIZATION_SYMBOL at initial_condition:z",
        repair_index=1,
        output_directory=output,
        adapter=adapter,
    )
    second = repair_raw_data_agent_candidate(
        config=config,
        inputs=inputs,
        original=original,
        diagnostics="INVALID_INITIALIZATION_SYMBOL at initial_condition:z",
        repair_index=1,
        output_directory=output,
        adapter=adapter,
    )

    assert adapter.repairs == 1
    assert first == second
    assert first.tool_call_count == 0
    assert (output / "repair_result_01.json").is_file()


def test_contract_repair_ignores_mismatched_legacy_checkpoint(
    tmp_path: Path,
) -> None:
    adapter = _Adapter()
    inputs = _inputs(tmp_path)
    config = _config(RawAgentProvider.OPENAI)
    output = tmp_path / "run"
    original = run_raw_data_agent(
        config=config,
        inputs=inputs,
        output_directory=output,
        adapter=adapter,
    )
    (output / "repair_result_01.json").write_text(
        original.model_dump_json(), encoding="utf-8"
    )

    repaired = repair_raw_data_agent_candidate(
        config=config,
        inputs=inputs,
        original=original,
        diagnostics="new deterministic diagnostic",
        repair_index=1,
        output_directory=output,
        adapter=adapter,
    )

    assert repaired.request_hash != original.request_hash
    assert adapter.repairs == 1
    assert len(list(output.glob("repair_result_01_*.json"))) == 1


class _OpenAIFiles:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.deleted: list[str] = []

    def create(self, *, file: Any, purpose: str) -> SimpleNamespace:
        assert purpose == "user_data"
        item = SimpleNamespace(id=f"file-{len(self.created)}")
        self.created.append(Path(file.name).name)
        return item

    def delete(self, file_id: str) -> None:
        self.deleted.append(file_id)


class _OpenAIResponses:
    def __init__(self, parsed: object | None = None) -> None:
        self.kwargs: dict[str, Any] = {}
        self.parsed = parsed or _proposal()

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(
            id="response-openai",
            status="completed",
            output_parsed=self.parsed,
            output=[SimpleNamespace(type="code_interpreter_call")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
            model_dump=lambda **_: {"id": "response-openai"},
        )


class _OpenAIResponsesWithNonterminalToolRecord(_OpenAIResponses):
    def parse(self, **kwargs: Any) -> SimpleNamespace:
        response = super().parse(**kwargs)
        response.output = [
            SimpleNamespace(type="code_interpreter_call", status="completed"),
            SimpleNamespace(type="code_interpreter_call", status="interpreting"),
        ]
        return response


def test_openai_adapter_attaches_files_and_bounds_tools(tmp_path: Path) -> None:
    files = _OpenAIFiles()
    responses = _OpenAIResponses()
    client = SimpleNamespace(files=files, responses=responses)

    result = OpenAIRawDataAgent(client).call(
        config=_config(RawAgentProvider.OPENAI),
        inputs=_inputs(tmp_path),
        system_prompt="system",
        user_prompt="user",
    )

    assert files.created == ["train.csv", "validation.csv"]
    assert files.deleted == ["file-0", "file-1"]
    assert responses.kwargs["max_tool_calls"] == 7
    assert responses.kwargs["reasoning"] == {"effort": "high"}
    assert responses.kwargs["store"] is False
    assert responses.kwargs["tools"][0]["type"] == "code_interpreter"
    assert result.tool_call_count == 1
    assert result.usage is not None and result.usage.total_tokens == 30

    OpenAIRawDataAgent(client).repair(
        config=_config(RawAgentProvider.OPENAI),
        system_prompt="repair-system",
        user_prompt="repair-user",
    )
    assert "tools" not in responses.kwargs
    assert responses.kwargs["input"] == "repair-user"


def test_openai_adapter_excludes_nonterminal_tool_records(tmp_path: Path) -> None:
    files = _OpenAIFiles()
    responses = _OpenAIResponsesWithNonterminalToolRecord()
    client = SimpleNamespace(files=files, responses=responses)

    result = OpenAIRawDataAgent(client).call(
        config=_config(RawAgentProvider.OPENAI),
        inputs=_inputs(tmp_path),
        system_prompt="system",
        user_prompt="user",
    )

    assert result.tool_call_count == 1


def test_openai_adapter_returns_agent_fitted_parameter_vector(
    tmp_path: Path,
) -> None:
    files = _OpenAIFiles()
    responses = _OpenAIResponses(_fitted_model())
    client = SimpleNamespace(files=files, responses=responses)
    config = _config(RawAgentProvider.OPENAI).model_copy(
        update={"output_contract": RawAgentOutputContract.FITTED_MODEL}
    )

    result = OpenAIRawDataAgent(client).call(
        config=config,
        inputs=_inputs(tmp_path),
        system_prompt="system",
        user_prompt="user",
    )

    assert responses.kwargs["text_format"] is RawAgentFittedModel
    assert result.parsed == _proposal()
    assert result.fitted_parameter_values == {"k": 0.25}
    assert result.fit_method_summary is not None


class _GeminiFiles:
    def __init__(self) -> None:
        self.uploaded: list[str] = []
        self.deleted: list[str] = []

    def upload(self, *, file: str) -> SimpleNamespace:
        name = f"files/{len(self.uploaded)}"
        self.uploaded.append(Path(file).name)
        return SimpleNamespace(name=name)

    def delete(self, *, name: str) -> None:
        self.deleted.append(name)


class _GeminiModels:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(
            text=_proposal().model_dump_json(),
            response_id="response-gemini",
            usage_metadata=SimpleNamespace(
                prompt_token_count=11,
                candidates_token_count=22,
                total_token_count=33,
            ),
            candidates=[],
            model_dump=lambda **_: {"response_id": "response-gemini"},
        )


def test_gemini_adapter_uses_code_execution_and_seed(tmp_path: Path) -> None:
    files = _GeminiFiles()
    models = _GeminiModels()
    client = SimpleNamespace(files=files, models=models)

    result = GeminiRawDataAgent(client).call(
        config=_config(RawAgentProvider.GEMINI),
        inputs=_inputs(tmp_path),
        system_prompt="system",
        user_prompt="user",
    )

    assert files.uploaded == ["train.csv", "validation.csv"]
    assert files.deleted == ["files/0", "files/1"]
    assert models.kwargs["config"]["tools"] == [{"code_execution": {}}]
    assert models.kwargs["config"]["seed"] == 2
    assert result.usage is not None and result.usage.total_tokens == 33

    GeminiRawDataAgent(client).repair(
        config=_config(RawAgentProvider.GEMINI),
        system_prompt="repair-system",
        user_prompt="repair-user",
    )
    assert "tools" not in models.kwargs["config"]
    assert models.kwargs["config"]["seed"] == 10_002


def test_request_hash_covers_data_and_repetition(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    config = _config(RawAgentProvider.OPENAI)
    system = raw_agent_system_prompt(inputs)
    user = raw_agent_user_prompt(inputs, config.repetition)
    first = raw_agent_request_hash(config, inputs, system, user)

    inputs.train_path.write_text(
        "trajectory_id,time,x,u\na,0,3,0\n", encoding="utf-8"
    )
    changed = raw_agent_request_hash(config, inputs, system, user)

    assert first != changed
    assert "No test data" in system
    assert "glucose" not in system.lower()
    assert "validation.csv" in user


def test_fitted_model_prompt_requires_train_only_values(tmp_path: Path) -> None:
    prompt = raw_agent_system_prompt(
        _inputs(tmp_path), RawAgentOutputContract.FITTED_MODEL
    )

    assert "using train only" in prompt
    assert "without parameter optimization" in prompt
    assert "RawAgentFittedModel" in prompt


def test_fitted_model_evaluation_does_not_optimize_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal_payload = _proposal().model_dump(mode="json")
    proposal_payload["parameters"].append(
        {"name": "unused_removed", "bounds": {"lower": 0.0, "upper": 1.0}}
    )
    proposal_with_unused = ProposerCandidateV2.model_validate(proposal_payload)
    original_candidate = enrich_proposal_v2(proposal_with_unused, ("x",))
    candidate = enrich_proposal_v2(_proposal(), ("x",))
    artifact = RawAgentArtifact(
        request_hash="request",
        provider=RawAgentProvider.OPENAI,
        model="frontier",
        repetition=0,
        latency_seconds=1.0,
        tool_call_count=1,
        compact_candidate=proposal_with_unused,
        candidate=original_candidate,
        output_contract=RawAgentOutputContract.FITTED_MODEL,
        # An unused declaration can be removed by deterministic normalization;
        # only the compiled model's exact vector may reach simulation.
        fitted_parameter_values={"k": 0.25, "unused_removed": 0.75},
        fit_method_summary="Train-only fit.",
        raw_response_sha256="digest",
    )
    compiled = SimpleNamespace(
        validated=SimpleNamespace(candidate=candidate, warnings=())
    )
    monkeypatch.setattr(
        "autoformalism.baselines.raw_data_agent.repair_protected_declarations",
        lambda value, context: (candidate, ("removed unused parameter",)),
    )
    monkeypatch.setattr(
        "autoformalism.baselines.raw_data_agent.compile_candidate",
        lambda value, context: compiled,
    )

    class _Scaler:
        scales: ClassVar = {
            "target:x": SimpleNamespace(standard_deviation=2.0)
        }

        def fit(self, split):
            return self

    monkeypatch.setattr(
        "autoformalism.baselines.raw_data_agent.TrainingScaler", _Scaler
    )
    calls = []

    def fake_evaluate(model, split, **kwargs):
        calls.append(kwargs)
        return {}, EvaluationMetrics(0.5, {"x": 0.5})

    monkeypatch.setattr(
        "autoformalism.baselines.raw_data_agent.evaluate_fitted_candidate",
        fake_evaluate,
    )
    dataset = SimpleNamespace(train=object(), validation=object())
    context = SimpleNamespace(targets=("x",))

    _, _, training, validation, _ = evaluate_raw_agent_fitted_model(
        artifact=artifact,
        dataset=dataset,
        context=context,
    )

    assert training.normalized_mse == validation.normalized_mse == 0.5
    assert len(calls) == 2
    assert all(call["global_parameters"] == {"k": 0.25} for call in calls)
    assert all(call["fit_trajectory_initial_conditions"] is False for call in calls)
