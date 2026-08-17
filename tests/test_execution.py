"""User-facing dry-run, mock execution, resume, and summary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoformalism.execution import (
    ExecutionArguments,
    _latent_ablation_prompt,
    _numeric_declared_channels,
    _prediction_protocol_prompt,
    _symbol_contract,
    arguments_from_namespace,
    build_experiment_parser,
    execute,
)
from autoformalism.expressions import ValidationContext
from autoformalism.llm import OllamaThinking


def _arguments(
    tmp_path: Path,
    *,
    dry_run: bool,
    resume: bool = False,
    development_only: bool = False,
):
    return ExecutionArguments(
        data_root=tmp_path,
        benchmark_id="synthetic",
        tier="easy",
        seed=31,
        proposer_model=None,
        judge_model=None,
        iteration_budget=1,
        beam_size=1,
        output_root=tmp_path / "runs",
        resume=resume,
        dry_run=dry_run,
        mock_llm=True,
        use_clean_observations=False,
        development_only=development_only,
    )


def test_dry_run_creates_no_output_and_exposes_no_test_fingerprint(
    tmp_path: Path,
) -> None:
    result = execute(_arguments(tmp_path, dry_run=True))

    assert result["status"] == "dry_run"
    assert set(result["split_fingerprints"]) == {"train", "validation"}
    assert not (tmp_path / "runs").exists()


def test_mock_execution_and_resume_are_idempotent(tmp_path: Path) -> None:
    first = execute(_arguments(tmp_path, dry_run=False))
    resumed = execute(_arguments(tmp_path, dry_run=False, resume=True))

    assert first == resumed
    assert first["status"] == "complete"
    assert first["test_failed_trajectories"] == []


def test_mock_development_only_execution_omits_test_metrics(tmp_path: Path) -> None:
    result = execute(
        _arguments(tmp_path, dry_run=False, development_only=True)
    )

    assert result["status"] == "complete"
    assert result["evaluation_stage"] == "development_selection_frozen"
    assert not any(key.startswith("test_") for key in result)


def test_cli_timeout_defaults_to_900_and_accepts_override() -> None:
    parser = build_experiment_parser(description="test")

    default = arguments_from_namespace(parser.parse_args(["--mock-llm", "--dry-run"]))
    changed = arguments_from_namespace(
        parser.parse_args(
            [
                "--mock-llm",
                "--dry-run",
                "--llm-timeout-seconds",
                "1200",
            ]
        )
    )

    assert default.llm_timeout_seconds == 900.0
    assert changed.llm_timeout_seconds == 1200.0
    assert default.llm_max_output_tokens == 2048
    assert default.fit_starts == 1
    assert default.fit_max_nfev == 50
    assert default.fit_timeout_seconds == 300.0
    assert default.final_fit_max_nfev == 150
    assert default.final_fit_timeout_seconds == 300.0
    assert default.forbid_latent_states is False
    assert default.use_derivative_fit_fast_path is True
    assert default.llm_cache_only is False
    assert default.llm_cache_root is None
    assert default.development_only is False
    assert default.ollama_base_url == "http://127.0.0.1:11434"
    assert default.ollama_thinking is OllamaThinking.AUTO
    assert default.ollama_temperature == 0.0
    assert default.ollama_seed is None
    assert default.stagnation_iterations is None
    assert default.selection_policy == "validation_only"
    assert default.judge_weight == 0.25
    assert default.judge_score_epsilon == 0.05


def test_cli_accepts_weighted_selection_and_rejects_no_judge() -> None:
    parser = build_experiment_parser(description="test")
    arguments = arguments_from_namespace(
        parser.parse_args(
            [
                "--mock-llm",
                "--dry-run",
                "--selection-policy",
                "normalized_weighted_sum",
                "--judge-weight",
                "0.5",
            ]
        )
    )

    assert arguments.selection_policy == "normalized_weighted_sum"
    assert arguments.judge_weight == 0.5

    with pytest.raises(SystemExit, match="requires judge"):
        arguments_from_namespace(
            parser.parse_args(
                [
                    "--mock-llm",
                    "--dry-run",
                    "--selection-policy",
                    "normalized_weighted_sum",
                    "--no-judge",
                ]
            )
        )


def test_cli_accepts_development_only_mode() -> None:
    parser = build_experiment_parser(description="test")
    arguments = arguments_from_namespace(
        parser.parse_args(["--mock-llm", "--dry-run", "--development-only"])
    )

    assert arguments.development_only is True


def test_cli_accepts_custom_ollama_endpoint() -> None:
    parser = build_experiment_parser(description="test")
    arguments = arguments_from_namespace(
        parser.parse_args(
            [
                "--mock-llm",
                "--dry-run",
                "--ollama-base-url",
                "http://127.0.0.1:23456",
            ]
        )
    )

    assert arguments.ollama_base_url == "http://127.0.0.1:23456"


def test_cli_accepts_explicit_ollama_thinking_level() -> None:
    parser = build_experiment_parser(description="test")
    arguments = arguments_from_namespace(
        parser.parse_args(
            ["--mock-llm", "--dry-run", "--ollama-thinking", "medium"]
        )
    )

    assert arguments.ollama_thinking is OllamaThinking.MEDIUM


def test_cli_accepts_ollama_sampling_and_stagnation_controls() -> None:
    parser = build_experiment_parser(description="test")
    arguments = arguments_from_namespace(
        parser.parse_args(
            [
                "--mock-llm",
                "--dry-run",
                "--ollama-temperature",
                "0.2",
                "--ollama-seed",
                "4",
                "--stagnation-iterations",
                "12",
            ]
        )
    )

    assert arguments.ollama_temperature == 0.2
    assert arguments.ollama_seed == 4
    assert arguments.stagnation_iterations == 12


def test_cli_accepts_shared_cache_only_mode(tmp_path: Path) -> None:
    parser = build_experiment_parser(description="test")
    arguments = arguments_from_namespace(
        parser.parse_args(
            [
                "--dry-run",
                "--proposer-model",
                "openai:example",
                "--llm-cache-only",
                "--llm-cache-root",
                str(tmp_path / "resolved"),
            ]
        )
    )

    assert arguments.llm_cache_only is True
    assert arguments.llm_cache_root == (tmp_path / "resolved").resolve()


def test_cli_accepts_forbid_latent_states() -> None:
    parser = build_experiment_parser(description="test")
    arguments = arguments_from_namespace(
        parser.parse_args(["--mock-llm", "--dry-run", "--forbid-latent-states"])
    )

    assert arguments.forbid_latent_states is True
    assert "do not declare latent dynamic states" in _latent_ablation_prompt(arguments)


def test_cli_can_disable_derivative_fit_fast_path() -> None:
    parser = build_experiment_parser(description="test")
    arguments = arguments_from_namespace(
        parser.parse_args(
            [
                "--mock-llm",
                "--dry-run",
                "--disable-derivative-fit-fast-path",
            ]
        )
    )

    assert arguments.use_derivative_fit_fast_path is False


def test_cli_rejects_nonpositive_timeout() -> None:
    parser = build_experiment_parser(description="test")
    namespace = parser.parse_args(
        ["--mock-llm", "--dry-run", "--llm-timeout-seconds", "0"]
    )

    with pytest.raises(SystemExit, match="must be positive"):
        arguments_from_namespace(namespace)


def test_cli_accepts_and_validates_max_output_tokens() -> None:
    parser = build_experiment_parser(description="test")
    accepted = arguments_from_namespace(
        parser.parse_args(
            ["--mock-llm", "--dry-run", "--llm-max-output-tokens", "1024"]
        )
    )
    rejected = parser.parse_args(
        ["--mock-llm", "--dry-run", "--llm-max-output-tokens", "127"]
    )

    assert accepted.llm_max_output_tokens == 1024
    with pytest.raises(SystemExit, match="at least 128"):
        arguments_from_namespace(rejected)


def test_symbol_contract_uses_exact_runtime_identifiers() -> None:
    contract = _symbol_contract(
        ValidationContext(
            targets=("Gp",),
            auxiliaries=("EGP", "Gt"),
            external_inputs=("meal_event_g",),
            fixed_covariates=("body_weight_kg",),
        )
    )

    assert "Target channels (generate at each predicted slot): Gp" in contract
    assert "Causally available one-slot-lagged targets: (none)" in contract
    assert "Supplied auxiliary trajectories: EGP, Gt" in contract
    assert "External input trajectories: meal_event_g" in contract
    assert "Fixed numeric covariates: body_weight_kg" in contract
    assert "Time symbol in expressions: t" in contract
    assert "meal_amount" not in contract
    assert "deliberately promoted to modeled states" in contract
    assert "Never redeclare external inputs or fixed covariates" in contract


def test_numeric_declared_channels_omit_structured_metadata() -> None:
    bounds = {"numeric_input": (0.0, 1.0), "numeric_covariate": (3.0, 3.0)}

    assert _numeric_declared_channels(("numeric_input", "input_schedule"), bounds) == (
        "numeric_input",
    )
    assert _numeric_declared_channels(
        ("numeric_covariate", "meal_schedule"), bounds
    ) == ("numeric_covariate",)


def test_prediction_protocol_prompt_overrides_legacy_wording() -> None:
    prompt = _prediction_protocol_prompt(
        ValidationContext(targets=("Gp",), lagged_targets=("Gp",))
    )

    assert "one-step-ahead" in prompt
    assert "strictly prior" in prompt
    assert "supersedes" in prompt
