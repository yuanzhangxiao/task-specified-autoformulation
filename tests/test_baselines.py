"""Leakage, fitting, and adapter tests for experiment baselines."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.baselines.core import target_scales
from autoformalism.baselines.d3 import _d3_system_prompt, run_d3_native_no_tools
from autoformalism.baselines.d3_native import (
    NativeD3Error,
    fit_native_d3,
    validate_native_candidate,
)
from autoformalism.baselines.models import BaselineConfig, BaselineRunStatus
from autoformalism.baselines.pysr import _restore_feature_names, fit_pysr
from autoformalism.baselines.runner import _select_pysr, run_baseline
from autoformalism.data import (
    DatasetSplit,
    DevelopmentDataset,
    SplitName,
    Trajectory,
)
from autoformalism.data.models import TierRoles
from autoformalism.expressions import ValidationContext
from autoformalism.llm import MockLLMClient
from autoformalism.schemas import CandidateModel
from scripts.run_baseline import build_parser


def _split(name: SplitName, rate: float = 0.4) -> DatasetSplit:
    time = np.linspace(0.0, 2.0, 41)
    target = np.exp(-rate * time)
    trajectory = Trajectory(
        trajectory_id=name.value,
        time=time,
        targets={"x": target},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={"x": -rate * target},
    )
    return DatasetSplit(name, (trajectory,), f"{name.value}-fingerprint")


def _dataset() -> DevelopmentDataset:
    return DevelopmentDataset(
        "synthetic_baseline",
        "easy",
        TierRoles(targets=("x",)),
        _split(SplitName.TRAIN),
        _split(SplitName.VALIDATION),
    )


def test_persistence_opens_test_exactly_once() -> None:
    calls = 0

    def load_test() -> DatasetSplit:
        nonlocal calls
        calls += 1
        return _split(SplitName.TEST)

    result = run_baseline(
        BaselineConfig(method="persistence"),
        _dataset(),
        load_test,
        ValidationContext(targets=("x",), lagged_targets=("x",)),
    )

    assert calls == 1
    assert result.test_normalized_mse > 0.0


def test_sindy_recovers_derivative_and_evaluates_test_once() -> None:
    calls = 0

    def load_test() -> DatasetSplit:
        nonlocal calls
        calls += 1
        return _split(SplitName.TEST)

    result = run_baseline(
        BaselineConfig(method="sindy", sindy_thresholds=(1e-3, 1e-2)),
        _dataset(),
        load_test,
        ValidationContext(targets=("x",), lagged_targets=("x",)),
    )

    assert calls == 1
    assert "x" in result.equations["x"]
    assert result.test_normalized_mse < 1e-8


def test_llm_feature_sindy_uses_one_proposer_call() -> None:
    proposal = CandidateModel.model_validate(
        {
            "candidate_id": "features",
            "parent_candidate_id": None,
            "change_summary": "One-shot feature proposal.",
            "states": [{"name": "x", "kind": "observed"}],
            "processes": [{"name": "x_squared", "expression": "x * x"}],
            "state_equations": [{"state": "x", "rhs": "0"}],
            "observation_mappings": [{"channel": "x", "expression": "x"}],
            "parameters": [],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "x"}
            ],
        }
    )
    client = MockLLMClient(proposer_responses=[proposal])
    result = run_baseline(
        BaselineConfig(method="llm_feature_sindy", sindy_thresholds=(1e-3,)),
        _dataset(),
        lambda: _split(SplitName.TEST),
        ValidationContext(targets=("x",), lagged_targets=("x",)),
        llm_client=client,
        proposer_prompt="Discover the dynamics.",
    )

    assert [call["role"] for call in client.calls] == ["proposer"]
    assert result.selected_hyperparameters["proposed_feature_count"] == 1


def test_pysr_pareto_expression_is_selected_on_validation_rollout() -> None:
    dataset = _dataset()
    context = ValidationContext(targets=("x",), lagged_targets=("x",))

    selected = _select_pysr(
        {"x": ("0", "-0.4 * x")},
        dataset,
        context,
        target_scales(dataset.train, context.targets),
    )

    assert selected == {"x": "-0.4 * x"}


def test_pysr_cli_options_are_propagated() -> None:
    args = build_parser().parse_args(
        [
            "--method",
            "pysr",
            "--pysr-iterations",
            "2",
            "--maximum-expression-size",
            "12",
        ]
    )

    config = BaselineConfig(
        method=args.method,
        pysr_iterations=args.pysr_iterations,
        maximum_expression_size=args.maximum_expression_size,
    )

    assert config.pysr_iterations == 2
    assert config.maximum_expression_size == 12


def test_pysr_receives_native_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRegressor:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.equations_ = {"sympy_format": ["-0.4*af_x0"]}

        def fit(self, x, y, *, variable_names):
            captured["variable_names"] = variable_names

    monkeypatch.setitem(
        sys.modules, "pysr", SimpleNamespace(PySRRegressor=FakeRegressor)
    )

    equations, metadata = fit_pysr(
        np.ones((4, 1)),
        np.ones((4, 1)),
        ("x",),
        ("x",),
        iterations=2,
        seed=0,
        maximum_expression_size=10,
        timeout_seconds=123.0,
    )

    assert captured["timeout_in_seconds"] == 123.0
    assert equations == {"x": ("-0.4*x",)}
    assert metadata["x"]["timeout_seconds"] == 123.0


def test_pysr_restores_dataset_names_that_conflict_with_sympy() -> None:
    expression = _restore_feature_names(
        "af_x0 + af_x1**2 + tanh(af_x10)",
        ("af_x0", "af_x1", "af_x10"),
        ("E", "Gp", "body_weight_kg"),
    )

    assert expression == "E + Gp**2 + tanh(body_weight_kg)"


def test_native_d3_iterates_without_judge_or_test_feedback(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    proposal = CandidateModel.model_validate(
        {
            "candidate_id": "d3_linear",
            "parent_candidate_id": None,
            "change_summary": "Observed-state D3 proposal.",
            "states": [{"name": "x", "kind": "observed"}],
            "state_equations": [
                {"state": "x", "rhs": "(exp(-0.4 * 0.05) - 1) * x"}
            ],
            "observation_mappings": [{"channel": "x", "expression": "x"}],
            "parameters": [],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "x"}
            ],
        }
    )
    client = MockLLMClient(proposer_responses=[proposal])
    test_calls = 0

    def load_test() -> DatasetSplit:
        nonlocal test_calls
        test_calls += 1
        return _split(SplitName.TEST)

    result = run_d3_native_no_tools(
        BaselineConfig(method="d3_native_no_tools", d3_generations=1),
        _dataset(),
        load_test,
        ValidationContext(targets=("x",), lagged_targets=("x",)),
        task_prompt="Discover x dynamics.",
        llm_client=client,
        work_directory=tmp_path,
    )

    assert [call["role"] for call in client.calls] == ["proposer"]
    assert test_calls == 1
    assert result.test_normalized_mse < 1e-5
    assert result.selected_hyperparameters["parameter_fitting"] == "pytorch_adam"
    assert result.selected_hyperparameters["state_update"] == (
        "teacher_forced_forward_euler"
    )
    checkpoint = json.loads((tmp_path / "d3_checkpoint.json").read_text())
    assert "test" not in json.dumps(checkpoint).lower()

    resumed = run_d3_native_no_tools(
        BaselineConfig(method="d3_native_no_tools", d3_generations=1),
        _dataset(),
        load_test,
        ValidationContext(targets=("x",), lagged_targets=("x",)),
        task_prompt="Discover x dynamics.",
        llm_client=client,
        work_directory=tmp_path,
    )

    assert [call["role"] for call in client.calls] == ["proposer"]
    assert resumed.validation_normalized_mse == result.validation_normalized_mse


def test_native_d3_requires_all_observed_channels_as_states() -> None:
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "overdeclared",
            "parent_candidate_id": None,
            "change_summary": "Auxiliary declared as a state.",
            "states": [
                {"name": "x", "kind": "observed"},
                {"name": "u", "kind": "observed"},
            ],
            "state_equations": [
                {"state": "x", "rhs": "u - k * x"},
                {"state": "u", "rhs": "-ku * u"},
            ],
            "observation_mappings": [{"channel": "x", "expression": "x"}],
            "parameters": [
                {
                    "name": name,
                    "scope": "global",
                    "bounds": {"lower": 0.01, "upper": 1.0},
                    "initialization_range": {"lower": 0.1, "upper": 0.8},
                }
                for name in ("k", "ku")
            ],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "x"},
                {"state": "u", "scope": "global", "expression": "u"},
            ],
        }
    )

    validate_native_candidate(
        candidate,
        ("x", "u"),
        (),
    )

    target_only = candidate.model_copy(
        update={
            "states": candidate.states[:1],
            "state_equations": candidate.state_equations[:1],
        }
    )
    with np.testing.assert_raises(NativeD3Error):
        validate_native_candidate(target_only, ("x", "u"), ())


def test_native_d3_prompt_describes_interface_without_latent_prohibition() -> None:
    prompt = _d3_system_prompt(
        "Discover the system.",
        ValidationContext(
            targets=("x",), auxiliaries=("u",), lagged_targets=("x",)
        ),
    )

    assert "fixed state vector" in prompt
    assert "need not appear in other equations" in prompt
    assert "algebraic features" in prompt
    assert "no latent states" not in prompt


def test_native_d3_broadcasts_constant_state_equation() -> None:
    pytest.importorskip("torch")
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "constant_state",
            "parent_candidate_id": None,
            "change_summary": "Constant derivative output.",
            "states": [{"name": "x", "kind": "observed"}],
            "state_equations": [{"state": "x", "rhs": "0"}],
            "observation_mappings": [{"channel": "x", "expression": "x"}],
            "parameters": [],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "x"}
            ],
        }
    )

    fit = fit_native_d3(
        candidate,
        _dataset().train,
        _dataset().validation,
        targets=("x",),
        seed=0,
    )

    assert np.isfinite(fit.validation_mse)


def test_baseline_run_status_schema_supports_timeout() -> None:
    status = BaselineRunStatus(
        status="timed_out",
        elapsed_wall_seconds=30.1,
        wall_timeout_seconds=30.0,
        error="baseline wall-clock limit reached",
    )

    assert status.status == "timed_out"


def test_baseline_cli_hard_timeout_writes_status(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_baseline.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-root",
            str(tmp_path),
            "--benchmark-id",
            "missing",
            "--method",
            "persistence",
            "--output-root",
            str(tmp_path / "outputs"),
            "--wall-timeout-seconds",
            "0.000001",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    status_path = (
        tmp_path
        / "outputs"
        / "persistence"
        / "missing_easy_seed0"
        / "run_status.json"
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))

    assert completed.returncode == 124
    assert status["status"] == "timed_out"
