"""Leakage, fitting, and adapter tests for experiment baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from autoformalism.baselines.core import target_scales
from autoformalism.baselines.d3 import _project_d3_structure, run_d3_no_tools
from autoformalism.baselines.models import BaselineConfig
from autoformalism.baselines.pysr import _restore_feature_names
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


def test_pysr_restores_dataset_names_that_conflict_with_sympy() -> None:
    expression = _restore_feature_names(
        "af_x0 + af_x1**2 + tanh(af_x10)",
        ("af_x0", "af_x1", "af_x10"),
        ("E", "Gp", "body_weight_kg"),
    )

    assert expression == "E + Gp**2 + tanh(body_weight_kg)"


def test_d3_request_excludes_test_and_disables_external_tools(
    tmp_path: Path,
) -> None:
    bridge = (
        "import json,sys;"
        "p=json.load(open(sys.argv[1]));"
        "assert p['external_tools_enabled'] is False;"
        "assert p['candidate_submission_enabled'] is True;"
        "assert 'test' not in p;"
        "json.dump({'equations':{'x':'-0.4*x'}},open(sys.argv[2],'w'))"
    )
    result = run_d3_no_tools(
        BaselineConfig(method="d3_no_tools"),
        _dataset(),
        lambda: _split(SplitName.TEST),
        ValidationContext(targets=("x",), lagged_targets=("x",)),
        task_prompt="Discover x dynamics.",
        command=(sys.executable, "-c", bridge),
        work_directory=tmp_path,
    )

    request = json.loads((tmp_path / "d3_request.json").read_text())
    assert request["external_tools_enabled"] is False
    assert result.test_normalized_mse < 1e-8


def test_safe_d3_iterates_without_judge_or_test_feedback(tmp_path: Path) -> None:
    proposal = CandidateModel.model_validate(
        {
            "candidate_id": "d3_linear",
            "parent_candidate_id": None,
            "change_summary": "Observed-state D3 proposal.",
            "states": [{"name": "x", "kind": "observed"}],
            "state_equations": [{"state": "x", "rhs": "-k * x"}],
            "observation_mappings": [{"channel": "x", "expression": "x"}],
            "parameters": [
                {
                    "name": "k",
                    "scope": "global",
                    "bounds": {"lower": 0.01, "upper": 1.0},
                    "initialization_range": {"lower": 0.1, "upper": 0.8},
                }
            ],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "x"}
            ],
        }
    )
    client = MockLLMClient(proposer_responses=[proposal, proposal])
    test_calls = 0

    def load_test() -> DatasetSplit:
        nonlocal test_calls
        test_calls += 1
        return _split(SplitName.TEST)

    result = run_d3_no_tools(
        BaselineConfig(method="d3_no_tools", d3_generations=2),
        _dataset(),
        load_test,
        ValidationContext(targets=("x",), lagged_targets=("x",)),
        task_prompt="Discover x dynamics.",
        llm_client=client,
        work_directory=tmp_path,
    )

    assert [call["role"] for call in client.calls] == ["proposer", "proposer"]
    assert test_calls == 1
    assert result.test_normalized_mse < 1e-8
    checkpoint = json.loads((tmp_path / "d3_checkpoint.json").read_text())
    assert "test" not in json.dumps(checkpoint).lower()

    resumed = run_d3_no_tools(
        BaselineConfig(method="d3_no_tools", d3_generations=2),
        _dataset(),
        load_test,
        ValidationContext(targets=("x",), lagged_targets=("x",)),
        task_prompt="Discover x dynamics.",
        llm_client=client,
        work_directory=tmp_path,
    )

    assert [call["role"] for call in client.calls] == ["proposer", "proposer"]
    assert resumed.validation_normalized_mse == result.validation_normalized_mse


def test_d3_projects_supplied_auxiliary_states_to_forcing() -> None:
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

    projected, repairs = _project_d3_structure(
        candidate,
        ValidationContext(
            targets=("x",), auxiliaries=("u",), lagged_targets=("x",)
        ),
    )

    assert [state.name for state in projected.states] == ["x"]
    assert [parameter.name for parameter in projected.parameters] == ["k"]
    assert "u" in repairs[0]
