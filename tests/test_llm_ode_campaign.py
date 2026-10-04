"""Supplying Phase-B data to LLM-ODE's search without altering it.

Upstream builds its own data by integrating a ground-truth equation, assumes
one trajectory, and hands its selector the sealed test trajectory. None of
those hold here, so this module's job is to accommodate the benchmark while
leaving the search, the derivative order and the selection rule alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.rebuttal.llm_ode_campaign import (
    cell_arrays,
    finite_difference,
    observed_channels,
    select_system,
    specification_block,
)
from autoformalism.schemas.candidate import CandidateModel


def _trajectory(identifier: str, count: int = 12, *, step: float = 0.1) -> Trajectory:
    time = np.arange(count) * step
    return Trajectory(
        trajectory_id=identifier,
        time=time,
        targets={"y": time**2},
        auxiliaries={"u": 2.0 * time},
        external_inputs={},
        fixed_covariates={},
        derivatives={},
    )


def _split(*trajectories: Trajectory) -> DatasetSplit:
    return DatasetSplit(SplitName.TRAIN, trajectories, "train")


def test_the_difference_is_fourth_order_as_upstream_requests() -> None:
    """Upstream asks findiff for acc=4, which is exact on a quartic."""
    time = np.linspace(0.0, 2.0, 40)
    values = np.column_stack([time**4 - 3 * time**2, 2 * time**3])
    exact = np.column_stack([4 * time**3 - 6 * time, 6 * time**2])
    got = finite_difference(values, float(time[1] - time[0]))
    assert np.max(np.abs(got - exact)) < 1e-9
    # a second-order rule would not be exact here, so this distinguishes them
    second_order = np.gradient(values, float(time[1] - time[0]), axis=0)
    assert np.max(np.abs(second_order - exact)) > 1e-4


def test_each_trajectory_is_differentiated_before_the_rows_are_stacked() -> None:
    """Differentiating across a join would invent a derivative at the seam."""
    split = _split(_trajectory("a"), _trajectory("b"))
    arrays = cell_arrays(split)
    assert arrays.states.shape == (24, 2)
    assert arrays.derivatives.shape == (24, 2)
    assert arrays.trajectory_bounds == ((0, 12), (12, 24))
    assert arrays.channels == ("y", "u")
    # the seam carries each trajectory's own endpoint derivative, not a blend
    single = cell_arrays(_split(_trajectory("a")))
    assert np.allclose(arrays.derivatives[:12], single.derivatives)
    assert np.allclose(arrays.derivatives[12:], single.derivatives)


def test_a_nonuniform_grid_is_refused() -> None:
    """The stencil assumes a fixed step; silently applying it would be wrong."""
    time = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    trajectory = Trajectory(
        trajectory_id="a",
        time=time,
        targets={"y": time**2},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={},
    )
    with pytest.raises(ValueError, match="uniform grid"):
        cell_arrays(_split(trajectory))


def test_channel_identities_must_agree_across_trajectories() -> None:
    other = Trajectory(
        trajectory_id="b",
        time=np.arange(12) * 0.1,
        targets={"z": np.arange(12, dtype=float)},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={},
    )
    with pytest.raises(ValueError, match="channel identities differ"):
        observed_channels(_split(_trajectory("a"), other))


def test_the_specification_names_the_variables_and_the_derivative() -> None:
    """The adaptation must be unambiguous about what is being asked."""
    block = specification_block("Recover the glucose flux.", ("y", "u"), 1)
    assert "Recover the glucose flux." in block
    # Upstream's SYSTEM_TEMPLATE says "Independent variables: x_0, x_1", so the
    # specification must use the same spelling or one prompt names a variable
    # two ways.
    assert "x_0 = y" in block and "x_1 = u" in block
    assert "derivative of x_1 (u)" in block
    # nothing about our own pipeline leaks into their prompt
    for forbidden in ("Autoformalism", "judge", "schema", "proposer"):
        assert forbidden.lower() not in block.lower()


def test_selection_ranks_the_product_of_the_per_variable_frontiers() -> None:
    """Upstream assembles systems from the frontiers and keeps the best."""
    frontiers = (("a1", "a2"), ("b1", "b2"))
    scores = {("a1", "b1"): 5.0, ("a1", "b2"): 2.0, ("a2", "b1"): 9.0}
    assert select_system(frontiers, lambda item: scores.get(item)) == ("a1", "b2")
    # a candidate that will not integrate is skipped, not ranked as zero
    assert select_system(frontiers, lambda item: None) is None


def test_an_exploding_product_is_refused_rather_than_truncated() -> None:
    """Ranking a subset and reporting it as the search would be a false claim."""
    frontiers = tuple(tuple(f"v{index}_{n}" for n in range(30)) for index in range(4))
    with pytest.raises(ValueError, match="exceed the"):
        select_system(frontiers, lambda item: 1.0, maximum_combinations=1000)


# --- plan freeze, resume and reporting ------------------------------------


def test_only_the_predicted_channels_are_searched() -> None:
    """Auxiliaries are supplied over the horizon, so searching them is waste."""
    from autoformalism.rebuttal.llm_ode_campaign import target_indices

    assert target_indices(("y", "u", "w"), ("y", "w")) == (0, 2)
    with pytest.raises(ValueError, match="absent from the observed channels"):
        target_indices(("y", "u"), ("y", "missing"))


def test_the_plan_identity_excludes_the_served_port() -> None:
    """Preparation runs before the server exists, as it does for D3."""
    from autoformalism.rebuttal.llm_ode_campaign import environment_identity

    first = environment_identity()
    assert first["provider"] == "vllm"
    assert not first["endpoint_kind"].startswith("http")
    assert environment_identity() == first


def test_a_task_without_a_searcher_refuses_rather_than_inventing_one(
    tmp_path: Path, monkeypatch
) -> None:
    """The search must come from the pinned checkout, never from here."""
    from autoformalism.rebuttal import llm_ode_campaign as campaign

    sealed = {
        "protocol": campaign.PROTOCOL,
        "environment": campaign.environment_identity(),
        "public_root": str(tmp_path),
        "artifact_sha256": "a" * 64,
        "rows": [
            {
                "index": 0,
                "benchmark_id": "phase_b_cell",
                "tier": "easy",
                "repetition": 0,
                "public_identity": {"train": "t", "validation": "v", "prompt": "p"},
            }
        ],
    }
    monkeypatch.setattr(campaign, "sealed_read", lambda path: sealed)
    monkeypatch.setattr(
        campaign,
        "load_public",
        lambda *args: (None, None, sealed["rows"][0]["public_identity"]),
    )
    monkeypatch.setattr(campaign, "cell_arrays", lambda split: None)
    with pytest.raises(ValueError, match="searcher factory"):
        campaign.run(tmp_path, 0)


def test_a_changed_checkout_names_its_remedy(tmp_path: Path, monkeypatch) -> None:
    from autoformalism.rebuttal import llm_ode_campaign as campaign

    monkeypatch.setattr(
        campaign,
        "sealed_read",
        lambda path: {"protocol": campaign.PROTOCOL, "environment": {"stale": True}},
    )
    with pytest.raises(ValueError, match="delete"):
        campaign.run(tmp_path, 0)


def test_only_the_public_scientific_sections_reach_a_vendored_prompt() -> None:
    """Our modelling requirements and response format are not theirs to obey."""
    from autoformalism.rebuttal.llm_ode_campaign import public_task_specification

    prompt = "\n".join(
        [
            "A. Task specification",
            "Recover the absorption flux.",
            "",
            "B. Available data",
            "Target channels:",
            "- y: observed",
            "",
            "C. Modeling requirements",
            "1. Propose an explicit continuous-time model.",
            "",
            "D. Response format",
            "Return JSON matching the candidate schema.",
        ]
    )
    specification = public_task_specification(prompt)
    assert "Recover the absorption flux." in specification
    assert "- y: observed" in specification
    for withheld in ("Modeling requirements", "Response format", "JSON", "schema"):
        assert withheld not in specification


def test_a_reordered_public_prompt_is_refused_rather_than_passed_whole() -> None:
    """A silent prefix would hand over the whole prompt if the format changed."""
    from autoformalism.rebuttal.llm_ode_campaign import public_task_specification

    with pytest.raises(ValueError, match="boundary must be re-established"):
        public_task_specification("A. Task specification\nB. Available data\nno C")


def test_a_completed_search_persists_a_model_the_evaluator_can_adapt(
    tmp_path: Path, monkeypatch
) -> None:
    """Accounting without the model would make the whole campaign unusable.

    The frozen evaluator adapts a vendored campaign through
    native-selection.json beside the result. A run that recorded only its
    status and token counts would look successful and carry nothing to score.
    """
    from autoformalism.baselines.models import BaselineDevelopmentResult
    from autoformalism.expressions import ValidationContext
    from autoformalism.rebuttal import llm_ode_campaign as campaign

    context = ValidationContext(targets=("G",), auxiliaries=("I",))
    sealed = {
        "protocol": campaign.PROTOCOL,
        "environment": campaign.environment_identity(),
        "public_root": str(tmp_path),
        "artifact_sha256": "b" * 64,
        "search_config": {"n_islands": 4},
        "plan": {"budget": {"declared": 200}},
        "rows": [
            {
                "index": 0,
                "benchmark_id": "phase_b_cell",
                "tier": "easy",
                "repetition": 0,
                "public_identity": {"train": "t", "validation": "v", "prompt": "p"},
                "prompt": "",
            }
        ],
    }
    monkeypatch.setattr(campaign, "sealed_read", lambda path: sealed)
    monkeypatch.setattr(
        campaign,
        "load_public",
        lambda *args: (
            SimpleNamespace(train=object(), validation=object()),
            context,
            sealed["rows"][0]["public_identity"],
        ),
    )
    monkeypatch.setattr(campaign, "cell_arrays", lambda split: None)

    def searcher(**kwargs) -> dict:
        return {
            "status": "complete",
            "equations": {"G": "-0.5 * G + I"},
            "development_rollout_error": 0.25,
            "training_rollout_error": 0.125,
            "accounting": {"llm_queries": 2400, "search_seconds": 42.0},
        }

    result = campaign.run(tmp_path, 0, search=searcher)
    assert result["status"] == "complete"
    assert result["equations"] == {"G": "-0.5 * G + I"}

    saved = json.loads(
        (tmp_path / "results" / "0" / "native-selection.json").read_text()
    )
    assert saved["plan_sha256"] == sealed["artifact_sha256"]
    selection = BaselineDevelopmentResult.model_validate(saved["selection"])
    assert selection.method == "llm_ode"
    assert selection.validation_normalized_mse == 0.25
    assert selection.training_normalized_mse == 0.125
    # the payload the adapter reads must carry a usable candidate
    candidate = CandidateModel.model_validate(selection.selection_payload["candidate"])
    assert [item.state for item in candidate.state_equations] == ["G"]
    # the island count is the sealed one, not an environment override
    assert selection.selected_hyperparameters["n_islands"] == 4


def test_the_report_persists_a_summary_with_coverage_before_scores(
    tmp_path: Path, monkeypatch
) -> None:
    """Both campaigns must write summary.json, or a reader infers a difference.

    D3 persisted one and LLM-ODE did not, so an identical-looking directory
    meant different things depending on which method produced it.
    """
    from autoformalism.rebuttal import llm_ode_campaign as campaign

    sealed = {
        "protocol": campaign.PROTOCOL,
        "artifact_sha256": "c" * 64,
        "reporting_qualifications": ["declared prompt adaptation"],
        "rows": [
            {"index": index, "benchmark_id": f"cell{index}", "tier": "easy",
             "repetition": 0}
            for index in range(3)
        ],
    }
    monkeypatch.setattr(campaign, "sealed_read", lambda path: sealed)
    (tmp_path / "results").mkdir()

    value = campaign.report(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary == value
    # nothing has run, so the campaign is pending and claims no successes
    assert value["status"] == "pending"
    assert value["expected"] == 3
    assert value["not_started"] == 3
    assert value["terminal_success"] == 0
    assert value["frozen_models"] == 0


def test_measured_inputs_and_covariates_are_variables_an_equation_can_use() -> None:
    """A meal or an inflow must be usable, as it is by every other baseline.

    Upstream's systems are autonomous, so the adapter once passed only targets
    and auxiliaries and every forced mechanism was unrepresentable.
    """
    from autoformalism.baselines.core import baseline_validation_context
    from autoformalism.data.models import DevelopmentDataset, TierRoles
    from autoformalism.rebuttal.llm_ode_driver import development_rollout_error
    from autoformalism.rebuttal.llm_ode_upstream import to_state_equations

    def trajectory(identifier: str, start: float) -> Trajectory:
        time = np.linspace(0.0, 3.0, 31)
        return Trajectory(
            trajectory_id=identifier,
            time=time,
            targets={"h": start * np.exp(-time)},
            auxiliaries={},
            external_inputs={"inflow": np.zeros_like(time)},
            fixed_covariates={"area": 2.0},
            derivatives={},
        )

    train = DatasetSplit(
        SplitName.TRAIN, (trajectory("a", 1.0), trajectory("b", 2.0)), "train"
    )
    validation = DatasetSplit(SplitName.VALIDATION, (trajectory("c", 1.5),), "val")
    arrays = cell_arrays(train)
    assert arrays.channels == ("h", "inflow", "area")
    assert np.all(arrays.states[:, 2] == 2.0)
    context = baseline_validation_context(
        DevelopmentDataset(
            benchmark_id="toy",
            tier="easy",
            roles=TierRoles(targets=("h",), auxiliaries=()),
            train=train,
            validation=validation,
        ),
        SimpleNamespace(
            external_inputs=("inflow",),
            fixed_covariates=("area",),
            one_step_target_history=False,
        ),
    )
    equations = to_state_equations(("-x_0*x_2/2.0 + x_1",), arrays.channels, ("h",))
    assert equations == {"h": "-h*area/2.0 + inflow"}
    assert development_rollout_error(equations, context, train, validation) < 1e-4


def test_basin_prompts_pass_the_science_and_withhold_the_protocol() -> None:
    """Basin sections A-D are all science; E is the evaluation protocol."""
    from autoformalism.rebuttal.llm_ode_campaign import public_task_specification

    prompt = "\n".join(
        [
            "# Stormwater detention basins",
            "## A. Scientific task",
            "Model the downstream level.",
            "## B. Observations and known forcing",
            "Target h_down.",
            "## C. Surveyed geometry and preparation",
            "Covariates area_up and initial_up.",
            "## D. Required mechanisms",
            "- Represent threshold-dependent overflow.",
            "## E. Evaluation boundaries",
            "Fit parameters using training only.",
        ]
    )
    specification = public_task_specification(prompt)
    for passed in ("downstream level", "Target h_down", "initial_up", "overflow"):
        assert passed in specification
    assert "Evaluation boundaries" not in specification
    assert "training only" not in specification


def test_the_unsectioned_negative_control_withholds_its_protocol_paragraph() -> None:
    from autoformalism.rebuttal.llm_ode_campaign import public_task_specification

    prompt = "\n\n".join(
        [
            "# Independent stormwater basins - negative control v1",
            "Predict downstream basin depth `h_down` from `inflow_down`.",
            "Represent downstream accumulation and a threshold outlet.",
            "Fixed covariates `area_up` and `initial_up` are surveyed geometry.",
            "Fit using training only. Validation-specific fitting is forbidden.",
        ]
    )
    specification = public_task_specification(prompt)
    assert "threshold outlet" in specification
    assert "initial_up" in specification
    assert "training only" not in specification


def test_a_basin_prompt_with_sections_out_of_order_is_refused() -> None:
    from autoformalism.rebuttal.llm_ode_campaign import public_task_specification

    prompt = "\n".join(
        [
            "## A. Scientific task",
            "## B. Observations and known forcing",
            "## E. Evaluation boundaries",
            "## C. Surveyed geometry and preparation",
            "## D. Required mechanisms",
        ]
    )
    with pytest.raises(ValueError, match="out of order"):
        public_task_specification(prompt)


def test_a_real_power_and_a_sympy_function_are_scored_not_refused() -> None:
    """Upstream fits exponents and SymPy prints cos; both now roll out."""
    from autoformalism.baselines.core import baseline_validation_context
    from autoformalism.data.models import DevelopmentDataset, TierRoles
    from autoformalism.rebuttal.llm_ode_driver import development_rollout_error
    from autoformalism.rebuttal.llm_ode_upstream import to_state_equations

    def trajectory(identifier: str, start: float) -> Trajectory:
        time = np.linspace(0.0, 3.0, 31)
        return Trajectory(
            trajectory_id=identifier,
            time=time,
            targets={"h": start * np.exp(-time)},
            auxiliaries={},
            external_inputs={"inflow": np.zeros_like(time)},
            fixed_covariates={},
            derivatives={},
        )

    train = DatasetSplit(
        SplitName.TRAIN, (trajectory("a", 1.0), trajectory("b", 2.0)), "train"
    )
    validation = DatasetSplit(SplitName.VALIDATION, (trajectory("c", 1.5),), "val")
    context = baseline_validation_context(
        DevelopmentDataset(
            benchmark_id="toy",
            tier="easy",
            roles=TierRoles(targets=("h",), auxiliaries=()),
            train=train,
            validation=validation,
        ),
        SimpleNamespace(
            external_inputs=("inflow",),
            fixed_covariates=(),
            one_step_target_history=False,
        ),
    )
    equations = to_state_equations(
        ("-(x_0**0.5)**2.0*cos(x_1) + x_1",), ("h", "inflow"), ("h",)
    )
    assert development_rollout_error(equations, context, train, validation) < 1e-4
