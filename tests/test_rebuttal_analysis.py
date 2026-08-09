"""Focused tests for rebuttal artifact and deterministic metric utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoformalism.data.models import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.adversarial import MutationRecipe, mutate_candidate
from autoformalism.rebuttal.artifacts import CandidateArtifact
from autoformalism.rebuttal.benchmark_audit import (
    audit_excitation,
    audit_response_phases,
    audit_shortcuts,
    downsample_split,
)
from autoformalism.rebuttal.dalla_man import STATE_NAMES, simulate_dalla_man
from autoformalism.rebuttal.hidden import hidden_mechanism_nmse
from autoformalism.rebuttal.intervention_evaluation import (
    FrozenModel,
    align_hidden_states,
    evaluate_frozen_model,
    load_frozen_model,
    qualitative_response_metrics,
)
from autoformalism.rebuttal.interventions import (
    InterventionCase,
    InterventionSuite,
    load_intervention_suite,
    simulate_reference,
)
from autoformalism.rebuttal.llm_assets import audit_llm_caches, resolve_llm_caches
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.rebuttal.objectives import (
    compare_ratio_and_weighted_sum,
    select_frozen_candidate,
)
from autoformalism.rebuttal.observability import (
    empirical_dalla_observability,
    empirical_dalla_parameter_sensitivity,
)
from autoformalism.rebuttal.statistics import (
    holm_adjust,
    paired_log_comparison,
    wilson_interval,
)
from autoformalism.rebuttal.structure import pairwise_similarities
from autoformalism.schemas import CandidateModel
from scripts.analyze_judge_calibration import (
    _average_repetitions,
    _expand_categories,
    _fixed_score_metrics,
    fit_ridge_logistic,
    predict_ridge_logistic,
)
from scripts.analyze_learning_curves import _iteration_summary
from scripts.analyze_surviving_terms import summarize_candidate
from scripts.audit_failure_recovery import (
    cache_replay_candidate,
    classify_failure,
    expected_cells,
)
from scripts.evaluate_failed_candidate_selection import validate_frozen_manifest
from scripts.freeze_selector_confirmation import build_confirmation
from scripts.refit_failed_candidate_pool import initial_parameters
from scripts.run_intervention_batch import resolve_cohort, summarize
from scripts.summarize_selector_confirmation import (
    assemble_policy_outcomes,
    summarize_policy_outcomes,
)


def _audit_split(name: SplitName) -> DatasetSplit:
    time = np.arange(8, dtype=np.float64)
    input_values = np.asarray([0, 0, 1, 0, 0, 2, 0, 0], dtype=np.float64)
    target = np.zeros(8, dtype=np.float64)
    for index in range(7):
        target[index + 1] = 0.8 * target[index] + 2.0 * input_values[index]
    trajectory = Trajectory(
        trajectory_id=name.value,
        time=time,
        targets={"y": target},
        auxiliaries={},
        external_inputs={"u": input_values},
        fixed_covariates={},
        derivatives={"y": np.gradient(target)},
    )
    return DatasetSplit(name=name, trajectories=(trajectory,), fingerprint=name.value)


def test_benchmark_audit_exposes_shortcuts_and_excitation() -> None:
    train = _audit_split(SplitName.TRAIN)
    validation = _audit_split(SplitName.VALIDATION)

    records = audit_shortcuts(train, validation, "y", horizons=(1, 3))
    by_key = {(record.model, record.horizon): record for record in records}

    assert by_key[("arx", 1)].raw_mse < 1e-20
    assert by_key[("arx", 3)].raw_mse < by_key[("persistence", 3)].raw_mse
    assert by_key[("persistence", 3)].event_sample_fraction > 0
    excitation = audit_excitation(train, "y")
    assert excitation.distinct_input_levels == 3
    assert 0 < excitation.input_change_fraction < 1
    assert excitation.increment_variance_ratio > 0


def test_benchmark_audit_rejects_invalid_horizon() -> None:
    split = _audit_split(SplitName.TRAIN)
    with pytest.raises(ValueError, match="positive"):
        audit_shortcuts(split, split, "y", horizons=(0,))


def test_benchmark_audit_downsampling_and_response_phases() -> None:
    train = _audit_split(SplitName.TRAIN)
    sampled = downsample_split(train, 2)

    assert sampled.trajectories[0].time.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert sampled.fingerprint.endswith("stride=2")
    phases = audit_response_phases(train, train, "y", horizons=(1,))
    assert {record.phase for record in phases} >= {"rise", "peak"}
    with pytest.raises(ValueError, match="positive"):
        downsample_split(train, 0)


def test_empirical_dalla_observability_reports_scaled_spectrum() -> None:
    result = empirical_dalla_observability(
        "T1", meals=((0.0, 90.0),), duration=30.0, dt=2.0
    )

    assert result.outputs == ("Gp",)
    assert result.hidden_states == ("Qsto1", "Qsto2", "Qgut")
    assert len(result.singular_values) == 3
    assert 0 <= result.rank_at_1e3 <= result.rank_at_1e6 <= 3
    with pytest.raises(ValueError, match="unknown"):
        empirical_dalla_observability("T5", meals=((0.0, 90.0),))


def test_empirical_dalla_parameter_sensitivity_reports_flux_spectrum() -> None:
    result = empirical_dalla_parameter_sensitivity(
        "T2",
        meals=((0.0, 90.0),),
        duration=30.0,
        dt=2.0,
        quantity_kind="fluxes",
    )

    assert result.quantities == ("Ra", "U")
    assert len(result.singular_values) == len(result.parameters)
    assert result.rank_at_1e3 <= result.rank_at_1e6


def test_intervention_suite_is_hashed_and_enforces_leakage_flags() -> None:
    case = InterventionCase(
        case_id="case",
        benchmark_id="benchmark6",
        shift_types=("multiple_events",),
        protocol={"kind": "step", "start": 1.0, "end": 2.0, "amplitude": 1.0},
        duration=3.0,
        dt=0.1,
        initial_state=(0.0,) * 6,
    )
    suite = InterventionSuite(
        suite_id="suite",
        frozen_before_evaluation=True,
        uses_private_reference=True,
        available_to_proposal_fit_or_selection=False,
        cases=(case,),
    )

    assert len(suite.fingerprint) == 64
    with pytest.raises(ValueError):
        InterventionSuite.model_validate(
            {
                **suite.model_dump(),
                "available_to_proposal_fit_or_selection": True,
            }
        )


def test_benchmark5_private_reference_is_deterministic() -> None:
    case = InterventionCase(
        case_id="b5",
        benchmark_id="benchmark5",
        shift_types=("multiple_events",),
        protocol={
            "kind": "piecewise",
            "segments": [{"start": 0.2, "end": 0.6, "delta": [0.1, 2.0, -2.0]}],
        },
        duration=1.0,
        dt=0.1,
        initial_state=(0.26193232096817415, 365.13129525446493, 347.56564762723247),
    )
    spec = {
        "parameters": {
            "k0": 72_000_000_000.0,
            "E_over_R": 8750.0,
            "flow_rate": 1.0,
            "source_gain": 80.0,
            "exchange_rate": 2.5,
            "secondary_flow_rate": 1.5,
            "secondary_exchange_rate": 1.5,
            "C_feed_base": 1.0,
            "T_feed_base": 350.0,
            "T_secondary_feed_base": 330.0,
        }
    }

    first = simulate_reference(case, system_spec=spec)
    second = simulate_reference(case, system_spec=spec)
    assert first == second
    assert len(first.time) == 11
    assert len(first.forcing[0]) == 3


def test_benchmark6_private_reference_applies_sampling_and_noise() -> None:
    case = InterventionCase(
        case_id="b6",
        benchmark_id="benchmark6",
        shift_types=("noise", "sparse"),
        protocol={"kind": "step", "start": 0.2, "end": 0.6, "amplitude": 1.0},
        duration=1.0,
        dt=0.1,
        initial_state=(0.0,) * 6,
        observation_stride=2,
        noise_fraction=0.05,
        noise_seed=7,
    )
    spec = {
        "n_latent": 5,
        "decay": [0.1] * 5,
        "skew": [[0.0] * 5 for _ in range(5)],
        "tanh_terms": [[] for _ in range(5)],
        "product_terms": [[] for _ in range(5)],
        "input_vector": [1.0, 0.0, 0.0, 0.0, 0.0],
        "input_scale": 1.0,
        "output_decay": 0.1,
        "output_terms": [{"source": 0, "coefficient": 1.0, "scale": 1.0}],
        "output_product_terms": [],
    }

    result = simulate_reference(case, system_spec=spec)
    assert len(result.time) == 6
    assert result.states_observed != result.states_clean


def test_baseline_artifact_adapter_and_intervention_evaluation(tmp_path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text(
        json.dumps(
            {
                "method": "sindy",
                "equations": {"v02": "gain*u01 - decay*v02"},
                "selected_hyperparameters": {
                    "selected_parameters": json.dumps({"gain": 1.0, "decay": 0.1})
                },
                "test_normalized_mse": 0.2,
            }
        ),
        encoding="utf-8",
    )
    frozen = load_frozen_model(artifact, target="v02")
    case = InterventionCase(
        case_id="b6_eval",
        benchmark_id="benchmark6",
        shift_types=("step",),
        protocol={"kind": "step", "start": 0.2, "end": 0.6, "amplitude": 1.0},
        duration=1.0,
        dt=0.1,
        initial_state=(0.0,) * 6,
    )
    spec = {
        "n_latent": 5,
        "decay": [0.1] * 5,
        "skew": [[0.0] * 5 for _ in range(5)],
        "tanh_terms": [[] for _ in range(5)],
        "product_terms": [[] for _ in range(5)],
        "input_vector": [1.0, 0.0, 0.0, 0.0, 0.0],
        "input_scale": 1.0,
        "output_decay": 0.1,
        "output_terms": [{"source": 0, "coefficient": 1.0, "scale": 1.0}],
        "output_product_terms": [],
        "secret_mapping": {
            "semantic_to_opaque": {
                "z1": "v01",
                "z2": "v05",
                "z3": "v06",
                "z4": "v03",
                "z5": "v04",
                "y": "v02",
            }
        },
    }
    reference = simulate_reference(case, system_spec=spec)
    result = evaluate_frozen_model(
        frozen,
        case=case,
        reference=reference,
        context=ValidationContext(
            targets=("v02",),
            external_inputs=("u01",),
            lagged_targets=("v02",),
        ),
        tier="hard",
        system_spec=spec,
        fallback_target_scale=1.0,
    )

    assert result.success
    assert result.target_mse is not None
    assert result.nmse_degradation_ratio is not None
    assert result.hidden_alignment_nmse is None
    assert result.hidden_state_coverage == 0.0
    assert result.hidden_reference_states == 5


def test_hidden_intervention_alignment_handles_permutation_sign_and_missing_state(
    tmp_path,
) -> None:
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "latent_alignment",
            "parent_candidate_id": None,
            "states": [
                {"name": "target", "kind": "observed"},
                {"name": "memory_a", "kind": "latent"},
                {"name": "memory_b", "kind": "latent"},
            ],
            "state_equations": [
                {"state": "target", "rhs": "0"},
                {"state": "memory_a", "rhs": "0"},
                {"state": "memory_b", "rhs": "0"},
            ],
            "observation_mappings": [{"channel": "target", "expression": "target"}],
            "initial_conditions": [
                {"state": "target", "scope": "global", "expression": "target"},
                {"state": "memory_a", "scope": "global", "fixed_value": 0.0},
                {"state": "memory_b", "scope": "global", "fixed_value": 0.0},
            ],
        }
    )
    model = FrozenModel(
        method="autoformalism",
        source=tmp_path / "final.json",
        candidate=candidate,
        parameters={},
        initial_conditions={},
        in_distribution_nmse=None,
        target_scales={},
    )
    time = np.linspace(0.0, 2.0, 21)
    reference = np.column_stack((np.sin(time), np.cos(time), time**2))
    states = np.vstack(
        (
            np.zeros_like(time),
            -3.0 * reference[:, 1] + 4.0,
            2.0 * reference[:, 0] - 7.0,
        )
    )

    result = align_hidden_states(model=model, model_states=states, reference=reference)

    assert result.matched_states == 2
    assert result.reference_states == 3
    assert result.coverage == pytest.approx(2 / 3)
    assert result.nmse == pytest.approx(1 / 3, abs=1e-12)


def test_qualitative_response_metrics_score_direction_shape_and_timing() -> None:
    time = np.arange(5, dtype=float)
    reference = np.asarray([0.0, 1.0, 3.0, 1.0, 0.0])
    prediction = np.asarray([0.0, 0.5, 1.0, 2.0, 0.5])

    direction, correlation, timing = qualitative_response_metrics(
        time, prediction, reference
    )

    assert direction is True
    assert correlation is not None and correlation > 0.0
    assert timing == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("variant", "reference_path"),
    [
        (
            "original",
            "data_raw/benchmark1_original_dalla_man/full/full_test.csv",
        ),
        (
            "perturbed_b1",
            "data_raw/benchmark2_perturbed_dalla_man/"
            "B1_meal_appearance/full/full_test.csv",
        ),
    ],
)
def test_extracted_dalla_man_simulator_reproduces_generator(
    variant, reference_path
) -> None:
    reference = pd.read_csv(reference_path)

    simulated = simulate_dalla_man(
        meals=((0.0, 120.0),),
        duration=300.0,
        dt=1.0,
        variant=variant,
    )

    assert simulated.time == pytest.approx(reference["time"].to_numpy())
    assert simulated.states == pytest.approx(
        reference[list(STATE_NAMES)].to_numpy(), rel=2e-7, abs=2e-7
    )
    assert simulated.derived["Ra"] == pytest.approx(
        reference["Ra"].to_numpy(), rel=2e-7, abs=2e-7
    )


def test_dalla_man_template_suite_expands_to_all_registered_representations() -> None:
    suite = load_intervention_suite(
        Path("configs/interventions/phase_a3_dalla_man_v1.json")
    )

    assert len(suite.cases) == 16
    assert {case.benchmark_id for case in suite.cases} == {
        "original_b1",
        "perturbed_b1",
        "obfuscated_original_case01",
        "obfuscated_perturbed_case01",
    }
    reference = simulate_reference(suite.cases[0], system_spec={})
    assert len(reference.time) == 361
    assert len(reference.states_clean[0]) == 12


def test_intervention_cohort_resolution_and_summary(tmp_path) -> None:
    artifact = tmp_path / "runs" / "benchmark5_hard_seed0" / "result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"seed": 0, "status": "complete"}), encoding="utf-8")
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps(
            {
                "tier": "hard",
                "methods": {
                    "Method": {
                        "expected_seeds": [0, 1],
                        "patterns": {
                            "benchmark5": [
                                "runs/benchmark5_hard_seed{seed}/result.json"
                            ]
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    specifications, manifest = resolve_cohort(cohort, project_root=tmp_path)
    assert len(specifications) == 1
    assert [item["status"] for item in manifest] == ["complete", "missing"]

    summary = summarize(
        [
            {
                "model_label": "benchmark5:Method:0",
                "benchmark_id": "benchmark5",
                "case_id": "case",
                "success": True,
                "target_mse": 2.0,
                "target_nmse": 4.0,
                "nmse_degradation_ratio": 2.0,
            },
            {
                "model_label": "benchmark5:Method:1",
                "benchmark_id": "benchmark5",
                "case_id": "case",
                "success": False,
                "target_mse": None,
                "target_nmse": None,
                "nmse_degradation_ratio": None,
            },
        ]
    )
    assert summary[0]["completion_rate"] == 0.5
    assert summary[0]["target_mse_mean"] == 2.0


def _candidate(identifier: str = "candidate") -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": identifier,
            "parent_candidate_id": None,
            "states": [
                {
                    "name": "memory",
                    "kind": "latent",
                    "mechanisms": ["input_memory"],
                },
                {"name": "target", "kind": "observed"},
            ],
            "processes": [],
            "state_equations": [
                {"state": "memory", "rhs": "input_u - memory"},
                {"state": "target", "rhs": "memory - target"},
            ],
            "observation_mappings": [{"channel": "target", "expression": "target"}],
            "initial_conditions": [
                {"state": "memory", "scope": "global", "fixed_value": 0.0},
                {"state": "target", "scope": "global", "fixed_value": 0.0},
            ],
        }
    )


def _artifact(identifier: str, loss: float, score: float) -> CandidateArtifact:
    return CandidateArtifact(
        artifact_id=identifier,
        source_checkpoint=f"/{identifier}.json",
        run_directory=f"/{identifier}",
        benchmark_id="synthetic",
        tier="hard",
        seed=0,
        round_index=0,
        structural_hash=identifier,
        candidate=_candidate(identifier),
        validation_mse=loss,
        training_mse=loss,
        judge_score=score,
        judge_category_scores={},
        state_count=2,
        latent_state_count=1,
        process_count=0,
        parameter_count=0,
        term_count=4,
        use_judge=True,
    )


def test_ratio_and_weighted_sum_comparison_is_development_only() -> None:
    result = compare_ratio_and_weighted_sum(
        (_artifact("a", 0.01, 0.9), _artifact("b", 0.02, 0.2)),
        lambda_multiplier=1.0,
    )

    assert result.candidate_count == 2
    assert result.ratio_selected_artifact_id == "a"
    assert result.lambda_value == 0.015


def test_normalized_weighted_selection_can_trade_fit_for_judge_and_sparsity() -> None:
    complex_fit = _artifact("complex", 0.01, 0.2).model_copy(update={"term_count": 20})
    simple_judged = _artifact("simple", 0.0105, 0.95).model_copy(
        update={"term_count": 2}
    )

    validation = select_frozen_candidate(
        (complex_fit, simple_judged), policy="validation_only"
    )
    weighted = select_frozen_candidate(
        (complex_fit, simple_judged),
        policy="normalized_weighted_sum",
        judge_weight=1.0,
        sparsity_weight=1.0,
    )

    assert validation.artifact_id == "complex"
    assert weighted.artifact_id == "simple"


def test_epsilon_selector_uses_judge_only_inside_fit_tolerance() -> None:
    best = _artifact("best", 0.01, 0.2)
    near = _artifact("near", 0.0104, 0.9)
    far = _artifact("far", 0.02, 1.0)

    selected = select_frozen_candidate(
        (best, near, far),
        policy="epsilon_constrained",
        epsilon_fraction=0.05,
    )

    assert selected.artifact_id == "near"
    assert selected.eligible_count == 2


def test_pareto_selector_excludes_dominated_candidate() -> None:
    dominant = _artifact("dominant", 0.01, 0.9).model_copy(update={"term_count": 2})
    dominated = _artifact("dominated", 0.02, 0.8).model_copy(update={"term_count": 3})

    selected = select_frozen_candidate(
        (dominant, dominated), policy="pareto_compromise"
    )

    assert selected.artifact_id == "dominant"
    assert selected.eligible_count == 1


def test_frozen_selector_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        select_frozen_candidate(
            (_artifact("a", 0.01, 0.5),),
            policy="normalized_weighted_sum",
            judge_weight=-1.0,
        )


def test_confirmation_manifest_contains_only_changed_alternatives() -> None:
    rows = []
    for config_id, artifact_id in (
        ("validation_only", "baseline"),
        ("normalized_weighted_sum__j0.5__s0.1", "weighted"),
        ("epsilon_constrained__d0.2", "baseline"),
    ):
        rows.append(
            {
                "run_directory": "/run",
                "benchmark": "synthetic",
                "tier": "hard",
                "seed": 0,
                "config_id": config_id,
                "artifact_id": artifact_id,
            }
        )

    frozen, changed = build_confirmation(pd.DataFrame(rows))

    assert len(frozen) == 3
    assert changed.confirmation_policy.tolist() == ["weighted_j0.5_s0.1"]
    assert changed.validation_artifact_id.tolist() == ["baseline"]


def test_selector_confirmation_keeps_outcomes_multidimensional() -> None:
    selections = pd.DataFrame(
        [
            {
                "run_directory": "/run",
                "benchmark": "synthetic",
                "tier": "hard",
                "seed": 0,
                "confirmation_policy": policy,
                "artifact_id": artifact,
                "term_count": terms,
            }
            for policy, artifact, terms in (
                ("validation_only", "baseline", 5),
                ("weighted_j0.5_s0.1", "alternative", 3),
                ("epsilon_d0.2", "baseline", 5),
            )
        ]
    )
    authoritative = pd.DataFrame(
        [
            {
                "method": "full",
                "benchmark": "synthetic",
                "tier": "hard",
                "seed": 0,
                "test_mse": 2.0,
            }
        ]
    )
    production_structure = pd.DataFrame(
        [
            {
                "method": "full",
                "benchmark": "synthetic",
                "tier": "hard",
                "seed": 0,
                "structural_validity": 0.5,
            }
        ]
    )
    production_hidden = pd.DataFrame(
        [
            {
                "method": "full",
                "benchmark": "synthetic",
                "tier": "hard",
                "seed": 0,
                "hidden_mse": 3.0,
            }
        ]
    )
    source = "/tmp/alternative/result.json"
    alternative_runs = pd.DataFrame(
        [
            {
                "artifact_id": "alternative",
                "status": "complete",
                "test_mse": 1.0,
                "source": source,
            }
        ]
    )
    alternative_structure = pd.DataFrame(
        [{"source": source, "structural_validity": 0.75}]
    )
    alternative_hidden = pd.DataFrame([{"source": source, "hidden_mse": 2.0}])

    details = assemble_policy_outcomes(
        selections=selections,
        authoritative=authoritative,
        production_structure=production_structure,
        production_hidden=production_hidden,
        alternative_runs=alternative_runs,
        alternative_structure=alternative_structure,
        alternative_hidden=alternative_hidden,
    )
    summary, paired = summarize_policy_outcomes(details)

    weighted = details[details.confirmation_policy == "weighted_j0.5_s0.1"].iloc[0]
    assert weighted.test_mse == 1.0
    assert weighted.structural_validity == 0.75
    assert weighted.hidden_mse == 2.0
    assert weighted.term_count == 3
    assert len(summary) == 3
    assert paired.iloc[0].alternative_pareto_dominates == 1


def test_ridge_logistic_calibration_separates_synthetic_categories() -> None:
    features = np.asarray([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]], dtype=float)
    labels = np.asarray([0, 0, 1, 1])

    fitted = fit_ridge_logistic(features, labels)
    probabilities = predict_ridge_logistic(fitted, features)

    assert probabilities[labels == 1].min() > probabilities[labels == 0].max()


def test_judge_primary_scope_excludes_prose_only_mutation() -> None:
    category_payload = dict.fromkeys(
        (
            "task_output_coverage",
            "mechanism_state_adequacy",
            "mathematical_completeness",
            "data_causal_consistency",
            "constraint_compliance",
            "parsimony_interpretability",
        ),
        0.9,
    )
    rows = []
    for pair_id, mutation in (
        ("dynamic", "wrong_causal_driver"),
        ("narrative", "narrative_equation_mismatch"),
    ):
        for label, score in (("valid", 0.9), ("adversarial", 0.2)):
            rows.append(
                {
                    "pair_id": pair_id,
                    "benchmark_id": "synthetic",
                    "mutation_type": mutation,
                    "known_label": label,
                    "judge_model": "example",
                    "repetition": 0,
                    "aggregate_score": score,
                    "category_scores": json.dumps(category_payload),
                }
            )
    averaged = _average_repetitions(_expand_categories(pd.DataFrame(rows)))

    metrics = _fixed_score_metrics(averaged, samples=100)

    assert set(metrics.scope) == {"all", "dynamics_only"}
    assert set(metrics[metrics.scope == "dynamics_only"].pair_count) == {1}


def test_failure_audit_classifies_recoverable_and_terminal_failures() -> None:
    rollout = "ValueError: no PySR expression passed safe validation rollout"
    metadata = "ValueError: could not convert string to float: '[{}]'"

    assert classify_failure(rollout) == "safe_rollout_failure"
    assert classify_failure(metadata) == "structured_metadata_ingestion"
    assert not cache_replay_candidate(rollout)
    assert not cache_replay_candidate(metadata)


def test_failure_audit_expected_matrix_has_486_core_cells() -> None:
    expected = expected_cells()

    assert len(expected) == 486
    assert ("full", "benchmark6", "hard", 4) in expected
    assert ("sindy", "benchmark6", "hard", 1) not in expected


def test_failed_candidate_refit_uses_declared_or_near_zero_starts() -> None:
    candidate = _candidate()
    payload = candidate.model_dump(mode="json")
    payload["parameters"] = [
        {
            "name": "signed",
            "scope": "global",
            "bounds": {"lower": -100.0, "upper": 100.0},
            "initialization_range": {"lower": -2.0, "upper": 4.0},
        },
        {
            "name": "positive",
            "scope": "global",
            "bounds": {"lower": 1e-6, "upper": 100.0},
            "initialization_range": {"lower": 0.01, "upper": 2.0},
        },
    ]
    candidate = CandidateModel.model_validate(payload)

    assert initial_parameters(candidate, "declared_midpoint") == {
        "signed": 1.0,
        "positive": 1.005,
    }
    assert initial_parameters(candidate, "scale_aware_near_zero") == {
        "signed": 0.0,
        "positive": 0.01,
    }


def test_failure_recovery_evaluator_requires_development_freeze() -> None:
    manifest = {
        "stage": "development_selection_frozen",
        "uses_test_data": False,
        "selection_metric": "validation_normalized_mse",
    }

    validate_frozen_manifest(manifest)
    with pytest.raises(ValueError, match="uses_test_data=false"):
        validate_frozen_manifest({**manifest, "uses_test_data": True})


def test_paired_log_comparison_reports_ratio_and_reproducible_interval() -> None:
    result = paired_log_comparison(
        np.asarray([1.0, 2.0, 4.0]),
        np.asarray([2.0, 4.0, 8.0]),
        bootstrap_samples=500,
        permutation_samples=500,
        random_seed=7,
    )

    assert result.pair_count == 3
    assert result.first_win_rate == 1.0
    assert result.geometric_mean_ratio == pytest.approx(0.5)
    assert result.geometric_ratio_ci_low == pytest.approx(0.5)
    assert result.geometric_ratio_ci_high == pytest.approx(0.5)


def test_holm_and_wilson_statistics_are_bounded() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    low, high = wilson_interval(8, 10)

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert 0.0 < low < 0.8 < high < 1.0


def test_llm_cache_audit_deduplicates_and_detects_conflicts(tmp_path) -> None:
    request_hash = "a" * 64
    payload = {
        "request_hash": request_hash,
        "provider": "openai",
        "model": "example-model",
        "parsed_response": {"candidate_id": "one"},
        "raw_response": {"id": "response-one"},
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    for root in (tmp_path / "first" / "llm_cache", tmp_path / "second" / "cache"):
        root.mkdir(parents=True)
        (root / f"{request_hash}.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8"
        )

    audit = audit_llm_caches((tmp_path,))

    assert audit.cache_file_count == 2
    assert audit.unique_request_count == 1
    assert audit.duplicate_copy_count == 1
    assert audit.conflicting_request_count == 0
    assert audit.total_tokens == 15

    payload["raw_response"] = {"id": "response-two"}
    conflict = tmp_path / "third" / "llm_cache"
    conflict.mkdir(parents=True)
    (conflict / f"{request_hash}.json").write_text(
        __import__("json").dumps(payload), encoding="utf-8"
    )

    metadata_conflict = audit_llm_caches((tmp_path,))
    assert metadata_conflict.metadata_only_conflicting_hashes == (request_hash,)
    assert metadata_conflict.semantic_conflicting_hashes == ()
    metadata_resolution = resolve_llm_caches(
        metadata_conflict, tmp_path / "resolved_metadata"
    )
    assert metadata_resolution.resolved_request_count == 1
    assert (tmp_path / "resolved_metadata" / f"{request_hash}.json").is_file()

    payload["parsed_response"] = {"candidate_id": "two"}
    semantic = tmp_path / "fourth" / "llm_cache"
    semantic.mkdir(parents=True)
    (semantic / f"{request_hash}.json").write_text(
        __import__("json").dumps(payload), encoding="utf-8"
    )

    semantic_conflict = audit_llm_caches((tmp_path,))
    assert semantic_conflict.semantic_conflicting_hashes == (request_hash,)
    resolution = resolve_llm_caches(semantic_conflict, tmp_path / "resolved")
    assert resolution.resolved_request_count == 0
    assert resolution.excluded_semantic_hashes == (request_hash,)


def test_iteration_summary_tracks_best_so_far_fit_and_judge() -> None:
    first = _artifact("first", 0.2, 0.4)
    second = _artifact("second", 0.1, 0.3).model_copy(
        update={"round_index": 1, "run_directory": first.run_directory}
    )
    third = _artifact("third", 0.15, 0.9).model_copy(
        update={"round_index": 2, "run_directory": first.run_directory}
    )

    rows = _iteration_summary(
        [first, second, third], {("synthetic", "hard"): 2.0}, iterations=3
    )

    assert [row["validation_raw_mse_mean"] for row in rows] == pytest.approx(
        [0.8, 0.4, 0.4]
    )
    assert [row["judge_score_mean"] for row in rows] == pytest.approx([0.4, 0.4, 0.9])


def test_surviving_term_summary_uses_frozen_state_equations() -> None:
    summary = summarize_candidate(_candidate())

    assert summary["target_equations"] == "d(target)/dt = memory - target"
    assert summary["dynamic_terms"] == 4
    assert "d(memory)/dt = input_u - memory" in summary["state_equations"]


def test_mechanism_coverage_requires_driver_memory_and_target_path() -> None:
    spec = MechanismEvaluationSpec.model_validate(
        {
            "benchmark_id": "synthetic",
            "tier": "hard",
            "required_mechanisms": [
                {
                    "id": "input_memory",
                    "required_drivers": ["input_u"],
                    "required_targets": ["target"],
                    "requires_dynamic_memory": True,
                }
            ],
        }
    )

    result = evaluate_mechanisms(_candidate(), spec)

    assert result.mechanism_coverage == 1.0
    assert result.structural_validity == 1.0
    assert result.manual_review_required is False


def test_mechanism_tags_ignore_case_and_separator_variation() -> None:
    candidate = _candidate().model_copy(
        update={
            "states": (
                _candidate()
                .states[0]
                .model_copy(update={"mechanisms": ("Input-Memory",)}),
                _candidate().states[1],
            )
        }
    )
    spec = MechanismEvaluationSpec.model_validate(
        {
            "benchmark_id": "synthetic",
            "tier": "hard",
            "required_mechanisms": [
                {
                    "id": "input_memory",
                    "required_drivers": ["input_u"],
                    "required_targets": ["target"],
                    "requires_dynamic_memory": True,
                }
            ],
        }
    )

    assert evaluate_mechanisms(candidate, spec).structural_validity == 1.0


def test_dynamic_memory_does_not_count_observed_target_state() -> None:
    payload = _candidate().model_dump(mode="json")
    payload["states"] = [
        state for state in payload["states"] if state["name"] == "target"
    ]
    payload["states"][0]["mechanisms"] = ["input_memory"]
    payload["state_equations"] = [{"state": "target", "rhs": "input_u - target"}]
    payload["initial_conditions"] = [
        initial
        for initial in payload["initial_conditions"]
        if initial["state"] == "target"
    ]
    spec = MechanismEvaluationSpec.model_validate(
        {
            "benchmark_id": "synthetic",
            "tier": "hard",
            "required_mechanisms": [
                {
                    "id": "input_memory",
                    "required_drivers": ["input_u"],
                    "required_targets": ["target"],
                    "requires_dynamic_memory": True,
                }
            ],
        }
    )

    result = evaluate_mechanisms(CandidateModel.model_validate(payload), spec)

    assert result.mechanism_coverage == 1.0
    assert result.structural_validity == pytest.approx(2 / 3)


def test_process_inheriting_latent_state_counts_as_dynamic_memory() -> None:
    payload = _candidate().model_dump(mode="json")
    payload["states"][0]["mechanisms"] = []
    payload["processes"] = [
        {
            "name": "memory_effect",
            "expression": "memory",
            "mechanisms": ["input_memory"],
        }
    ]
    payload["state_equations"][1]["rhs"] = "memory_effect - target"
    spec = MechanismEvaluationSpec.model_validate(
        {
            "benchmark_id": "synthetic",
            "tier": "hard",
            "required_mechanisms": [
                {
                    "id": "input_memory",
                    "required_drivers": ["input_u"],
                    "required_targets": ["target"],
                    "requires_dynamic_memory": True,
                }
            ],
        }
    )

    result = evaluate_mechanisms(CandidateModel.model_validate(payload), spec)

    assert result.mechanism_coverage == 1.0
    assert result.structural_validity == 1.0


def test_hidden_metric_is_invariant_to_positive_affine_coordinate() -> None:
    train_reference = np.asarray([0.0, 1.0, 2.0, 3.0])
    test_reference = np.asarray([4.0, 5.0])
    train_candidate = (train_reference - 7.0) / 2.5
    test_candidate = (test_reference - 7.0) / 2.5

    result = hidden_mechanism_nmse(
        train_candidate,
        train_reference,
        test_candidate,
        test_reference,
    )

    assert result.scale == pytest.approx(2.5)
    assert result.offset == pytest.approx(7.0)
    assert result.test_nmse == pytest.approx(0.0, abs=1e-20)


def test_structure_similarity_ignores_alpha_renaming() -> None:
    first = _candidate("first")
    payload = first.model_dump(mode="json")
    payload["candidate_id"] = "second"
    replacements = {"memory": "buffer", "target": "output"}
    for state in payload["states"]:
        state["name"] = replacements[state["name"]]
    for equation in payload["state_equations"]:
        equation["state"] = replacements[equation["state"]]
        for old, new in replacements.items():
            equation["rhs"] = equation["rhs"].replace(old, new)
    payload["observation_mappings"][0]["expression"] = "output"
    for initial in payload["initial_conditions"]:
        initial["state"] = replacements[initial["state"]]
    second = CandidateModel.model_validate(payload)

    result = pairwise_similarities((("first", first), ("second", second)))[0]

    assert result.edge_jaccard == 1.0
    assert result.term_jaccard == 1.0


def test_adversarial_symbol_replacement_is_ast_scoped() -> None:
    changed = mutate_candidate(
        _candidate(),
        MutationRecipe(
            mutation_type="replace_symbol",
            component="memory",
            old_symbol="input_u",
            new_symbol="wrong_input",
        ),
    )

    equations = {item.state: item.rhs for item in changed.state_equations}
    assert "wrong_input" in equations["memory"]
    assert "input_u" not in equations["memory"]


def test_adversarial_memory_replacement_removes_dynamic_state() -> None:
    changed = mutate_candidate(
        _candidate(),
        MutationRecipe(
            mutation_type="replace_state_with_algebraic",
            component="memory",
            replacement_expression="input_u",
        ),
    )

    assert {item.name for item in changed.states} == {"target"}
    assert {item.name for item in changed.processes} == {"memory"}
