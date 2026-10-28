"""Finite premise inference, independent logical projections, and durable resume."""

import inspect
import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from autoformalism.research import premise_reference as reference


def snapshot(root: Path) -> dict[str, bytes]:
    """Read all durable bytes without changing the experiment."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def scientific_summary(summary: dict) -> dict:
    """Compare scientific outcomes independently of runtime accounting."""
    return {
        key: summary[key]
        for key in (
            "expected_datasets",
            "recorded_datasets",
            "completed_worlds",
            "missing_worlds",
            "complete",
            "totals",
            "per_world",
        )
    }


@pytest.mark.parametrize("mask", range(512))
def test_all_retained_subsets_match_diagnosis(mask):
    assert reference.project(mask) == reference.diagnosis(mask)


def test_retaining_truth_implies_every_projected_claim_is_valid():
    for mask in range(1, 512):
        claims = reference.project(mask)
        for world in reference.catalog():
            if not mask & (1 << world.world_id):
                continue
            assert not claims.support_bits & (3 ^ world.premise_bits)
            assert not claims.reject_bits & world.premise_bits
            for subset in (1, 2, 3):
                if claims.conflict_bits & (1 << (subset - 1)):
                    assert world.premise_bits & subset != subset
            assert 3 ^ world.premise_bits in claims.violation_patterns


def test_joint_conflict_keeps_single_and_mixed_explanations():
    claims = reference.project(84)
    assert claims.support_bits == 0
    assert claims.reject_bits == 0
    assert claims.conflict_bits == 4
    assert claims.violation_patterns == (1, 2, 3)


@pytest.mark.parametrize("method", [reference.project, reference.diagnosis])
def test_empty_set_suppresses_every_claim(method):
    claims = method(0)
    assert claims.support_bits == claims.reject_bits == claims.conflict_bits == 0
    assert claims.violation_patterns == ()


def test_catalog_truth_and_claims_are_immutable():
    worlds = reference.catalog()
    assert isinstance(worlds, tuple)
    assert len(worlds) == 9
    assert [world.world_id for world in worlds] == list(range(9))
    assert worlds[0].premise_bits == 3
    assert worlds[4].premise_bits == 0
    assert worlds[2].sum_index == worlds[4].sum_index == worlds[6].sum_index == 2
    assert worlds[2].mu_x == worlds[4].mu_x == worlds[6].mu_x
    with pytest.raises((FrozenInstanceError, AttributeError)):
        worlds[0].kappa = 9.0
    with pytest.raises((FrozenInstanceError, AttributeError)):
        reference.project(84).support_bits = 3


def test_classification_preserves_exact_ambiguity_then_resolves_catalog():
    worlds = reference.catalog()
    x = np.array([world.mu_x for world in worlds])
    ligand = np.array([world.mu_ligand for world in worlds])
    first, second = reference.classify(x, ligand)
    expected_first = [
        sum(
            1 << other.world_id
            for other in worlds
            if other.sum_index == world.sum_index
        )
        for world in worlds
    ]
    np.testing.assert_array_equal(first, expected_first)
    np.testing.assert_array_equal(second, 1 << np.arange(9))
    assert first[2] == first[4] == first[6] == 84
    assert "truth" not in inspect.signature(reference.classify).parameters
    assert "truth_id" not in inspect.signature(reference.classify).parameters


def test_score_counts_false_joint_conflict_without_individual_accusation():
    first = np.array([0, 84, 16, 1, 511], dtype=np.uint16)
    second = np.array([0, 0, 16, 1, 1], dtype=np.uint16)
    assert reference.score_masks(first, second, 0) == {
        "stage1_true_retained": 2,
        "stage2_true_retained": 2,
        "stage1_empty": 1,
        "stage2_empty": 2,
        "final_correct_singleton": 2,
        "final_wrong_singleton": 1,
        "any_false_support": 0,
        "any_false_rejection": 1,
        "any_false_conflict": 2,
        "any_false_claim": 2,
    }


def test_false_support_persists_in_run_error_after_final_empty_set():
    metrics = reference.score_masks(
        np.array([1, 16, 0], dtype=np.uint16),
        np.array([0, 16, 0], dtype=np.uint16),
        4,
    )
    assert metrics["any_false_support"] == metrics["any_false_claim"] == 1
    assert metrics["any_false_rejection"] == metrics["any_false_conflict"] == 0
    assert metrics["stage2_empty"] == 2
    assert metrics["final_correct_singleton"] == 1
    assert metrics["final_wrong_singleton"] == 0


@pytest.mark.parametrize(
    "observed_world,truth_id,support,rejection,conflict",
    [(0, 4, 1, 0, 0), (4, 0, 0, 1, 1)],
)
def test_adversarial_observations_produce_detectable_false_claims(
    observed_world, truth_id, support, rejection, conflict
):
    misleading = reference.catalog()[observed_world]
    first, second = reference.classify([misleading.mu_x], [misleading.mu_ligand])
    metrics = reference.score_masks(first, second, truth_id)
    assert metrics["stage1_true_retained"] == 0
    assert metrics["stage2_true_retained"] == 0
    assert metrics["final_wrong_singleton"] == 1
    assert metrics["any_false_support"] == support
    assert metrics["any_false_rejection"] == rejection
    assert metrics["any_false_conflict"] == conflict
    assert metrics["any_false_claim"] == 1


@pytest.mark.parametrize("field", ["trials_per_world", "seed"])
@pytest.mark.parametrize("value", [True, False, 2.5, "5", None])
def test_config_rejects_noninteger_values(field, value):
    with pytest.raises((ValueError, TypeError)):
        reference.ReferenceConfig(**{field: value})


@pytest.mark.parametrize(
    "settings", [{"trials_per_world": 0}, {"trials_per_world": -1}, {"seed": -1}]
)
def test_config_rejects_invalid_integer_domains(settings):
    with pytest.raises((ValueError, TypeError)):
        reference.ReferenceConfig(**settings)


def test_config_is_frozen_and_seed_zero_is_valid():
    config = reference.ReferenceConfig(trials_per_world=3, seed=0)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        config.seed = 8


@pytest.mark.parametrize("mask", [-1, 512, True, 1.5, "1"])
@pytest.mark.parametrize("method", [reference.project, reference.diagnosis])
def test_invalid_retained_masks_fail_closed(method, mask):
    with pytest.raises((ValueError, TypeError)):
        method(mask)


@pytest.mark.parametrize(
    "x,ligand",
    [
        ([math.nan], [0.0]),
        ([0.0], [math.inf]),
        ([[0.0]], [[0.0]]),
        ([0.0, 1.0], [0.0]),
    ],
)
def test_classification_rejects_invalid_observation_arrays(x, ligand):
    with pytest.raises((ValueError, TypeError)):
        reference.classify(np.asarray(x), np.asarray(ligand))


@pytest.mark.parametrize(
    "first,second,truth",
    [([1], [2], 0), ([512], [0], 0), ([1], [1], 9), ([1], [1], True)],
)
def test_scoring_rejects_invalid_or_nonnested_inputs(first, second, truth):
    with pytest.raises((ValueError, TypeError)):
        reference.score_masks(np.asarray(first), np.asarray(second), truth)


@pytest.mark.parametrize(
    "rate,time,b",
    [(-0.1, 5.0, 1.0), (0.1, -1.0, 1.0), (0.1, 5.0, 0.0), (math.nan, 5, 1)],
)
def test_response_rejects_invalid_domains(rate, time, b):
    with pytest.raises((ValueError, TypeError)):
        reference.response(rate, time=time, b=b)


@pytest.mark.parametrize("rate", [0.0, 0.2, 1.0, 1.0 - 1e-10, 1.0 + 1e-10])
def test_response_agrees_with_independent_dynamics(rate):
    solved = solve_ivp(
        lambda _time, state: [-rate * state[0], state[0] - state[1]],
        (0.0, 5.0),
        (1.0, 0.0),
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    assert solved.success
    assert reference.response(rate) == pytest.approx(solved.y[1, -1], abs=2e-10)


def test_response_limit_zero_time_and_large_rate_do_not_overflow():
    assert reference.response(1.0) == pytest.approx(5.0 * math.exp(-5.0))
    assert reference.response(0.2, time=0.0) == 0.0
    expected = math.exp(-5.0) / 999_999.0
    assert reference.response(1e6) == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("time", [0.1, 1.2])
@pytest.mark.parametrize("rate", [0.0, 5e-324])
def test_response_preserves_time_limit_at_subnormal_rates(rate, time):
    assert reference.response(rate, time=time, b=5e-324) == pytest.approx(
        time, rel=1e-15, abs=0.0
    )


def test_analytic_probabilities_include_empty_runs_and_rare_errors():
    rows = reference.analytic_reference()
    assert [row["world_id"] for row in rows] == list(range(9))
    for row in rows:
        correct = row["final_correct_singleton_probability"]
        wrong = row["final_wrong_singleton_probability"]
        empty = row["stage2_empty_probability"]
        assert correct == pytest.approx(0.975**2, abs=2e-13)
        assert correct + wrong + empty == pytest.approx(1.0, abs=2e-13)
        assert 0 <= row["any_false_claim_probability"] <= 1 - 0.975**2 + 2e-13
        assert 0 <= row["stage1_empty_probability"] <= 0.025 + 2e-13
        assert wrong >= 0 and empty >= 0


def test_partial_resume_matches_full_run_and_preserves_denominator(tmp_path):
    config = reference.ReferenceConfig(trials_per_world=7, seed=20261028)
    resumed_root, full_root = tmp_path / "resumed", tmp_path / "full"
    partial = reference.run_control(resumed_root, config, max_new_worlds=2)
    assert partial["expected_datasets"] == 63
    assert partial["recorded_datasets"] == 14
    assert partial["completed_worlds"] == [0, 1]
    assert partial["missing_worlds"] == list(range(2, 9))
    assert not partial["complete"]
    before = snapshot(resumed_root)
    reported = reference.report_control(resumed_root)
    assert scientific_summary(reported) == scientific_summary(partial)
    assert snapshot(resumed_root) == before
    resumed = reference.run_control(resumed_root, config)
    full = reference.run_control(full_root, config)
    assert scientific_summary(resumed) == scientific_summary(full)
    assert resumed["complete"] and resumed["recorded_datasets"] == 63
    assert resumed["completed_worlds"] == list(range(9))
    assert resumed["missing_worlds"] == []
    assert sum(row["trials"] for row in resumed["per_world"]) == 63
    assert snapshot(resumed_root) == snapshot(full_root)


def test_zero_budget_report_has_no_observed_denominator(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Zero-world run sampled a dataset")

    monkeypatch.setattr(reference, "_draw_world", forbidden)
    config = reference.ReferenceConfig(trials_per_world=3, seed=0)
    summary = reference.run_control(tmp_path, config, max_new_worlds=0)
    assert summary["expected_datasets"] == 27
    assert summary["recorded_datasets"] == 0
    assert summary["completed_worlds"] == summary["per_world"] == []
    assert summary["missing_worlds"] == list(range(9))
    assert not summary["complete"]
    assert all(count == 0 for count in summary["totals"].values())
    before = snapshot(tmp_path)
    assert reference.report_control(tmp_path) == summary
    assert snapshot(tmp_path) == before


def test_completed_resume_preserves_data_and_artifacts(tmp_path, monkeypatch):
    config = reference.ReferenceConfig(trials_per_world=3, seed=7)
    initial = reference.run_control(tmp_path, config)
    before = snapshot(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Completed resume regenerated a dataset")

    monkeypatch.setattr(reference, "_draw_world", forbidden)
    resumed = reference.run_control(tmp_path, config)
    assert scientific_summary(resumed) == scientific_summary(initial)
    assert snapshot(tmp_path) == before


def test_resume_replays_world_interrupted_after_rng_draws(tmp_path, monkeypatch):
    config = reference.ReferenceConfig(trials_per_world=7, seed=23)
    resumed_root, full_root = tmp_path / "interrupted", tmp_path / "full"
    original = reference._draw_world

    def interrupted(rng, world, settings):
        values = original(rng, world, settings)
        if world.world_id == 2:
            raise RuntimeError("interruption after draws, before checkpoint")
        return values

    with monkeypatch.context() as patch:
        patch.setattr(reference, "_draw_world", interrupted)
        with pytest.raises(RuntimeError, match="interruption"):
            reference.run_control(resumed_root, config)
    partial = reference.report_control(resumed_root)
    assert partial["recorded_datasets"] == 14
    assert partial["completed_worlds"] == [0, 1]
    assert not (resumed_root / "world-2.json").exists()
    resumed = reference.run_control(resumed_root, config)
    full = reference.run_control(full_root, config)
    assert resumed == full
    assert snapshot(resumed_root) == snapshot(full_root)


def test_resume_rejects_mismatched_plan_without_overwriting(tmp_path):
    config = reference.ReferenceConfig(trials_per_world=4, seed=7)
    reference.run_control(tmp_path, config, max_new_worlds=1)
    before = snapshot(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        reference.run_control(
            tmp_path, reference.ReferenceConfig(trials_per_world=4, seed=8)
        )
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("drift", ["source", "runtime"])
def test_source_or_runtime_drift_blocks_resume_and_report(
    tmp_path, monkeypatch, drift
):
    config = reference.ReferenceConfig(trials_per_world=2, seed=4)
    reference.run_control(tmp_path, config, max_new_worlds=1)
    before = snapshot(tmp_path)
    original_plan = reference._plan

    def changed_plan(settings):
        plan = original_plan(settings)
        if drift == "source":
            plan["source_sha256"] = "changed implementation digest"
        else:
            plan["runtime"]["numpy"] = "changed numerical runtime"
        return plan

    monkeypatch.setattr(reference, "_plan", changed_plan)
    with pytest.raises((ValueError, RuntimeError)):
        reference.run_control(tmp_path, config)
    with pytest.raises((ValueError, RuntimeError)):
        reference.report_control(tmp_path)
    assert snapshot(tmp_path) == before


def test_tampered_world_checkpoint_blocks_report_and_resume(tmp_path):
    config = reference.ReferenceConfig(trials_per_world=4, seed=8)
    reference.run_control(tmp_path, config, max_new_worlds=1)
    path = tmp_path / "world-0.json"
    sealed = json.loads(path.read_text())
    assert set(sealed) == {"sha256", "value"}
    sealed["value"]["unauthorized_change"] = True
    path.write_text(json.dumps(sealed) + "\n")
    before = snapshot(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        reference.report_control(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        reference.run_control(tmp_path, config)
    assert snapshot(tmp_path) == before


def test_stale_completion_summary_cannot_hide_missing_world(tmp_path):
    config = reference.ReferenceConfig(trials_per_world=2, seed=1)
    reference.run_control(tmp_path, config)
    (tmp_path / "world-8.json").unlink()
    before = snapshot(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        reference.report_control(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        reference.run_control(tmp_path, config)
    assert snapshot(tmp_path) == before
