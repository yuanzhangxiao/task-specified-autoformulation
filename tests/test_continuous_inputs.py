"""Input mass, true-reference/production-rollout agreement and release isolation."""

import json
from pathlib import Path

import numpy as np
import pytest

from autoformalism.benchmarks.phase_b_generation import (
    phase_b_protocols,
    simulate_phase_b,
)
from autoformalism.benchmarks.phase_b_public import (
    phase_b_public_spec,
    render_phase_b_prompts,
    write_public_production_bundle,
    write_public_staging_bundle,
)
from autoformalism.benchmarks.reference_audit import audit_one, check_identities
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, Trajectory
from autoformalism.data.exceptions import ChannelRoleError
from autoformalism.data.registry import BenchmarkRegistry, continuous_phase_b_specs
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.input_schedule import LinearInputSchedule, meal_rate_schedule
from autoformalism.rebuttal.dalla_man import STATE_INDEX, simulate_dalla_man
from autoformalism.reference_integration import ReferenceSolver
from autoformalism.schemas import CandidateModel

CONTRACT = "continuous-rates-1"


@pytest.mark.parametrize("dt", [0.5, 1, 5, 20])
@pytest.mark.parametrize(
    "meals", [((0, 60),), ((0.37, 60),), ((20, 60),), ((0, 30), (0, 30), (2.37, 45))]
)
def test_mass_independent_of_sampling_start_overlap_and_end(dt, meals):
    schedule = meal_rate_schedule(meals, 40, dt, ingestion_minutes=20)
    mass = np.sum(
        np.diff(schedule.time) * (schedule.values[:-1, 0] + schedule.values[1:, 0]) / 2
    )
    assert mass == pytest.approx(sum(m for _, m in meals), abs=1e-10)
    assert np.min(schedule.values) >= 0
    assert schedule.values[0, 0] == 0
    for start, _ in meals:
        assert start in schedule.time
        assert start + 10 in schedule.time
        assert start + 20 in schedule.time


@pytest.mark.parametrize(
    "meals,duration",
    [(((25, 60),), 30), (((-1, 60),), 30), (((0, -1),), 30), (((np.nan, 1),), 30)],
)
def test_invalid_or_truncated_meals_fail_instead_of_losing_mass(meals, duration):
    with pytest.raises(ValueError):
        meal_rate_schedule(meals, duration, 1)


def test_invalid_input_tables_and_outside_queries_are_rejected():
    with pytest.raises(ValueError):
        LinearInputSchedule(np.array([0, 0]), np.zeros((2, 1)))
    with pytest.raises(ValueError):
        LinearInputSchedule(np.array([0, 1]), np.array([[0], [np.nan]]))
    schedule = meal_rate_schedule((), 10, 1)
    with pytest.raises(ValueError):
        schedule.at(11)


def test_decimal_knot_does_not_create_zero_length_solver_segment():
    schedule = meal_rate_schedule(((0.3, 1),), 2, 0.1, 0.2)
    assert np.diff(schedule.time).min() > 0.09
    result = simulate_dalla_man(
        meals=((0.3, 1),),
        duration=2,
        dt=0.1,
        variant="original",
        continuous_inputs=True,
        ingestion_minutes=0.2,
        input_dt=0.1,
    )
    assert np.isfinite(result.states).all()


@pytest.mark.parametrize("method", ["Radau", "DOP853"])
def test_production_rollout_matches_finite_ingestion_reference(method):
    reference = simulate_dalla_man(
        meals=((0, 30), (2.37, 30)),
        duration=30,
        dt=5,
        variant="original",
        continuous_inputs=True,
        input_dt=5,
        solver=ReferenceSolver(method="LSODA"),
    )
    q = reference.states[:, STATE_INDEX["Qsto1"]]
    assert q[0] == 0  # The meal at zero does not reset a state.
    row = Trajectory(
        "pulse",
        reference.time,
        {"q": q},
        {},
        {"u": reference.derived["meal_rate_g_per_min"]},
        {},
        {},
    )
    model = CandidateModel.model_validate(
        {
            "candidate_id": "numerical_oracle",
            "parent_candidate_id": None,
            "states": [{"name": "q", "kind": "observed"}],
            "state_equations": [{"state": "q", "rhs": "1000*u - 0.0558*q"}],
            "observation_mappings": [{"channel": "q", "expression": "q"}],
            "initial_conditions": [
                {"state": "q", "expression": "0", "scope": "global"}
            ],
        }
    )
    compiled = compile_candidate(
        model, ValidationContext(targets=("q",), external_inputs=("u",))
    )
    result = simulate_trajectory(
        compiled,
        row,
        {},
        {},
        FitConfig(
            integration_method=method,
            relative_tolerance=1e-10,
            absolute_tolerance=1e-12,
        ),
        reset_observed_states=False,
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.predictions["q"], q, rtol=2e-8, atol=2e-5)


@pytest.mark.parametrize(
    "family,task", [("dalla_man", "T2"), ("cstr", None), ("alien_device", None)]
)
def test_continuous_protocol_independent_solver_grid_and_checkpoint(
    tmp_path, family, task
):
    protocol = phase_b_protocols(family, task=task, input_contract=CONTRACT)[1]
    record, reference = audit_one(
        tmp_path, protocol, "canonical", Path("data_raw"), "rates-test"
    )
    assert record["passed"]
    assert reference.input_contract == CONTRACT
    assert (
        record
        == audit_one(tmp_path, protocol, "canonical", Path("data_raw"), "rates-test")[0]
    )
    assert check_identities(protocol, reference)["passed"]


def test_new_public_contract_and_obfuscation_do_not_change_old_prompts(tmp_path):
    legacy = phase_b_public_spec("dalla_man", "easy", "named")
    spec = phase_b_public_spec("dalla_man", "easy", "named", input_contract=CONTRACT)
    hidden = phase_b_public_spec(
        "dalla_man", "easy", "obfuscated", input_contract=CONTRACT
    )
    assert "meal_event_g(t)" in render_phase_b_prompts(legacy)[0]
    prompt = render_phase_b_prompts(spec)[0]
    assert "meal_rate_g_per_min(t)" in prompt and "g min^-1" in prompt
    assert "piecewise-linear" in prompt and "state jumps" in prompt
    hidden_prompt = render_phase_b_prompts(hidden)[0]
    assert "meal_rate_g_per_min" not in hidden_prompt and "u01(t)" in hidden_prompt
    p = phase_b_protocols("dalla_man", input_contract=CONTRACT)[1]
    reference = simulate_phase_b(p)
    with pytest.raises(ValueError, match="input contract"):
        write_public_staging_bundle(tmp_path / "bad", legacy, (reference,))
    write_public_staging_bundle(tmp_path / "good", spec, (reference,))
    manifest = json.loads((tmp_path / "good/manifest.json").read_text())
    assert manifest["input_contract"] == CONTRACT
    assert not (tmp_path / "good/test.csv").exists()


def test_explicit_registry_loads_new_irregular_grid_and_rejects_wrong_contract(
    tmp_path,
):
    protocols = phase_b_protocols("dalla_man", input_contract=CONTRACT)
    base = protocols[1].model_copy(
        update={"duration": 30, "specification": {"meals": [[0.37, 30]]}}
    )
    rows = tuple(
        simulate_phase_b(
            base.model_copy(
                update={
                    "protocol_id": f"{split}_unit",
                    "split": split,
                }
            )
        )
        for split in ("train", "validation", "test")
    )
    spec = phase_b_public_spec("dalla_man", "easy", "named", input_contract=CONTRACT)
    path = tmp_path / "phase_b_continuous_inputs_v1" / spec.benchmark_id
    write_public_production_bundle(path, spec, rows)
    registry = BenchmarkRegistry(continuous_phase_b_specs())
    assert len(registry.identifiers()) == 40
    assert len(BenchmarkRegistry().identifiers()) == 46
    config = DataConfig(root=tmp_path, benchmark_id=spec.benchmark_id, tier="easy")
    loader = BenchmarkLoader(registry)
    dataset = loader.load_development(config)
    assert dataset.train.trajectories[0].number_of_rows > 31
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_contract"] = "legacy-events-1"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ChannelRoleError, match="input contract"):
        loader.load_development(config)
