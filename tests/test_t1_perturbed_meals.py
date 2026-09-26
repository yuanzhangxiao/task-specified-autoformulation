"""Intervention diagnostics must freeze models, retain controls and separate truth."""

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest

from scripts import compare_t1_perturbed_meals as probe


def test_fixed_grid_and_matched_mass():
    cases = {row["id"]: row for row in probe.cases()}
    assert len(cases) == 11
    for case in cases.values():
        for field in ("control", "comparison"):
            if case[field]:
                assert cases[case[field]]["duration"] == case["duration"]
    for gap in (15, 30, 60, 120):
        assert sum(m[1] for m in cases[f"split_gap{gap}"]["meals"]) == 60
    assert cases["washout_pair"]["meals"] == [[60, 60], [540, 60]]


def test_conditional_probe_changes_only_input_and_does_not_mutate():
    base = {
        "trajectory_id": "base",
        "time": [0, 1],
        "targets": {"v01": [3, 4]},
        "auxiliaries": {"v02": [5, 6]},
        "external_inputs": {"u01": [0, 0]},
    }
    pulse = copy.deepcopy(base)
    pulse["external_inputs"]["u01"] = [0, 60]
    pulse["targets"]["v01"] = [3, 90]
    pulse["auxiliaries"]["v02"] = [5, 90]
    mixed = probe.conditional_row(pulse, base)
    assert mixed["targets"] == base["targets"]
    assert mixed["auxiliaries"] == base["auxiliaries"]
    assert mixed["external_inputs"] == pulse["external_inputs"]
    mixed["auxiliaries"]["v02"][1] = 100
    assert base["auxiliaries"]["v02"][1] == 6
    pulse["time"] = [0, 2]
    with pytest.raises(ValueError, match="grids differ"):
        probe.conditional_row(pulse, base)


def test_reference_uses_perturbation_and_two_solvers_agree():
    plan = {
        "reference_parameters": asdict(probe.reference.DallaManParameters()),
        "reference_variant": "perturbed_b1",
        "dt": 1,
        "reference_rtol": 1e-10,
        "reference_atol": 1e-12,
        "reference_max_step": 0.5,
    }
    case = {"id": "short", "duration": 12, "meals": [[1, 60]]}
    a = probe.generate_reference(case, plan, "Radau")
    b = probe.generate_reference(case, plan, "DOP853")
    canonical = probe.generate_reference(
        case, dict(plan, reference_variant="original"), "Radau"
    )
    np.testing.assert_allclose(
        a["targets"]["v01"], b["targets"]["v01"], atol=2e-5, rtol=0
    )
    assert not np.allclose(a["targets"]["v01"], canonical["targets"]["v01"], atol=1e-4)
    assert a["external_inputs"]["u01"][1] == 60


def test_freeze_rejects_missing_endpoint_before_any_replay(tmp_path):
    models = tmp_path / "models.json"
    models.write_text(json.dumps([{"label": probe.IDS[0]}]))
    with pytest.raises(ValueError, match="all three Sol"):
        probe.freeze(models, tmp_path / "absent", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_paired_score_requires_reliable_control():
    from scripts.verify_t1_perturbed_meals import metric_reliability

    case = {"id": "pair", "control": "fasting", "comparison": "single"}
    flags = metric_reliability(case, {"pair": True, "fasting": False, "single": True})
    assert flags == {
        "absolute_verified": True,
        "response_verified": False,
        "schedule_difference_verified": True,
    }
    assert not any(metric_reliability(case, {}).values())


def test_small_complete_run_and_resume(tmp_path, monkeypatch):
    from autoformalism.expressions import ValidationContext
    from autoformalism.schemas import CandidateModel
    from scripts import replay_t1_curves

    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "synthetic",
            "parent_candidate_id": None,
            "states": [{"name": "v01", "kind": "observed"}],
            "state_equations": [{"state": "v01", "rhs": "u01"}],
            "observation_mappings": [{"channel": "v01", "expression": "v01"}],
            "initial_conditions": [
                {"state": "v01", "expression": "v01", "scope": "global"}
            ],
        }
    )
    context = ValidationContext(targets=("v01",), external_inputs=("u01",)).model_dump(
        mode="json"
    )
    fixed = [
        {
            "id": "fasting_300",
            "duration": 2,
            "meals": [],
            "control": None,
            "comparison": None,
        },
        {
            "id": "single_60",
            "duration": 2,
            "meals": [[1, 1]],
            "control": "fasting_300",
            "comparison": None,
        },
    ]
    plan = {
        "identity": probe.identity(),
        "models": [
            {
                "id": "test",
                "candidate": candidate.model_dump(mode="json"),
                "context": context,
                "parameters": {},
                "initials": {},
            }
        ],
        "cases": fixed,
        "training_scale": 1.0,
        "reference_solvers": ["Radau", "DOP853"],
        "agreement_tolerance": 2e-5,
        "scope": "test",
    }
    probe.sealed_write(tmp_path / "plan.json", plan)

    def fake_reference(case, plan, method):
        return {
            "trajectory_id": case["id"],
            "time": [0.0, 1.0, 2.0],
            "targets": {"v01": [2.0, 2.0, 2.0]},
            "auxiliaries": {},
            "external_inputs": {"u01": [0.0, 1.0 if case["meals"] else 0.0, 0.0]},
        }

    monkeypatch.setattr(probe, "generate_reference", fake_reference)
    result = probe.run(tmp_path)
    assert result["status"] == "complete"
    assert result["conditional"][0]["max_delta"] == pytest.approx(1.0)
    physical = probe.sealed_read(tmp_path / "replays/test/single_60.json")
    assert physical["predicted"][-1] == pytest.approx(3.0)  # Never reset to measured 2.
    conditional = probe.sealed_read(tmp_path / "conditional/test.json")
    assert "nmse" not in conditional and "observed" not in conditional
    monkeypatch.setattr(
        replay_t1_curves, "simulate_trajectory", lambda *a, **kw: pytest.fail("cached")
    )
    monkeypatch.setattr(probe, "generate_reference", lambda *a: pytest.fail("cached"))
    assert probe.run(tmp_path) == result
    # A tampered frozen plan is not silently accepted.
    path = tmp_path / "plan.json"
    value = json.loads(path.read_text())
    value["training_scale"] = 3
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="digest differs"):
        probe.run(tmp_path)
