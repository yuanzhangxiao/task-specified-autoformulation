"""Phase B mechanism scores: pathways, missing pathways, failures, official NMSE."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from scripts import score_phase_b_mechanisms as pb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw"
RELEASE = Path(
    "/Users/yuanzhangxiao/Projects/autoformalism/outputs/release/phase-b-nine-case-v1"
)
T1_HARD = "phase_b_dalla_man_t1_canonical_named_hard"
T2_HARD = "phase_b_dalla_man_t2_canonical_named_hard"
T2_EASY = "phase_b_dalla_man_t2_canonical_named_easy"


def _row(cell, equations, targets, auxiliaries, inputs, parameters):
    """A ready row of our grammar with the given state equations."""
    states = [
        {
            "description": "s",
            "kind": "observed" if s in targets else "latent",
            "mechanisms": [],
            "name": s,
            "unit": "unspecified",
        }
        for s in equations
    ]
    candidate = {
        "candidate_id": "t",
        "change_summary": "t",
        "schema_version": "1",
        "parent_candidate_id": None,
        "states": states,
        "processes": [],
        "state_equations": [{"state": s, "rhs": r} for s, r in equations.items()],
        "initial_conditions": [
            {
                "expression": s["name"] if s["kind"] == "observed" else "0",
                "fixed_value": None,
                "initialization_range": None,
                "scope": "global",
                "state": s["name"],
            }
            for s in states
        ],
        "observation_mappings": [
            {"channel": t, "expression": t, "unit": "unspecified"} for t in targets
        ],
        "parameters": [
            {
                "bounds": None,
                "description": "unspecified",
                "domain": "real",
                "initialization_range": None,
                "name": n,
                "role": "coefficient",
                "scope": "global",
                "unit": "unspecified",
            }
            for n in parameters
        ],
        "constraints": [],
        "equations": [],
    }
    context = {
        "auxiliaries": list(auxiliaries),
        "external_inputs": list(inputs),
        "fitted_initialization": True,
        "fixed_covariates": [],
        "forbid_latent_states": False,
        "forcing_bounds": {},
        "lagged_targets": [],
        "targets": list(targets),
        "time_symbol": "t",
        "unavailable_observed_channels": [],
    }
    model = {
        "candidate": candidate,
        "context": context,
        "parameters": dict.fromkeys(parameters, 0.1),
        "initials": {},
        "semantics": "continuous_time",
        "grammar": "default",
    }
    return {
        "method": "full",
        "benchmark_id": cell,
        "repetition": 0,
        "status": "ready",
        "model": model,
    }


def _groups(row, channels):
    refs = {"names": {c: c for c in channels}, "cases": []}
    entry = pb.score_row(row, refs, {}, None)
    return {g["expression"]: g["group"] for g in entry["groups"]}, entry


def test_time_and_deviations_neither_mix_nor_split_pathways():
    row = _row(
        T2_HARD,
        {
            "Gp": "a * Qg - b * Gp * (X - c) - Uii + d * t",
            "I": "insulin_pmol_per_kg_min - q * I",
            "Qg": "k * meal_event_g - k * Qg",
            "X": "p * (I - X)",
        },
        ("Gp", "I"),
        ("Uii",),
        ("meal_event_g", "insulin_pmol_per_kg_min"),
        ("a", "b", "c", "d", "k", "p", "q"),
    )
    groups, entry = _groups(
        row, ("Gp", "I", "Uii", "meal_event_g", "insulin_pmol_per_kg_min")
    )
    assert groups == {
        "a * Qg": "meal",
        "b * Gp * (X - c)": "insulin",
        "Uii": "supplied",
        "d * t": "remainder",
    }
    assert entry["meal_states"] == ["Qg"] and entry["insulin_states"] == ["X"]
    assert entry["missing"] == []


def test_insulin_acting_through_a_modeled_disposal_is_the_insulin_pathway():
    row = _row(
        T2_EASY,
        {
            "Gp": "EGP - U + a * meal_event_g",
            "I": "insulin_pmol_per_kg_min - q * I",
            "U": "r * (Uii + v * X * Gp - U)",
            "X": "p * (I - X)",
        },
        ("Gp", "I", "U"),
        ("EGP", "Uii"),
        ("meal_event_g", "insulin_pmol_per_kg_min"),
        ("a", "p", "q", "r", "v"),
    )
    groups, entry = _groups(
        row, ("Gp", "I", "U", "EGP", "Uii", "meal_event_g", "insulin_pmol_per_kg_min")
    )
    assert groups == {"EGP": "supplied", "U": "insulin", "a * meal_event_g": "meal"}
    assert "U" in entry["insulin_carriers"] and entry["missing"] == []


def test_a_model_without_meal_or_insulin_terms_misses_those_pathways():
    row = _row(
        T2_HARD,
        {"Gp": "c - k * Gp", "I": "insulin_pmol_per_kg_min - q * I"},
        ("Gp", "I"),
        ("Uii",),
        ("meal_event_g", "insulin_pmol_per_kg_min"),
        ("c", "k", "q"),
    )
    groups, entry = _groups(
        row, ("Gp", "I", "Uii", "meal_event_g", "insulin_pmol_per_kg_min")
    )
    assert set(groups.values()) == {"remainder"}
    assert entry["missing"] == ["meal", "insulin"]


def test_failed_cases_count_as_worst_and_nmse_is_the_trajectory_mean():
    good = {"A": 1.0, "RSE": 0.2, "S": 0.1}
    entry = {
        "status": "ready",
        "cases": [
            {"status": "complete", "nmse": 0.5, "target": good, "meal": good},
            {"status": "complete", "nmse": 1.5, "target": good},
            {"status": "failed: x", "target": pb.FAILED, "meal": pb.FAILED},
        ],
    }
    s = pb.summary(entry)
    assert s["failed"] == 1 and s["test_nmse"] == math.inf
    assert s["meal"] == (0.5, (0.2 + math.inf) / 2, 0.55)
    assert s["target"] == pytest.approx((2 / 3, 0.2, 0.4))
    entry["cases"].pop()
    assert pb.summary(entry)["test_nmse"] == pytest.approx(1.0)
    assert pb.summary({"status": "method_failure"}) is None


def test_windows_bracket_the_command():
    u = np.zeros(10)
    u[3:5] = 1.0
    pulse, after = pb.windows(u)
    assert np.flatnonzero(pulse).tolist() == [2, 3, 4, 5]
    assert np.flatnonzero(after).tolist() == [6, 7, 8, 9]
    assert not any(a.any() for a in pb.windows(np.zeros(5)))


@pytest.mark.skipif(
    not (DATA.exists() and RELEASE.exists()), reason="Phase B data absent"
)
def test_a_zero_model_reproduces_the_official_test_nmse(tmp_path):
    """The submitted SINDy's zero model on T1-hard scored 2.136 (mean of six)."""
    from autoformalism.rebuttal import phase_b_nine_case as nine
    from autoformalism.rebuttal import phase_c_baseline_mechanisms as mechanisms
    from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate

    context = nine.load_cell(RELEASE, T1_HARD, context="baseline").context
    model = mechanisms._model(
        equation_candidate("sindy", {"Gp": "0"}, context),
        {},
        context,
        "continuous_time",
    )
    row = {
        "method": "sindy",
        "benchmark_id": T1_HARD,
        "repetition": 0,
        "status": "ready",
        "model": model,
    }
    record = pb.main({}, RELEASE, DATA, tmp_path, extra_rows=[row])
    (entry,) = record["entries"]
    assert entry["summary"]["test_nmse"] == pytest.approx(2.136011151353723, rel=1e-6)
    assert entry["missing"] == ["meal"] and "meal" not in entry["summary"]
    assert entry["summary"]["target"] == (0.0, 1.0, 1.0)


def test_more_insulin_action_makes_the_true_insulin_terms_more_negative():
    from types import SimpleNamespace

    from autoformalism.rebuttal.dalla_man import (
        STATE_INDEX,
        DallaManParameters,
        compute_dalla_man_basal,
    )

    basal = compute_dalla_man_basal(DallaManParameters()).initial_state
    names = tuple(sorted(STATE_INDEX, key=STATE_INDEX.get))
    states = np.tile(np.asarray(basal, float), (3, 1))
    states[1, STATE_INDEX["X"]] += 0.01
    states[2, STATE_INDEX["Id"]] *= 1.5
    private = SimpleNamespace(states=states, state_names=names)
    terms = pb.insulin_terms(private)
    assert terms[1] < terms[0] and terms[2] < terms[0]
    assert pb.insulin_terms(SimpleNamespace(states=states[:1], state_names=names))[
        0
    ] == pytest.approx(terms[0])


@pytest.mark.skipif(not DATA.exists(), reason="Phase B data absent")
def test_phase_c_cells_render_the_same_six_test_protocols():
    for cell in (
        "phase_c_dalla_man_t2_canonical_named_hard_rates_v1",
        "phase_c_alien_device_unknown_device_mechanism_canonical_functional_easy_reset_v1",
    ):
        refs = pb.references(cell, DATA)
        assert len(refs["cases"]) == 6
    assert pb.CONTRACT[cell] == "continuous-rates-1"
    assert sorted(refs["names"]) == ["u01", "v01", "v02", "v03"]
