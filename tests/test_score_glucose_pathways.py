"""Plasma-glucose pathways of a frozen T1 model, by the names each term uses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import score_glucose_pathways as pathways

EXPORT = Path(
    "/Users/yuanzhangxiao/Projects/autoformalism/output/iclr-t1-perturbed-full-2353678"
)
CELL = "phase_c_dalla_man_t1_perturbed_named_easy_rates_v1"
CONTEXT = {
    "auxiliaries": ["EGP", "Uii", "E", "Gt"],
    "external_inputs": ["meal_rate_g_per_min"],
    "fitted_initialization": True,
    "fixed_covariates": [],
    "forbid_latent_states": False,
    "forcing_bounds": {},
    "lagged_targets": [],
    "targets": ["Gp"],
    "time_symbol": "t",
    "unavailable_observed_channels": [],
}


def _parameter(name: str) -> dict:
    return {
        "bounds": None,
        "description": "unspecified",
        "domain": "nonnegative",
        "initialization_range": None,
        "name": name,
        "role": "nonnegative_coefficient",
        "scope": "global",
        "unit": "unspecified",
    }


def _row(gp_rhs: str, *, extra_process: dict | None = None) -> dict:
    names = ["a", "b", "c", "d", "e", "k1", "k2", "g", "init_M"]
    processes = [
        {
            "description": "supplied production",
            "expression": "(EGP)",
            "mechanisms": [],
            "name": "production",
            "unit": "unspecified",
        }
    ] + ([extra_process] if extra_process else [])
    return {
        "status": "ready",
        "repetition": 0,
        "candidate": {
            "candidate_id": "test",
            "change_summary": "test",
            "schema_version": "1",
            "parent_candidate_id": None,
            "states": [
                {
                    "description": "gut",
                    "kind": "latent",
                    "mechanisms": [],
                    "name": "M",
                    "unit": "unspecified",
                },
                {
                    "description": "plasma glucose mass",
                    "kind": "observed",
                    "mechanisms": [],
                    "name": "Gp",
                    "unit": "unspecified",
                },
            ],
            "processes": processes,
            "state_equations": [
                {"state": "M", "rhs": "a * meal_rate_g_per_min - b * M"},
                {"state": "Gp", "rhs": gp_rhs},
            ],
            "initial_conditions": [
                {
                    "expression": "init_M",
                    "fixed_value": None,
                    "initialization_range": None,
                    "scope": "global",
                    "state": "M",
                },
                {
                    "expression": "Gp",
                    "fixed_value": None,
                    "initialization_range": None,
                    "scope": "global",
                    "state": "Gp",
                },
            ],
            "observation_mappings": [
                {"channel": "Gp", "expression": "Gp", "unit": "unspecified"}
            ],
            "parameters": [_parameter(n) for n in names],
            "constraints": [],
            "equations": [],
        },
        "context": CONTEXT,
        "parameters": dict.fromkeys(names, 0.1),
        "initials": {},
    }


def test_terms_are_assigned_by_the_names_they_use():
    row = _row("c * M + d * production - e * Uii - k1 * Gp + k2 * Gt")
    assignment = pathways.attribute(row)
    assert assignment["meal_states"] == ["M"]
    by_pathway = {}
    for term in assignment["terms"]:
        by_pathway.setdefault(term["pathway"], []).append(
            (term["sign"], term["expression"])
        )
    assert by_pathway["meal"] == [(1, "c * M")]
    assert by_pathway["supplied"] == [(1, "d * production"), (-1, "e * Uii")]
    assert by_pathway["exchange"] == [(-1, "k1 * Gp"), (1, "k2 * Gt")]
    assert assignment["exchange_is_a_law_of_gp_gt"]
    assert not assignment["represented"]["mixed"]


def test_a_term_mixing_the_meal_with_glucose_is_mixed_and_a_latent_exchange_is_no_law():
    row = _row("c * M * Gp + g * M - k1 * Gp")
    row["candidate"]["state_equations"][0]["rhs"] = (
        "a * meal_rate_g_per_min - b * M * Gp"
    )
    assignment = pathways.attribute(row)
    assert assignment["meal_states"] == []
    assert {t["pathway"] for t in assignment["terms"]} == {"exchange"}
    assert not assignment["exchange_is_a_law_of_gp_gt"]
    row = _row("c * M * Gt + k1 * Gp")
    assignment = pathways.attribute(row)
    assert [t["pathway"] for t in assignment["terms"]] == ["mixed", "exchange"]


def test_measures_of_a_scaled_response():
    time = np.linspace(0, 10, 101)
    true = np.sin(time / 10 * np.pi)
    result = pathways.measures(0.5 * true, true, time)
    assert result["rse"] == pytest.approx(0.25)
    assert result["A"] == pytest.approx(1.0)
    assert result["S"] == pytest.approx(0.0, abs=1e-12)
    assert result["gain"] == pytest.approx(2.0)
    assert pathways.measures(None, true, time) is None


@pytest.mark.skipif(not EXPORT.exists(), reason="perturbed T1 export absent")
def test_the_frozen_t1_models_reproduce_their_panel_scores():
    out = pathways.score(
        EXPORT / "frozen/frozen_models.json", EXPORT / "panel/references" / CELL
    )
    seed2 = next(m for m in out["models"] if m["seed"] == 2)
    meal = seed2["contrasts"]["fed - fasting"]
    assert meal["Gp"]["rse"] == pytest.approx(0.0474, abs=5e-4)
    assert meal["meal"]["integral"] == pytest.approx(78.4, abs=0.1)
    assert meal["meal"]["true_integral"] == pytest.approx(524.8, abs=0.1)
    assert not seed2["assignment"]["represented"]["exchange"]
    assert meal["exchange"]["integral"] == 0.0
    assert out["best_linear_law"]["explained_variance"] < 0.8
