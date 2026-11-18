"""Memory scores of procedural-device models: inputs, windows, term groups, truth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import score_alien_memory as memory

STRESS = Path(
    "/Users/yuanzhangxiao/Projects/autoformalism/output/iclr-stress-final-20261115"
)
TIME = np.arange(0.0, 24.0 + 1e-9, 0.25)


def _parameter(name):
    return {
        "bounds": None,
        "description": "unspecified",
        "domain": "real",
        "initialization_range": None,
        "name": name,
        "role": "coefficient",
        "scope": "global",
        "unit": "unspecified",
    }


def _row(state_equations, processes=(), observation="v01", states=("m",)):
    names = ["a", "b", "c", "k"]
    all_states = [
        {
            "description": "s",
            "kind": "latent",
            "mechanisms": [],
            "name": s,
            "unit": "unspecified",
        }
        for s in states
    ]
    if any(e["state"] == "v01" for e in state_equations):
        all_states.append(
            {
                "description": "output",
                "kind": "observed",
                "mechanisms": [],
                "name": "v01",
                "unit": "unspecified",
            }
        )
    return {
        "status": "ready",
        "method": "full",
        "repetition": 0,
        "candidate": {
            "candidate_id": "t",
            "change_summary": "t",
            "schema_version": "1",
            "parent_candidate_id": None,
            "states": all_states,
            "processes": [
                {
                    "description": "p",
                    "expression": e,
                    "mechanisms": [],
                    "name": n,
                    "unit": "unspecified",
                }
                for n, e in processes
            ],
            "state_equations": state_equations,
            "initial_conditions": [
                {
                    "expression": "0",
                    "fixed_value": None,
                    "initialization_range": None,
                    "scope": "global",
                    "state": s["name"],
                }
                for s in all_states
            ],
            "observation_mappings": [
                {"channel": "v01", "expression": observation, "unit": "unspecified"}
            ],
            "parameters": [_parameter(n) for n in names],
            "constraints": [],
            "equations": [],
        },
        "context": {
            "auxiliaries": [],
            "external_inputs": ["u01"],
            "fitted_initialization": True,
            "fixed_covariates": [],
            "forbid_latent_states": False,
            "forcing_bounds": {},
            "lagged_targets": [],
            "targets": ["v01"],
            "time_symbol": "t",
            "unavailable_observed_channels": [],
        },
        "parameters": dict.fromkeys(names, 0.5),
        "initials": {},
    }


def test_suite_commands_follow_the_input_rule():
    u = memory.command({"segments": [[4.0, 5.0, 1.0], [4.5, 6.0, 0.5]]}, TIME)
    assert (
        u[TIME == 4.0][0] == 1.0
        and u[TIME == 4.5][0] == 1.5
        and u[TIME == 6.0][0] == 0.5
    )
    assert u[TIME == 3.75][0] == 0.0 and u[TIME == 6.25][0] == 0.0
    s = memory.command(
        {"sine": {"start": 4.0, "period": 8.0, "amplitude": 0.6, "cycles": 1}}, TIME
    )
    assert s[TIME == 6.0][0] == pytest.approx(0.6) and s[TIME == 10.0][
        0
    ] == pytest.approx(-0.6)
    assert s[TIME == 13.0][0] == 0.0


def test_windows_bracket_the_command_with_its_zero_samples():
    u = memory.command({"segments": [[4.0, 5.0, 1.0]]}, TIME)
    pulse, after = memory.windows(u, TIME)
    assert TIME[pulse][0] == 3.75 and TIME[pulse][-1] == 5.25
    assert TIME[after][0] == 5.5 and TIME[after][-1] == 24.0
    step = memory.command({"segments": [[5.0, 24.0, 0.5]]}, TIME)
    assert not memory.windows(step, TIME)[1].any()


def test_output_terms_are_grouped_by_memory_direct_command_and_output_only():
    readout = _row(
        [{"state": "m", "rhs": "a * u01 - b * m"}],
        processes=[("v01", "c * u01 + k * m")],
    )
    dynamic, groups = memory.output_terms(readout)
    assert not dynamic and [g for _, _, g in groups] == ["direct_input", "memory"]
    ode = _row(
        [
            {"state": "m", "rhs": "a * u01"},
            {"state": "v01", "rhs": "k * m - b * v01 + c"},
        ]
    )
    dynamic, groups = memory.output_terms(ode)
    assert dynamic and [g for _, _, g in groups] == [
        "memory",
        "output_only",
        "output_only",
    ]


def test_measures_of_a_scaled_and_an_absent_response():
    true = np.sin(np.linspace(0, np.pi, 50))
    m = memory.measures(0.5 * true, true)
    assert (
        m["RSE"] == pytest.approx(0.25)
        and m["A"] == 1.0
        and m["S"] == pytest.approx(0.0, abs=1e-12)
    )
    assert memory.measures(np.zeros(50), true) == {"A": 0.0, "RSE": 1.0, "S": 1.0}
    assert memory.measures(true, np.zeros(50)) is None


@pytest.mark.skipif(not STRESS.exists(), reason="stress packet absent")
def test_private_equations_reproduce_a_saved_held_out_record():
    device = "device_6d366c058b5d"
    spec = json.loads((STRESS / f"design/evaluator/{device}/system.json").read_text())[
        "value"
    ]["equations"]
    record = json.loads((STRESS / f"evaluation/references/{device}.json").read_text())[
        "value"
    ]["records"][1]
    y = memory.truth(spec, record["perturbed"])["y"]
    assert np.max(np.abs(y - np.asarray(record["perturbed"]["targets"]["v01"]))) < 1e-6
