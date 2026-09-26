"""Package handoff preserves actual execution and does not overclaim evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from autoformalism.data import Trajectory
from autoformalism.rebuttal.mechanism_probes import ProbeSettings, rollout

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "package_mechanisms", SCRIPTS / "assess_results_package_mechanisms.py"
)
assessment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessment)


def source():
    return {
        "method": "sindy",
        "benchmark_id": "fixture",
        "tier": "easy",
        "repetition": 0,
        "request_id": "example",
        "terminal_status": "complete",
        "execution_semantics": None,
        "public_mechanism": {"public_prompt_sha256": "a" * 64},
        "model": {
            "execution_semantics": None,
            "source_provenance": {"adapter": "sindy_result"},
            "validation_context": {"targets": ["y"], "external_inputs": ["u"]},
            "parameterization": {
                "global_parameters": {"k": 1.0},
                "global_initial_conditions": {},
            },
            "candidate": {
                "candidate_id": "example",
                "parent_candidate_id": None,
                "states": [{"name": "y", "kind": "observed"}],
                "state_equations": [{"state": "y", "rhs": "k*u-y"}],
                "observation_mappings": [{"channel": "y", "expression": "y"}],
                "initial_conditions": [
                    {"state": "y", "scope": "global", "expression": "y"}
                ],
                "parameters": [
                    {
                        "name": "k",
                        "role": "rate",
                        "scope": "global",
                        "domain": "positive",
                    }
                ],
            },
        },
    }


def config():
    return {
        "cells": {
            "fixture": {
                "public_prompt_sha256": "a" * 64,
                "requirements": [
                    {
                        "id": "input",
                        "public_requirement": "Input drives output",
                        "kind": "input_path",
                        "driver": "u",
                        "target": "y",
                        "interpretation": "Necessary dependency",
                    }
                ],
            }
        }
    }


def test_saved_parameter_zero_removes_path_and_negative_is_separate():
    saved = source()
    saved["model"]["parameterization"]["global_parameters"]["k"] = 0
    zero = assessment.adapt(saved, config())
    assert zero["status"] == "ready"
    assert zero["fitted_equation_requirements"]["counts"]["fail"] == 1
    saved["model"]["parameterization"]["global_parameters"]["k"] = -1
    negative = assessment.adapt(saved, config())
    assert negative["fitted_equation_requirements"]["counts"]["pass"] == 1
    assert len(negative["declaration_violations"]) == 1
    assert negative["semantics_basis"] == "legacy_adapter_contract_inference"


def test_native_auxiliary_law_is_ignored_and_replay_stays_discrete():
    saved = source()
    saved["method"] = "d3_native_no_tools"
    saved["execution_semantics"] = "discrete_increment_recursive_rollout"
    model = saved["model"]
    model["validation_context"]["auxiliaries"] = ["a"]
    c = model["candidate"]
    c["states"].append({"name": "a", "kind": "observed"})
    c["state_equations"] = [{"state": "y", "rhs": "a"}, {"state": "a", "rhs": "k*u"}]
    c["initial_conditions"].append({"state": "a", "scope": "global", "expression": "a"})
    original = copy.deepcopy(saved)
    row = assessment.adapt(saved, config())
    assert saved == original
    assert row["status"] == "ready"
    assert row["fitted_equation_requirements"]["counts"]["fail"] == 1
    trajectory = Trajectory(
        "test",
        np.array([0.0, 3.0, 6.0]),
        {"y": np.array([0.0, 99.0, 99.0])},
        {"a": np.ones(3)},
        {"u": np.ones(3)},
        {},
        {},
    )
    assert rollout(row, trajectory, "native", ProbeSettings())["y"] == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_native",
        "unknown_legacy",
        "conflict",
        "missing_parameter",
        "wrong_prompt",
    ],
)
def test_reject_ambiguous_or_incomplete_model_inputs(mutation):
    saved = source()
    if mutation == "missing_native":
        saved["method"] = "d3_native_no_tools"
    elif mutation == "unknown_legacy":
        saved["model"]["source_provenance"]["adapter"] = "unrecognized"
    elif mutation == "conflict":
        saved["execution_semantics"] = "continuous_ode_free_rollout"
        saved["model"]["execution_semantics"] = "discrete_increment_recursive_rollout"
    elif mutation == "missing_parameter":
        saved["model"]["parameterization"]["global_parameters"] = {}
    else:
        saved["public_mechanism"]["public_prompt_sha256"] = "b" * 64
    assert assessment.adapt(saved, config())["status"] == "input_failed"


def test_missing_hash_and_unbound_case_are_not_certified():
    saved = source()
    saved["public_mechanism"]["public_prompt_sha256"] = None
    row = assessment.adapt(saved, config())
    assert row["status"] == "ready"
    assert not row["source_public_prompt_verified"]
    assert row["prompt_binding_basis"] == "case_roster_only"
    saved["benchmark_id"] = "phase_b_dalla_man_t2_canonical_named_easy"
    assert assessment.adapt(saved, config())["status"] == "requirements_unbound"


def test_missing_models_are_not_dropped_from_confirmed_evidence_denominator():
    rows = []
    for method in assessment.METHODS:
        for rep in range(3):
            r = assessment.adapt(source(), config())
            r.update(method=method, repetition=rep)
            if rep != 0:
                r.pop("fitted_equation_requirements")
                r["status"] = "unavailable"
            rows.append(r)
    result = assessment.aggregate(rows, ("fixture",), config())
    assert all(r["confirmed_fraction_macro"]["median"] == 0 for r in result)
    assert result[0]["cases"][0]["evidence_counts"]["unassessed"] == 2


def test_hard_t1_binding_matches_committed_public_contract():
    public = json.loads(
        (
            SCRIPTS.parent
            / "configs/mechanism_eval/phase_b_v1/specs"
            / f"{assessment.HARD_T1}.json"
        ).read_text()
    )
    rule = assessment.assessment_config()["cells"][assessment.HARD_T1]
    assert rule["public_prompt_sha256"] == public["public_prompt_sha256"]
    assert rule["requirements"][0]["public_requirement"] in {
        r["public_requirement"] for r in public["required_mechanisms"]
    }


def test_cli_sealed_outputs_resume_identically(tmp_path, monkeypatch):
    rows = []
    for method in assessment.METHODS:
        for rep in range(3):
            r = source()
            r.update(method=method, repetition=rep)
            rows.append(r)
    monkeypatch.setattr(
        assessment,
        "load_package",
        lambda _: (
            rows,
            {"schema_version": "phase-b-results-package-2"},
            {"archive_sha256": "a" * 64},
        ),
    )
    monkeypatch.setattr(assessment, "read_roster", lambda *_: ("fixture",))
    monkeypatch.setattr(assessment, "summarize", lambda *_: [])
    monkeypatch.setattr(assessment, "assessment_config", config)
    monkeypatch.setattr(
        sys,
        "argv",
        ["assessment", "--package", "unused.tar.gz", "--output", str(tmp_path)],
    )
    assessment.main()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assessment.main()
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
