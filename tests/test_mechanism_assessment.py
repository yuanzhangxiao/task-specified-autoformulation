"""Deterministic evidence layers, native semantics, no leakage and resume."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.rebuttal import mechanism_assessment as campaign
from autoformalism.rebuttal import mechanism_probes as probes
from autoformalism.rebuttal.mechanism_audit_sources import BUNDLE_PROTOCOL
from autoformalism.rebuttal.mechanism_checks import (
    EquationRequirement,
    equation_evidence,
)
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.schemas import CandidateModel

PROMPT = """A. Task specification
- input-driven memory and its causal contribution to the observed output.
B. Available data
Target y; input u.
C. Modeling requirements
1. Propose an explicit continuous-time model.
"""


def model(rhs="-m + b*u"):
    return CandidateModel.model_validate(
        {
            "candidate_id": "saved",
            "parent_candidate_id": None,
            "change_summary": "ignored",
            "states": [
                {"name": "m", "kind": "latent"},
                {"name": "y", "kind": "observed"},
            ],
            "state_equations": [
                {"state": "m", "rhs": rhs},
                {"state": "y", "rhs": "m-y"},
            ],
            "observation_mappings": [{"channel": "y", "expression": "y"}],
            "parameters": [{"name": "b", "scope": "global"}],
            "initial_conditions": [
                {"state": "m", "scope": "global", "fixed_value": 0},
                {"state": "y", "scope": "global", "expression": "y"},
            ],
        }
    )


def rule(kind="dynamic_memory"):
    return EquationRequirement(
        id="memory",
        public_requirement=PROMPT.splitlines()[1][2:],
        kind=kind,
        driver="u",
        target="y",
        interpretation="Input-driven memory.",
    )


def row(b=1.0):
    return {
        "method": "fixture",
        "benchmark_id": "fixture",
        "tier": "easy",
        "repetition": 0,
        "status": "ready",
        "candidate": model().model_dump(mode="json"),
        "parameters": {"b": b},
        "initials": {},
        "semantics": "continuous_time",
        "context": ValidationContext(targets=("y",), external_inputs=("u",)).model_dump(
            mode="json"
        ),
        "requirements": [rule().model_dump()],
        "public_prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
    }


def data():
    def trajectory(name, end, amplitude):
        t = np.linspace(0, end, 21)
        y = amplitude * (t - 2 + (t + 2) * np.exp(-t))
        return Trajectory(name, t, {"y": y}, {}, {"u": amplitude * t}, {}, {})

    return (
        DatasetSplit(SplitName.TRAIN, (trajectory("train", 4, 1),), "train_hash"),
        DatasetSplit(SplitName.VALIDATION, (trajectory("val", 5, 0.5),), "val_hash"),
    )


def test_fitted_zero_removes_path_without_overclaiming():
    context = ValidationContext.model_validate(row()["context"])
    assert equation_evidence(model(), context, [rule()])["counts"]["pass"] == 1
    zero = equation_evidence(model(), context, [rule()], parameters={"b": 0})
    assert zero["counts"]["fail"] == 1
    assert not zero["scientific_correctness_certified"]


@pytest.mark.parametrize(
    "rhs,present",
    [
        ("-m+b*u+y*y", True),
        ("-m+b*u+0*y*y", False),
        ("-m+b*u+(y*y-y*y)", False),
        ("-m+b*u*u+y", False),
        ("-m+b*u+b*y*y", True),
    ],
)
def test_feedback_uses_nonlinearity_on_actual_cycle(rhs, present):
    result = equation_evidence(
        model(rhs),
        ValidationContext.model_validate(row()["context"]),
        [rule("nonlinear_feedback")],
    )
    assert (result["counts"]["pass"] == 1) is present


def test_nonlinear_feedback_at_zero_gain_is_not_active_syntax():
    result = equation_evidence(
        model("-m+u+b*y*y"),
        ValidationContext.model_validate(row()["context"]),
        [rule("nonlinear_feedback")],
        parameters={"b": 0},
    )
    assert result["counts"]["fail"] == 1


def test_incomplete_fitted_vector_rejected():
    with pytest.raises(ValueError, match="complete"):
        equation_evidence(
            model(),
            ValidationContext.model_validate(row()["context"]),
            [rule()],
            parameters={},
        )


def test_three_layers_recover_truth_and_resume_without_rollouts(tmp_path, monkeypatch):
    train, val = data()
    settings = probes.ProbeSettings(trajectory_seconds=10)
    first = probes.numerical_assessment(row(), train, val, settings, tmp_path)
    assert first["fitted_activity"]["counts"]["pass"] == 1
    assert first["response_behavior"]["counts"]["pass"] == 1
    assert first["response_behavior"]["trajectories"][0]["nmse"] < 1e-10

    def forbidden(*args, **kwargs):
        raise AssertionError("resume performed a new numerical call")

    monkeypatch.setattr(probes, "rollout", forbidden)
    assert probes.numerical_assessment(row(), train, val, settings, tmp_path) == first
    assert not first["test_data_opened"] and not first["parameter_refit_applied"]


def test_zero_gain_activity_unresolved_and_bad_response(tmp_path):
    train, val = data()
    result = probes.numerical_assessment(
        row(0), train, val, probes.ProbeSettings(), tmp_path
    )
    assert result["fitted_activity"]["counts"]["unresolved"] == 1
    assert result["response_behavior"]["counts"]["fail"] == 1


def test_test_split_rejected_before_any_rollout(tmp_path):
    train, val = data()
    with pytest.raises(ValueError, match="never TEST"):
        probes.numerical_assessment(
            row(),
            train,
            replace(val, name=SplitName.TEST),
            probes.ProbeSettings(),
            tmp_path,
        )
    assert not list(tmp_path.iterdir())


def test_probes_preserve_initial_values_and_training_range():
    train, _ = data()
    original, trials, _ = probes.probe_trajectory(train, "u", probes.ProbeSettings())
    for _, t in trials:
        assert t.external_inputs["u"][0] == original.external_inputs["u"][0]
        assert np.array_equal(t.targets["y"], original.targets["y"])
        assert (
            np.min(t.external_inputs["u"]) >= 0 and np.max(t.external_inputs["u"]) <= 4
        )
    assert len(trials) == 2


def test_flat_input_is_unexcited_not_a_mechanism_failure(tmp_path):
    train, _ = data()
    t = replace(train.trajectories[0], external_inputs={"u": np.zeros(21)})
    result = probes.assess_activity(
        row(),
        replace(train, trajectories=(t,)),
        row()["requirements"],
        {"y": 1},
        probes.ProbeSettings(),
        tmp_path,
    )
    assert result["counts"]["unresolved"] == 1
    assert not list(tmp_path.iterdir())


def test_solver_disagreement_does_not_pass_response(tmp_path, monkeypatch):
    _, val = data()

    def disagree(row, trajectory, solver, settings):
        return {"y": (trajectory.targets["y"] + (1 if solver == "BDF" else 0)).tolist()}

    monkeypatch.setattr(probes, "rollout", disagree)
    result = probes.assess_responses(
        row(), val, {"y": 1}, probes.ProbeSettings(), tmp_path
    )
    assert result["counts"]["unresolved"] == 1


def test_native_increment_has_no_dt_or_target_teacher_forcing():
    value = row()
    candidate = value["candidate"]
    candidate["states"] = [candidate["states"][1]]
    candidate["state_equations"] = [{"state": "y", "rhs": "b*u"}]
    candidate["initial_conditions"] = [candidate["initial_conditions"][1]]
    value["semantics"] = "native_increment"
    train, _ = data()
    t = train.trajectories[0]
    actual = probes.rollout(value, t, "native", probes.ProbeSettings())["y"]
    np.testing.assert_allclose(actual, np.r_[0, np.cumsum(t.external_inputs["u"][:-1])])
    modified = t.targets["y"].copy()
    modified[1:] = 99999
    second = probes.rollout(
        value, replace(t, targets={"y": modified}), "native", probes.ProbeSettings()
    )["y"]
    assert actual == second


def fixture(tmp_path, monkeypatch):
    train, val = data()
    source = row()
    public = tmp_path / "public"
    (public / "fixture").mkdir(parents=True)
    (public / "fixture/proposer_prompt.txt").write_text(PROMPT)
    bundle = tmp_path / "bundle.json"
    sealed_write(bundle, {"protocol": BUNDLE_PROTOCOL, "rows": [source]})
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "numerical_settings": {"trajectory_seconds": 10},
                "cells": {
                    "fixture": {
                        "public_prompt_sha256": source["public_prompt_sha256"],
                        "requirements": [rule().model_dump()],
                    }
                },
            }
        )
    )
    identity = {
        "train": train.fingerprint,
        "validation": val.fingerprint,
        "prompt": source["public_prompt_sha256"],
    }
    monkeypatch.setattr(
        campaign,
        "load_public",
        lambda *args: (
            SimpleNamespace(train=train, validation=val),
            ValidationContext.model_validate(source["context"]),
            identity,
        ),
    )
    return bundle, public, tmp_path / "output", config


def test_campaign_frozen_results_and_missing_denominators(tmp_path, monkeypatch):
    bundle, public, root, config = fixture(tmp_path, monkeypatch)
    plan = campaign.prepare([bundle], public, root, config)
    assert plan["rows"][0]["status"] == "ready"
    assert campaign.report(root)["groups"][0]["response_behavior"]["unresolved"] == 1
    first = campaign.run(root, 0)
    assert first["status"] == "complete"
    assert campaign.run(root, 0) == first
    assert campaign.prepare([bundle], public, root, config) == plan
    assert campaign.report(root)["groups"][0]["fitted_activity"]["pass"] == 1


def test_prompt_mismatch_retains_failed_row(tmp_path, monkeypatch):
    bundle, public, root, config = fixture(tmp_path, monkeypatch)
    (public / "fixture/proposer_prompt.txt").write_text(PROMPT + "changed")
    plan = campaign.prepare([bundle], public, root, config)
    assert plan["rows"][0]["status"] == "input_failed"
    assert (
        campaign.report(root)["groups"][0]["equation_requirements"]["unresolved"] == 1
    )


def test_duplicate_models_are_not_selected_by_score(tmp_path, monkeypatch):
    bundle, public, root, config = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="duplicate"):
        campaign.prepare([bundle, bundle], public, root, config)


def test_cstr_single_mixed_term_does_not_prove_distinct_contributions():
    candidate = model("-m+b*u*y")
    context = ValidationContext.model_validate(row()["context"])
    rules = [rule("balance_channel")]
    assert equation_evidence(candidate, context, rules)["counts"]["pass"] == 1
    rules.append(
        EquationRequirement(**{**rules[0].model_dump(), "id": "other", "driver": "y"})
    )
    assert equation_evidence(candidate, context, rules)["counts"]["unresolved"] >= 1


def test_equation_ast_rejects_untrusted_code():
    bad = copy.deepcopy(model().model_dump(mode="json"))
    bad["state_equations"][0]["rhs"] = "__import__('os').system('echo hacked')"
    with pytest.raises(ModelValidationError):
        equation_evidence(
            CandidateModel.model_validate(bad),
            ValidationContext.model_validate(row()["context"]),
            [rule()],
        )


def test_timeout_checkpoint_does_not_gain_a_new_attempt(tmp_path, monkeypatch):
    calls = []

    def timeout(*args):
        calls.append(1)
        raise TimeoutError("trajectory budget")

    monkeypatch.setattr(probes, "rollout", timeout)
    train, val = data()
    first = probes.numerical_assessment(
        row(), train, val, probes.ProbeSettings(), tmp_path
    )
    n = len(calls)
    second = probes.numerical_assessment(
        row(), train, val, probes.ProbeSettings(), tmp_path
    )
    assert first == second and len(calls) == n
    assert first["response_behavior"]["counts"]["unresolved"] == 1


def test_ode_free_rollout_does_not_read_future_targets():
    train, _ = data()
    t = train.trajectories[0]
    modified = t.targets["y"].copy()
    modified[1:] = -99999
    settings = probes.ProbeSettings()
    actual = probes.rollout(row(), t, "Radau", settings)
    other = probes.rollout(
        row(), replace(t, targets={"y": modified}), "Radau", settings
    )
    assert actual == other


def test_observed_memory_is_allowed_without_inventing_hidden_state_rule():
    value = row()
    payload = value["candidate"]
    payload["states"] = [payload["states"][1]]
    payload["state_equations"] = [{"state": "y", "rhs": "-y+b*u"}]
    payload["initial_conditions"] = [payload["initial_conditions"][1]]
    result = equation_evidence(
        CandidateModel.model_validate(payload),
        ValidationContext.model_validate(value["context"]),
        [rule()],
    )
    assert result["counts"]["pass"] == 1


def test_native_supplied_auxiliary_law_cannot_create_false_driver_path():
    value = row()
    payload = value["candidate"]
    payload["states"][0]["kind"] = "observed"
    payload["state_equations"] = [{"state": "y", "rhs": "m"}]
    value["native_auxiliary_equations_ignored"] = [{"state": "m", "rhs": "b*u"}]
    value["context"] = ValidationContext(
        targets=("y",), auxiliaries=("m",), external_inputs=("u",)
    ).model_dump(mode="json")
    value["semantics"] = "native_increment"
    train, _ = data()
    t = replace(train.trajectories[0], auxiliaries={"m": np.ones(21)})
    pred = probes.rollout(value, t, "native", probes.ProbeSettings())["y"]
    np.testing.assert_allclose(pred, np.arange(21))
    result = equation_evidence(
        CandidateModel.model_validate(payload),
        ValidationContext.model_validate(value["context"]),
        [rule()],
        semantics="native_increment",
    )
    assert result["counts"]["fail"] == 1
    assert result["continuous_time_representation"] == "fail"


def test_native_zero_increment_is_not_feedback_evidence():
    value = row()
    payload = value["candidate"]
    payload["states"] = [payload["states"][1]]
    payload["state_equations"] = [{"state": "y", "rhs": "b*u"}]
    result = equation_evidence(
        CandidateModel.model_validate(payload),
        ValidationContext.model_validate(value["context"]),
        [rule("nonlinear_feedback")],
        semantics="native_increment",
    )
    assert result["counts"]["fail"] == 1


def test_extreme_response_has_explicit_unresolved_metrics(tmp_path, monkeypatch):
    _, val = data()
    monkeypatch.setattr(probes, "rollout", lambda *args: {"y": [1e300] * 21})
    result = probes.assess_responses(
        row(), val, {"y": 1}, probes.ProbeSettings(), tmp_path
    )
    assert result["counts"]["unresolved"] == 1
    assert result["normalized_mse"] is None
    assert probes._rms(np.array([1e300, -1e300])) == 1e300


@pytest.mark.parametrize("reply", ["123;aces", "uncertain reply"])
def test_submission_records_intent_without_duplicate_jobs(tmp_path, reply):
    root, binaries = tmp_path / "output", tmp_path / "bin"
    root.mkdir()
    binaries.mkdir()
    (root / "plan.json").write_text(json.dumps({"rows": [{}, {}]}))
    fake = binaries / "sbatch"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['AF_FAKE_LOG'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"print({reply!r})\n"
    )
    fake.chmod(0o755)
    repo = Path(__file__).resolve().parents[1]
    log = tmp_path / "submissions.jsonl"
    environment = {
        **os.environ,
        "AF_REPO_ROOT": str(repo),
        "AF_PYTHON": sys.executable,
        "AF_OUTPUT_ROOT": str(root),
        "AF_FAKE_LOG": str(log),
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
    }
    command = ["bash", str(repo / "scripts/hpc/submit_mechanism_assessment_aces.sh")]
    first = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=10
    )
    assert (first.returncode == 0) == (reply == "123;aces")
    calls = log.read_text()
    assert len(calls.splitlines()) == (2 if first.returncode == 0 else 1)
    second = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=10
    )
    assert second.returncode != 0 and log.read_text() == calls
    assert (root / "submission-intent").is_dir()
