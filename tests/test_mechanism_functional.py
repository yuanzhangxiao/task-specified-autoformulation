"""Positive/negative scientific controls and numerical/provenance boundaries."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal import mechanism_functional as probe
from autoformalism.rebuttal import mechanism_functional_campaign as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel


def row(equations=None, outputs=None, inputs=("u",), auxiliary=(), native=False):
    equations = equations or {"y": "u-y"}
    outputs = outputs or {"y": "y"}
    states = set(equations)
    model = CandidateModel.model_validate(
        {
            "candidate_id": "control",
            "parent_candidate_id": None,
            "states": [
                {"name": n, "kind": "observed" if n in outputs else "latent"}
                for n in equations
            ],
            "state_equations": [{"state": n, "rhs": e} for n, e in equations.items()],
            "observation_mappings": [
                {"channel": n, "expression": e} for n, e in outputs.items()
            ],
            "initial_conditions": [
                {
                    "state": n,
                    "scope": "global",
                    **({"expression": n} if n in outputs else {"fixed_value": 1.0}),
                }
                for n in states
            ],
        }
    )
    return {
        "status": "ready",
        "candidate": model.model_dump(mode="json"),
        "parameters": {},
        "initials": {},
        "semantics": "native_increment" if native else "continuous_time",
        "context": ValidationContext(
            targets=tuple(outputs), external_inputs=inputs, auxiliaries=auxiliary
        ).model_dump(mode="json"),
        "method": "fixture",
        "benchmark_id": "fixture",
        "tier": "easy",
        "repetition": 0,
    }


def rule(kind="causal_response", driver="u", target="y", sign=1, **kw):
    return {
        "id": "response",
        "public_requirement": "Input drives output",
        "interpretation": "Test fixture",
        "kind": kind,
        "driver": driver,
        "target": target,
        "expected_sign": sign,
        "readout": "output",
        **kw,
    }


def local(source, requirement=None, forcing=None, values=None):
    runtime = probe.Runtime(source)
    names = {
        *runtime.context.targets,
        *runtime.context.external_inputs,
        *runtime.context.auxiliaries,
    }
    x = np.array([values.get(n, 1.0) if values else 1.0 for n in runtime.names])
    inputs = forcing or dict.fromkeys(
        (*runtime.context.external_inputs, *runtime.context.auxiliaries), 1.0
    )
    return probe.local_response(
        runtime,
        requirement or rule(),
        x,
        inputs,
        0.0,
        dict.fromkeys(names, 1.0),
        3.0,
        probe.Settings(),
    )


@pytest.mark.parametrize("expression,status", [("u-y", "pass"), ("-u-y", "fail")])
def test_effective_sign_is_stronger_than_graph_presence(expression, status):
    source = row({"y": expression})
    assert probe.structural_check(source, rule()) is None
    assert local(source)["status"] == status


def test_no_hidden_sign_for_anonymous_case():
    assert local(row({"y": "-u-y"}), rule(sign=0))["status"] == "pass"


def test_zero_fitted_path_fails_and_parallel_dynamics_cancel():
    assert probe.structural_check(row({"y": "0*u-y"}), rule())["status"] == "fail"
    source = row({"a": "u-a", "b": "u-b"}, {"y": "a-b"})
    assert probe.structural_check(source, rule()) is None
    assert local(source)["status"] == "inactive"  # no false pass from graph paths


def test_feedthrough_does_not_establish_memory():
    source = row({"z": "-z"}, {"y": "u"})
    assert local(source)["status"] == "pass"
    assert local(source, rule(kind="memory"))["status"] == "fail"


def test_delayed_generated_driver_is_clamped_at_its_model_state():
    r = rule(kind="delayed_action", driver="I", target="U")
    delayed = row({"I": "u-I", "x": "I-x"}, {"I": "I", "U": "x"})
    result = local(delayed, r)
    assert result["status"] == "pass"
    assert result["held_states"] == ["I"]
    assert result["generated_driver_uses_model_state"]
    direct = row({"I": "u-I"}, {"I": "I", "U": "I"})
    assert local(direct, r)["status"] == "fail"
    alias = row({"plasma": "u-plasma", "x": "plasma-x"}, {"I": "plasma", "U": "x"})
    assert probe.structural_check(alias, r) is None
    assert local(alias, r)["status"] == "pass"
    two_stage = row({"I": "u-I", "a": "I-a", "b": "a-b"}, {"I": "I", "U": "b"})
    assert local(two_stage, r)["status"] == "pass"
    assert local(two_stage, r)["first_dynamic_order"] == 1


def test_glucose_storage_integration_is_not_insulin_action_delay():
    r = rule(
        kind="delayed_action", driver="I", target="Gp", sign=-1, readout="balance_rate"
    )
    direct = row({"I": "u-I", "Gp": "-I"}, {"I": "I", "Gp": "Gp"})
    value = local(direct, r)
    assert value["status"] == "fail"
    assert set(value["held_states"]) == {"I", "Gp"}
    for decoy in ("u-z", "Gp-z"):
        unrelated = row({"I": "u-I", "Gp": "-I+z", "z": decoy}, {"I": "I", "Gp": "Gp"})
        assert local(unrelated, r)["status"] == "fail"
    delayed = row({"I": "u-I", "Gp": "-x", "x": "I-x"}, {"I": "I", "Gp": "Gp"})
    scaled = row({"I": "u-I", "Gp": "-0.01*z", "z": "100*I-z"}, {"I": "I", "Gp": "Gp"})
    assert local(delayed, r)["status"] == local(scaled, r)["status"] == "pass"
    wrong = row({"I": "u-I", "Gp": "x", "x": "I-x"}, {"I": "I", "Gp": "Gp"})
    assert local(wrong, r)["status"] == "fail"


def test_nonidentity_generated_driver_is_not_guessed():
    source = row(
        {"plasma": "u-plasma", "x": "plasma-x"}, {"I": "exp(plasma)", "U": "x"}
    )
    r = rule(kind="delayed_action", driver="I", target="U")
    assert probe.structural_check(source, r)["status"] == "unresolved"


def test_native_increment_does_not_gain_sample_time_multiplier():
    native = row({"y": "u-0.5*y"}, native=True)
    ode = row({"y": "u-0.5*y"})
    assert local(native)["dynamic_markov_coefficients"][0] == pytest.approx(1.0)
    assert local(ode)["dynamic_markov_coefficients"][0] == pytest.approx(3.0)


def test_thermal_boundaries_and_directions():
    r = {
        "id": "balance",
        "public_requirement": "Heat balance",
        "kind": "thermal_balance",
        "target": "T",
        "feed": "Tf",
        "jacket": "Tj",
        "reactant": "C",
    }

    def check(expression):
        runtime = probe.Runtime(
            row({"T": expression}, {"T": "T"}, inputs=("Tf",), auxiliary=("Tj", "C"))
        )
        return probe.thermal_balance(
            runtime,
            r,
            np.array([300.0]),
            {"Tf": 310.0, "Tj": 290.0, "C": 1.0},
            0.0,
            {"T": 1.0},
            1.0,
            probe.Settings(),
        )

    assert check("(Tf-T)+(Tj-T)+C")["status"] == "pass"
    assert check("(Tf-T)-(Tj-T)+C")["checks"]["jacket_heats_in_its_direction"] == "fail"
    assert check("(Tf-T)+(Tj-T)-C")["checks"]["reaction_heat_generation"] == "fail"
    assert (
        check("Tf+Tj+C")["checks"]["equal_temperature_zero_reactant_balance"] == "fail"
    )
    assert check("Tf-T")["status"] == "unresolved"
    cross = check("(Tf-T)+(Tj-T)+C+(Tf-T)*(Tj-T)")
    assert cross["checks"]["additive_public_heat_channels"] == "fail"
    assert check("2*(Tf+Tj-2*T)+C")["status"] == "pass"


def training(targets=("y",), inputs=("u",), values=None):
    t = np.array([0.0, 0.1, 0.2])
    trajectory = Trajectory(
        "a",
        t,
        {n: np.ones(3) for n in targets},
        {},
        {n: np.ones(3) for n in inputs},
        {},
        {},
    )
    return DatasetSplit("train", (trajectory,), "test-train-fingerprint")


def test_real_solver_replay_resume_and_unidentified_hard_proxy(tmp_path):
    simple = row()
    rules = [rule()]
    data = training()
    result = probe.assess(simple, rules, data, probe.Settings(), tmp_path / "simple")
    assert result[0]["status"] == "pass"
    assert (
        probe.assess(simple, rules, data, probe.Settings(), tmp_path / "simple")
        == result
    )
    delayed = row({"I": "u-I", "Gp": "-x", "x": "I-x"}, {"I": "I", "Gp": "Gp"})
    r = rule(
        kind="delayed_action",
        driver="I",
        target="Gp",
        sign=-1,
        readout="balance_rate",
        identification_limit="source versus disposal",
    )
    output = probe.assess(
        delayed, [r], training(("I", "Gp")), probe.Settings(), tmp_path / "hard"
    )
    assert output[0]["status"] == "unresolved"
    assert output[0]["proxy_test_status"] == "pass"


def test_solver_disagreement_prevents_confirmation(tmp_path, monkeypatch):
    def fake_baseline(runtime, trajectory, solver, *args):
        return {
            "status": "complete",
            "states": [[1.0, 1.0, 1.0 if solver == "Radau" else 2.0]],
        }

    monkeypatch.setattr(probe, "baseline", fake_baseline)
    result = probe.assess(row(), [rule()], training(), probe.Settings(), tmp_path)
    assert result[0]["status"] == "unresolved"
    assert result[0]["points"][0]["reason"] == "state replay disagreement"


def test_interrupted_rollout_does_not_get_another_budget(tmp_path, monkeypatch):
    from autoformalism.staged_topology import content_hash

    source, settings = row(), probe.Settings()
    runtime = probe.Runtime(source)
    trajectory = training().trajectories[0]
    key = content_hash([source, "data", "a", "Radau", settings.model_dump(mode="json")])
    sealed_write(tmp_path / f"{key}.started.json", {"identity": key})
    monkeypatch.setattr(
        runtime, "trajectory", lambda *args: pytest.fail("must not rerun")
    )
    assert (
        probe.baseline(runtime, trajectory, "Radau", settings, tmp_path, "data")[
            "status"
        ]
        == "unresolved"
    )


def test_known_failures_zero_missing_evidence_bounded_and_median_mad():
    source = row()
    source["mechanisms"] = [rule()]
    source.update(status="unavailable", source_terminal_status="failed")
    failed = campaign.unavailable(source, "no model")
    assert failed[0]["status"] == "fail"
    source["source_terminal_status"] = None
    missing = campaign.unavailable(source, "no evidence")
    assert missing[0]["status"] == "unresolved"
    rows = [
        {"method": "m", "benchmark_id": c, "mechanisms": [{"status": status}]}
        for c, status in (("a", "pass"), ("b", "fail"), ("c", "unresolved"))
    ]
    value = campaign.aggregate(rows, ["a", "b", "c"])[0]
    assert value["confirmed"] == {"median": 0.0, "mad": 0.0}
    assert value["possible"] == {"median": 1.0, "mad": 0.0}
    assert campaign.robust([0.0, 0.5, 1.0]) == {"median": 0.5, "mad": 0.5}


def test_frozen_rubric_covers_nine_cases_eleven_public_obligations():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/mechanism_functional_v1.json").read_text())
    assert len(cfg["cells"]) == 9
    assert sum(len(c["mechanisms"]) for c in cfg["cells"].values()) == 11
    probe.Settings.model_validate(cfg["numerical_settings"])
    for name, cell in cfg["cells"].items():
        spec = json.loads(
            (
                root / "configs/mechanism_eval/phase_b_v1/specs" / f"{name}.json"
            ).read_text()
        )
        assert cell["public_prompt_sha256"] == spec["public_prompt_sha256"]
        assert {r["public_requirement"] for r in cell["mechanisms"]} == {
            r["public_requirement"] for r in spec["required_mechanisms"]
        }
        for r in cell["mechanisms"]:
            probe.Rule.model_validate(r)
            if "anonymous" in name or "alien" in name:
                assert r["expected_sign"] == 0


@pytest.mark.parametrize(
    "kw",
    [
        {"operating_fractions": []},
        {"operating_fractions": [2.0]},
        {"difference_step": float("inf")},
    ],
)
def test_invalid_settings_rejected(kw):
    with pytest.raises(ValueError):
        probe.Settings(**kw)


def test_plan_idempotence_changed_sources_and_missing_public_files(
    tmp_path, monkeypatch
):
    prompt = """A. Task specification
- Input drives output
B. Available data
Target y, input u.
C. Modeling requirements
1. Generate y.
D. Constraints and plausibility requirements
Use causal equations.
E. Intended use
Prediction.
"""
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    source = row()
    source["public_prompt_sha256"] = digest
    cfg = {
        "protocol": campaign.PROTOCOL,
        "numerical_settings": probe.Settings().model_dump(mode="json"),
        "cells": {"fixture": {"public_prompt_sha256": digest, "mechanisms": [rule()]}},
    }
    data = SimpleNamespace(train=training())

    def public(*args):
        return (
            data,
            ValidationContext.model_validate(source["context"]),
            {"prompt": digest},
        )

    monkeypatch.setattr(campaign, "_public_prompt", lambda *args: prompt)
    monkeypatch.setattr(campaign, "load_public", public)
    monkeypatch.setattr(campaign, "code_identity", lambda: {"code": "v1"})
    root = tmp_path / "campaign"
    first = campaign.prepare([source], cfg, root, tmp_path, {"input": "fixture"})
    assert (
        campaign.prepare([source], cfg, root, tmp_path, {"input": "fixture"}) == first
    )
    result = campaign.run(root, 0)
    assert result["status"] == "assessed"
    assert result["mechanisms"][0]["status"] == "pass"
    assert campaign.run(root, 0) == result
    assert campaign.report(root)["methods"][0]["confirmed"]["median"] == 1.0
    changed = copy.deepcopy(cfg)
    changed["numerical_settings"]["difference_step"] = 1e-4
    with pytest.raises(ValueError):
        campaign.prepare([source], changed, root, tmp_path, {"input": "fixture"})
    monkeypatch.setattr(campaign, "code_identity", lambda: {"code": "v2"})
    with pytest.raises(ValueError, match="source changed"):
        campaign.run(root, 0)
    with pytest.raises(ValueError, match="source changed"):
        campaign.report(root)
    assert not sealed_read(root / "plan.json")["test_data_opened"]


def test_cli_prepare_run_report_smoke(tmp_path, monkeypatch):
    """Standalone commands reuse one frozen bundle and real two-solver evidence."""
    from autoformalism.rebuttal.mechanism_audit_sources import BUNDLE_PROTOCOL

    script = (
        Path(__file__).resolve().parents[1] / "scripts/assess_functional_mechanisms.py"
    )
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("functional_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    public_prompt = "public fixture"
    digest = hashlib.sha256(public_prompt.encode()).hexdigest()
    source = row()
    source["public_prompt_sha256"] = digest
    bundle = tmp_path / "bundle.json"
    sealed_write(bundle, {"protocol": BUNDLE_PROTOCOL, "rows": [source]})
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "protocol": campaign.PROTOCOL,
                "numerical_settings": probe.Settings().model_dump(mode="json"),
                "cells": {
                    "fixture": {"public_prompt_sha256": digest, "mechanisms": [rule()]}
                },
            }
        )
    )
    monkeypatch.setattr(campaign, "_public_prompt", lambda *args: public_prompt)
    monkeypatch.setattr(
        campaign,
        "rubric",
        lambda *args: [
            SimpleNamespace(
                category="task_mechanism", text=rule()["public_requirement"]
            )
        ],
    )
    monkeypatch.setattr(
        campaign,
        "load_public",
        lambda *args: (
            SimpleNamespace(train=training()),
            ValidationContext.model_validate(source["context"]),
            {"prompt": digest},
        ),
    )
    root = tmp_path / "run"
    for args in (
        [
            "prepare",
            "--bundle",
            str(bundle),
            "--config",
            str(config),
            "--public-root",
            str(tmp_path),
        ],
        ["run", "--index", "0"],
        ["report"],
    ):
        monkeypatch.setattr(sys, "argv", [str(script), *args, "--root", str(root)])
        cli.main()
    result = json.loads((root / "summary.json").read_text())
    assert result["status_counts"] == {"assessed": 1}
    assert result["free_rollouts_started"] == 2
    assert result["methods"][0]["fully_confirmed_runs"] == 1
    assert result["llm_calls"] == result["optimizer_calls"] == 0
