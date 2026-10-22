"""Phase C external baselines: grammar, the D3 discrete identity, LLM-SR
transcriptions, and the per-method adapters of the public mechanism tests."""

import ast
import hashlib
import json

import numpy as np
import pytest

from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import (
    ModelValidationError,
    ValidationContext,
    baseline_validator,
    compile_candidate,
)
from autoformalism.fitting.public_fitting import pack_split
from autoformalism.rebuttal import basin_algebra as algebra
from autoformalism.rebuttal import basin_equation_checks as equations
from autoformalism.rebuttal import llm_sr_transcriptions as transcriptions
from autoformalism.rebuttal import phase_c_baseline_mechanisms as mechanisms
from autoformalism.rebuttal.mechanism_checks import effective_trees
from autoformalism.rebuttal.mechanism_functional import Settings
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research import basin_construction_assessment as basin
from autoformalism.schemas import CandidateModel

SURVEY = {
    "area_up": 2.0,
    "area_down": 3.0,
    "crest_up": 0.4,
    "crest_down": 0.2,
    "initial_up": 0.6,
    "warning_depth": 1.0,
}
CONTEXT = ValidationContext(
    targets=("h_down",),
    external_inputs=("inflow_up", "inflow_down"),
    fixed_covariates=tuple(SURVEY),
)
NAME = "phase_c_detention_coupled_noise0_v1"
SETTINGS = Settings(maximum_training_trajectories=1, trajectory_seconds=5)
SOURCE = (
    "def equation(h_down, inflow_down, params):\n"
    "    return params[0] * inflow_down\n"
)
OUTLET = "Max(0, h_down - crest_down)"


def candidate(rhs: dict[str, str], parameters=(), latent=()) -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": "baseline",
            "parent_candidate_id": None,
            "states": [
                {"name": n, "kind": "latent" if n in latent else "observed"}
                for n in rhs
            ],
            "state_equations": [{"state": n, "rhs": v} for n, v in rhs.items()],
            "observation_mappings": [{"channel": "h_down", "expression": "h_down"}],
            "parameters": [{"name": p, "scope": "global"} for p in parameters],
            "initial_conditions": [
                {
                    "state": n,
                    "scope": "global",
                    "expression": "initial_up" if n in latent else n,
                }
                for n in rhs
            ],
        }
    )


def training(rows: int = 4, step: float = 4.0) -> DatasetSplit:
    return DatasetSplit(
        name=SplitName.TRAIN,
        fingerprint="synthetic-basin-training",
        trajectories=(
            Trajectory(
                trajectory_id="training",
                time=np.arange(rows) * step,
                targets={"h_down": np.full(rows, 0.5)},
                auxiliaries={},
                derivatives={},
                external_inputs={
                    "inflow_up": np.ones(rows),
                    "inflow_down": np.ones(rows),
                },
                fixed_covariates=SURVEY,
            ),
        ),
    )


def coupled_cell() -> dict:
    prompt = "\n".join(basin.COUPLED_QUOTES)
    _, _, rules = basin.contracts(
        NAME, prompt, hashlib.sha256(prompt.encode()).hexdigest()
    )
    return {
        "public_specification": {"benchmark_id": NAME},
        "independent_rules": rules,
        "assessment_policy": basin.POLICY,
        "context": CONTEXT.model_dump(mode="json"),
        "training": pack_split(training()).model_dump(mode="json"),
    }


def native_row(rhs: str, **extra) -> dict:
    return {
        "candidate": candidate({"h_down": rhs}).model_dump(mode="json"),
        "context": CONTEXT.model_dump(mode="json"),
        "parameters": {},
        "initials": {},
        "semantics": "native_increment",
        "grammar": "baseline",
        "increment_step": 4.0,
        **extra,
    }


# --- grammar ---------------------------------------------------------------


def test_formal_algebra_keeps_real_and_symbolic_exponents_opaque():
    def poly(text):
        return algebra.formal(algebra.expanded(text, {}, "baseline"))

    assert poly("x ** 2") == {(("x", 2),): 1}
    assert poly("x ** 1.5") == {(("x ** 1.5", 1),): 1}
    assert poly("x ** beta") == {(("x ** beta", 1),): 1}
    assert algebra.coefficient(poly("Max(0, x) * u"), "u", "baseline") == {
        (("Max(0, x)", 1),): 1
    }
    with pytest.raises(ModelValidationError):
        algebra.expanded("x ** 1.5", {})


def test_baseline_models_are_read_with_the_baseline_grammar():
    model = candidate(
        {"h_down": "-Max(0, h_down - crest_down) ** beta"}, parameters=("beta",)
    )
    with pytest.raises(ModelValidationError):
        effective_trees(model, {"beta": 1.5})
    trees = effective_trees(model, {"beta": 1.5}, grammar="baseline")
    assert ast.unparse(trees["h_down"]) == "-Max(0, h_down - crest_down) ** 1.5"
    with pytest.raises(ValueError, match="grammar"):
        effective_trees(model, {"beta": 1.5}, grammar="sympy")


# --- D3's discrete input conversion ------------------------------------------


def conversion(rhs: str, step: float | None) -> dict:
    evidence = equations.assess(
        candidate({"h_down": rhs}),
        CONTEXT,
        "independent",
        [SURVEY],
        {},
        grammar="baseline",
        increment_step=step,
    )
    code = "inflow_conversion:h_down"
    return next(c for c in evidence["checks"] if c["code"] == code)


def test_a_native_increment_uses_the_discrete_input_conversion():
    passing = conversion(f"4.0 * inflow_down / area_down - {OUTLET}", 4.0)
    assert passing["status"] == "pass"
    assert passing["evidence"]["required_identity"] == (
        "area_down * coefficient(inflow_down) = 4"
    )
    assert passing["evidence"]["equation_semantics"] == "native_increment"
    # An ODE's conversion is wrong for a map that steps 4 minutes at a time.
    assert conversion("inflow_down / area_down", 4.0)["status"] == "fail"
    assert conversion("inflow_down / area_down", None)["status"] == "pass"
    assert conversion("4.0 * inflow_down / area_down", None)["status"] == "fail"
    with pytest.raises(ValueError, match="step"):
        conversion("inflow_down / area_down", 0.0)


def test_native_d3_row_on_the_coupled_basin(tmp_path):
    rhs = (
        "4.0 * (inflow_down + 0.5 * inflow_up) / area_down"
        " - Max(0, h_down - crest_down)"
    )
    found = {
        f["id"]: f["status"]
        for f in basin.assess(
            native_row(rhs), coupled_cell(), training(), SETTINGS, tmp_path
        )
    }
    assert found["local_inflow_conversion"] == "pass"
    assert found["threshold_outlet"] == "pass"
    # No upstream state: no memory, and nothing to convert or cancel.
    assert found["upstream_memory"] == "fail"
    assert found["upstream_inflow_conversion"] == "unresolved"
    assert found["transfer_cancellation"] == "unresolved"


def test_a_native_basin_row_needs_its_step(tmp_path):
    with pytest.raises(ValueError, match="step"):
        basin.assess(
            native_row("0.0 * h_down", increment_step=None),
            coupled_cell(),
            training(),
            SETTINGS,
            tmp_path,
        )


# --- LLM-SR transcriptions ----------------------------------------------------


def item(**changes) -> transcriptions.Transcription:
    value = {
        "task": 12,
        "benchmark_id": NAME,
        "repetition": 0,
        "programs": {
            "h_down": {
                "function_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
                "parameters": [0.5],
            }
        },
        "parameters": {"k": 0.1},
        "processes": {"Q_over": "k * max(h_up - crest_up, 0.0)"},
        "hidden_states": {
            "h_up": {
                "rhs": "(inflow_up - Q_over) / area_up / 4.0",
                "initial": "max(0.0, initial_up)",
                "program_name": "h_up[t+1] = h_up[t] + (inflow_up - Q_ov)/area_up",
                "rule": "euler",
            }
        },
        "targets": {"h_down": "(inflow_down + Q_over) / area_down"},
        "rules": ["euler", "initial"],
    }
    value.update(changes)
    return transcriptions.Transcription.model_validate(value)


def test_a_transcription_binds_to_the_exact_program_and_coefficients():
    good = {"h_down": {"function": SOURCE, "parameters": [0.5]}}
    transcriptions.bind(item(), good)
    with pytest.raises(ValueError, match="program text"):
        transcriptions.bind(
            item(), {"h_down": {"function": SOURCE + "\n", "parameters": [0.5]}}
        )
    with pytest.raises(ValueError, match="coefficients"):
        transcriptions.bind(
            item(), {"h_down": {"function": SOURCE, "parameters": [0.5000001]}}
        )
    with pytest.raises(ValueError, match="covers"):
        transcriptions.bind(item(), {**good, "h_up": good["h_down"]})


def test_a_transcription_is_an_ode_with_latent_program_states():
    context = transcriptions.context_for(CONTEXT)
    assert context.fitted_initialization
    model = transcriptions.candidate(item(), context)
    assert [(s.name, s.kind.value) for s in model.states] == [
        ("h_down", "observed"),
        ("h_up", "latent"),
    ]
    compiled = compile_candidate(model, context, validator=baseline_validator())
    assert compiled.state_names == ("h_down", "h_up")
    with pytest.raises(ValueError, match="targets"):
        transcriptions.candidate(item(), context.model_copy(update={"targets": ("x",)}))


def test_a_transcription_cites_declared_rules_once_per_task():
    with pytest.raises(ValueError, match="rule"):
        item(rules=["initial"])
    with pytest.raises(ValueError, match="distinct"):
        item(parameters={"Q_over": 1.0})
    document = {
        "protocol": transcriptions.PROTOCOL,
        "status": "proposed_pending_review",
        "source_run": "run",
        "source_plan_sha256": "0" * 64,
        "rules": {"euler": "...", "initial": "..."},
        "transcriptions": [item().model_dump(), item().model_dump()],
    }
    with pytest.raises(ValueError, match="twice"):
        transcriptions.TranscriptionSet.model_validate(document)
    document["transcriptions"] = [item().model_dump()]
    document["rules"] = {"initial": "..."}
    with pytest.raises(ValueError, match="undeclared"):
        transcriptions.TranscriptionSet.model_validate(document)


# --- adapters -------------------------------------------------------------------


def selection(method: str, payload: dict) -> dict:
    return BaselineDevelopmentResult(
        method=method,
        benchmark_id=NAME,
        tier="fixed",
        seed=0,
        equations={"h_down": "see payload"},
        selected_hyperparameters={},
        selection_payload=payload,
        training_normalized_mse=0.0,
        validation_normalized_mse=0.0,
    ).model_dump(mode="json")


def campaign(tmp_path, status: str, method: str, payload: dict | None = None):
    run = tmp_path / method
    plan = sealed_write(
        run / "plan.json",
        {
            "rows": [
                {"index": 0, "benchmark_id": NAME, "tier": "fixed", "repetition": 0}
            ]
        },
    )
    sealed_write(
        run / "results/0/result.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "index": 0,
            "status": status,
            "error": None if status == "complete" else "it stopped",
            "test_data_opened": False,
        },
    )
    if payload is not None:
        sealed_write(
            run / "results/0/native-selection.json",
            {
                "plan_sha256": plan["artifact_sha256"],
                "selection": selection(method, payload),
            },
        )
    return run, plan


def test_classical_rows_keep_failed_runs(tmp_path):
    root = tmp_path / "classical"
    tasks = [
        {"task_index": i, "method": m, "benchmark_id": NAME, "tier": "fixed",
         "repetition": 0}
        for i, m in enumerate(("sindy", "pysr"))
    ]
    (root / "frozen").mkdir(parents=True)
    (root / "frozen/task_plan.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in tasks)
    )
    sindy = root / "runs/sindy" / f"{NAME}_fixed_seed0"
    sindy.mkdir(parents=True)
    result = selection("sindy", {})
    result["equations"] = {"h_down": "inflow_down / area_down - 0.1 * h_down"}
    (sindy / "result.json").write_text(json.dumps(result))
    pysr = root / "runs/pysr" / f"{NAME}_fixed_seed0"
    pysr.mkdir(parents=True)
    (pysr / "run_status.json").write_text(
        json.dumps({"status": "timed_out", "error": "baseline wall-clock limit"})
    )
    rows = mechanisms.classical_rows(root, {NAME: coupled_cell()})
    assert [r["status"] for r in rows] == ["ready", "method_failure"]
    assert rows[0]["model"]["grammar"] == "baseline"
    assert rows[0]["model"]["semantics"] == "continuous_time"
    assert rows[1]["failure"] == "timed_out"


def test_d3_and_llm_ode_rows(tmp_path):
    cells = {NAME: coupled_cell()}
    increment = candidate({"h_down": "k * inflow_down / area_down"}, parameters=("k",))
    run, _ = campaign(
        tmp_path,
        "complete",
        "d3_native_no_tools",
        {"candidate": increment.model_dump(mode="json"), "parameters": {"k": 4.0}},
    )
    (row,) = mechanisms.campaign_rows("d3", run, cells)
    assert row["status"] == "ready"
    assert row["model"]["semantics"] == "native_increment"
    assert row["model"]["increment_step"] == 4.0
    system = candidate({"h_down": "inflow_down / area_down - 0.1 * h_down"})
    run, _ = campaign(
        tmp_path, "complete", "llm_ode", {"candidate": system.model_dump(mode="json")}
    )
    (row,) = mechanisms.campaign_rows("llm_ode", run, cells)
    assert row["model"]["semantics"] == "continuous_time"
    assert row["model"]["parameters"] == {}


def test_failures_count_and_infrastructure_stops_are_rerun(tmp_path):
    cells = {NAME: coupled_cell()}
    run, _ = campaign(tmp_path / "a", "rollout_failed", "llm_sr")
    reviewed = transcriptions.TranscriptionSet.model_validate(
        {
            "protocol": transcriptions.PROTOCOL,
            "status": "proposed_pending_review",
            "source_run": "run",
            "source_plan_sha256": json.loads((run / "plan.json").read_text())[
                "artifact_sha256"
            ],
            "rules": {"euler": "...", "initial": "..."},
            "transcriptions": [],
        }
    )
    (row,) = mechanisms.campaign_rows("llm_sr", run, cells, reviewed)
    assert (row["status"], row["failure"]) == ("method_failure", "rollout_failed")
    run, _ = campaign(tmp_path / "b", "endpoint_unavailable", "llm_ode")
    with pytest.raises(ValueError, match="rerun"):
        mechanisms.campaign_rows("llm_ode", run, cells)


def test_llm_sr_rows_use_their_bound_transcription(tmp_path):
    cells = {NAME: coupled_cell()}
    program = {
        "function": SOURCE,
        "parameters": [0.5],
        "variables": ["h_down", "inflow_down"],
        "look_ahead": {"reads_later_rows": False},
    }
    run, plan = campaign(
        tmp_path, "complete", "llm_sr", {"programs": {"h_down": program}}
    )
    reviewed = transcriptions.TranscriptionSet.model_validate(
        {
            "protocol": transcriptions.PROTOCOL,
            "status": "proposed_pending_review",
            "source_run": "run",
            "source_plan_sha256": plan["artifact_sha256"],
            "rules": {"euler": "...", "initial": "..."},
            "transcriptions": [item(task=0).model_dump()],
        }
    )
    (row,) = mechanisms.campaign_rows("llm_sr", run, cells, reviewed)
    assert row["model"]["transcription"]["hidden_states"] == ["h_up"]
    assert row["model"]["context"]["fitted_initialization"] is True
    other = reviewed.model_copy(update={"source_plan_sha256": "1" * 64})
    with pytest.raises(ValueError, match="another LLM-SR run"):
        mechanisms.campaign_rows("llm_sr", run, cells, other)
    # A pick that reads later rows is LLM-SR's failure (decided 2026-10-15).
    program["look_ahead"] = {"reads_later_rows": True}
    run, _ = campaign(
        tmp_path / "ahead", "complete", "llm_sr", {"programs": {"h_down": program}}
    )
    reviewed = reviewed.model_copy(
        update={
            "source_plan_sha256": json.loads((run / "plan.json").read_text())[
                "artifact_sha256"
            ]
        }
    )
    (row,) = mechanisms.campaign_rows("llm_sr", run, cells, reviewed)
    assert (row["status"], row["failure"]) == ("method_failure", "look_ahead")


def test_assess_and_report_keep_failures_in_the_denominator(tmp_path):
    cell = coupled_cell()
    ready = {
        "row_id": f"d3:{NAME}:0",
        "method": "d3",
        "benchmark_id": NAME,
        "tier": "fixed",
        "repetition": 0,
        "status": "ready",
        "model": native_row(f"4.0 * inflow_down / area_down - {OUTLET}"),
    }
    failed = {
        **{k: v for k, v in ready.items() if k != "model"},
        "row_id": f"d3:{NAME}:1",
        "repetition": 1,
        "status": "method_failure",
        "failure": "timed_out",
    }
    sealed_write(
        tmp_path / "plan.json",
        {
            "protocol": mechanisms.PROTOCOL,
            "settings": SETTINGS.model_dump(mode="json"),
            "cells": {NAME: cell},
            "rows": [ready, failed],
            "llm_sr_transcriptions": None,
        },
    )
    first = mechanisms.assess(tmp_path, 0)
    assert first["status"] == "assessed"
    assert mechanisms.assess(tmp_path, 0) == first  # reused, not rerun
    second = mechanisms.assess(tmp_path, 1)
    assert second["status"] == "method_failure"
    assert second["confirmed"] == 0 and second["possible"] == 1
    summary = mechanisms.report(tmp_path)
    aggregate = summary["methods"]["d3"]["all_cells"]
    assert aggregate["runs"] == 2 and aggregate["completed_runs"] == 1
    assert aggregate["confirmed_mean"] == pytest.approx(first["confirmed"] / 2)
    assert (tmp_path / "SUMMARY.md").read_text().startswith("# Phase C external")
