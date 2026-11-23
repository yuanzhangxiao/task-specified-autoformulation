"""Train-only discharge proposals, numerical semantics and persisted call budgets."""

import json

import numpy as np
import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.expressions.compiler import _evaluate
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import basin_process_proposals as pilot
from scripts import fit_basin_process_proposals as scoring
from scripts import trajectory_functions as trial
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_trajectory_function_families import event, packet


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    survey = {
        "area_up": 800,
        "area_down": 1200,
        "crest_up": 0.35,
        "crest_down": 0.1,
        "initial_up": 0.15,
    }
    ctx = ValidationContext(
        targets=("h_down",),
        external_inputs=("inflow_down", "inflow_up"),
        fixed_covariates=tuple(survey),
    )
    events = [
        event(
            f"train_{i:03}",
            [0, 1, 2],
            {"h_down": np.full(3, 0.3 + i)},
            {"inflow_up": np.full(3, i), "inflow_down": np.ones(3)},
            survey,
        )
        for i in (0, 1)
    ]
    c = "basin_seed0_downstream"
    case = packet(
        events,
        ctx,
        case=c,
        initial_rules={"h_down": {"mode": "observed_initial"}},
        dependencies_exhaustive=True,
        shared_parameters=True,
        open_loop=True,
        reset_impulse_free=True,
        uniqueness_assumed=True,
        absolute_error_bounds=dict.fromkeys(
            (*ctx.targets, *ctx.external_inputs, *ctx.fixed_covariates), 0
        ),
        interpolation=dict.fromkeys(ctx.external_inputs, "linear_complete_knots"),
    )
    case.update(
        target="h_down",
        histories=list(ctx.external_inputs),
        initials=["initial_up"],
        required_histories=list(ctx.external_inputs),
        provenance={"case": pilot.BASIN},
        measured_initial_bindings={"h_up": "initial_up"},
        public_brief={
            "scientific_context": "Model both water balances.",
            "public_variables": [
                {"name": n, "data_role": role}
                for role, names in (
                    ("external_input", ctx.external_inputs),
                    ("target", ctx.targets),
                    ("covariate", ctx.fixed_covariates),
                )
                for n in names
            ],
            "requirements": [],
        },
    )
    rows = case["evidence"].pop("trajectories")
    graph = {
        "variable_types": dict.fromkeys(("h_down", "h_up"), "differential"),
        "sources": {"h_down": ["h_up"], "h_up": ["inflow_up"]},
        "initializers": {
            "h_down": {"mode": "observed_initial", "channel": "h_down"},
            "h_up": {"mode": "observed_initial", "channel": "initial_up"},
        },
    }
    monkeypatch.setattr(pilot, "PARENT", content_hash(graph))
    data = sealed_write(
        tmp_path / "campaign/intake.json",
        {
            "source": "training",
            "train": {pilot.BASIN: rows},
            "observations": {pilot.BASIN: []},
            "cases": {c: case},
            "graphs": [
                {
                    "dataset": pilot.BASIN,
                    "case": c,
                    "graph": graph,
                    "structure_sha256": pilot.PARENT,
                }
            ],
        },
    )
    monkeypatch.setattr(pilot, "INTAKE", data["artifact_sha256"])
    monkeypatch.setattr(pilot, "runtime_source_hash", lambda: "test")
    root = tmp_path / "result"
    pilot.prepare(root, tmp_path / "campaign")
    return root


def test_physical_model_and_safe_dry_head(run_dir):
    _, data = pilot.load(run_dir)
    result = pilot.assemble(
        data,
        {
            "upstream_factor": "k*head**0.5",
            "downstream_factor": "d",
            "parameters": [{"name": "k", "initial": -3}, {"name": "d", "initial": 2}],
        },
    )
    assert result["status"] == "eligible"
    assert result["balance_check"]["status"] == "pass"
    formula = result["formula"]
    rhs = {
        n: RestrictedParser().parse(s, location=n)
        for n, s in formula["equations"].items()
    }
    environment = {
        "inflow_up": 0,
        "inflow_down": 0,
        "area_up": 8,
        "area_down": 10,
        "crest_up": 0.35,
        "crest_down": 0.1,
        "h_up": 0.35,
        "h_down": 0.09,
        "k": -3,
        "d": 2,
    }
    assert (
        _evaluate(rhs["h_up"], environment)
        == _evaluate(rhs["h_down"], environment)
        == 0
    )
    environment.update(h_up=4.35, h_down=1.1)
    assert _evaluate(rhs["h_up"], environment) == pytest.approx(-24 / 8)
    assert _evaluate(rhs["h_down"], environment) == pytest.approx((24 - 2) / 10)
    assert "initial_up" not in result["revised_graph"]["sources"]["h_down"]


@pytest.mark.parametrize(
    "factor",
    [
        "1/head",
        "head**-1",
        "log(head)",
        "sqrt(-1)",
        "Q_ch",
        "h_up",
        "__import__('sys')",
    ],
)
def test_rejects_undefined_or_unavailable_factor(factor):
    with pytest.raises((ValueError, ModelValidationError)):
        pilot._factor(factor, "down", {})


def test_generation_resume_budget_and_train_only(run_dir):
    values = ["k", "unknown", "1/head"]
    calls = []

    def provider(url, body, timeout):
        data = json.loads(body["messages"][1]["content"])
        assert set(data) == {
            "scientific_task",
            "public_requirements",
            "training_observations",
            "outlet_evidence",
        }
        points = data["outlet_evidence"]["rows"]
        assert points and {s["event"] for s in points} == {"train_000"}
        assert all(s["mean_outlet_m3_per_min"] == 1 for s in points)
        calls.append(body["seed"])
        return reply(
            {
                "upstream_factor": values[len(calls) - 1],
                "downstream_factor": "k",
                "parameters": [{"name": "k", "initial": 4}],
            }
        )

    result = pilot.run(
        run_dir, "http://localhost", transport=provider, token_transport=tokenizer
    )
    assert len(set(calls)) == 3
    assert result["status"] == "complete" and result["physical_requests"] == 3
    assert result["candidate_statuses"] == {"eligible": 1, "invalid_formula": 2}

    def duplicate(*args):
        pytest.fail("A settled generation was repeated")

    assert pilot.run(run_dir, "http://localhost", transport=duplicate) == result
    recorded = sealed_read(run_dir / "candidates/002.json")
    assert "undefined at zero head" in recorded["reason"]
    with pytest.raises(ValueError, match="training"):
        pilot._outlet_observations([{"trajectory_id": "val_001"}])
    (physical,) = (run_dir / "steps/slot-000/calls").glob("*.json")
    physical.write_text(physical.read_text() + "\n")
    with pytest.raises(ValueError, match="receipt"):
        pilot.status(run_dir)


def test_fit_requires_freeze_and_keeps_validation_separate(
    run_dir, tmp_path, monkeypatch
):
    out = tmp_path / "numeric"
    with pytest.raises(ValueError, match="Generate all"):
        scoring.train(run_dir, out)

    def mock(url, body, timeout):
        return reply(
            {"upstream_factor": "1", "downstream_factor": "1", "parameters": []}
        )

    pilot.run(run_dir, "http://localhost", transport=mock, token_transport=tokenizer)
    _, data = pilot.load(run_dir)
    val = tmp_path / "val.json"

    cohort = sealed_write(
        val,
        {
            "training_intake_sha256": "training",
            "rows": {
                pilot.BASIN: [
                    {
                        **event,
                        "trajectory_id": event["trajectory_id"].replace("train", "val"),
                    }
                    for event in data["train"][pilot.BASIN]
                ]
            },
        },
    )
    monkeypatch.setattr(trial, "VALIDATION", cohort["artifact_sha256"])
    ranking = scoring.train(run_dir, out)
    assert ranking["training_order"] == [0, 1, 2]
    assert not (out / "validation").exists()
    assert scoring.train(run_dir, out) == ranking
    wrong = tmp_path / "wrong.json"
    sealed_write(wrong, {"split": "wrong"})
    with pytest.raises(ValueError, match="Unexpected"):
        scoring.validate(run_dir, out, wrong)
    summary = scoring.validate(run_dir, out, val)
    assert [r["validation_status"] for r in summary["rows"]] == ["scored"] * 3
    assert sealed_read(out / "ranking.json") == ranking
    assert scoring.validate(run_dir, out, val) == summary
