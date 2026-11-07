"""The depleted-state feedback is replayable and only qualified checks block."""

import json
from copy import deepcopy

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.research import insulin_boundary_revision as boundary
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from scripts import propose_meal_insulin_revision as cli
from scripts.prepare_insulin_boundary_intake import _windows
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_construction_revision_edits import problem
from tests.test_meal_insulin_proposer import forbidden, patch, source
from tests.test_meal_revision_metrics import evaluate, training


def with_rhs(rhs):
    data = problem().model_dump(mode="json")
    next(p for p in data["base_candidate"]["state_equations"] if p["state"] == "I")[
        "rhs"
    ] = rhs
    return PrefixFitProblem.model_validate(data)


@pytest.fixture
def point():
    return {
        "trajectory": "train_003",
        "time_min": 2,
        "inputs": {boundary.MEAL: 0.0, boundary.INFUSION: 0.0},
        "state_at_boundary": {"I": 0.0, "Gp": 1.5, "mem": 2.0, "dep": 0.0},
        "reference_rate_pmol_L_min": -2.0,
    }


def test_boundary_proofs_use_assembled_process_and_complete_rhs(point):
    parent = with_rhs("-q")
    with pytest.raises(cli.RevisionValidationError, match="UNCHANGED_OUTWARD"):
        boundary.check(parent, with_rhs("-q - a*I"), point)
    with pytest.raises(cli.RevisionValidationError, match="NEGATIVE_INSULIN_BOUNDARY"):
        boundary.check(parent, with_rhs("-1 - I*q"), point)
    with pytest.raises(cli.RevisionValidationError, match="NEGATIVE_INSULIN_BOUNDARY"):
        boundary.check(parent, with_rhs("-a*Gp - I*q"), point)
    out = boundary.check(parent, with_rhs("a*Gp - I*q"), point)
    assert out["boundary_sign"] == "nonnegative_under_stated_assumptions"
    assert (
        out["candidate_at_inherited_values_and_new_parameter_guesses"][
            "rate_pmol_L_min"
        ]
        == 1.5
    )
    assert not out["continuous_trajectory_positivity_verified"]


def test_unknowns_and_unfitted_point_estimates_defer(point):
    parent = with_rhs("-q")
    # A revised boundary at an inherited parameter point can need refitting.
    changed = boundary.check(parent, with_rhs("Gp-2*q"), point)
    assert changed["boundary_sign"] == "deferred"
    assert (
        changed["candidate_at_inherited_values_and_new_parameter_guesses"][
            "rate_pmol_L_min"
        ]
        < 0
    )
    unsupported = boundary.check(parent, with_rhs("softplus(Gp)-I*q"), point)
    assert unsupported["boundary_sign"] == "deferred"
    missing = deepcopy(point)
    del missing["state_at_boundary"]["dep"]
    missing_rate = boundary.check(parent, with_rhs("dep-I*q"), missing)
    assert missing_rate["candidate_at_inherited_values_and_new_parameter_guesses"] == {
        "status": "deferred",
        "unavailable_symbols": ["dep"],
    }
    with pytest.raises(ValueError, match="does not reproduce"):
        boundary.check(
            parent, with_rhs("Gp"), {**point, "reference_rate_pmol_L_min": -100.0}
        )


def test_ingestion_window_recovers_prediction_change_in_channel_units():
    train = training()
    # U and Gp have unrelated residuals. The insulin error grows by 3 units
    # per minute after rescaling, irrespective of the trajectory's onset time.
    residual = np.concatenate(
        [
            np.column_stack(
                (
                    np.ones(len(row.time)),
                    -np.ones(len(row.time)),
                    1.5 * np.array(row.time),
                )
            ).ravel()
            for row in train.rows
        ]
    )
    terminal = {
        "metrics": {
            "channels": ["U", "Gp", "I"],
            "residual": residual,
            "scales": {"I": 2.0},
            "metrics": evaluate(train),
        },
        "retained": {"parameters": {"c_I": 2.0}},
    }
    windows = _windows(train, terminal)
    assert len(windows) == 3
    for row in windows:
        assert row["fitted_rise_pmol_L"] - row["observed_rise_pmol_L"] == pytest.approx(
            30
        )
        assert row["direct_ingestion_contribution_pmol_L"] == pytest.approx(180)


def test_boundary_protocol_pins_intake_renders_actual_request_and_resumes(
    tmp_path, monkeypatch, point
):
    data = source()
    data["fitted_problem"] = with_rhs("-q").model_dump(mode="json")
    data["public_brief"] = data["public_brief"].model_dump(mode="json")
    data["boundary_feedback"] = {
        "depletion": point,
        "ingestion_windows": [{"trajectory": "train_006"}],
    }
    del data["comparison"]["routing"]
    intake = {"protocol": boundary.INTAKE_PROTOCOL, "arguments": data}
    monkeypatch.setattr(boundary, "INTAKE_SHA256", cli.content_sha256(intake))
    path = tmp_path / "source.json"
    root = tmp_path / "boundary"
    seal(path, intake)
    plan = cli.prepare_source(root, path)
    assert plan["protocol"] == boundary.PROTOCOL
    assert cli.prepare_source(root, path) == plan
    body = patch(
        edit={
            "component": "I",
            "kind": "dynamic",
            "expression": "a*Gp - q*I",
            "sources": ["Gp", "q", "I"],
        },
        insulin_rate_at_zero="a*Gp is nonnegative.",
        expected_ingestion_window_change="No direct meal-rate source.",
    )
    first = cli.run(
        root,
        "http://127.0.0.1",
        transport=lambda *args: reply(body),
        token_transport=tokenizer,
    )
    assert first["status"] == "eligible_for_refit", first
    assert (
        first["insulin_boundary_check"]["boundary_sign"]
        == "nonnegative_under_stated_assumptions"
    )
    assert first["refit_handoff"]["fitted_parent_reference"] == data["result_identity"]
    assert "constrained endpoint" in first["refit_handoff"]["comparison"]
    (generated,) = (root / "calls").glob("*.json")
    request = json.loads(generated.read_text())["request"]["body"]
    assert request["messages"][0]["content"] == cli.BOUNDARY_SYSTEM
    assert (
        json.loads(request["messages"][1]["content"])["training_evidence"]
        == data["boundary_feedback"]
    )
    (root / "result.json").unlink()
    assert (
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == first
    )
    assert cli.inspect(root)["physical_requests_started"] == 1
    html = (root / "PROMPT.html").read_text()
    assert "clipping" in html.lower() and "insulin_rate_at_zero" in html
    assert "When ingestion ends, what could sustain" not in html
    data["boundary_feedback"]["ingestion_windows"][0]["trajectory"] = "val_001"
    seal(tmp_path / "changed.json", intake)
    with pytest.raises(ValueError, match="reviewed insulin boundary intake"):
        cli.prepare_source(tmp_path / "bad", tmp_path / "changed.json")
    with pytest.raises(ValueError, match="public training"):
        cli.prepare(tmp_path / "bad", **data)
