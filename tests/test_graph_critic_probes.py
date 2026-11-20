"""Frozen graph selection, training-only substitution, and numerical abstention."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import graph_critic_probes as critic
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory
from tests.test_candidate_validation import candidate_payload


def example(*, all_correct=False, algebraic=False):
    payload = candidate_payload()
    payload.update(parameters=[], constraints=[])
    payload["processes"][0]["expression"] = "x"
    payload["state_equations"] = [
        {"state": "x", "rhs": "2"},
        {"state": "y", "rhs": "flux"},
    ]
    payload["observation_mappings"] = [
        {"channel": "driver", "expression": "x+1" if algebraic else "x"},
        {"channel": "response", "expression": "y"},
    ]
    model = compile_candidate(
        CandidateModel.model_validate(payload),
        ValidationContext(targets=("response", "driver")),
    )
    t = np.arange(10, dtype=float)
    x = t * (2 if all_correct else 1)
    y = x * t / 2
    row = PublicTrajectory(
        trajectory_id="train_a",
        time=tuple(t),
        targets={"response": tuple(y), "driver": tuple(x + int(algebraic))},
    )
    train = PublicSplit(
        name="train",
        fingerprint="fixture",
        rows=(row.model_copy(update={"trajectory_id": "train_b"}), row),
    )
    return model, train, {}, {r.trajectory_id: [0.0, 0.0] for r in train.rows}


def test_all_windows_ranked_and_only_graph_connected_readouts_tested():
    result = critic.probe_endpoint(*example())
    assert result["status"] == "ready"
    assert result["compared_windows"] == 12
    window = result["window"]
    assert (window["trajectory_id"], window["target"], window["band"]) == (
        "train_a",
        "response",
        "late",
    )
    assert window["observed_change"] == pytest.approx(22.5)
    assert window["predicted_change"] == pytest.approx(45.0)
    (probe,) = result["probes"]
    assert probe["paths"] == {
        "response": ["state:x", "process:flux", "state:y", "target:response"]
    }
    out = probe["outcomes"]["response"]
    assert out["observed_terminal"] == pytest.approx(40.5)
    assert out["free_terminal"] == pytest.approx(81.0)
    # y remains at its fitted value 36 at t=6; only the x drive is supplied.
    assert out["substituted_terminal"] == pytest.approx(36.0 + 22.5)
    assert out["terminal_error_reduction"] == pytest.approx(22.5)
    assert out["rmse_reduction"] > 0


def test_withhold_on_accurate_parent_or_nonidentity_measurement():
    good = critic.probe_endpoint(*example(all_correct=True))
    assert good["status"] == "withheld" and not good["probes"]
    assert good["training_nmse"] < 1e-14
    algebraic = critic.probe_endpoint(*example(algebraic=True))
    assert algebraic["status"] == "ready" and not algebraic["probes"]
    assert algebraic["balance"]["status"] == "ready"


def test_split_root_and_saved_score_checks():
    model, train, p, initials = example()
    with pytest.raises(ValueError, match="training split"):
        critic.probe_endpoint(
            model, train.model_copy(update={"name": "val"}), p, initials
        )
    with pytest.raises(ValueError, match="coverage"):
        critic.probe_endpoint(model, train, p, {})
    mismatch = critic.probe_endpoint(model, train, p, initials, expected_nmse=0.1)
    assert mismatch["window"] is None and "saved endpoint" in mismatch["reason"]


@pytest.mark.parametrize("clamped", [False, True])
def test_solver_disagreement_withholds_numerical_evidence(monkeypatch, clamped):
    old = critic._trace

    def altered(model, row, p, initial, method, first, last, driver=None, **kwargs):
        trace = old(model, row, p, initial, method, first, last, driver, **kwargs)
        if method == "DOP853" and (driver is not None) == clamped:
            trace.values += 0.3
        return trace

    monkeypatch.setattr(critic, "_trace", altered)
    result = critic.probe_endpoint(*example())
    if clamped:
        assert result["balance"]["status"] == "ready"
        (probe,) = result["probes"]
        assert probe["status"] == "withheld" and not probe["outcomes"]
        assert "solvers disagree" in probe["reason"]
    else:
        assert result["status"] == "withheld"
        assert "solvers disagree" in result["reason"] and result["window"] is None


def test_training_score_weights_samples_and_constant_target_exclusion():
    model, train, p, initials = example()
    one, two = deepcopy(train.rows)
    t = two.time[:2]
    train = train.model_copy(
        update={
            "rows": (
                one,
                two.model_copy(
                    update={
                        "time": t,
                        "targets": {c: row[:2] for c, row in two.targets.items()},
                    }
                ),
            )
        }
    )
    result = critic.probe_endpoint(model, train, p, initials)
    assert result["compared_windows"] == 6
    truth = np.concatenate(
        [
            np.column_stack([r.targets[c] for c in model.validated.context.targets])
            for r in train.rows
        ]
    )
    assert result["training_nmse"] == pytest.approx(
        np.mean((truth / truth.std(axis=0)) ** 2)
    )
    constant = train.model_copy(
        update={
            "rows": tuple(
                r.model_copy(
                    update={"targets": {**r.targets, "driver": (0.0,) * len(r.time)}}
                )
                for r in train.rows
            )
        }
    )
    result = critic.probe_endpoint(model, constant, p, initials)
    assert result["constant_targets"] == ["driver"]


def test_rejected_solver_work_is_charged(monkeypatch):
    monkeypatch.setitem(critic.RULES, "max_rhs_calls_per_solver_trace", 2)
    result = critic.probe_endpoint(*example())
    assert result["status"] == "withheld"
    assert "exceeded 2 RHS calls" in result["reason"]
    assert result["compute"] == {
        "solver_traces_started": 1,
        "baseline_rhs": 2,
        "clamped_rhs": 0,
    }


def test_signed_integral_closure_and_coarse_quadrature_withholding():
    model, train, p, initials = example()
    payload = model.validated.candidate.model_dump(mode="json")
    payload["state_equations"][1]["rhs"] = "-(-flux) + 2 - 1"
    model = compile_candidate(
        CandidateModel.model_validate(payload), model.validated.context
    )
    balance = critic.probe_endpoint(model, train, p, initials)["balance"]
    assert balance["status"] == "ready"
    assert balance["closure_error"] == pytest.approx(0, abs=1e-10)
    assert [term["sign"] for term in balance["terms"]] == [1, 1, -1]
    assert [term["signed_integral"] for term in balance["terms"]] == pytest.approx(
        [45, 6, -3]
    )
    payload["state_equations"][1]["rhs"] = "flux ** 4"
    model = compile_candidate(
        CandidateModel.model_validate(payload), model.validated.context
    )
    result = critic.probe_endpoint(model, train, p, initials)
    assert result["window"]["target"] == "response"
    assert result["balance"]["status"] == "withheld"
    assert "quadrature" in result["balance"]["reason"]
    assert result["balance"]["terms"] == []


def test_alias_targets_never_validate_their_own_substitution():
    model, train, p, initials = example()
    payload = model.validated.candidate.model_dump(mode="json")
    # The alias uses a named identity process, but it measures the same x state.
    payload["observation_mappings"].append(
        {"channel": "alias", "expression": "flux", "unit": "relative"}
    )
    context = ValidationContext(targets=("alias", "driver", "response"))
    model = compile_candidate(CandidateModel.model_validate(payload), context)
    train = train.model_copy(
        update={
            "rows": tuple(
                r.model_copy(
                    update={"targets": {**r.targets, "alias": r.targets["driver"]}}
                )
                for r in train.rows
            )
        }
    )
    result = critic.probe_endpoint(model, train, p, initials)
    assert {r["driver"] for r in result["probes"]} == {"driver", "alias"}
    assert all(set(r["outcomes"]) == {"response"} for r in result["probes"])
