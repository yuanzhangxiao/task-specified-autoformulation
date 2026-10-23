"""Offline model failures remain visible without weakening the training boundary."""

import json
from copy import deepcopy

import pytest
from scipy import optimize

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import simulation
from autoformalism.llm import staged_topology
from autoformalism.staged_topology import content_hash
from scripts import audit_construction_functions as campaign
from tests.test_construction_function_consistency import evidence, problem


def inputs():
    """A small authenticated training fixture with a provable slope conflict."""
    source = problem()
    training = evidence(source)
    return {
        "protocol": "construction-function-science-inputs-1",
        "models": [
            {
                "model_id": "candidate",
                "source_role": "candidate",
                "problem": source.model_dump(mode="json"),
            }
        ],
        "evidence": training,
        "source_sha256": "1" * 64,
        "evidence_sha256": content_hash(training),
        "test_data_opened": False,
    }


def forbidden(*_args, **_kwargs):
    raise AssertionError("numerical/provider activity or invalid-envelope assessment")


def run(tmp_path, value):
    source = tmp_path / "inputs.json"
    seal(source, value)
    return campaign.audit(source, tmp_path / "report")


def test_train_only_successful_proof_without_calls_or_rollouts(tmp_path, monkeypatch):
    monkeypatch.setattr(simulation, "simulate_trajectory", forbidden)
    monkeypatch.setattr(simulation, "solve_ivp", forbidden)
    monkeypatch.setattr(staged_topology, "http_transport", forbidden)
    monkeypatch.setattr(staged_topology.StagedTopologyClient, "call", forbidden)
    monkeypatch.setattr(optimize, "least_squares", forbidden)
    monkeypatch.setattr(optimize, "minimize", forbidden)
    monkeypatch.setattr(campaign, "runtime_source_hash", lambda: "2" * 64)
    value = inputs()
    before = deepcopy(value)
    result = run(tmp_path, value)
    assert result["status_counts"] == {"blocked": 1}
    assert result["unavailable_model_count"] == 0
    assert (
        result["llm_calls"]
        == result["optimizer_calls"]
        == result["solver_rollouts"]
        == 0
    )
    assert result["test_data_opened"] is False
    assert result["fitted_metrics_used"] is False
    assert value == before
    saved = json.loads((tmp_path / "report" / "audit.json").read_text())
    functions = saved["models"][0]["training_consistency"]["function_consistency"]
    assert functions["conflicts"][0]["witnesses"][0]["required_change"] == {
        "comparison": "at_least",
        "value": 2,
    }
    assert "Training witness" in (tmp_path / "report" / "REPORT.html").read_text()


@pytest.mark.parametrize(
    "field", ["validation", "test", "metrics", "fitted_parameters"]
)
def test_non_training_and_fitted_payloads_rejected_at_outer_boundary(
    tmp_path, monkeypatch, field
):
    value = inputs()
    value[field] = {"untrusted": "must not be consumed"}
    monkeypatch.setattr(campaign.construction_revision_consistency, "assess", forbidden)
    with pytest.raises(ValueError, match="Extra inputs"):
        run(tmp_path, value)
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("split", ["validation", "test"])
def test_non_training_evidence_rejected_before_assessing_any_model(
    tmp_path, monkeypatch, split
):
    value = inputs()
    value["evidence"]["split"] = split
    value["evidence_sha256"] = content_hash(value["evidence"])
    monkeypatch.setattr(campaign.construction_revision_consistency, "assess", forbidden)
    with pytest.raises(ValueError):
        run(tmp_path, value)
    assert not (tmp_path / "report").exists()


def test_duplicate_ids_reject_entire_envelope(tmp_path, monkeypatch):
    value = inputs()
    value["models"].append(deepcopy(value["models"][0]))
    monkeypatch.setattr(campaign.construction_revision_consistency, "assess", forbidden)
    with pytest.raises(ValueError, match="duplicate model identity"):
        run(tmp_path, value)


@pytest.mark.parametrize("change", ["digest", "inner_digest", "context", "test_flag"])
def test_invalid_evidence_hash_context_or_access_flag_fail_before_model_checks(
    tmp_path, monkeypatch, change
):
    value = inputs()
    if change == "digest":
        value["evidence_sha256"] = "f" * 64
    elif change == "inner_digest":
        value["evidence"]["sha256"] = "f" * 64
        value["evidence_sha256"] = content_hash(value["evidence"])
    elif change == "context":
        value["models"][0]["problem"]["context"]["auxiliaries"] = ["different"]
    else:
        value["test_data_opened"] = True
    monkeypatch.setattr(campaign.construction_revision_consistency, "assess", forbidden)
    with pytest.raises(ValueError):
        run(tmp_path, value)
    assert not (tmp_path / "report").exists()


def test_changed_seal_rejected_before_model_checks(tmp_path, monkeypatch):
    value = inputs()
    source = tmp_path / "inputs.json"
    seal(source, value)
    raw = json.loads(source.read_text())
    raw["value"]["models"][0]["model_id"] = "changed"
    source.write_text(json.dumps(raw))
    monkeypatch.setattr(campaign.construction_revision_consistency, "assess", forbidden)
    with pytest.raises(ValueError, match="sealed artifact digest differs"):
        campaign.audit(source, tmp_path / "report")


def test_invalid_candidate_formula_is_retained_and_other_models_continue(
    tmp_path, monkeypatch
):
    value = inputs()
    bad = deepcopy(value["models"][0])
    bad["model_id"] = "invalid-candidate"
    bad["problem"]["base_candidate"]["state_equations"][0]["rhs"] = "missing_name+x"
    value["models"].insert(0, bad)
    monkeypatch.setattr(campaign, "runtime_source_hash", lambda: "2" * 64)
    result = run(tmp_path, value)
    assert result["model_count"] == 2
    assert result["status_counts"] == {"unavailable": 1, "blocked": 1}
    assert result["unavailable_model_count"] == 1
    saved = json.loads((tmp_path / "report" / "audit.json").read_text())
    failed, valid = saved["models"]
    assert failed["model_id"] == "invalid-candidate"
    assert failed["diagnostic"]["error_type"] == "ModelValidationError"
    assert failed["diagnostic"]["stage"] == "matched_condition_equality"
    assert failed["diagnostic"]["model_diagnostics"]
    assert failed["training_consistency"]["status"] == "unavailable"
    assert valid["status"] == "blocked"
    html = (tmp_path / "report" / "REPORT.html").read_text()
    assert (
        "invalid-candidate" in html and "Assessment unavailable or incomplete" in html
    )


def test_unexpected_programming_failure_is_not_relabelled_unavailable(
    tmp_path, monkeypatch
):
    def bug(*_args, **_kwargs):
        raise RuntimeError("programming failure")

    monkeypatch.setattr(campaign.construction_function_consistency, "assess", bug)
    with pytest.raises(RuntimeError, match="programming failure"):
        run(tmp_path, inputs())
    assert not (tmp_path / "report").exists()
