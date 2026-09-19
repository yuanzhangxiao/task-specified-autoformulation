"""Syntax evidence never certifies transfers or substitutes parameters."""

import copy
import json
import re

import pytest

from autoformalism.expressions import ModelValidationError
from autoformalism.rebuttal.mechanism_audit_sources import BUNDLE_PROTOCOL
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.shared_process_audit import audit_bundle, inventory
from autoformalism.schemas import CandidateModel
from tests.test_repair_comparison import CONTEXT, candidate


def model(m_rhs, y_rhs, *, processes=None):
    payload = candidate().model_dump(mode="json")
    payload["state_equations"] = [
        {"state": "m", "rhs": m_rhs},
        {"state": "y", "rhs": y_rhs},
    ]
    if processes is not None:
        payload["processes"] = processes
    symbols = set(re.findall(r"\b[a-zA-Z_]\w*\b", m_rhs + " " + y_rhs))
    payload["parameters"] = [p for p in payload["parameters"] if p["name"] in symbols]
    return CandidateModel.model_validate(payload)


def test_named_shared_law_is_reused_without_assuming_conservation():
    value = model("-f+gain*u01", "f-decay*y")
    before = value.model_dump(mode="json")
    report = inventory(value, CONTEXT)
    assert report["existing_shared_processes"] == [
        {
            "process": "f",
            "expression": "sigmoid(m)",
            "consumers": ["state:m", "state:y"],
        }
    ]
    assert report["physical_transfer_status"] == "not_inferred_from_syntax"
    assert report["missing_shared_mechanisms"] is None
    assert value.model_dump(mode="json") == before


def test_outer_sign_normalization_preserves_internal_differences():
    value = model("(-rate)*(m-y)+gain*u01", "rate*(m-y)-decay*y")
    report = inventory(value, CONTEXT)
    groups = report["exact_repeated_terms"]
    assert len(groups) == 1
    assert {w["outer_sign"] for w in groups[0]["witnesses"]} == {-1, 1}
    assert all(w["expression"] == "rate * (m - y)" for w in groups[0]["witnesses"])


def test_independent_parameters_are_only_similarity_evidence():
    value = model("-rate*m+gain*u01", "decay*m-y")
    report = inventory(value, CONTEXT)
    assert report["exact_repeated_terms"] == []
    assert len(report["similar_terms_review_only"]) == 1
    witnesses = report["similar_terms_review_only"][0]["witnesses"]
    assert {tuple(w["parameters"]) for w in witnesses} == {("rate",), ("decay",)}


def test_repeated_response_with_distinct_outer_scales_is_reported():
    value = model("-rate*tanh(m-y)+gain*u01", "decay*tanh(m-y)-y")
    report = inventory(value, CONTEXT)
    assert len(report["exact_repeated_nonlinear_calls"]) == 1
    assert not report["exact_repeated_terms"]


def test_sign_changes_within_response_are_not_normalized_away():
    report = inventory(model("tanh(m-y)+u01", "tanh(y-m)-y"), CONTEXT)
    assert not report["exact_repeated_nonlinear_calls"]
    assert not report["similar_terms_review_only"]


def test_audit_rejects_untrusted_or_cyclic_expressions():
    with pytest.raises(ModelValidationError):
        inventory(model("__import__('os')", "m-y"), CONTEXT)
    with pytest.raises(ModelValidationError):
        inventory(
            model("f+u01", "m-y", processes=[{"name": "f", "expression": "f+m"}]),
            CONTEXT,
        )


def bundle(tmp_path):
    path = tmp_path / "bundle.json"
    sealed_write(
        path,
        {
            "protocol": BUNDLE_PROTOCOL,
            "test_data_opened": False,
            "private_reference_opened": False,
            "rows": [
                {
                    "status": "ready",
                    "semantics": "continuous_time",
                    "benchmark_id": "demo",
                    "method": "autoformalism:full",
                    "repetition": 0,
                    "candidate": candidate().model_dump(mode="json"),
                    "context": CONTEXT.model_dump(mode="json"),
                },
                {
                    "status": "unavailable",
                    "benchmark_id": "demo",
                    "method": "autoformalism:brief_only",
                },
            ],
        },
    )
    return path


def test_resume_is_deterministic_and_input_and_bad_rows_are_preserved(tmp_path):
    path = bundle(tmp_path)
    before = path.read_bytes()
    root = tmp_path / "audit"
    result = audit_bundle(path, root)
    assert result["status_counts"] == {"complete": 1, "unavailable": 1}
    times = {p: p.stat().st_mtime_ns for p in root.rglob("*.json")}
    assert audit_bundle(path, root) == result
    assert {p: p.stat().st_mtime_ns for p in times} == times
    assert path.read_bytes() == before
    assert not result["trajectory_tables_opened"]
    assert (
        result["llm_calls"] == result["optimizer_calls"] == result["model_changes"] == 0
    )
    changed = copy.deepcopy(sealed_read(path))
    changed.pop("artifact_sha256")
    changed["rows"].pop()
    next_path = tmp_path / "changed.json"
    sealed_write(next_path, changed)
    with pytest.raises(ValueError, match="frozen artifact differs"):
        audit_bundle(next_path, root)


def test_corrupted_checkpoint_and_private_bundle_are_rejected(tmp_path):
    path = bundle(tmp_path)
    root = tmp_path / "audit"
    audit_bundle(path, root)
    (root / "rows" / "0000.json").write_text(json.dumps({"status": "complete"}))
    with pytest.raises(ValueError, match="artifact digest"):
        audit_bundle(path, root)
    value = sealed_read(path)
    value.pop("artifact_sha256")
    value["private_reference_opened"] = True
    other = tmp_path / "private.json"
    sealed_write(other, value)
    with pytest.raises(ValueError, match="test/private-free"):
        audit_bundle(other, tmp_path / "other")
