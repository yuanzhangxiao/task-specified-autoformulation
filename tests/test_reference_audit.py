"""Audit failure detection, immutable checkpoints and release separation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism import reference_integration as integration
from autoformalism.benchmarks import reference_audit as audit
from autoformalism.benchmarks.phase_b_generation import (
    phase_b_protocols,
    simulate_phase_b,
    write_private_bundle,
)
from autoformalism.benchmarks.phase_b_public import (
    phase_b_public_spec,
    write_public_staging_bundle,
)


def test_failed_or_truncated_integration_is_not_a_valid_reference(monkeypatch):
    for result in (
        SimpleNamespace(success=False, message="failed", y=[], t=[]),
        SimpleNamespace(success=True, message="truncated", y=np.zeros((1, 0)), t=[]),
    ):
        monkeypatch.setattr(integration, "solve_ivp", lambda *a, r=result, **k: r)
        with pytest.raises(RuntimeError, match="reference simulation failed"):
            integration.integrate_reference(
                lambda t, x: x, np.array([0.0, 1.0]), np.zeros(1)
            )


def test_audit_resume_does_not_resimulate_and_detects_record_tampering(
    tmp_path, monkeypatch
):
    protocol = phase_b_protocols("cstr")[0]
    record, trajectory = audit.audit_one(
        tmp_path, protocol, "canonical", Path("data_raw"), "source-1"
    )
    assert record["passed"]
    monkeypatch.setattr(
        audit, "simulate_phase_b", lambda *a, **k: pytest.fail("resimulated")
    )
    resumed, restored = audit.audit_one(
        tmp_path, protocol, "canonical", Path("data_raw"), "source-1"
    )
    assert record == resumed
    np.testing.assert_array_equal(trajectory.states, restored.states)
    with pytest.raises(ValueError, match="checkpoint changed"):
        audit.audit_one(tmp_path, protocol, "canonical", Path("data_raw"), "source-2")
    path = tmp_path / f"{protocol.protocol_id}.json"
    altered = json.loads(path.read_text())
    altered["passed"] = False
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="checkpoint changed"):
        audit.audit_one(tmp_path, protocol, "canonical", Path("data_raw"), "source-1")


def test_audit_array_tampering_is_rejected(tmp_path):
    protocol = phase_b_protocols("cstr")[0]
    audit.audit_one(tmp_path, protocol, "canonical", Path("data_raw"), "source-1")
    path = tmp_path / f"{protocol.protocol_id}.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        audit.audit_one(tmp_path, protocol, "canonical", Path("data_raw"), "source-1")


def test_comparison_catches_wrong_reference_and_misaligned_inputs():
    protocol = phase_b_protocols("cstr")[0]
    original = simulate_phase_b(protocol)
    altered = original.model_copy(update={"states": original.states + 1})
    assert not audit.compare_references(original, altered)["passed"]
    with pytest.raises(ValueError, match="inputs differ"):
        audit.compare_references(
            original, original.model_copy(update={"inputs": original.inputs + 1})
        )


def test_public_projection_is_checked_against_actual_csv_not_just_manifest(tmp_path):
    protocol = phase_b_protocols("cstr")[0]
    trajectory = simulate_phase_b(protocol)
    spec = phase_b_public_spec("cstr", "easy", "obfuscated")
    write_public_staging_bundle(tmp_path, spec, (trajectory,))
    path = tmp_path / "train.csv"
    assert audit.check_projection(path, spec, (trajectory,))
    text = path.read_text()
    rows = text.splitlines()
    cells = rows[1].split(",")
    cells[-1] = "123456"
    rows[1] = ",".join(cells)
    path.write_text("\n".join(rows) + "\n")
    assert not audit.check_projection(path, spec, (trajectory,))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["reference_generation_protocol"] == integration.REFERENCE_PROTOCOL
    assert protocol.protocol_id not in json.dumps(manifest["reference_solver_settings"])


def test_writers_refuse_to_overwrite_previous_benchmarks(tmp_path):
    protocol = phase_b_protocols("cstr")[0]
    trajectory = simulate_phase_b(protocol)
    private, public = tmp_path / "private", tmp_path / "public"
    write_private_bundle(private, (protocol,), (trajectory,))
    spec = phase_b_public_spec("cstr", "easy", "named")
    write_public_staging_bundle(public, spec, (trajectory,))
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="overwrite"):
        write_private_bundle(private, (protocol,), (trajectory,))
    with pytest.raises(FileExistsError, match="overwrite"):
        write_public_staging_bundle(public, spec, (trajectory,))
    assert before == {
        str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }
