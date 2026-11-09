"""Check portability and atomic handoff of rebuttal package files."""

from __future__ import annotations

import hashlib
from zipfile import ZipFile

import pytest

from scripts.package_iclr_dynamics_evidence import check_csv_fields, pack


def test_metadata_hash_cannot_hide_a_mismatched_csv_score():
    check_csv_fields(
        {"response_rse": "0.0", "peak": ""}, {"response_rse": 0.0, "peak": None}
    )
    with pytest.raises(ValueError, match="disagree on response_rse"):
        check_csv_fields({"response_rse": "0.1"}, {"response_rse": 0.0})


def test_rebuilding_package_preserves_checksums_and_bytes(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"measurement\n")
    output = tmp_path / "dynamics.zip"
    files = {"meal-appearance/scores.csv": source}
    pack(files, output)
    original = output.read_bytes()
    pack(files, output)
    assert output.read_bytes() == original
    with ZipFile(output) as z:
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        assert (
            z.read("SHA256SUMS").decode() == f"{expected}  meal-appearance/scores.csv\n"
        )
        assert z.read("meal-appearance/scores.csv") == b"measurement\n"


def test_missing_input_preserves_previous_package(tmp_path):
    out = tmp_path / "evidence.zip"
    out.write_bytes(b"prior")
    with pytest.raises(FileNotFoundError):
        pack({"input": tmp_path / "missing"}, out)
    assert out.read_bytes() == b"prior"
