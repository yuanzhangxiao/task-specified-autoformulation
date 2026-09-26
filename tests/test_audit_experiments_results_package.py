"""Independent aggregation must preserve missing cases and endpoint semantics."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "package_audit",
    Path(__file__).resolve().parent.parent
    / "scripts/audit_experiments_results_package.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def rows(roster: tuple[str, ...]) -> list[dict]:
    return [
        {
            "method": method,
            "benchmark_id": case,
            "repetition": repetition,
            "request_id": f"{method}/{case}/{repetition}",
            "target_nmse": float(index + 1),
            "target_status": "available",
            "mechanism_compliance": 1.0,
            "mechanism_coverage": 0.0,
            "mechanism_status": "available",
            "runtime_valid": True,
            "complexity_terms": 2,
            "equation_count": 1,
            "equations": {"x": "untrusted_text_is_never_executed"},
        }
        for method in audit.METHODS
        for index, case in enumerate(roster)
        for repetition in range(3)
    ]


def test_case_medians_precede_macro_and_missing_cases_are_not_dropped():
    roster = ("a", "b", "c")
    records = rows(roster)
    for row in records:
        if row["method"] == "sindy" and row["benchmark_id"] == "b":
            row["target_nmse"] = None
        if row["method"] == "pysr" and row["benchmark_id"] == "a":
            row["target_nmse"] = [0.0, 100.0, 1000.0][row["repetition"]]
    result = audit.summarize(records, roster)
    assert result[0]["macros"]["target_nmse"]["median"] is None
    assert result[0]["macros"]["target_nmse"]["missing_cases"] == ["b"]
    # Case medians are 100, 2, 3; pooling the repetitions would give 2.
    assert result[1]["macros"]["target_nmse"]["median"] == 3.0
    assert result[1]["macros"]["target_nmse"]["mad"] == 1.0


def test_graph_availability_does_not_erase_a_recorded_prediction():
    records = rows(("a",))
    row = next(r for r in records if r["method"] == "d3_native_no_tools")
    row.update(
        runtime_valid=False,
        mechanism_compliance=None,
        mechanism_status="invalid_runtime",
    )
    result = audit.summarize(records, ("a",))[2]["macros"]
    assert result["target_nmse"]["record_coverage"] == 3
    assert result["mechanism_compliance"]["record_coverage"] == 2
    assert result["mechanism_compliance"]["median"] == 1.0
    # Annotation coverage is zero in all rows and must not replace graph scores.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, -1])
def test_nonfinite_boolean_or_negative_scores_are_rejected(bad):
    records = rows(("a",))
    records[0]["target_nmse"] = bad
    with pytest.raises(ValueError, match="invalid numeric"):
        audit.summarize(records, ("a",))


def test_duplicate_or_missing_planned_repetition_is_rejected():
    records = rows(("a",))
    with pytest.raises(ValueError, match="planned three-repetition roster"):
        audit.summarize(records[:-1], ("a",))


def archive(tmp_path: Path, *, inconsistent: bool = False) -> Path:
    records = rows(("a",))
    flat = [
        {**r, "equations": json.dumps(r["equations"], sort_keys=True)} for r in records
    ]
    if inconsistent:
        flat[0]["target_nmse"] = 999
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
    contents = {
        "models.jsonl": "\n".join(json.dumps(r) for r in records),
        "models_and_metrics.csv": stream.getvalue(),
        "manifest.json": json.dumps(
            {
                "planned_total": len(records),
                "with_model": len(records),
                "with_score": len(records),
                "evaluations": [{"files": {}}],
            }
        ),
    }
    path = tmp_path / "package.tar.gz"
    with tarfile.open(path, "w:gz") as bundle:
        for name, content in contents.items():
            data = content.encode()
            info = tarfile.TarInfo(f"package/{name}")
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return path


def test_csv_and_jsonl_agree_and_payload_hashes_are_computed(tmp_path):
    records, _, integrity = audit.load_package(archive(tmp_path))
    assert len(records) == 12
    assert integrity["csv_jsonl_mismatches"] == 0
    assert len(integrity["computed_payload_sha256"]["models.jsonl"]) == 64


def test_conflicting_csv_is_not_silently_accepted(tmp_path):
    with pytest.raises(ValueError, match="CSV/JSONL field differs"):
        audit.load_package(archive(tmp_path, inconsistent=True))
