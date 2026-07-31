"""CSV, JSON, and JSONL reader tests."""

import json
from pathlib import Path

import pytest

from autoformalism.data.exceptions import (
    DataFileNotFoundError,
    DataFormatError,
    MissingColumnError,
)
from autoformalism.data.io import load_csv, load_json, load_jsonl


def test_loads_json_and_jsonl(tmp_path: Path) -> None:
    json_path = tmp_path / "one.json"
    jsonl_path = tmp_path / "many.jsonl"
    json_path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    jsonl_path.write_text('{"value": 1}\n\n{"value": 2}\n', encoding="utf-8")

    assert load_json(json_path) == {"value": 1}
    assert load_jsonl(jsonl_path) == [{"value": 1}, {"value": 2}]


def test_jsonl_reports_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(DataFormatError, match=r"bad\.jsonl:2"):
        load_jsonl(path)


def test_csv_reports_missing_file_and_column(tmp_path: Path) -> None:
    with pytest.raises(DataFileNotFoundError):
        load_csv(tmp_path / "missing.csv")

    path = tmp_path / "data.csv"
    path.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(MissingColumnError, match="b"):
        load_csv(path, ("a", "b"))

