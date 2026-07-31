"""Safe, explicit readers for supported tabular and metadata formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from autoformalism.data.exceptions import (
    DataFileNotFoundError,
    DataFormatError,
    MissingColumnError,
)


def require_file(path: Path) -> Path:
    """Return a file path or raise a domain-specific exception."""
    if not path.is_file():
        raise DataFileNotFoundError(f"required data file does not exist: {path}")
    return path


def load_csv(path: Path, required_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read a CSV and validate required columns."""
    require_file(path)
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataFormatError(f"could not read CSV {path}: {exc}") from exc
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise MissingColumnError(f"{path} is missing columns: {missing}")
    return frame


def load_json(path: Path) -> Any:
    """Read one UTF-8 JSON document."""
    require_file(path)
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataFormatError(f"could not read JSON {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[Any]:
    """Read newline-delimited JSON with line-specific diagnostics."""
    require_file(path)
    records: list[Any] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DataFormatError(
                        f"invalid JSONL at {path}:{line_number}: {exc.msg}"
                    ) from exc
    except (OSError, UnicodeError) as exc:
        raise DataFormatError(f"could not read JSONL {path}: {exc}") from exc
    return records

