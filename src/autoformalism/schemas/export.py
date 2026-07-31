"""Deterministic JSON Schema export."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from autoformalism.schemas.candidate import CandidateModel
from autoformalism.schemas.judge import JudgeResult

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "candidate.schema.json": CandidateModel,
    "judge.schema.json": JudgeResult,
}


def export_json_schemas(output_directory: Path) -> tuple[Path, ...]:
    """Write stable JSON Schema documents and return their paths."""
    output_directory.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for filename, model in SCHEMA_MODELS.items():
        path = output_directory / filename
        payload = json.dumps(
            model.model_json_schema(mode="validation"),
            indent=2,
            sort_keys=True,
        )
        path.write_text(f"{payload}\n", encoding="utf-8")
        exported.append(path)
    return tuple(exported)

