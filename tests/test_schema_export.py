"""Deterministic JSON Schema export tests."""

import json
from pathlib import Path

from autoformalism.schemas import CandidateModel, JudgeResult, export_json_schemas


def test_exports_deterministic_valid_json_schemas(tmp_path: Path) -> None:
    first_paths = export_json_schemas(tmp_path)
    first_contents = {path.name: path.read_bytes() for path in first_paths}
    second_paths = export_json_schemas(tmp_path)

    assert first_paths == second_paths
    assert {path.name for path in first_paths} == {
        "candidate.schema.json",
        "judge.schema.json",
    }
    assert {path.name: path.read_bytes() for path in second_paths} == first_contents

    candidate_schema = json.loads(first_contents["candidate.schema.json"])
    judge_schema = json.loads(first_contents["judge.schema.json"])
    assert candidate_schema == CandidateModel.model_json_schema(mode="validation")
    assert judge_schema == JudgeResult.model_json_schema(mode="validation")
    assert candidate_schema["additionalProperties"] is False
    assert judge_schema["additionalProperties"] is False


def test_checked_in_schemas_match_models() -> None:
    schema_directory = Path(__file__).resolve().parents[1] / "schemas"

    assert json.loads(
        (schema_directory / "candidate.schema.json").read_text(encoding="utf-8")
    ) == CandidateModel.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "judge.schema.json").read_text(encoding="utf-8")
    ) == JudgeResult.model_json_schema(mode="validation")
