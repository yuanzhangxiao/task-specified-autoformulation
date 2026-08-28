"""Deterministic JSON Schema export tests."""

import json
from pathlib import Path

from autoformalism.schemas import (
    AtomicJudgeResult,
    CandidateModel,
    HybridJudgeResult,
    JudgeResult,
    ProposerCandidate,
    ProposerCandidateV2,
    ScientificJudgeResult,
    export_json_schemas,
)


def test_exports_deterministic_valid_json_schemas(tmp_path: Path) -> None:
    first_paths = export_json_schemas(tmp_path)
    first_contents = {path.name: path.read_bytes() for path in first_paths}
    second_paths = export_json_schemas(tmp_path)

    assert first_paths == second_paths
    assert {path.name for path in first_paths} == {
        "atomic-judge-v1.schema.json",
        "candidate.schema.json",
        "comparative-judge-v1.schema.json",
        "judge.schema.json",
        "hybrid-judge-v1.schema.json",
        "proposer-candidate.schema.json",
        "proposer-candidate-v2.schema.json",
        "scientific-judge-v2.schema.json",
        "target-completeness-judge-v1.schema.json",
    }
    assert {path.name: path.read_bytes() for path in second_paths} == first_contents

    candidate_schema = json.loads(first_contents["candidate.schema.json"])
    atomic_judge_schema = json.loads(
        first_contents["atomic-judge-v1.schema.json"]
    )
    judge_schema = json.loads(first_contents["judge.schema.json"])
    proposer_schema = json.loads(first_contents["proposer-candidate.schema.json"])
    proposer_v2_schema = json.loads(
        first_contents["proposer-candidate-v2.schema.json"]
    )
    scientific_judge_schema = json.loads(
        first_contents["scientific-judge-v2.schema.json"]
    )
    hybrid_judge_schema = json.loads(
        first_contents["hybrid-judge-v1.schema.json"]
    )
    assert candidate_schema == CandidateModel.model_json_schema(mode="validation")
    assert atomic_judge_schema == AtomicJudgeResult.model_json_schema(
        mode="validation"
    )
    assert judge_schema == JudgeResult.model_json_schema(mode="validation")
    assert proposer_schema == ProposerCandidate.model_json_schema(mode="validation")
    assert proposer_v2_schema == ProposerCandidateV2.model_json_schema(
        mode="validation"
    )
    assert scientific_judge_schema == ScientificJudgeResult.model_json_schema(
        mode="validation"
    )
    assert hybrid_judge_schema == HybridJudgeResult.model_json_schema(
        mode="validation"
    )
    assert candidate_schema["additionalProperties"] is False
    assert atomic_judge_schema["additionalProperties"] is False
    assert judge_schema["additionalProperties"] is False
    assert proposer_schema["additionalProperties"] is False
    assert proposer_v2_schema["additionalProperties"] is False
    assert scientific_judge_schema["additionalProperties"] is False
    assert hybrid_judge_schema["additionalProperties"] is False


def test_checked_in_schemas_match_models() -> None:
    schema_directory = Path(__file__).resolve().parents[1] / "schemas"

    assert json.loads(
        (schema_directory / "atomic-judge-v1.schema.json").read_text(
            encoding="utf-8"
        )
    ) == AtomicJudgeResult.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "candidate.schema.json").read_text(encoding="utf-8")
    ) == CandidateModel.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "judge.schema.json").read_text(encoding="utf-8")
    ) == JudgeResult.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "scientific-judge-v2.schema.json").read_text(
            encoding="utf-8"
        )
    ) == ScientificJudgeResult.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "hybrid-judge-v1.schema.json").read_text(
            encoding="utf-8"
        )
    ) == HybridJudgeResult.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "proposer-candidate.schema.json").read_text(
            encoding="utf-8"
        )
    ) == ProposerCandidate.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "proposer-candidate-v2.schema.json").read_text(
            encoding="utf-8"
        )
    ) == ProposerCandidateV2.model_json_schema(mode="validation")
