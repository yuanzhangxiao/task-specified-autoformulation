"""Deterministic JSON Schema export tests."""

import json
from pathlib import Path

from autoformalism.schemas import (
    AtomicJudgeResult,
    CandidateModel,
    FunctionalCandidate,
    HybridJudgeResult,
    JudgeResult,
    PairedTargetCompletenessJudgeResult,
    ProposedConstructionFocus,
    ProposedConstructionIntent,
    ProposedFunctionalActionTransaction,
    ProposedFunctionalCandidate,
    ProposedTopologyActionTransaction,
    ProposedTopologyCandidate,
    ProposerCandidate,
    ProposerCandidateV2,
    ScientificJudgeResult,
    TopologyCandidate,
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
        "functional-candidate-v1.schema.json",
        "judge.schema.json",
        "hybrid-judge-v1.schema.json",
        "paired-target-completeness-judge-v1.schema.json",
        "proposer-candidate.schema.json",
        "proposer-candidate-v2.schema.json",
        "proposed-construction-intent-v1.schema.json",
        "proposed-construction-focus-v1.schema.json",
        "proposed-functional-candidate-v1.schema.json",
        "proposed-functional-action-transaction-v1.schema.json",
        "proposed-topology-action-transaction-v1.schema.json",
        "proposed-topology-candidate-v2.schema.json",
        "scientific-judge-v2.schema.json",
        "target-completeness-judge-v1.schema.json",
        "topology-candidate-v2.schema.json",
    }
    assert {path.name: path.read_bytes() for path in second_paths} == first_contents

    candidate_schema = json.loads(first_contents["candidate.schema.json"])
    atomic_judge_schema = json.loads(first_contents["atomic-judge-v1.schema.json"])
    judge_schema = json.loads(first_contents["judge.schema.json"])
    proposer_schema = json.loads(first_contents["proposer-candidate.schema.json"])
    proposer_v2_schema = json.loads(first_contents["proposer-candidate-v2.schema.json"])
    proposed_functional_schema = json.loads(
        first_contents["proposed-functional-candidate-v1.schema.json"]
    )
    proposed_intent_schema = json.loads(
        first_contents["proposed-construction-intent-v1.schema.json"]
    )
    proposed_focus_schema = json.loads(
        first_contents["proposed-construction-focus-v1.schema.json"]
    )
    proposed_functional_action_schema = json.loads(
        first_contents["proposed-functional-action-transaction-v1.schema.json"]
    )
    proposed_topology_action_schema = json.loads(
        first_contents["proposed-topology-action-transaction-v1.schema.json"]
    )
    proposed_topology_schema = json.loads(
        first_contents["proposed-topology-candidate-v2.schema.json"]
    )
    scientific_judge_schema = json.loads(
        first_contents["scientific-judge-v2.schema.json"]
    )
    hybrid_judge_schema = json.loads(first_contents["hybrid-judge-v1.schema.json"])
    functional_candidate_schema = json.loads(
        first_contents["functional-candidate-v1.schema.json"]
    )
    paired_target_schema = json.loads(
        first_contents["paired-target-completeness-judge-v1.schema.json"]
    )
    topology_candidate_schema = json.loads(
        first_contents["topology-candidate-v2.schema.json"]
    )
    assert candidate_schema == CandidateModel.model_json_schema(mode="validation")
    assert atomic_judge_schema == AtomicJudgeResult.model_json_schema(mode="validation")
    assert judge_schema == JudgeResult.model_json_schema(mode="validation")
    assert proposer_schema == ProposerCandidate.model_json_schema(mode="validation")
    assert proposer_v2_schema == ProposerCandidateV2.model_json_schema(
        mode="validation"
    )
    assert proposed_functional_schema == (
        ProposedFunctionalCandidate.model_json_schema(mode="validation")
    )
    assert proposed_intent_schema == ProposedConstructionIntent.model_json_schema(
        mode="validation"
    )
    assert proposed_focus_schema == ProposedConstructionFocus.model_json_schema(
        mode="validation"
    )
    assert proposed_functional_action_schema == (
        ProposedFunctionalActionTransaction.model_json_schema(mode="validation")
    )
    assert proposed_topology_action_schema == (
        ProposedTopologyActionTransaction.model_json_schema(mode="validation")
    )
    assert proposed_topology_schema == ProposedTopologyCandidate.model_json_schema(
        mode="validation"
    )
    assert scientific_judge_schema == ScientificJudgeResult.model_json_schema(
        mode="validation"
    )
    assert hybrid_judge_schema == HybridJudgeResult.model_json_schema(mode="validation")
    assert paired_target_schema == (
        PairedTargetCompletenessJudgeResult.model_json_schema(mode="validation")
    )
    assert functional_candidate_schema == FunctionalCandidate.model_json_schema(
        mode="validation"
    )
    assert topology_candidate_schema == TopologyCandidate.model_json_schema(
        mode="validation"
    )
    assert candidate_schema["additionalProperties"] is False
    assert atomic_judge_schema["additionalProperties"] is False
    assert judge_schema["additionalProperties"] is False
    assert proposer_schema["additionalProperties"] is False
    assert proposer_v2_schema["additionalProperties"] is False
    assert proposed_functional_schema["additionalProperties"] is False
    assert proposed_intent_schema["additionalProperties"] is False
    assert proposed_focus_schema["additionalProperties"] is False
    assert proposed_functional_action_schema["additionalProperties"] is False
    assert proposed_topology_action_schema["additionalProperties"] is False
    assert proposed_topology_schema["additionalProperties"] is False
    assert scientific_judge_schema["additionalProperties"] is False
    assert hybrid_judge_schema["additionalProperties"] is False
    assert paired_target_schema["additionalProperties"] is False
    assert functional_candidate_schema["additionalProperties"] is False
    assert topology_candidate_schema["additionalProperties"] is False


def test_checked_in_schemas_match_models() -> None:
    schema_directory = Path(__file__).resolve().parents[1] / "schemas"

    assert json.loads(
        (schema_directory / "atomic-judge-v1.schema.json").read_text(encoding="utf-8")
    ) == AtomicJudgeResult.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "candidate.schema.json").read_text(encoding="utf-8")
    ) == CandidateModel.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "functional-candidate-v1.schema.json").read_text(
            encoding="utf-8"
        )
    ) == FunctionalCandidate.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "judge.schema.json").read_text(encoding="utf-8")
    ) == JudgeResult.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "scientific-judge-v2.schema.json").read_text(
            encoding="utf-8"
        )
    ) == ScientificJudgeResult.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "hybrid-judge-v1.schema.json").read_text(encoding="utf-8")
    ) == HybridJudgeResult.model_json_schema(mode="validation")
    assert json.loads(
        (
            schema_directory / "paired-target-completeness-judge-v1.schema.json"
        ).read_text(encoding="utf-8")
    ) == PairedTargetCompletenessJudgeResult.model_json_schema(mode="validation")
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
    assert json.loads(
        (schema_directory / "proposed-construction-intent-v1.schema.json").read_text(
            encoding="utf-8"
        )
    ) == ProposedConstructionIntent.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "proposed-construction-focus-v1.schema.json").read_text(
            encoding="utf-8"
        )
    ) == ProposedConstructionFocus.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "proposed-functional-candidate-v1.schema.json").read_text(
            encoding="utf-8"
        )
    ) == ProposedFunctionalCandidate.model_json_schema(mode="validation")
    assert json.loads(
        (
            schema_directory / "proposed-functional-action-transaction-v1.schema.json"
        ).read_text(encoding="utf-8")
    ) == ProposedFunctionalActionTransaction.model_json_schema(mode="validation")
    assert json.loads(
        (
            schema_directory / "proposed-topology-action-transaction-v1.schema.json"
        ).read_text(encoding="utf-8")
    ) == ProposedTopologyActionTransaction.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "proposed-topology-candidate-v2.schema.json").read_text(
            encoding="utf-8"
        )
    ) == ProposedTopologyCandidate.model_json_schema(mode="validation")
    assert json.loads(
        (schema_directory / "topology-candidate-v2.schema.json").read_text(
            encoding="utf-8"
        )
    ) == TopologyCandidate.model_json_schema(mode="validation")
