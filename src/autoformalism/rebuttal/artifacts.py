"""Index completed candidate checkpoints without opening test data."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from autoformalism.schemas import CandidateModel


class CandidateArtifact(BaseModel):
    """One completed candidate and its development-only selection evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    source_checkpoint: str
    run_directory: str
    benchmark_id: str
    tier: str
    seed: int
    round_index: int
    structural_hash: str
    candidate: CandidateModel
    validation_mse: float
    training_mse: float
    judge_score: float | None
    judge_category_scores: dict[str, float]
    state_count: int
    latent_state_count: int
    process_count: int
    parameter_count: int
    term_count: int
    use_judge: bool


def index_artifacts(roots: tuple[Path, ...]) -> tuple[CandidateArtifact, ...]:
    """Return deterministic unique records for valid completed round checkpoints."""
    records: list[CandidateArtifact] = []
    seen_sources: set[Path] = set()
    for root in roots:
        for path in sorted(root.expanduser().resolve().rglob("round_*.json")):
            if path in seen_sources:
                continue
            seen_sources.add(path)
            payload = _read_json(path)
            if payload.get("stage") != "complete" or not payload.get("valid"):
                continue
            candidate_payload = payload.get("pruned_candidate")
            fit = payload.get("pruned_fit")
            judge = payload.get("postpruning_judge")
            record = payload.get("record")
            if isinstance(record, dict):
                candidate_payload = candidate_payload or record.get(
                    "pruned_candidate"
                )
                fit = fit or record.get("pruned_fit")
                judge = judge or record.get("postpruning_judge")
            if not all(isinstance(item, dict) for item in (candidate_payload, fit)):
                continue
            candidate = CandidateModel.model_validate(candidate_payload)
            run_directory = path.parent.parent
            config = _read_optional_json(run_directory / "run_config.json")
            benchmark_id, tier, seed = _run_identity(run_directory, config)
            validation = float(fit["validation_metrics"]["normalized_mse"])
            training = float(fit["training_metrics"]["normalized_mse"])
            use_judge = bool(config.get("use_judge", judge is not None))
            judge_score = (
                float(judge.get("aggregate_score", 0.0))
                if use_judge and isinstance(judge, dict)
                else None
            )
            categories = (
                _numeric_categories(judge.get("category_scores", {}))
                if isinstance(judge, dict) and use_judge
                else {}
            )
            structural_hash = str(
                (record or {}).get("structural_hash")
                or _stable_hash(candidate.model_dump(mode="json"))
            )
            round_index = _round_index(path)
            artifact_id = _stable_hash(
                {
                    "source": str(path),
                    "fingerprint": payload.get("fingerprint"),
                    "structure": structural_hash,
                }
            )
            records.append(
                CandidateArtifact(
                    artifact_id=artifact_id,
                    source_checkpoint=str(path),
                    run_directory=str(run_directory),
                    benchmark_id=benchmark_id,
                    tier=tier,
                    seed=seed,
                    round_index=round_index,
                    structural_hash=structural_hash,
                    candidate=candidate,
                    validation_mse=validation,
                    training_mse=training,
                    judge_score=judge_score,
                    judge_category_scores=categories,
                    state_count=len(candidate.states),
                    latent_state_count=sum(
                        state.kind.value == "latent" for state in candidate.states
                    ),
                    process_count=len(candidate.processes),
                    parameter_count=len(candidate.parameters),
                    term_count=sum(
                        _term_count(equation.rhs)
                        for equation in candidate.state_equations
                    ),
                    use_judge=use_judge,
                )
            )
    return tuple(sorted(records, key=lambda item: item.artifact_id))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_optional_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _run_identity(
    run_directory: Path, config: dict[str, Any]
) -> tuple[str, str, int]:
    if {"benchmark_id", "tier", "seed"} <= config.keys():
        return (
            str(config["benchmark_id"]),
            str(config["tier"]),
            int(config["seed"]),
        )
    match = re.match(r"(.+)_(easy|medium|hard)_seed(\d+)$", run_directory.name)
    if match is None:
        raise ValueError(f"cannot infer run identity: {run_directory}")
    return match.group(1), match.group(2), int(match.group(3))


def _round_index(path: Path) -> int:
    match = re.fullmatch(r"round_(\d+)\.json", path.name)
    if match is None:
        raise ValueError(f"invalid round checkpoint name: {path.name}")
    return int(match.group(1))


def _numeric_categories(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, float] = {}
    for name, raw in payload.items():
        value = raw.get("score") if isinstance(raw, dict) else raw
        result[str(name)] = float(value)
    return result


def _term_count(source: str) -> int:
    node = ast.parse(source, mode="eval").body
    return _count_additive(node)


def _count_additive(node: ast.AST) -> int:
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _count_additive(node.left) + _count_additive(node.right)
    return 1


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
