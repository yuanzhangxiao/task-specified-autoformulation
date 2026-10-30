"""Authenticate the frozen, training-only 23-model construction delivery.

No supplied script is executed and no source path embedded in model metadata is
followed. Numerical qualification belongs to the campaign, not this importer.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field

from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-portfolio-inputs-1"
DELIVERY_MANIFEST_SHA256 = (
    "a7f6b64c9cec04626bbaa1f7c31b2f64f8f206248934c271c569e8293dc15a3e"
)
DELIVERY_COMMIT = "3eb449bc2554a5cd1b81853dde774d1b46805612"
TRAINING_SHA256 = "169e2b72a1ba4145ac4ec5911875e9d7e064a525f7a1ace92cc2b151efe70795"
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CandidateId = Annotated[str, Field(pattern=r"^candidate-[0-9]{2}$")]
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024


class Manifest(StrictSchema):
    """The delivery inventory binds every consumed source byte."""

    protocol: Literal["scaled-readout-orion-delivery-1"]
    source_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    unique_models: int = Field(ge=1, le=99)
    static_interface_passed: int = Field(ge=1, le=99)
    numerical_qualification: Literal["pending"]
    optimizer_calls: Literal[0]
    llm_calls: Literal[0]
    solver_rollouts: Literal[0]
    test_data_opened: Literal[False]
    files: dict[str, Digest] = Field(min_length=1, max_length=1000)


class IntakeRow(StrictSchema):
    """One complete, eligible candidate in the authoritative ordered roster."""

    candidate_id: CandidateId
    family: Digest
    path: str
    sha256: Digest
    problem_sha256: Digest
    records: list[str] = Field(min_length=1, max_length=100)
    static_interface: Literal["passed"]
    numerical_qualification: Literal["pending"]


class Packet(StrictSchema):
    """Only the declared model and initialization boundary may enter fitting."""

    protocol: Literal["scaled-readout-fitting-intake-1"]
    candidate_id: CandidateId
    family: Digest
    problem: PrefixFitProblem
    audit: dict
    records: list[dict] = Field(min_length=1, max_length=100)
    static_interface: dict
    fixed_fitting_prefix_policy: PrefixPolicy
    training: Literal["../evidence/training.json"]
    compatibility_evidence: Literal["../evidence/compatibility.json"]
    public_context: Literal["../evidence/public-context.json"]
    fit_started: Literal[False]
    scientific_validity: Literal["not_established"]
    test_data_opened: Literal[False]


class Source(StrictSchema):
    """Verified byte identities, with the two digest conventions kept distinct."""

    manifest_sha256: Digest
    delivery_source_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    roster_sha256: Digest
    training_sha256: Digest
    source_hash_convention: Literal["sorted_json_default_separators"]
    fitting_hash_convention: Literal["sorted_json_compact_separators"]


class ModelRecord(StrictSchema):
    """A strict numerical projection without arbitrary upstream source paths."""

    candidate_id: CandidateId
    family: Digest
    problem: PrefixFitProblem
    source_problem_sha256: Digest
    fitting_problem_sha256: Digest
    source_packet_sha256: Digest
    records: list[str] = Field(min_length=1, max_length=100)


class Inputs(StrictSchema):
    """Sealed campaign inputs can be rechecked without rereading the delivery."""

    protocol: Literal["construction-portfolio-inputs-1"]
    source: Source
    training: PublicSplit
    prefix: PrefixPolicy
    models: list[ModelRecord] = Field(min_length=1, max_length=99)
    validation_exported: Literal[False]
    test_data_opened: Literal[False]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pairs(items: list[tuple[str, object]]) -> dict:
    """Reject duplicate JSON keys rather than silently replacing provenance."""
    result = {}
    for name, value in items:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _json(payload: bytes) -> dict:
    result = json.loads(payload, object_pairs_hook=_pairs)
    if not isinstance(result, dict):
        raise ValueError("delivery JSON must contain an object")
    return result


def _read(root: Path, name: str) -> bytes:
    """Read only a bounded regular file below the physical delivery directory."""
    relative = PurePosixPath(name)
    if (
        not name
        or relative.is_absolute()
        or str(relative) != name
        or any(part in {".", ".."} for part in relative.parts)
        or "\\" in name
    ):
        raise ValueError("unsafe delivery path")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("delivery paths must not contain symlinks")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_FILE_BYTES:
        raise ValueError("delivery entry is not a bounded regular file")
    payload = path.read_bytes()
    if len(payload) > MAX_FILE_BYTES:
        raise ValueError("delivery entry exceeds byte limit")
    return payload


def _allowed_manifest_path(name: str) -> bool:
    """The intake format contains public training evidence, never test payloads."""
    fixed = {
        "FEEDBACK.html",
        "FITTING_HANDOFF.md",
        "PROOF.html",
        "REPORT.html",
        "SOURCE_COMMIT",
        "VALIDATION.json",
        "build_report.py",
        "pytest-isolated-rerun.txt",
        "replay-summary.json",
        "replay_and_package.py",
        "report-verification.json",
        "scaled-witness.png",
        "seal_delivery.py",
        "evidence/compatibility.json",
        "evidence/public-context.json",
        "evidence/training.json",
    }
    return name in fixed or bool(
        re.fullmatch(r"candidate-[0-9]{2}\.html", name)
        or re.fullmatch(r"models/candidate-[0-9]{2}\.json", name)
        or re.fullmatch(r"(?:assessments|feedback)/[A-Za-z0-9_-]+\.json", name)
    )


def _verified_files(root: Path, manifest: Manifest) -> dict[str, bytes]:
    """Authenticate the inventory before parsing any model or training arrays."""
    if not all(_allowed_manifest_path(name) for name in manifest.files):
        raise ValueError("unexpected or unsafe delivery manifest path")
    files = {}
    total = 0
    for name, expected in manifest.files.items():
        payload = _read(root, name)
        total += len(payload)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("delivery exceeds total byte limit")
        if _sha(payload) != expected:
            raise ValueError(f"delivery digest differs: {name}")
        files[name] = payload
    required = {
        "replay-summary.json",
        "evidence/training.json",
        "evidence/public-context.json",
        "evidence/compatibility.json",
        "SOURCE_COMMIT",
    }
    if not required <= files.keys():
        raise ValueError("delivery manifest is missing required evidence")
    if files["SOURCE_COMMIT"].decode().strip() != manifest.source_commit:
        raise ValueError("delivery source commit differs")
    return files


def _check_rows(problem: PrefixFitProblem, training: PublicSplit, policy: PrefixPolicy):
    """Validate the same input roles and nonempty prefix/suffix for every model."""
    if set(policy.future_inputs) - set(problem.context.external_inputs) or set(
        policy.future_auxiliaries
    ) - set(problem.context.auxiliaries):
        raise ValueError("future forcing policy differs from declared input roles")
    for row in training.rows:
        if set(row.targets) != set(problem.context.targets):
            raise ValueError("training targets differ from model targets")
        if (
            set(policy.future_inputs) - row.external_inputs.keys()
            or set(policy.future_auxiliaries) - row.auxiliaries.keys()
        ):
            raise ValueError("training is missing authorized forcing schedules")
        cutoff = row.time[0] + policy.duration
        if not any(t > cutoff for t in row.time):
            raise ValueError("training has no post-prefix forecast samples")


def _model(row: IntakeRow, files: dict[str, bytes], training, prefix) -> dict:
    """Bind each packet to its exact roster identity and preserve original domains."""
    expected_path = f"models/{row.candidate_id}.json"
    if row.path != expected_path or expected_path not in files:
        raise ValueError("candidate path differs from the complete roster")
    payload = files[expected_path]
    raw = _json(payload)
    packet = Packet.model_validate(raw)
    if (
        row.sha256 != _sha(payload)
        or row.candidate_id != packet.candidate_id
        or row.family != packet.family
        or row.problem_sha256 != content_hash(raw["problem"])
    ):
        raise ValueError("candidate identity or source problem digest differs")
    if packet.fixed_fitting_prefix_policy != prefix:
        raise ValueError("candidate prefix policy differs from public contract")
    if [record.get("record") for record in packet.records] != row.records or any(
        record.get("current_status") != "eligible" or record.get("family") != row.family
        for record in packet.records
    ):
        raise ValueError("candidate source records differ from eligible roster")
    if len(set(row.records)) != len(row.records):
        raise ValueError("duplicate candidate source record")
    _check_rows(packet.problem, training, prefix)
    return {
        "candidate_id": row.candidate_id,
        "family": row.family,
        "problem": raw["problem"],
        "source_problem_sha256": row.problem_sha256,
        "fitting_problem_sha256": content_sha256(packet.problem),
        "source_packet_sha256": row.sha256,
        "records": list(row.records),
    }


def load_delivery(source: Path, *, expected_count: int = 23) -> dict:
    """Import all frozen candidates and training only, without numerical fitting.

    The default population pins the reviewed manifest, training and prefix. A
    smaller explicit count supports synthetic qualification fixtures; it never
    silently selects a subset from the real delivery.
    """
    if type(expected_count) is not int or not 1 <= expected_count <= 99:
        raise ValueError("expected_count must be between one and 99")
    if source.is_symlink() or not source.is_dir():
        raise ValueError("delivery must be a physical directory")
    root = source.resolve()
    manifest_bytes = _read(root, "manifest.json")
    manifest = Manifest.model_validate(_json(manifest_bytes))
    if expected_count == 23 and _sha(manifest_bytes) != DELIVERY_MANIFEST_SHA256:
        raise ValueError("manifest differs from reviewed 23-model delivery")
    if manifest.unique_models != expected_count or (
        manifest.static_interface_passed != expected_count
    ):
        raise ValueError("delivery must retain the complete expected roster")
    files = _verified_files(root, manifest)
    summary = _json(files["replay-summary.json"])
    rows = [IntakeRow.model_validate(row) for row in summary.get("intake", [])]
    expected_ids = [f"candidate-{i:02d}" for i in range(expected_count)]
    model_paths = {name for name in files if name.startswith("models/")}
    if [row.candidate_id for row in rows] != expected_ids or model_paths != {
        row.path for row in rows
    }:
        raise ValueError("complete candidate roster differs or contains duplicates")
    if {path.name for path in (root / "models").iterdir()} != {
        f"{name}.json" for name in expected_ids
    }:
        raise ValueError("unlisted candidate entry in model directory")
    if len({row.family for row in rows}) != expected_count:
        raise ValueError("duplicate model family in deduplicated roster")
    if len({row.problem_sha256 for row in rows}) != expected_count:
        raise ValueError("duplicate complete problem in deduplicated roster")
    if (
        any(
            summary.get(key) != expected_count
            for key in ("unique_models", "static_interface_passed")
        )
        or summary.get("test_data_opened") is not False
    ):
        raise ValueError("summary population or public data boundary differs")
    training = PublicSplit.model_validate(_json(files["evidence/training.json"]))
    context = _json(files["evidence/public-context.json"])
    prefix = PrefixPolicy.model_validate(context["fixed_fitting_prefix_policy"])
    if training.name != "train" or (
        content_sha256(training) != context.get("training_sha256")
    ):
        raise ValueError("training split or content identity differs")
    if expected_count == 23 and (
        manifest.source_commit != DELIVERY_COMMIT
        or content_sha256(training) != TRAINING_SHA256
        or prefix.duration != 8
        or len(training.rows) != 16
    ):
        raise ValueError("reviewed training/prefix contract differs")
    models = [_model(row, files, training, prefix) for row in rows]
    records = [record for model in models for record in model["records"]]
    if len(records) != len(set(records)):
        raise ValueError("source record occurs in multiple model packets")
    result = {
        "protocol": PROTOCOL,
        "source": {
            "manifest_sha256": _sha(manifest_bytes),
            "delivery_source_commit": manifest.source_commit,
            "roster_sha256": _sha(files["replay-summary.json"]),
            "training_sha256": content_sha256(training),
            "source_hash_convention": "sorted_json_default_separators",
            "fitting_hash_convention": "sorted_json_compact_separators",
        },
        "training": training.model_dump(mode="json"),
        "prefix": prefix.model_dump(mode="json"),
        "models": models,
        "validation_exported": False,
        "test_data_opened": False,
    }
    validate_inputs(result, expected_count=expected_count)
    return result


def validate_inputs(data: dict, *, expected_count: int = 23) -> None:
    """Recheck sealed numerical boundaries without source files or numerical work."""
    if type(expected_count) is not int or not 1 <= expected_count <= 99:
        raise ValueError("expected_count must be between one and 99")
    value = Inputs.model_validate(data)
    expected_ids = [f"candidate-{i:02d}" for i in range(expected_count)]
    if [row.candidate_id for row in value.models] != expected_ids:
        raise ValueError("sealed inputs must retain the complete ordered roster")
    if (
        len({row.family for row in value.models}) != expected_count
        or len({row.source_problem_sha256 for row in value.models}) != expected_count
    ):
        raise ValueError("duplicate model identity in sealed inputs")
    records = [record for model in value.models for record in model.records]
    if len(records) != len(set(records)):
        raise ValueError("duplicate source record in sealed inputs")
    if value.training.name != "train" or content_sha256(value.training) != (
        value.source.training_sha256
    ):
        raise ValueError("sealed training split or digest differs")
    if expected_count == 23 and (
        value.source.manifest_sha256 != DELIVERY_MANIFEST_SHA256
        or value.source.delivery_source_commit != DELIVERY_COMMIT
        or value.source.training_sha256 != TRAINING_SHA256
        or value.prefix
        != PrefixPolicy(
            duration=8,
            future_inputs=("meal_rate_g_per_min", "insulin_pmol_per_kg_min"),
            future_auxiliaries=("EGP", "Uii", "E", "Gt"),
        )
        or len(value.training.rows) != 16
    ):
        raise ValueError("sealed inputs differ from reviewed delivery contract")
    for raw, model in zip(data["models"], value.models, strict=True):
        if content_hash(raw["problem"]) != model.source_problem_sha256 or (
            content_sha256(model.problem) != model.fitting_problem_sha256
        ):
            raise ValueError("sealed model source/fitting digest differs")
        _check_rows(model.problem, value.training, value.prefix)
