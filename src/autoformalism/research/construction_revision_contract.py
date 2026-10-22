"""Bind complete training consistency evidence to authenticated construction inputs.

The existing training projection authenticates the source archive. This producer
reads selected sealed members in memory, cross-checks every frozen parent and
exports only the shared typed training contract. It never extracts archives,
opens held-out members, fits parameters or integrates a candidate.
"""

from __future__ import annotations

from pathlib import Path

from autoformalism.research.construction_revision_evidence import (
    _digest,
    _shape,
    validate_training_projection,
)
from autoformalism.research.interaction_source import _Reader, _sealed
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    validate_evidence,
)

PROTOCOL = "construction-revision-consistency-evidence-1"


def _members(projection: dict) -> set[str]:
    """Name the exact public source members needed for the sidecar provenance."""
    return {"plan.json"} | {
        path
        for name in projection["parents"]
        for path in (f"inputs/{name}.json", f"results/{name}/fit_handoff.json")
    }


def _evidence(value: dict, projection: dict) -> dict:
    """Validate the complete contract and exact ordered public training rows."""
    evidence = TrainingCompatibilityEvidence.model_validate(value)
    validate_evidence(evidence)
    canonical = evidence.model_dump(mode="json")
    if canonical != value:
        raise ValueError("canonical complete training evidence required")
    training = PublicSplit.model_validate(projection["training"])
    expected = [
        {
            "trajectory_id": row.trajectory_id,
            "time": list(row.time),
            "targets": {key: list(values) for key, values in row.targets.items()},
            "auxiliaries": {
                key: list(values) for key, values in row.auxiliaries.items()
            },
            "inputs": {
                key: list(values) for key, values in row.external_inputs.items()
            },
            "covariates": dict(row.fixed_covariates),
        }
        for row in training.rows
    ]
    if canonical["trajectories"] != expected:
        raise ValueError("consistency evidence differs from complete training rows")
    for parent in projection["parents"].values():
        problem = PrefixFitProblem.model_validate(parent["problem"])
        if evidence.context != problem.context:
            raise ValueError("consistency context differs from a frozen parent")
    return canonical


def validate(value: dict, projection: dict) -> dict:
    """Validate a portable sidecar and return typed train-only evidence as JSON.

    Raw producer output and verified ``sealed_read`` output are accepted. The
    archive is not reopened here: its member hashes are provenance frozen by
    ``build`` and the caller's sealed plan, not a new archive authentication.
    """
    validate_training_projection(projection)
    if isinstance(value, dict) and "artifact_sha256" in value:
        _sealed(value)
        value = {key: child for key, child in value.items() if key != "artifact_sha256"}
    _shape(
        value,
        "protocol training_sha256 construction_archive_sha256 evidence source_members",
        "training consistency sidecar",
    )
    if value["protocol"] != PROTOCOL:
        raise ValueError("unsupported training consistency protocol")
    for key in ("training_sha256", "construction_archive_sha256"):
        _digest(value[key], key)
        if value[key] != projection[key]:
            raise ValueError(f"training consistency {key} differs from projection")
    members = value["source_members"]
    if not isinstance(members, dict) or set(members) != _members(projection):
        raise ValueError("training consistency source member roster differs")
    for digest in members.values():
        _digest(digest, "source member")
    return _evidence(value["evidence"], projection)


def build(projection: dict, archive: Path) -> dict:
    """Read authenticated archive members and return a sidecar ready to seal."""
    validate_training_projection(projection)
    _digest(projection["construction_archive_sha256"], "construction archive")
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("regular construction archive required")
    reader = _Reader(archive)
    if reader.blob_sha256 != projection["construction_archive_sha256"]:
        raise ValueError("construction archive identity differs from projection")
    plan = reader.json("plan.json", "artifact_sha256")
    if plan.get("protocol") != "saved-function-handoff-repair-1":
        raise ValueError("unsupported construction source protocol")
    entries = plan.get("cases")
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) or not isinstance(entry.get("case_id"), str)
        for entry in entries
    ):
        raise ValueError("invalid construction source roster")
    names = [entry["case_id"] for entry in entries]
    if len(names) != len(set(names)) or set(names) != set(projection["parents"]):
        raise ValueError("all eight construction parents are required")
    common = None
    for entry in entries:
        name = entry["case_id"]
        parent = projection["parents"][name]
        saved = reader.json(f"inputs/{name}.json", "artifact_sha256")
        handoff = reader.json(f"results/{name}/fit_handoff.json", "artifact_sha256")
        if (
            saved["artifact_sha256"] != entry.get("input_sha256")
            or entry.get("source_model_id") != parent["source_model_id"]
            or entry.get("seed") != parent["proposal_seed"]
            or handoff["artifact_sha256"] != parent["source_packet_sha256"]
            or handoff.get("prefix_fit_problem") != parent["problem"]
        ):
            raise ValueError("construction member differs from frozen parent identity")
        source = saved["source"]
        case = _sealed(source["case"])
        original = _sealed(source["source_input"]["input"])
        evidence = _evidence(case["training_evidence"], projection)
        if (
            case != original
            or case["context"] != evidence["context"]
            or source["public_cell"]["context"] != evidence["context"]
            or source["entry"].get("case_id") != name
            or source["entry"].get("source_model_id") != parent["source_model_id"]
            or source["entry"].get("seed") != parent["proposal_seed"]
        ):
            raise ValueError("construction training source or context differs")
        if common is not None and evidence != common:
            raise ValueError("construction parents disagree on the training contract")
        common = evidence
    result = {
        "protocol": PROTOCOL,
        "training_sha256": projection["training_sha256"],
        "construction_archive_sha256": reader.blob_sha256,
        "evidence": common,
        "source_members": dict(sorted(reader.read_hashes.items())),
    }
    validate(result, projection)
    return result
