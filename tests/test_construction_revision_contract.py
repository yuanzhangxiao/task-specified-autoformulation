"""Only identical authenticated full training contracts enter revision checks."""

import hashlib
import io
import json
import tarfile
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_revision_contract as contract
from autoformalism.research.construction_revision_evidence import load_training_evidence
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from autoformalism.staged_topology import content_hash
from tests.test_construction_revision_evidence import originals

__all__ = ["originals"]


def _seal(value):
    raw = {key: child for key, child in value.items() if key != "artifact_sha256"}
    return {**raw, "artifact_sha256": content_hash(raw)}


def _resign_evidence(value):
    value["sha256"] = content_hash(
        {key: child for key, child in value.items() if key != "sha256"}
    )


def _archive(path, members, extra=None):
    with tarfile.open(path, "w:gz") as stream:
        for name, value in members.items():
            blob = json.dumps(value).encode() if isinstance(value, dict) else value
            member = tarfile.TarInfo(name)
            member.size = len(blob)
            stream.addfile(member, io.BytesIO(blob))
        if extra is not None:
            stream.addfile(extra)


@pytest.fixture
def prepared(tmp_path, originals):
    projection = load_training_evidence(*originals[:2])
    context = next(iter(projection["parents"].values()))["problem"]["context"]
    trajectories = [
        {
            "trajectory_id": row["trajectory_id"],
            "time": row["time"],
            "targets": row["targets"],
            "auxiliaries": row["auxiliaries"],
            "inputs": row["external_inputs"],
            "covariates": row["fixed_covariates"],
        }
        for row in projection["training"]["rows"]
    ]
    evidence = TrainingCompatibilityEvidence.model_validate(
        {
            "context": context,
            "contract": {
                "provenance": "Synthetic public runtime contract",
                "absolute_error_bounds": {"y": 0},
            },
            "trajectories": trajectories,
            "sha256": "0" * 64,
        }
    ).model_dump(mode="json")
    _resign_evidence(evidence)
    members, entries = {}, []
    for name, parent in projection["parents"].items():
        case = _seal({"context": context, "training_evidence": evidence})
        entry = {
            "case_id": name,
            "source_model_id": parent["source_model_id"],
            "seed": parent["proposal_seed"],
        }
        saved = _seal(
            {
                "source": {
                    "case": deepcopy(case),
                    "source_input": {"input": deepcopy(case)},
                    "entry": entry,
                    "public_cell": {"context": context},
                    "ignored_metadata": "FORBIDDEN unprojected source prose",
                }
            }
        )
        handoff = _seal({"prefix_fit_problem": parent["problem"]})
        parent["source_packet_sha256"] = handoff["artifact_sha256"]
        members[f"inputs/{name}.json"] = saved
        members[f"results/{name}/fit_handoff.json"] = handoff
        entries.append({**entry, "input_sha256": saved["artifact_sha256"]})
    members["plan.json"] = _seal(
        {"protocol": "saved-function-handoff-repair-1", "cases": entries}
    )
    members["private/test.json"] = b"MUST NEVER BE OPENED"
    archive = tmp_path / "construction.tar.gz"
    _archive(archive, members)
    projection["construction_archive_sha256"] = hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    return projection, archive, members


def _rewrite(prepared):
    projection, archive, members = prepared
    _archive(archive, members)
    projection["construction_archive_sha256"] = hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()


def _change_case(prepared, name, change):
    """Reseal local fixtures so a semantic test reaches the intended check."""
    _projection, _path, members = prepared
    saved = deepcopy(members[f"inputs/{name}.json"])
    case = saved["source"]["case"]
    change(case)
    case = _seal(case)
    saved["source"]["case"] = case
    saved["source"]["source_input"]["input"] = deepcopy(case)
    saved = _seal(saved)
    members[f"inputs/{name}.json"] = saved
    plan = members["plan.json"]
    next(e for e in plan["cases"] if e["case_id"] == name)["input_sha256"] = saved[
        "artifact_sha256"
    ]
    members["plan.json"] = _seal(plan)
    _rewrite(prepared)


def test_build_reads_only_bound_members_and_returns_sealable_training(
    prepared, tmp_path, monkeypatch
):
    from autoformalism.fitting import prefix_initialization
    from autoformalism.llm.construction import ConstructionClient

    projection, archive, _ = prepared
    before = archive.read_bytes()
    before_projection = deepcopy(projection)
    reads = []
    original = contract._Reader.read

    def forbidden(*args, **kwargs):
        pytest.fail("sidecar construction cannot call providers or numerical models")

    for name in ("fit_training", "infer_prefix", "symbolic_rollout"):
        monkeypatch.setattr(prefix_initialization, name, forbidden)
    monkeypatch.setattr(ConstructionClient, "__init__", forbidden)

    def checked(reader, name):
        assert name in contract._members(projection)
        reads.append(name)
        return original(reader, name)

    monkeypatch.setattr(contract._Reader, "read", checked)
    value = contract.build(projection, archive)
    evidence = contract.validate(value, projection)
    assert set(reads) == set(value["source_members"]) == contract._members(projection)
    assert len(reads) == 17
    assert len(evidence["trajectories"]) == 16
    assert evidence["split"] == "train"
    assert evidence["contract"]["uniqueness_assumed"] is None
    assert "FORBIDDEN" not in json.dumps(value)
    assert "MUST NEVER" not in json.dumps(value)
    assert archive.read_bytes() == before
    assert projection == before_projection
    destination = tmp_path / "consistency.json"
    sealed_write(destination, value)
    assert contract.validate(sealed_read(destination), projection) == evidence


@pytest.mark.parametrize("mutation", ["hash", "symlink", "directory"])
def test_source_archive_identity_is_required_before_read(prepared, tmp_path, mutation):
    projection, archive, _ = prepared
    if mutation == "hash":
        projection["construction_archive_sha256"] = "0" * 64
    elif mutation == "symlink":
        link = tmp_path / "linked.tar.gz"
        link.symlink_to(archive)
        archive = link
    else:
        archive = tmp_path
    with pytest.raises(ValueError, match="archive"):
        contract.build(projection, archive)


@pytest.mark.parametrize("kind", ["link", "traversal", "duplicate"])
def test_unsafe_tar_members_are_refused_without_extraction(prepared, kind):
    projection, archive, members = prepared
    extra = tarfile.TarInfo("link" if kind == "link" else "../escape")
    if kind == "link":
        extra.type = tarfile.SYMTYPE
        extra.linkname = "plan.json"
    elif kind == "duplicate":
        extra.name = "plan.json"
    _archive(archive, members, extra)
    projection["construction_archive_sha256"] = hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match=r"unsafe|duplicate"):
        contract.build(projection, archive)


@pytest.mark.parametrize(
    "mutation", ["seal", "plan_input", "packet", "problem", "source_model", "roster"]
)
def test_source_seals_roster_and_frozen_parent_bindings(prepared, mutation):
    projection, archive, members = prepared
    if mutation == "seal":
        members["inputs/model_0.json"]["changed"] = True
    elif mutation == "plan_input":
        plan = members["plan.json"]
        plan["cases"][0]["input_sha256"] = "0" * 64
        members["plan.json"] = _seal(plan)
    elif mutation == "packet":
        projection["parents"]["model_0"]["source_packet_sha256"] = "0" * 64
    elif mutation == "problem":
        handoff = members["results/model_0/fit_handoff.json"]
        handoff["prefix_fit_problem"] = {}
        members["results/model_0/fit_handoff.json"] = _seal(handoff)
        projection["parents"]["model_0"]["source_packet_sha256"] = members[
            "results/model_0/fit_handoff.json"
        ]["artifact_sha256"]
    else:
        plan = members["plan.json"]
        if mutation == "source_model":
            plan["cases"][0]["source_model_id"] = "other"
        else:
            plan["cases"].pop()
        members["plan.json"] = _seal(plan)
    _rewrite(prepared)
    with pytest.raises(ValueError):
        contract.build(projection, archive)


@pytest.mark.parametrize("mutation", ["row", "contract", "context", "digest", "split"])
def test_all_eight_complete_training_packets_must_agree(prepared, mutation):
    def change(case):
        value = case["training_evidence"]
        if mutation == "row":
            value["trajectories"][0]["targets"]["y"][0] += 1
        elif mutation == "contract":
            value["contract"]["absolute_error_bounds"]["y"] = 1
        elif mutation == "context":
            case["context"]["time_symbol"] = "changed"
        elif mutation == "split":
            value["split"] = "val"
        else:
            value["sha256"] = "0" * 64
            return
        _resign_evidence(value)

    _change_case(prepared, "model_7", change)
    with pytest.raises(ValueError):
        contract.build(*prepared[:2])


@pytest.mark.parametrize(
    "mutation", ["extra", "heldout", "training", "archive", "members", "rows", "seal"]
)
def test_portable_sidecar_validation_rejects_unbound_or_hidden_fields(
    prepared, mutation
):
    projection, archive, _ = prepared
    value = contract.build(projection, archive)
    if mutation == "extra":
        value["hidden_validation_score"] = 3
    elif mutation == "heldout":
        value["evidence"]["contract"]["hidden_validation_score"] = 3
    elif mutation in {"training", "archive"}:
        key = (
            "training_sha256"
            if mutation == "training"
            else "construction_archive_sha256"
        )
        value[key] = "0" * 64
    elif mutation == "members":
        value["source_members"]["private/test.json"] = "0" * 64
    elif mutation == "rows":
        value["evidence"]["trajectories"].pop()
        _resign_evidence(value["evidence"])
    else:
        value["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        contract.validate(value, projection)


def test_portable_validation_never_reopens_archive(prepared, monkeypatch):
    projection, archive, _ = prepared
    value = contract.build(projection, archive)
    monkeypatch.setattr(
        contract, "_Reader", lambda *a: pytest.fail("archive reopened at runtime")
    )
    assert contract.validate(value, projection) == value["evidence"]
