"""Saved-case retests preserve rejected status and verify immutable source data."""

import hashlib
import io
import json
import tarfile
from copy import deepcopy

import pytest

from autoformalism.research import construction_saved_revision_sources as source
from autoformalism.research.construction_revision_edits import apply_revision
from autoformalism.staged_topology import content_hash
from tests.test_construction_revision_edits import brief, problem, reply


def _seal(value):
    raw = {key: item for key, item in value.items() if key != "artifact_sha256"}
    return {**raw, "artifact_sha256": content_hash(raw)}


def _blob(value):
    return (
        json.dumps(value, sort_keys=True).encode() if isinstance(value, dict) else value
    )


def _sha(value):
    return hashlib.sha256(_blob(value)).hexdigest()


def _archive(path, members, extra=None):
    with tarfile.open(path, "w:gz") as stream:
        for name, value in members.items():
            blob = _blob(value)
            member = tarfile.TarInfo(name)
            member.size = len(blob)
            stream.addfile(member, io.BytesIO(blob))
        if extra is not None:
            stream.addfile(extra)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    # Projection/sidecar schemas have independent real-source and negative tests.
    # This fixture substitutes only that producer boundary; archive, completion,
    # call, patch, accepted handoff, and rejected-status checks remain real.
    monkeypatch.setattr(source, "validate_training_projection", lambda value: None)
    monkeypatch.setattr(source.contract, "validate", lambda value, projection: None)
    original = {
        "problem": problem().model_dump(mode="json"),
        "public_brief": brief().model_dump(mode="json"),
        "scientific_aliases": {},
        "endpoint": {"unresolved": "not for the new proposer"},
        "prompt_evidence": {"old_fitted_score": "not for the new proposer"},
    }
    inputs = {"parents": {f"parent_{i}": deepcopy(original) for i in range(8)}}
    evidence = _seal({"evidence": {"synthetic": True}})
    settings = {"model_revision": "frozen", "maximum_requests": 3}
    entries = []
    for i in range(8):
        for seed in range(3):
            entries.append(
                {
                    "case_id": f"parent_{i}__revision_seed{seed}",
                    "parent_id": f"parent_{i}",
                    "seed": seed,
                    "input_sha256": content_hash(original),
                    "namespace": f"old:{i}:{seed}",
                }
            )
    plan = _seal(
        {
            "protocol": "construction-feedback-revision-3",
            "cases": entries,
            "input_bundle_sha256": content_hash(inputs),
            "settings": settings,
            "consistency_evidence_sha256": evidence["artifact_sha256"],
            "optimizer_calls": 0,
            "solver_rollouts": 0,
            "automatic_followup": False,
            "model_promotion": False,
            "test_data_opened": False,
        }
    )
    members = {
        "plan.json": plan,
        "inputs.json": _seal(inputs),
        "consistency_evidence.json": evidence,
        "private/test.json": b"DO NOT OPEN",
    }
    specs = deepcopy(source.SELECTIONS)
    for spec in specs:
        entry = next(
            e
            for e in entries
            if e["parent_id"] == f"parent_{spec['parent_index']}"
            and e["seed"] == spec["seed"]
        )
        prefix = f"results/{entry['case_id']}"
        proposed = reply(evidence_ids=["training_consistency"])
        if spec["status"] == "retained":
            proposed["edits"] = []
        candidate = apply_revision(
            problem(),
            proposed,
            evidence_ids=["training_consistency"],
            brief=brief(),
            scientific_aliases={},
            tolerant_formatting=True,
        )["problem"].model_dump(mode="json")
        calls, attempts = [], []
        for attempt in range(spec["attempt"] + 1):
            request = {
                "namespace": entry["namespace"],
                "settings": settings,
                "body": {
                    "seed": int(
                        content_hash([entry["seed"], "training_revision", attempt])[:8],
                        16,
                    )
                    % (2**31)
                },
            }
            key = content_hash(request)
            call = {
                "request": request,
                "request_hash": key,
                "attempt": attempt,
                "step": "training_revision",
                "status": "responded",
                "budget_charge": 10,
                "observed_total_tokens": 10,
                "raw_response": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(proposed)},
                        }
                    ]
                },
            }
            members[f"{prefix}/calls/{key}.json"] = call
            calls.append(call)
            attempts.append(
                {
                    "attempt": attempt,
                    "request_hash": key,
                    "reply_sha256": content_hash(proposed),
                    "diagnostic": {"code": "FUNCTION_MODEL_DATA_CONFLICT"},
                }
            )
        result = {"status": spec["status"], "attempts": attempts, "problem": None}
        if spec["status"] != "rejected":
            result.update(problem=candidate, reply=proposed)
            members[f"{prefix}/fit_handoff.json"] = _seal(
                {
                    "protocol": "construction-revision-fit-handoff-3",
                    "plan_sha256": plan["artifact_sha256"],
                    "case_id": entry["case_id"],
                    "parent_id": entry["parent_id"],
                    "parent_problem_sha256": content_hash(original["problem"]),
                    "prefix_fit_problem": candidate,
                    "fitting_launched": False,
                }
            )
        members[f"{prefix}/result.json"] = result
        files = {"result.json": _sha(result)}
        if spec["status"] != "rejected":
            files["fit_handoff.json"] = _sha(members[f"{prefix}/fit_handoff.json"])
        completion = _seal(
            {
                "plan_sha256": plan["artifact_sha256"],
                "input_sha256": entry["input_sha256"],
                "calls_sha256": content_hash(
                    sorted(calls, key=lambda c: c["request_hash"])
                ),
                "tokenizations_sha256": content_hash([]),
                "status": spec["status"],
                "files": files,
            }
        )
        members[f"{prefix}/completion.json"] = completion
        spec.update(
            request_sha256=calls[-1]["request_hash"],
            call_file_sha256=_sha(calls[-1]),
            result_file_sha256=_sha(result),
            completion_sha256=completion["artifact_sha256"],
        )
    archive = tmp_path / "source.tar.gz"
    attestation = _seal(
        {
            "historical_archive_sha256": source.HISTORICAL_ARCHIVE_SHA256,
            "source_plan_sha256": plan["artifact_sha256"],
            "historical_archive_available": False,
            "test_data_opened": False,
        }
    )
    members["source-provenance.json"] = attestation
    _archive(archive, members)
    monkeypatch.setattr(source, "SOURCE_ARCHIVE_SHA256", _sha(archive.read_bytes()))
    monkeypatch.setattr(source, "SOURCE_PLAN_SHA256", plan["artifact_sha256"])
    monkeypatch.setattr(
        source, "SOURCE_ATTESTATION_SHA256", attestation["artifact_sha256"]
    )
    monkeypatch.setattr(source, "SELECTIONS", specs)
    return archive, members, specs


def test_recovers_six_without_promoting_historical_rejections(saved, monkeypatch):
    archive, _members, _specs = saved
    reads = []
    original_read = source._Reader.read

    def read(reader, name):
        reads.append(name)
        return original_read(reader, name)

    monkeypatch.setattr(source._Reader, "read", read)
    first = source.load(archive)
    assert first == source.load(archive)
    assert len(first["inputs"]["parents"]) == 8
    assert list(first["parents"]) == [s["selection_id"] for s in source.SELECTIONS]
    assert [s["source_role"] for s in first["selection"]].count(
        "rejected_candidate"
    ) == 2
    assert all(s["new_acceptance_required"] for s in first["selection"])
    assert all(not s["historical_acceptance_reassessed"] for s in first["selection"])
    assert all(
        "endpoint" not in p and "prompt_evidence" not in p
        for p in first["parents"].values()
    )
    assert "private/test.json" not in reads
    assert first["provenance"]["llm_calls"] == 0


def test_archive_change_fails_before_reconstruction(saved):
    archive, members, _ = saved
    members["extra.txt"] = b"changed"
    _archive(archive, members)
    with pytest.raises(ValueError, match="archive identity"):
        source.load(archive)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../b", "a\\b"])
def test_unsafe_paths_rejected(saved, monkeypatch, path):
    archive, members, _ = saved
    members[path] = b"no"
    _archive(archive, members)
    monkeypatch.setattr(source, "SOURCE_ARCHIVE_SHA256", _sha(archive.read_bytes()))
    with pytest.raises(ValueError, match="unsafe"):
        source.load(archive)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "duplicate"])
def test_links_and_duplicate_members_rejected(saved, monkeypatch, kind):
    archive, members, _ = saved
    extra = tarfile.TarInfo("plan.json" if kind == "duplicate" else "link")
    if kind != "duplicate":
        extra.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
        extra.linkname = "plan.json"
    _archive(archive, members, extra)
    monkeypatch.setattr(source, "SOURCE_ARCHIVE_SHA256", _sha(archive.read_bytes()))
    with pytest.raises(ValueError, match=r"unsafe|duplicate"):
        source.load(archive)


@pytest.mark.parametrize(
    "limit", ["MAX_ARCHIVE_BYTES", "MAX_UNPACKED_BYTES", "MAX_MEMBERS"]
)
def test_resource_limits(saved, monkeypatch, limit):
    archive, _, _ = saved
    monkeypatch.setattr(source, limit, 1)
    with pytest.raises(ValueError, match=r"bounded|resource"):
        source.load(archive)


def test_archive_symlink_rejected(saved):
    archive, _, _ = saved
    linked = archive.with_name("linked.tar.gz")
    linked.symlink_to(archive)
    with pytest.raises(ValueError, match="regular"):
        source.load(linked)


@pytest.mark.parametrize(
    "kind", ["call", "result", "completion", "handoff", "plan", "inputs"]
)
def test_inner_receipts_cannot_be_replaced_even_with_archive_repin(
    saved, monkeypatch, kind
):
    archive, members, _ = saved
    if kind in {"plan", "inputs"}:
        name = kind + ".json"
    else:
        suffix = {
            "result": "/result.json",
            "completion": "/completion.json",
            "handoff": "/fit_handoff.json",
        }.get(kind)
        name = next(
            n
            for n in members
            if ("/calls/" in n if kind == "call" else n.endswith(suffix))
        )
    members[name]["unexpected"] = "tampered"
    _archive(archive, members)
    monkeypatch.setattr(source, "SOURCE_ARCHIVE_SHA256", _sha(archive.read_bytes()))
    with pytest.raises(ValueError, match="differs"):
        source.load(archive)


def test_wrong_attempt_selection_is_rejected(saved, monkeypatch):
    archive, _members, specs = saved
    specs[0]["attempt"] = 1
    monkeypatch.setattr(source, "SELECTIONS", specs)
    with pytest.raises(ValueError, match="attempt"):
        source.load(archive)


def test_historical_rejection_cannot_be_relabeled(saved, monkeypatch):
    archive, _members, specs = saved
    specs[0]["status"] = "revised"
    monkeypatch.setattr(source, "SELECTIONS", specs)
    with pytest.raises(ValueError, match="outcome"):
        source.load(archive)


def test_reconstruction_drift_cannot_replace_accepted_candidate(saved, monkeypatch):
    archive, _, _ = saved
    original_apply = source.apply_revision

    def changed(*args, **kwargs):
        value = original_apply(*args, **kwargs)
        candidate = value["problem"].model_dump(mode="json")
        candidate["base_candidate"]["state_equations"][0]["rhs"] = "0"
        value["problem"] = source.PrefixFitProblem.model_validate(candidate)
        return value

    monkeypatch.setattr(source, "apply_revision", changed)
    with pytest.raises(ValueError, match="accepted saved candidate"):
        source.load(archive)


def test_source_loading_does_not_run_scientific_checks_or_numerics(saved, monkeypatch):
    from autoformalism.fitting import prefix_initialization
    from autoformalism.research import (
        construction_function_consistency,
        construction_revision_consistency,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("source authentication must not fit or reassess acceptance")

    for name in ("fit_training", "infer_prefix", "forecast", "symbolic_rollout"):
        monkeypatch.setattr(prefix_initialization, name, forbidden)
    monkeypatch.setattr(construction_function_consistency, "assess", forbidden)
    monkeypatch.setattr(construction_revision_consistency, "assess", forbidden)
    assert len(source.load(saved[0])["parents"]) == 6
