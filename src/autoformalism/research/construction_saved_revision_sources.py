"""Recover six pinned saved candidates without accepting historical rejections.

The source is a pinned bundle of preserved training-only inspection members,
with a separate historical replay attestation. Only data members are read, in
memory; no historical code is executed or extracted. Reconstruction checks the
patch interface, not the old or new scientific acceptance predicate.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.research import construction_revision_contract as contract
from autoformalism.research.construction_revision_edits import apply_revision
from autoformalism.research.construction_revision_evidence import (
    validate_training_projection,
)
from autoformalism.research.interaction_source import _Reader
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.staged_topology import content_hash

PROTOCOL = "saved-construction-revision-sources-1"
SOURCE_ARCHIVE_SHA256 = (
    "7c3ad7e7561b85d5da19b3ebb232ab19c2437b7c9bd1291eb5a3ee8ea7ff21c0"
)
HISTORICAL_ARCHIVE_SHA256 = (
    "1f1bf441684c8bc09c5fc18b44c042f78f6c719e90fccb14163f2d23f9b4c494"
)
SOURCE_ATTESTATION_SHA256 = (
    "f58e61db8f3463b37ddf6a7e1c9ec71769a27cadd1f05ac542e2e6ac5bc678ca"
)
SOURCE_PLAN_SHA256 = "c1e8a11d9402061b30980cea3b756867644f35b7239e75bc5c381998123f67cf"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_UNPACKED_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 1024

# Pxx indexes the ordered original parents in the pinned source plan. Each
# selected call and completion is independently pinned as well as the archive.
SELECTIONS = (
    {
        "selection_id": "P03-r0",
        "parent_index": 3,
        "seed": 0,
        "attempt": 2,
        "status": "rejected",
        "request_sha256": (
            "3c41992a2f8d754ef3fb17c7f569fee436d2d29155991b87993c8c26fa5abc3b"
        ),
        "call_file_sha256": (
            "4b670221f0208e2256afe00ea306e979bbcd3ed64abf60ff1efe0a0af0a77a70"
        ),
        "completion_sha256": (
            "8a9cd0af97e96203f812d017184f178417a4a1aa71b927fcc7bfed4f8f2c30e3"
        ),
        "result_file_sha256": (
            "114a5902fd2185af84be9495d8683fb6581850049b2ff01948e1eeca54a91974"
        ),
    },
    {
        "selection_id": "P06-r2",
        "parent_index": 6,
        "seed": 2,
        "attempt": 2,
        "status": "rejected",
        "request_sha256": (
            "587be4c3a5aa17db66406c8ef04b4a28b65a9b4a0c2561d5c33635ea4cf439d7"
        ),
        "call_file_sha256": (
            "b82b1e58602d09257b0458016d3f4e4623f0575c7a5c956a3fb5b5b5d59f2fd1"
        ),
        "completion_sha256": (
            "1ad3c38ac281fd8831c1135130944f34592bcf1b2bd31ca652020746c47c1a79"
        ),
        "result_file_sha256": (
            "0ba5987517933336810326de915b4ac5d38eabdb8094e3fbb92b0f320bc2cc2c"
        ),
    },
    {
        "selection_id": "P01-r0",
        "parent_index": 1,
        "seed": 0,
        "attempt": 0,
        "status": "revised",
        "request_sha256": (
            "6f9ef0267276b3ecc30efe8e1c2b3c51e55849e875e2863a940cb2f9da6e8e5c"
        ),
        "call_file_sha256": (
            "4621c2619518f9933b5defdbe780239cb86816e6fff08dd5b4be057316580273"
        ),
        "completion_sha256": (
            "9c1a537c0d2292145f86c863a42ded278797423f252574f9ba1c6a76678be29b"
        ),
        "result_file_sha256": (
            "8d19205c261332c0a4242893420a6454ae9c894a43817194d8200cf8f76e791b"
        ),
    },
    {
        "selection_id": "P05-r2",
        "parent_index": 5,
        "seed": 2,
        "attempt": 1,
        "status": "revised",
        "request_sha256": (
            "ce350d1c672e99bde96fdaf567de3ca665d3303f8ca508728906fec35defe7b2"
        ),
        "call_file_sha256": (
            "517039b2a69b7823758756afc19ae67996b5fea69ea6dcdc6d244aa1b14b96e6"
        ),
        "completion_sha256": (
            "6df9cf15eb7e9c1d163752c03d7ac0708efe602d8fd4df89ae6b98d12fb96daf"
        ),
        "result_file_sha256": (
            "d0f4f8373b3c4d017e98bc8dce85c1e8f2dc3775f9db18ba13b1831a5273023c"
        ),
    },
    {
        "selection_id": "P00-r1",
        "parent_index": 0,
        "seed": 1,
        "attempt": 0,
        "status": "retained",
        "request_sha256": (
            "6f4812104a5a052cba6854d8c2e2e45726ebfc91bbdd91fe3f204215bdd5f18c"
        ),
        "call_file_sha256": (
            "b4c51963c37e5e7099f85463d83eae99ebcf0afd267714b28e47e3ef84dffd5d"
        ),
        "completion_sha256": (
            "40f9acdddda6f1eb3f74844a614cf339f545997039e4b5f17256319b5839aeea"
        ),
        "result_file_sha256": (
            "53430183f6c4051767d92908ee02c603c96cf26bf0a928250ba6bb16d47010fe"
        ),
    },
    {
        "selection_id": "P06-r1",
        "parent_index": 6,
        "seed": 1,
        "attempt": 1,
        "status": "revised",
        "request_sha256": (
            "da3fc287485146431206e1a150862537068c6022535b381fd29c312cdd346d82"
        ),
        "call_file_sha256": (
            "84efc7050b95df443c52879f6ab37ced7d5d6c5f7f140040e371b75031be8c63"
        ),
        "completion_sha256": (
            "38836903af9260ea743e414af374f0ce0ea4176d08aef5c580a99bd614e4e2fb"
        ),
        "result_file_sha256": (
            "8cdb4abc122f3e4ba84887c914fc67056e35bbe7f3544894f7840d5053c92df2"
        ),
    },
)


def _reader(path: Path) -> _Reader:
    """Require a small regular archive and reject every unsafe member header."""
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_ARCHIVE_BYTES
    ):
        raise ValueError("a bounded regular saved-revision archive is required")
    # Authenticate compressed bytes before parsing even the archive headers.
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_ARCHIVE_SHA256:
        raise ValueError("saved-revision archive identity differs")
    reader = _Reader(path)
    if (
        len(reader.members) > MAX_MEMBERS
        or sum(member.size for member in reader.members.values()) > MAX_UNPACKED_BYTES
    ):
        raise ValueError("saved-revision archive exceeds resource limits")
    if reader.blob_sha256 != SOURCE_ARCHIVE_SHA256:
        raise ValueError("saved-revision archive identity differs")
    return reader


def _records(reader: _Reader, entry: dict, plan: dict, *, tokens=False) -> list[dict]:
    """Check immutable request/cache identities without making provider calls."""
    prefix = f"results/{entry['case_id']}/calls"
    if tokens:
        prefix += "/tokenization"
    records = []
    for name in sorted(reader.files(prefix)):
        record = reader.json(name)
        request = record["request"]
        key = content_hash(request)
        if Path(name).stem != key or request.get("namespace") != entry["namespace"]:
            raise ValueError("saved request namespace or digest differs")
        if tokens:
            if (
                request.get("policy") != "response-token-preflight-1"
                or request.get("model_revision") != plan["settings"]["model_revision"]
            ):
                raise ValueError("saved tokenizer policy differs")
        elif (
            record.get("request_hash") != key
            or request.get("settings") != plan["settings"]
            or request["body"].get("seed")
            != int(
                content_hash([entry["seed"], record["step"], record["attempt"]])[:8], 16
            )
            % (2**31)
            or type(record.get("budget_charge")) is not int
            or record["budget_charge"] < 0
        ):
            raise ValueError("saved generation provenance differs")
        records.append(record)
    if not tokens and len(records) > plan["settings"]["maximum_requests"]:
        raise ValueError("saved generation allowance exceeded")
    return records


def _candidate(
    reader: _Reader, plan: dict, entry: dict, original: dict, spec: dict
) -> tuple:
    """Recover the exact final mathematical patch, retaining historical rejection."""
    prefix = f"results/{entry['case_id']}"
    completion = reader.json(f"{prefix}/completion.json", "artifact_sha256")
    result = reader.json(f"{prefix}/result.json")
    calls = _records(reader, entry, plan)
    tokens = _records(reader, entry, plan, tokens=True)
    expected_files = {"result.json"} | (
        {"fit_handoff.json"} if spec["status"] != "rejected" else set()
    )
    if (
        completion["artifact_sha256"] != spec["completion_sha256"]
        or completion["plan_sha256"] != plan["artifact_sha256"]
        or completion["input_sha256"] != entry["input_sha256"]
        or completion["calls_sha256"] != content_hash(calls)
        or completion["tokenizations_sha256"] != content_hash(tokens)
        or completion["status"] != spec["status"]
        or result["status"] != spec["status"]
        or set(completion["files"]) != expected_files
        or completion["files"]["result.json"] != spec["result_file_sha256"]
    ):
        raise ValueError("selected completion or historical outcome differs")
    for name, digest in completion["files"].items():
        reader.read(f"{prefix}/{name}")
        if reader.read_hashes[f"{prefix}/{name}"] != digest:
            raise ValueError("completed selected artifact differs")
    attempts = result["attempts"]
    last = attempts[-1]
    if (
        [a["attempt"] for a in attempts] != list(range(spec["attempt"] + 1))
        or last["request_hash"] != spec["request_sha256"]
        or len(calls) != len(attempts)
        or {a["request_hash"] for a in attempts} != {c["request_hash"] for c in calls}
    ):
        raise ValueError("selected attempt or request roster differs")
    record = next(
        call for call in calls if call["request_hash"] == spec["request_sha256"]
    )
    call_path = f"{prefix}/calls/{spec['request_sha256']}.json"
    if (
        reader.read_hashes[call_path] != spec["call_file_sha256"]
        or record["attempt"] != spec["attempt"]
        or record["step"] != "training_revision"
    ):
        raise ValueError("selected call receipt differs")
    reply = visible_response(record)
    reconstructed = apply_revision(
        PrefixFitProblem.model_validate(original["problem"]),
        reply,
        evidence_ids=["public_task", "training_consistency", "parameter_audit"],
        brief=PublicScientificBrief.model_validate(original["public_brief"]),
        scientific_aliases=original["scientific_aliases"],
        tolerant_formatting=True,
    )
    problem = reconstructed["problem"].model_dump(mode="json")
    if reconstructed["audit"].get("static_eligibility") != "eligible":
        raise ValueError("saved patch is not statically reconstructible")
    if spec["status"] == "rejected":
        if (
            result.get("problem") is not None
            or last.get("diagnostic", {}).get("code") != "FUNCTION_MODEL_DATA_CONFLICT"
            or last.get("reply_sha256") != content_hash(reply)
        ):
            raise ValueError(
                "selected rejection is not the declared scientific conflict"
            )
    else:
        handoff = reader.json(f"{prefix}/fit_handoff.json", "artifact_sha256")
        if (
            result.get("problem") != problem
            or result.get("reply") != reply
            or reconstructed["status"] != spec["status"]
            or handoff.get("protocol") != "construction-revision-fit-handoff-3"
            or handoff.get("plan_sha256") != plan["artifact_sha256"]
            or handoff.get("case_id") != entry["case_id"]
            or handoff.get("parent_id") != entry["parent_id"]
            or handoff.get("parent_problem_sha256") != content_hash(original["problem"])
            or handoff.get("prefix_fit_problem") != problem
            or handoff.get("fitting_launched") is not False
        ):
            raise ValueError("accepted saved candidate or handoff differs")
    provenance = {
        **spec,
        "source_case_id": entry["case_id"],
        "source_parent_id": entry["parent_id"],
        "source_parent_problem_sha256": content_hash(original["problem"]),
        "selected_problem_sha256": content_hash(problem),
        "source_role": "rejected_candidate"
        if spec["status"] == "rejected"
        else "accepted_candidate",
        "historical_acceptance_reassessed": False,
        "new_acceptance_required": True,
        "historical_cost": _cost(calls),
    }
    parent = {
        "problem": problem,
        "problem_sha256": content_hash(problem),
        "public_brief": deepcopy(original["public_brief"]),
        "scientific_aliases": deepcopy(original["scientific_aliases"]),
        "selected_source": provenance,
    }
    return parent, provenance, calls


def load(path: Path) -> dict:
    """Return six source-bound candidates plus unchanged train-only evidence."""
    reader = _reader(Path(path))
    plan = reader.json("plan.json", "artifact_sha256")
    sealed_inputs = reader.json("inputs.json", "artifact_sha256")
    inputs = {
        key: value for key, value in sealed_inputs.items() if key != "artifact_sha256"
    }
    consistency = reader.json("consistency_evidence.json", "artifact_sha256")
    attestation = reader.json("source-provenance.json", "artifact_sha256")
    if (
        attestation["artifact_sha256"] != SOURCE_ATTESTATION_SHA256
        or attestation["historical_archive_sha256"] != HISTORICAL_ARCHIVE_SHA256
        or attestation["source_plan_sha256"] != SOURCE_PLAN_SHA256
        or attestation["historical_archive_available"] is not False
        or attestation["test_data_opened"] is not False
    ):
        raise ValueError("saved source reconstruction attestation differs")
    validate_training_projection(inputs)
    contract.validate(consistency, inputs)
    parent_names = list(dict.fromkeys(e["parent_id"] for e in plan["cases"]))
    if (
        plan["artifact_sha256"] != SOURCE_PLAN_SHA256
        or plan["protocol"] != "construction-feedback-revision-3"
        or plan["input_bundle_sha256"] != content_hash(inputs)
        or plan["consistency_evidence_sha256"] != consistency["artifact_sha256"]
        or len(parent_names) != 8
        or set(parent_names) != set(inputs["parents"])
        or len(plan["cases"]) != 24
        or plan["test_data_opened"] is not False
        or plan["optimizer_calls"] != 0
        or plan["solver_rollouts"] != 0
        or plan["automatic_followup"] is not False
        or plan["model_promotion"] is not False
    ):
        raise ValueError("saved revision source plan or training bindings differ")
    parents, selection, historical_calls = {}, [], []
    for spec in SELECTIONS:
        parent_id = parent_names[spec["parent_index"]]
        matches = [
            e
            for e in plan["cases"]
            if e["parent_id"] == parent_id and e["seed"] == spec["seed"]
        ]
        if len(matches) != 1:
            raise ValueError("saved selection roster differs")
        entry = matches[0]
        original = inputs["parents"][parent_id]
        if entry["input_sha256"] != content_hash(original):
            raise ValueError("selected original parent differs")
        parent, record, calls = _candidate(reader, plan, entry, original, spec)
        parents[spec["selection_id"]] = parent
        selection.append(record)
        historical_calls.extend(calls)
    return {
        "inputs": inputs,
        "consistency_source": consistency,
        "parents": parents,
        "selection": selection,
        "provenance": {
            "protocol": PROTOCOL,
            "source_archive_sha256": reader.blob_sha256,
            "source_plan_sha256": plan["artifact_sha256"],
            "historical_archive_sha256": HISTORICAL_ARCHIVE_SHA256,
            "source_reconstruction": attestation,
            "source_members": dict(sorted(reader.read_hashes.items())),
            "selection_policy": "six_saved_defect_examples_no_score_selection",
            "historical_cost": _cost(historical_calls),
            "historical_acceptance_reassessed": False,
            "optimizer_calls": 0,
            "llm_calls": 0,
            "solver_rollouts": 0,
            "test_data_opened": False,
            "model_promotion": False,
        },
    }
