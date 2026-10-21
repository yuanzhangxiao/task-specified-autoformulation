"""Verify returned construction packets and bind only their public development data.

Historical runtime identities are retained as provenance, not compared to today's
implementation. Every historical model, accepted reply and exported packet is
reconstructed from its sealed source before it can enter the new qualification.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import tempfile
from pathlib import Path

from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research.interaction_source import _Reader, _sealed
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.search.scoped_process_gains import verify_scoped_gain_handoff
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-fit-inputs-1"
_DIGEST = re.compile(r"[a-f0-9]{64}")
_ROLES = ("targets", "auxiliaries", "external_inputs", "fixed_covariates")
_SUPPLIED_HORIZON = (
    "All listed inputs and supplied auxiliary trajectories are available over "
    "each observation horizon."
)


def _public_cell(directory: Path) -> tuple[dict, PublicSplit, PublicSplit, dict]:
    """Read exactly four named public files; never traverse an evaluator tree."""
    reader = _Reader(directory)
    values = {}
    for name in ("specification", "train", "val"):
        wrapped = reader.json(f"{name}.json")
        if set(wrapped) != {"sha256", "value"} or wrapped[
            "sha256"
        ] != content_sha256(wrapped["value"]):
            raise ValueError("public file seal differs")
        values[name] = wrapped["value"]
    spec = values["specification"]
    prompt = reader.read("proposer_prompt.txt")
    if (
        spec.get("protocol") != "phase-c-development-2"
        or spec.get("test_released") is not False
        or spec.get("benchmark_id") != directory.name
        or prompt.decode() != spec["public_prompt"] + "\n"
        or _SUPPLIED_HORIZON not in spec["public_prompt"]
    ):
        raise ValueError("public development identity or forcing authorization differs")
    train, val = (
        PublicSplit.model_validate(values[name]) for name in ("train", "val")
    )
    for split, expected in ((train, "train"), (val, "val")):
        if split.name != expected:
            raise ValueError("public split name differs")
        for row in split.rows:
            for role in _ROLES:
                if set(getattr(row, role)) != set(spec[role]):
                    raise ValueError(f"public channel role differs: {role}")
    if {r.trajectory_id for r in train.rows} & {r.trajectory_id for r in val.rows}:
        raise ValueError("training and validation trajectory IDs overlap")
    return spec, train, val, dict(reader.read_hashes)


def _training_matches(value: dict, training: PublicSplit) -> None:
    """Match complete saved training evidence, not a selected prompt summary."""
    evidence = value["source"]["source_input"]["input"]["training_evidence"]
    if evidence.get("split") != "train":
        raise ValueError("construction evidence is not training data")
    expected = [
        {
            "trajectory_id": row.trajectory_id,
            "time": list(row.time),
            "targets": {k: list(v) for k, v in row.targets.items()},
            "auxiliaries": {k: list(v) for k, v in row.auxiliaries.items()},
            "inputs": {k: list(v) for k, v in row.external_inputs.items()},
            "covariates": dict(row.fixed_covariates),
        }
        for row in training.rows
    ]
    if evidence.get("trajectories") != expected:
        raise ValueError("public training rows differ from construction evidence")


def _unpack(reader: _Reader, root: Path) -> None:
    """Materialize already validated regular members for historical validators."""
    with tarfile.open(fileobj=io.BytesIO(reader.blob), mode="r:gz") as archive:
        for member in archive:
            if member.isfile():
                target = root / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                assert stream is not None
                target.write_bytes(stream.read())


def _returned_plan(root: Path) -> tuple[dict, dict]:
    """Reconstruct historical content while retaining its original code identity."""
    from scripts import repair_function_handoff as repair

    plan = sealed_read(root / "plan.json")
    if plan.get("protocol") != repair.PROTOCOL:
        raise ValueError("unsupported returned construction protocol")
    expected, cases = repair._prepared(root / "historical/source.tar.gz")
    # Today's fitting capability must not rewrite or impersonate historical code.
    for key in ("runtime_source_sha256", "entrypoints"):
        identity = plan.get(key)
        digests = list(identity.values()) if isinstance(identity, dict) else [identity]
        if not digests or any(
            not isinstance(digest, str) or not _DIGEST.fullmatch(digest)
            for digest in digests
        ):
            raise ValueError("invalid historical code identity")
        expected[key] = identity
    if plan != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("returned source plan differs from reconstructed history")
    return plan, cases


def _record(root: Path, plan: dict, entry: dict, expected: dict) -> tuple[dict, dict]:
    """Require a complete receipt and accepted provider choices for every model."""
    from scripts import construction_functions as original
    from scripts import repair_function_handoff as repair

    name = entry["case_id"]
    value = sealed_read(root / "inputs" / f"{name}.json")
    if value != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("returned input differs from historical source")
    calls = original._calls(root, entry, plan)
    receipt = original._receipt(root, entry, plan, calls)
    if (
        not receipt
        or receipt["status"] != "complete"
        or set(receipt["files"]) != {"result.json", "fit_handoff.json"}
    ):
        raise ValueError("every source case requires a complete fitting packet")
    directory = root / "results" / name
    result = _sealed(
        json.loads((directory / "result.json").read_text()), "result_sha256"
    )
    lowered = repair._lower(value)
    repair._verify_result(directory, entry, value, lowered, result, calls)
    handoff = sealed_read(directory / "fit_handoff.json")
    verify_scoped_gain_handoff(
        lowered, result, handoff, provenance=repair._provenance(plan, entry)
    )
    problem = PrefixFitProblem.model_validate(handoff["prefix_fit_problem"])
    return {
        "problem": problem.model_dump(mode="json"),
        "source_packet_sha256": handoff["artifact_sha256"],
        "source_model_id": entry["source_model_id"],
        "proposal_seed": entry["seed"],
        "metadata": {
            "source_result_sha256": result["result_sha256"],
            "source_readiness": handoff["readiness"],
            "parameter_scopes": handoff["parameter_scopes"],
            "historical_action": entry["action"],
            "original_model_unchanged": True,
            "mechanism_compliance_assessed": False,
        },
    }, value


def load_inputs(source: Path, public_cell: Path) -> dict:
    """Return all verified packets with exact train/validation data and provenance."""
    if not source.is_file() or source.is_symlink():
        raise ValueError("a regular returned inspection archive is required")
    reader = _Reader(source)
    spec, train, val, public_hashes = _public_cell(public_cell)
    records = {}
    with tempfile.TemporaryDirectory(prefix="af-fit-inputs-") as temporary:
        root = Path(temporary)
        _unpack(reader, root)
        plan, expected_cases = _returned_plan(root)
        for entry in plan["cases"]:
            name = entry["case_id"]
            record, value = _record(root, plan, entry, expected_cases[name])
            target_contract = value["source"]["public_cell"]["target_contract"]
            if (
                target_contract["benchmark_id"] != spec["benchmark_id"]
                or target_contract["public_prompt_sha256"]
                != public_hashes["proposer_prompt.txt"]
            ):
                raise ValueError("public prompt differs from construction contract")
            _training_matches(value, train)
            context = record["problem"]["context"]
            for role in _ROLES:
                if set(context[role]) != set(spec[role]):
                    raise ValueError(
                        f"fitting context differs from public role: {role}"
                    )
            records[name] = record
        if len(records) != len(plan["cases"]) or not records:
            raise ValueError("source roster is empty or contains duplicate cases")
    return {
        "protocol": PROTOCOL,
        "records": records,
        "training": train.model_dump(mode="json"),
        "validation": val.model_dump(mode="json"),
        "source": {
            "archive_sha256": reader.blob_sha256,
            "plan_sha256": plan["artifact_sha256"],
            "historical_runtime_sha256": plan["runtime_source_sha256"],
            "historical_function_cost": plan["historical_function_cost"],
            "historical_initializer_cost": plan["historical_initializer_cost"],
            "benchmark_id": spec["benchmark_id"],
            "public_files": public_hashes,
            "permitted_future_auxiliaries": spec["auxiliaries"],
            "permitted_future_inputs": spec["external_inputs"],
            "forcing_authorization": _SUPPLIED_HORIZON,
            "initialization_scope": (
                "Opt-in observation-prefix qualification authorized separately; "
                "not the historical benchmark's no-validation-initial-fit score."
            ),
            "selection": "all complete source cases; no numerical preselection",
        },
        "test_data_opened": False,
    }
