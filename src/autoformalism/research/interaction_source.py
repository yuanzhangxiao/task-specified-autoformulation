"""Verify saved initializer episodes and export their frozen function sources.

Only the completed scope-aware diagnostic is accepted. Historical archives are
read in memory without extraction; source code, datasets and reference files are
never opened. No provider request, model selection or numerical fit occurs here.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import signed_processes
from autoformalism.search.construction_graph_repair import GraphDependencyReply
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
    validate_handoff,
)
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.search.training_evidence import (
    TrainingEvidence,
    validate_training_evidence,
)
from autoformalism.staged_topology import content_hash, lower_topology

_MAX_BYTES = 256 * 1024 * 1024
_MAX_MEMBERS = 4096
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
_DIGEST = re.compile(r"[a-f0-9]{64}")


def _safe_name(name: str) -> str:
    """Reject ambiguous or escaping archive paths before any member access."""
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError("unsafe source member path")
    return name


class _Reader:
    """Bounded selected-member reader for a directory or in-memory tar archive."""

    def __init__(self, source: Path | bytes):
        self.root = source if isinstance(source, Path) and source.is_dir() else None
        self.read_hashes: dict[str, str] = {}
        self.blob = None
        self.members = {}
        self.blob_sha256 = None
        if self.root is not None:
            if self.root.is_symlink():
                raise ValueError("source directory cannot be a symlink")
            return
        if isinstance(source, Path):
            if source.is_symlink() or source.stat().st_size > _MAX_BYTES:
                raise ValueError("source archive is unsafe or too large")
            source = source.read_bytes()
        if len(source) > _MAX_BYTES:
            raise ValueError("source archive is too large")
        self.blob_sha256 = hashlib.sha256(source).hexdigest()
        self.blob = source
        total = 0
        with tarfile.open(fileobj=io.BytesIO(source), mode="r:gz") as archive:
            for index, member in enumerate(archive):
                name = member.name.rstrip("/") if member.isdir() else member.name
                _safe_name(name)
                total += member.size
                if (
                    index >= _MAX_MEMBERS
                    or total > _MAX_BYTES
                    or not (member.isfile() or member.isdir())
                    or name in self.members
                ):
                    raise ValueError("unsafe, duplicate or oversized source archive")
                self.members[name] = member

    def read(self, name: str) -> bytes:
        """Read one explicitly requested regular file, retaining its digest."""
        _safe_name(name)
        if self.root is not None:
            path = self.root
            for part in PurePosixPath(name).parts:
                path = path / part
                if path.is_symlink():
                    raise ValueError("source member cannot be a symlink")
            if not path.is_file() or path.stat().st_size > _MAX_BYTES:
                raise ValueError("source member is absent or too large")
            blob = path.read_bytes()
        else:
            member = self.members.get(name)
            if member is None or not member.isfile():
                raise ValueError(f"missing regular source member: {name}")
            with tarfile.open(fileobj=io.BytesIO(self.blob), mode="r:gz") as archive:
                stream = archive.extractfile(member)
                assert stream is not None
                blob = stream.read()
        self.read_hashes[name] = hashlib.sha256(blob).hexdigest()
        return blob

    def json(self, name: str, seal: str | None = None) -> dict[str, Any]:
        """Parse an object and optionally verify its content-addressed seal."""
        value = json.loads(self.read(name))
        if not isinstance(value, dict):
            raise ValueError("source JSON must be an object")
        return _sealed(value, seal) if seal else value

    def files(self, prefix: str) -> set[str]:
        """List direct JSON children, without loading unrelated archive content."""
        _safe_name(prefix)
        if self.root is not None:
            directory = self.root / prefix
            if directory.is_symlink():
                raise ValueError("source calls cannot be a symlink")
            return {f"{prefix}/{p.name}" for p in directory.glob("*.json")}
        return {
            name
            for name in self.members
            if PurePosixPath(name).parent.as_posix() == prefix
            and name.endswith(".json")
        }


def _sealed(value: dict, field: str = "artifact_sha256") -> dict:
    """Check historical content without requiring its historical executable code."""
    if value.get(field) != content_hash(
        {key: item for key, item in value.items() if key != field}
    ):
        raise ValueError(f"source {field} differs")
    return value


def _history(reader: _Reader, plan: dict) -> tuple[_Reader, dict, dict]:
    """Follow exactly two hash-bound historical archives to authentic full cases."""
    graph_blob = reader.read("historical/source.tar.gz")
    if hashlib.sha256(graph_blob).hexdigest() != plan["source_archive_sha256"]:
        raise ValueError("historical graph archive digest differs")
    graph = _Reader(graph_blob)
    graph_plan = graph.json("plan.json", "artifact_sha256")
    if (
        graph_plan["artifact_sha256"] != plan["source_plan_sha256"]
        or graph_plan["protocol"] != "saved-construction-initialization-graph-1"
    ):
        raise ValueError("historical graph plan identity differs")
    full_blob = graph.read("historical/source.tar.gz")
    if hashlib.sha256(full_blob).hexdigest() != graph_plan["source_archive_sha256"]:
        raise ValueError("historical full-model archive digest differs")
    original = _Reader(full_blob).json("plan.json", "artifact_sha256")
    if original["artifact_sha256"] != graph_plan["source_plan_sha256"]:
        raise ValueError("historical full-model plan identity differs")
    return graph, graph_plan, original["cases"]


def _recover(model: dict, historical: dict) -> tuple[PublicScientificBrief, dict]:
    """Verify full declarations against the preserved initializer-only projection."""
    case = model["input"]
    original = _sealed(historical["source_result"])
    full = original["draft"]
    brief = PublicScientificBrief.model_validate(historical["cell"]["brief"])
    if (
        case["source_result_sha256"] != original["artifact_sha256"]
        or case["source_draft_sha256"] != content_hash(full)
        or historical["start"]["draft"] != full
        or historical["cell"]["context"] != case["context"]
        or brief.scientific_context != case["scientific_context"]
        or historical["cell"]["training_evidence"] != case["training_evidence"]
    ):
        raise ValueError("full historical model differs from saved projection")
    projected = {
        key: value for key, value in full.items() if key in ledger.Draft.model_fields
    }
    omitted = {
        key: content_hash(value)
        for key, value in full.items()
        if key not in ledger.Draft.model_fields
    }
    if (
        ledger.Draft.model_validate(projected).model_dump(mode="json") != case["draft"]
        or omitted != case["projection"]["omitted_fields"]
        or sorted(projected) != case["projection"]["retained_fields"]
    ):
        raise ValueError("historical draft projection differs")
    return brief, full


def _calls(reader: _Reader, plan: dict, entry: dict, result: dict) -> list[dict]:
    """Verify every consumed physical attempt and exact accepted provider choice."""
    attempts = result["attempts"]
    if not attempts or len(attempts) > plan["settings"]["maximum_requests"]:
        raise ValueError("source attempt count differs")
    records, paths = [], set()
    for attempt in attempts:
        key = attempt["request_hash"]
        if not isinstance(key, str) or not _DIGEST.fullmatch(key):
            raise ValueError("invalid source request hash")
        path = f"results/{entry['case_id']}/calls/{key}.json"
        record = reader.json(path)
        request = record["request"]
        payload = json.loads(
            request["body"]["messages"][1]["content"].split("\n", 1)[1]
        )
        if (
            key != content_hash(request)
            or key != record.get("request_hash")
            or request.get("namespace") != entry["namespace"]
            or request.get("settings") != plan["settings"]
            or record.get("attempt") != attempt["attempt"]
            or request["body"].get("seed")
            != int(
                content_hash([entry["seed"], record["step"], record["attempt"]])[:8], 16
            )
            % (2**31)
            or type(record.get("budget_charge")) is not int
            or record["budget_charge"] < 0
            or payload.get("construction_identity") != result["identity"]
            or payload.get("runtime_initialization_contract") != result["contract"]
            or content_hash(payload.get("public_context"))
            != result["contract"]["public_context_sha256"]
        ):
            raise ValueError("source cached request provenance differs")
        records.append(record)
        paths.add(path)
    if len(paths) != len(attempts) or paths != reader.files(
        f"results/{entry['case_id']}/calls"
    ):
        raise ValueError("source cached calls differ from consumed attempts")
    cost = _cost(records)
    if cost["budget_charge"] > plan["settings"]["maximum_total_tokens"]:
        raise ValueError("source charged-token ceiling exceeded")
    for result_key, cost_key in (
        ("physical_requests", "physical_requests"),
        ("budget_charge", "budget_charge"),
        ("observed_total_tokens", "observed_tokens"),
        ("unmeasured_requests", "requests_with_unknown_usage"),
    ):
        if result.get(result_key) != cost[cost_key]:
            raise ValueError("source result cost differs from cached calls")
    if result["accepted"]:
        accepted = [i for i, event in enumerate(attempts) if event["accepted"]]
        reply = GraphDependencyReply.model_validate(visible_response(records[-1]))
        handoff = GraphDependencyHandoff(records=reply.records).model_dump(mode="json")
        if (
            accepted != [len(attempts) - 1]
            or reply.decision != "revise"
            or handoff != result["handoff"]
            or attempts[-1].get("candidate") != handoff
        ):
            raise ValueError("accepted graph differs from cached provider reply")
    return records


def _function_source(
    brief: PublicScientificBrief, case: dict, result: dict, full: dict
) -> dict:
    """Compile exactly the retained signed processes and ordinary topology terms."""
    context = ValidationContext.model_validate(case["context"])
    draft = ledger.Draft.model_validate(case["draft"])
    inventory = ledger.inventory(brief, draft)
    equations = ledger.assembled_equations(brief, draft)
    topology, _ = lower_topology(brief, inventory, equations, context)
    return {
        "complete_topology": True,
        "inventory": [v.model_dump(mode="json") for v in inventory],
        "equations": [e.model_dump(mode="json") for e in equations],
        "topology": topology.model_dump(mode="json"),
        "shared_process_contract": {
            "protocol": signed_processes.POLICY,
            "bindings": [signed_processes.binding_for(p) for p in draft.processes],
        },
        "initialization_graph_contract": result["contract"],
        "initialization_graph_handoff": result["handoff"],
        "source_draft_sha256": content_hash(full),
        "scientific_declarations": full,
        "initializer_result_sha256": result["result_sha256"],
        "initializer_acceptance": result["acceptance"],
        "scientific_correctness": "unverified",
    }


def load_sources(source: Path) -> dict:
    """Load every accepted scope-aware episode, with the entire original roster.

    Returns source_plan/source_summary, source_sha256, roster (including rejected
    episodes), and cases keyed by accepted episode ID. Each case retains entry,
    source_input, case, result, brief, full_draft and compiled function_source.
    Identical graphs remain distinct proposal episodes.
    """
    reader = _Reader(source)
    plan = reader.json("plan.json", "artifact_sha256")
    summary = reader.json("summary.json")
    if (
        plan.get("protocol") != "saved-construction-initialization-repair-2"
        or plan.get("feedback_policy") != "scope-aware-2"
        or summary.get("plan_sha256") != plan["artifact_sha256"]
        or summary.get("status") != "complete"
        or summary.get("availability_status") != "available"
        or not 1 <= len(plan.get("source_models", [])) <= 2
        or not 1 <= len(plan.get("cases", [])) <= 10
        or any(
            value.get("test_data_opened") is not False
            or value.get("optimizer_calls") != 0
            or value.get("automatic_followup") is not False
            for value in (plan, summary)
        )
    ):
        raise ValueError("requires a completed scope-aware initialization source")
    graph, graph_plan, historical = _history(reader, plan)
    graph_entries = {entry["case_id"]: entry for entry in graph_plan["cases"]}
    models, briefs, full_drafts = {}, {}, {}
    for name in plan["source_models"]:
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in models:
            raise ValueError("unsafe or duplicate source model name")
        model = reader.json(f"inputs/{name}.json", "artifact_sha256")
        _sealed(model["input"])
        graph_case = graph.json(f"inputs/{name}.json", "artifact_sha256")
        graph_result = graph.json(f"results/{name}/result.json", "result_sha256")
        if (
            model["source_model_id"] != name
            or model["source_input_sha256"] != model["input"]["artifact_sha256"]
            or graph_entries[name]["input_sha256"] != graph_case["artifact_sha256"]
            or graph_case != model["input"]
            or model["source_result_sha256"] != graph_result["result_sha256"]
            or model["initial_handoff"] != graph_result["handoff"]
            or model["source_model_seed"] != graph_entries[name]["seed"]
        ):
            raise ValueError("source model input identity differs")
        briefs[name], full_drafts[name] = _recover(model, historical[name])
        models[name] = model
    cases, roster, all_calls = {}, [], []
    seen = set()
    for entry in plan["cases"]:
        name, model_id = entry["case_id"], entry["source_model_id"]
        if (
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or name in seen
            or model_id not in models
            or entry["input_sha256"] != models[model_id]["artifact_sha256"]
        ):
            raise ValueError("source episode input identity differs")
        seen.add(name)
        result = reader.json(f"results/{name}/result.json", "result_sha256")
        case = models[model_id]["input"]
        if (
            result.get("protocol") != "initialization-dependency-repair-1"
            or result.get("feedback_policy") != "scope-aware-2"
            or result.get("contract") != case["contract"]
            or result.get("accepted") != (result.get("status") == "complete")
            or result.get("test_data_opened") is not False
            or result.get("optimizer_calls") != 0
            or result.get("automatic_followup") is not False
            or result.get("status")
            not in {"complete", "needs_topology_revision", "repair_exhausted"}
        ):
            raise ValueError("source episode result contract differs")
        calls = _calls(reader, plan, entry, result)
        all_calls.extend(calls)
        roster.append(
            {
                "case_id": name,
                "source_model_id": model_id,
                "seed": entry["seed"],
                "accepted": result["accepted"],
                "status": result["status"],
                "result_sha256": result["result_sha256"],
            }
        )
        if not result["accepted"]:
            continue
        if (
            not result["structural_valid"]
            or result["conflict_count"] != 0
            or result["acceptance"].get("eligible") is not True
            or result["acceptance"].get("blocked") is not False
            or result["assessment"] != result["acceptance"]["assessment"]
        ):
            raise ValueError("accepted source retains a blocking conflict")
        evidence = TrainingCompatibilityEvidence.model_validate(
            case["training_evidence"]
        )
        validate_evidence(evidence)
        if evidence.context.model_dump(mode="json") != case["context"]:
            raise ValueError("source evidence context differs")
        validate_handoff(
            GraphDependencyHandoff.model_validate(result["handoff"]),
            GraphConstructionContract.model_validate(result["contract"]),
            draft=ledger.Draft.model_validate(case["draft"]),
            context=evidence.context,
            scientific_context=case["scientific_context"],
        )
        brief, full = briefs[model_id], full_drafts[model_id]
        training = historical[model_id]["cell"].get("evidence")
        if training is not None:
            validate_training_evidence(
                TrainingEvidence.model_validate(training), evidence.context
            )
        cases[name] = {
            "entry": entry,
            "source_input": models[model_id],
            "case": case,
            "result": result,
            "brief": brief.model_dump(mode="json"),
            "full_draft": full,
            "public_cell": historical[model_id]["cell"],
            "training_evidence": training,
            "function_source": _function_source(brief, case, result, full),
        }
    if (
        summary.get("cost") != _cost(all_calls)
        or summary.get("accepted_episodes") != len(cases)
        or {row["case_id"] for row in summary["rows"]} != seen
        or len(summary["rows"]) != len(seen)
    ):
        raise ValueError("source summary roster or costs differ")
    by_id = {entry["case_id"]: entry for entry in roster}
    for row in summary["rows"]:
        expected = by_id[row["case_id"]]
        if (
            any(
                row[field] != expected[field]
                for field in ("accepted", "status", "source_model_id")
            )
            or row["proposal_seed"] != expected["seed"]
        ):
            raise ValueError("source summary episode differs")
    for model_id in models:
        if sorted(
            row["seed"] for row in roster if row["source_model_id"] == model_id
        ) != sorted(plan["proposal_seeds"]):
            raise ValueError("source proposal-seed roster differs")
    return {
        "source_plan": plan,
        "source_summary": summary,
        "source_sha256": reader.blob_sha256 or content_hash(reader.read_hashes),
        "roster": roster,
        "cases": cases,
    }
