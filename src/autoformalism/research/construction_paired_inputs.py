"""Authenticate saved parent/child models without rewriting historical identities.

Only public training and validation files are read. All revision episodes are
replayed before exporting the complete accepted roster; retained replies alias
their unchanged parent and rejected replies remain provenance, not fit tasks.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research import construction_fit_inputs as original
from autoformalism.research import construction_revision_contract as consistency
from autoformalism.research.interaction_source import _Reader
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-paired-fit-inputs-1"
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
TaskId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")]
Population = Literal["returned-24", "synthetic-smoke"]


class Record(StrictSchema):
    """An unchanged full problem, with an explicit parent pairing."""

    role: Literal["parent", "child"]
    parent_id: TaskId
    parent_task_id: TaskId
    case_id: TaskId
    problem: PrefixFitProblem
    source_packet_sha256: Digest


class Episode(StrictSchema):
    """Preserve unsuccessful episodes and retained-parent aliases as well."""

    case_id: TaskId
    parent_id: TaskId
    parent_task_id: TaskId
    revision_seed: int = Field(ge=0)
    status: Literal["revised", "retained", "rejected", "unavailable"]
    completion_sha256: Digest
    result_sha256: Digest
    fit_task_id: TaskId | None


class Source(StrictSchema):
    """Allowlisted provenance; historical and importing runtimes stay separate."""

    original_archive_sha256: Digest
    original_plan_sha256: Digest
    original_runtime_source_sha256: Digest
    benchmark_id: TaskId
    public_files: dict[str, Digest]
    forcing_authorization: str
    permitted_future_inputs: tuple[str, ...]
    permitted_future_auxiliaries: tuple[str, ...]
    revision_plan_sha256: Digest
    revision_inputs_sha256: Digest
    revision_runtime_source_sha256: Digest
    revision_entrypoints: dict[str, Digest]
    consistency_evidence_sha256: Digest
    revision_members: dict[str, Digest]
    verifier_source_sha256: Digest
    provenance_policy: Literal["historical-source-preserved-exact-replay-1"] = (
        "historical-source-preserved-exact-replay-1"
    )


class Inputs(StrictSchema):
    """Positive-schema boundary from saved construction to numerical fitting."""

    protocol: Literal["construction-paired-fit-inputs-1"] = PROTOCOL
    records: dict[TaskId, Record]
    training: PublicSplit
    validation: PublicSplit
    prefix_policy: PrefixPolicy
    episodes: tuple[Episode, ...]
    source: Source
    test_data_opened: Literal[False] = False


def validate_inputs(data: dict, population: Population = "returned-24") -> dict:
    """Return canonical input after checking every pairing and public boundary."""
    from autoformalism.fitting import prefix_initialization as kernel

    if population not in {"returned-24", "synthetic-smoke"}:
        raise ValueError("unknown paired fitting population")
    value = Inputs.model_validate(data)
    train, val = value.training, value.validation
    if train.name != "train" or val.name != "val" or not train.rows or not val.rows:
        raise ValueError("nonempty public training/validation splits required")
    if {r.trajectory_id for r in train.rows} & {r.trajectory_id for r in val.rows}:
        raise ValueError("training and validation trajectory IDs overlap")
    for row in (*train.rows, *val.rows):
        # These constructors only project and validate data; no model compilation,
        # numerical integration, or root inference is performed here.
        kernel.make_prefix(row, value.prefix_policy)
        kernel.make_forecast_schedule(row, value.prefix_policy)
    parents = {k: v for k, v in value.records.items() if v.role == "parent"}
    children = {k: v for k, v in value.records.items() if v.role == "child"}
    if not parents or not children:
        raise ValueError("both unchanged parents and revised children are required")
    if len({r.case_id for r in value.records.values()}) != len(value.records):
        raise ValueError("duplicate model case identity")
    for task_id, record in value.records.items():
        parent = parents.get(record.parent_task_id)
        if parent is None or parent.parent_id != record.parent_id:
            raise ValueError("record has no matching unchanged parent")
        if record.role == "parent" and (
            record.parent_task_id != task_id or record.case_id != record.parent_id
        ):
            raise ValueError("parent identity must refer to itself")
        context = record.problem.context
        for row in (*train.rows, *val.rows):
            for role in (
                "targets",
                "auxiliaries",
                "external_inputs",
                "fixed_covariates",
            ):
                if set(getattr(row, role)) != set(getattr(context, role)):
                    raise ValueError(f"model/public {role} differ")
            if row.time[-1] <= row.time[0] + value.prefix_policy.duration:
                raise ValueError("common forecast suffix must be nonempty")
        if record.role == "child" and (
            record.problem.context != parent.problem.context
            or record.problem.graph != parent.problem.graph
        ):
            raise ValueError("revision changed fixed context or initialization")
    if len({e.case_id for e in value.episodes}) != len(value.episodes):
        raise ValueError("duplicate revision episode")
    represented_children = []
    for episode in value.episodes:
        parent = parents.get(episode.parent_task_id)
        if parent is None or parent.parent_id != episode.parent_id:
            raise ValueError("episode parent identity differs")
        if episode.status == "revised":
            child = children.get(episode.fit_task_id)
            if child is None or (
                child.case_id != episode.case_id
                or child.parent_task_id != episode.parent_task_id
            ):
                raise ValueError("revised episode has no exact child")
            represented_children.append(episode.fit_task_id)
        elif episode.status == "retained":
            if episode.fit_task_id != episode.parent_task_id:
                raise ValueError("retained episode must alias its parent")
        elif episode.fit_task_id is not None:
            raise ValueError("rejected episode cannot create a fit task")
    if len(set(represented_children)) != len(represented_children) or set(
        represented_children
    ) != set(children):
        raise ValueError("all children require one accepted revision episode")
    policy = value.prefix_policy
    if set(policy.future_inputs) != set(value.source.permitted_future_inputs) or set(
        policy.future_auxiliaries
    ) != set(value.source.permitted_future_auxiliaries):
        raise ValueError("prefix forcing differs from frozen source permission")
    if population == "returned-24":
        if (len(parents), len(children), len(train.rows), len(val.rows)) != (
            8,
            16,
            16,
            4,
        ):
            raise ValueError("all 8 parents, 16 children and 16/4 public rows required")
        if Counter(e.status for e in value.episodes) != {
            "revised": 16,
            "retained": 1,
            "rejected": 7,
        } or any(
            sorted(e.revision_seed for e in value.episodes if e.parent_task_id == name)
            != [0, 1, 2]
            for name in parents
        ):
            raise ValueError("complete 24-episode revision population required")
        if (
            policy.duration != 8
            or set(value.source.public_files)
            != {"specification.json", "proposer_prompt.txt", "train.json", "val.json"}
            or value.source.forcing_authorization != original._SUPPLIED_HORIZON
        ):
            raise ValueError("frozen eight-minute public information contract differs")
    return value.model_dump(mode="json")


def _historical_plan(reader: _Reader, projection: dict) -> dict:
    """Check immutable content while retaining, never replacing, old code hashes."""
    from scripts import construction_feedback_revision as revision

    plan = reader.json("plan.json", "artifact_sha256")
    raw = {k: v for k, v in projection.items() if k != "artifact_sha256"}
    revision._validate_inputs(raw)
    cases = []
    for name, parent in sorted(raw["parents"].items()):
        for seed in revision.SEEDS:
            identity = {
                "parent_id": name,
                "seed": seed,
                "input_sha256": content_hash(parent),
            }
            cases.append(
                {
                    **identity,
                    "case_id": f"{name}__revision_seed{seed}",
                    "namespace": f"{revision.V2_PROTOCOL}:{content_hash(identity)}",
                }
            )
    digest = plan.get("runtime_source_sha256", "")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("invalid historical runtime identity")
    sidecar = reader.json("consistency_evidence.json", "artifact_sha256")
    expected = {
        "protocol": revision.V2_PROTOCOL,
        "cases": cases,
        "settings": revision.SETTINGS.model_dump(mode="json"),
        "input_bundle_sha256": content_hash(raw),
        "runtime_source_sha256": digest,
        "entrypoints": revision._entrypoints(),
        "selection": "all eight parents, three seeds each; unchanged controls exported",
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "model_promotion": False,
        "policies": revision.V2_POLICIES,
        "consistency_evidence_sha256": sidecar["artifact_sha256"],
    }
    if plan != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("historical revision plan/roster/policy/entrypoint differs")
    return plan


def _episode_members(reader: _Reader, case_id: str) -> None:
    """Precheck every replayed path; no arbitrary directory members are opened."""
    prefix = f"results/{case_id}"
    reader.read(f"{prefix}/completion.json")
    result = reader.json(f"{prefix}/result.json")
    if result.get("problem"):
        reader.read(f"{prefix}/fit_handoff.json")
    for directory in (f"{prefix}/calls", f"{prefix}/calls/tokenization"):
        for name in sorted(reader.files(directory)):
            if not re.fullmatch(r"[a-f0-9]{64}\.json", Path(name).name):
                raise ValueError("unexpected provider receipt filename")
            reader.read(name)


def load_inputs(revision_root: Path, source_archive: Path, public_cell: Path) -> dict:
    """Reconstruct all source packets and 24 episodes without fitting or calls."""
    from scripts import construction_feedback_revision as revision

    original_inputs = original.load_inputs(source_archive, public_cell)
    reader = _Reader(revision_root)
    if reader.root is None:
        raise ValueError("revision root must be an extracted regular directory")
    projection = reader.json("inputs.json", "artifact_sha256")
    plan = _historical_plan(reader, projection)
    raw = {k: v for k, v in projection.items() if k != "artifact_sha256"}
    rebuilt = consistency.build(raw, source_archive)
    sidecar = reader.json("consistency_evidence.json", "artifact_sha256")
    if sidecar != {**rebuilt, "artifact_sha256": content_hash(rebuilt)}:
        raise ValueError("consistency evidence differs from original construction")
    if projection["training"] != original_inputs["training"] or set(
        projection["parents"]
    ) != set(original_inputs["records"]):
        raise ValueError("revision training or parent roster differs from source")
    records, parent_tasks, episodes = {}, {}, []
    for index, (name, parent) in enumerate(sorted(projection["parents"].items())):
        saved = original_inputs["records"][name]
        if parent["problem"] != saved["problem"] or (
            parent["source_packet_sha256"] != saved["source_packet_sha256"]
        ):
            raise ValueError("revision parent differs from original function packet")
        control = {
            "parent_id": name,
            "problem": parent["problem"],
            "problem_sha256": content_hash(parent["problem"]),
            "control_unchanged": True,
            "fitting_launched": False,
        }
        if reader.json(f"controls/{name}.json", "artifact_sha256") != {
            **control,
            "artifact_sha256": content_hash(control),
        }:
            raise ValueError("unchanged parent control differs")
        task_id = f"parent-{index:02d}"
        parent_tasks[name] = task_id
        records[task_id] = {
            "role": "parent",
            "parent_id": name,
            "parent_task_id": task_id,
            "case_id": name,
            "problem": parent["problem"],
            "source_packet_sha256": saved["source_packet_sha256"],
        }
    child_index = 0
    for entry in plan["cases"]:
        case_id, parent_id = entry["case_id"], entry["parent_id"]
        _episode_members(reader, case_id)
        receipt = revision._completion(revision_root, entry, plan)
        if receipt is None:
            raise ValueError("every revision episode requires a completed receipt")
        result = reader.json(f"results/{case_id}/result.json")
        fit_task_id = None
        if result["status"] == "revised":
            packet = reader.json(
                f"results/{case_id}/fit_handoff.json", "artifact_sha256"
            )
            fit_task_id = f"child-{child_index:02d}"
            child_index += 1
            records[fit_task_id] = {
                "role": "child",
                "parent_id": parent_id,
                "parent_task_id": parent_tasks[parent_id],
                "case_id": case_id,
                "problem": packet["prefix_fit_problem"],
                "source_packet_sha256": packet["artifact_sha256"],
            }
        elif result["status"] == "retained":
            if result["problem"] != projection["parents"][parent_id]["problem"]:
                raise ValueError("retained episode changed its parent")
            fit_task_id = parent_tasks[parent_id]
        episodes.append(
            {
                "case_id": case_id,
                "parent_id": parent_id,
                "parent_task_id": parent_tasks[parent_id],
                "revision_seed": entry["seed"],
                "status": result["status"],
                "completion_sha256": receipt["artifact_sha256"],
                "result_sha256": reader.read_hashes[f"results/{case_id}/result.json"],
                "fit_task_id": fit_task_id,
            }
        )
    hashes = dict(reader.read_hashes)
    for name, digest in hashes.items():
        if hashlib.sha256(reader.read(name)).hexdigest() != digest:
            raise ValueError("revision source changed during replay")
    source = original_inputs["source"]
    result = {
        "protocol": PROTOCOL,
        "records": records,
        "training": original_inputs["training"],
        "validation": original_inputs["validation"],
        "prefix_policy": projection["fitting_prefix_policy"],
        "episodes": episodes,
        "source": {
            "original_archive_sha256": source["archive_sha256"],
            "original_plan_sha256": source["plan_sha256"],
            "original_runtime_source_sha256": source["historical_runtime_sha256"],
            "benchmark_id": source["benchmark_id"],
            "public_files": source["public_files"],
            "forcing_authorization": source["forcing_authorization"],
            "permitted_future_inputs": source["permitted_future_inputs"],
            "permitted_future_auxiliaries": source["permitted_future_auxiliaries"],
            "revision_plan_sha256": plan["artifact_sha256"],
            "revision_inputs_sha256": projection["artifact_sha256"],
            "revision_runtime_source_sha256": plan["runtime_source_sha256"],
            "revision_entrypoints": plan["entrypoints"],
            "consistency_evidence_sha256": sidecar["artifact_sha256"],
            "revision_members": hashes,
            "verifier_source_sha256": runtime_source_hash(),
        },
        "test_data_opened": False,
    }
    return validate_inputs(result)
