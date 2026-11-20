"""Freeze the train-only seed-102 topology and its matched-trajectory evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research.t2_breadth import check_topology, named_topology_forms
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "fresh-102-matched-topology-repair-1"
PLAN = "fffca64ecc87f8d6fb23c22d6aaadf13e2d0cc3c3a94609a224d24581627a3e4"
REQUEST = "92043bf06887486083a9b44876160023d023a179cb3a4e6af625bec541510401"
REPLY = "1d7264aaf3c0967dee903d6698b2d8050603c2f856ecd5d9e05b92d5b67de8b8"
SLOT = "fresh-102-topology_correction-2"
PAIR = ("train_005", "train_011")
TARGET = "I"
WITNESS_TIME = 95.0
# Binds the selected public fields when the parent archive is absent on ACES.
INPUT_BINDING = "b0f8afc5093d3d213a2a1f2536383c4fec03212286a3a61bd0e92b21adea65f2"


def _named_graph(parent, targets):
    schema, _ = named_topology_forms(tuple(targets))
    return schema.model_validate(parent).as_topology()


def draft_from_parent(parent: dict) -> Draft:
    """Project unsigned source unions without inferring terms or function forms."""
    graph = _named_graph(parent, tuple(parent["targets"]))
    variables, equations = [], []
    for component in graph.components:
        variables.append(
            {
                "name": component.name,
                "definition": "differential"
                if component.kind == "dynamic"
                else "algebraic",
                "scientific_role": component.role,
            }
        )
        equations.append(
            {
                "name": component.name,
                "terms": [
                    {
                        "sources": component.sources,
                        "outer_weight_sign": "unrestricted",
                        "scientific_role": "Source union only.",
                    }
                ],
            }
        )
    return Draft.model_validate({"variables": variables, "equations": equations})


def _source_evidence(left, right, name: str, end: int) -> dict:
    """Compare simultaneous train measurements, without imputing hidden state data."""
    for group in ("targets", "inputs", "auxiliaries"):
        channels = getattr(left, group)
        if name not in channels:
            continue
        a, b = channels[name], getattr(right, group)[name]
        first = next((i for i in range(end + 1) if a[i] != b[i]), None)
        return {
            "channel_role": group,
            "initial_values": [a[0], b[0]],
            "witness_values": [a[end], b[end]],
            "first_different_sample": None
            if first is None
            else {"time": left.time[first], "values": [a[first], b[first]]},
            "minimum_through_witness": [min(a[: end + 1]), min(b[: end + 1])],
            "maximum_through_witness": [max(a[: end + 1]), max(b[: end + 1])],
        }
    return {"channel_role": "generated_unmeasured", "sample_comparison": "unavailable"}


def public_packet(case: dict) -> dict:
    """Render the shared observations independently of the repair generator."""
    evidence = TrainingCompatibilityEvidence.model_validate(case["evidence"])
    traj = {r.trajectory_id: r for r in evidence.trajectories}
    left, right = (traj[key] for key in case["pair"])
    if left.time != right.time or case["witness_time"] not in left.time:
        raise ValueError(
            "Matched evidence requires the same sampled grid and witness time."
        )
    end = left.time.index(case["witness_time"])
    sources = {
        name: _source_evidence(left, right, name, end)
        for name in case["allowed_sources"]
    }
    return {
        "split": evidence.split,
        "public_task": case["public_brief"]["scientific_context"],
        "required_pathways": case["public_brief"]["requirements"],
        "required_dependencies": case["public_brief"]["target_dependencies"],
        "current_topology": case["parent"],
        "allowed_sources": case["allowed_sources"],
        "source_use": (
            "Generated public targets use their model values in rollout. "
            "Declared future inputs and supplied auxiliaries are available as forcing."
        ),
        "contract": evidence.contract.model_dump(mode="json", exclude={"provenance"}),
        "latent_initialization": (
            "Initialization rules apply to differential states. "
            "The saved topology supplies no latent initial values. "
            "Equality of an unobserved initial is unknown without a specified rule."
        ),
        "target": case["target"],
        "training_pair_in_column_order": case["pair"],
        "witness_time": case["witness_time"],
        "number_of_aligned_samples_through_witness": end + 1,
        "paired_values": sources,
        "observation": (
            "Equal initial states and identical complete forcing histories give the "
            "same trajectory for a closed subsystem with shared dynamics and "
            "unique solutions. "
            "The recorded target values at the witness time differ."
        ),
    }


def validate_case(case: dict) -> None:
    """Verify the case, its embedded train-only evidence and its 20B projection."""
    if case.get("case_sha256") != content_hash(
        {k: v for k, v in case.items() if k != "case_sha256"}
    ):
        raise ValueError("Case digest differs.")
    if (case["protocol"], case["pair"], case["witness_time"], case["target"]) != (
        PROTOCOL,
        list(PAIR),
        WITNESS_TIME,
        TARGET,
    ):
        raise ValueError("Unexpected saved-case selection.")
    if content_hash(case["parent"]) != REPLY:
        raise ValueError("Saved seed-102 topology differs.")
    evidence = TrainingCompatibilityEvidence.model_validate(case["evidence"])
    validate_evidence(evidence)
    binding = {
        "evidence": evidence.sha256,
        "public_brief": case["public_brief"],
        "allowed_sources": case["allowed_sources"],
        "provenance": case["provenance"],
    }
    if content_hash(binding) != INPUT_BINDING:
        raise ValueError("Public source permissions, evidence or provenance differ.")
    graph = _named_graph(case["parent"], evidence.context.targets)
    check_topology(
        graph,
        evidence.context,
        PublicScientificBrief.model_validate(case["public_brief"]),
    )
    if case["prompt_evidence"] != public_packet(case):
        raise ValueError("The shared public packet differs from its train records.")


def load_case(path: Path) -> dict:
    """Load the frozen case and reject incomplete or stale copies."""
    value = json.loads(path.read_text())
    validate_case(value)
    return value


def prepare_case(returned: Path) -> dict:
    """Authenticate the accepted graph against its live receipt and sealed intake."""
    plan, inputs = (
        sealed_read(returned / file) for file in ("plan.json", "inputs.json")
    )
    if (plan["artifact_sha256"], plan["intake_sha256"], inputs["test_data_opened"]) != (
        PLAN,
        inputs["artifact_sha256"],
        False,
    ):
        raise ValueError("Expected the pinned train-only seed-102 plan and intake.")
    result = sealed_read(returned / "results" / "fresh-102.json")
    (step,) = [s for s in result["history"] if s["outcome"] == "topology_ready"]
    order = sealed_read(returned / "call-order.json")["calls"][step["through_call"] - 1]
    if (
        result["plan"],
        step["reply_sha256"],
        order["slot"],
        order["request_hash"],
        order["source_plan"],
    ) != (PLAN, REPLY, SLOT, REQUEST, PLAN):
        raise ValueError(
            "Saved branch history does not identify the accepted topology."
        )
    folder = returned / "steps" / SLOT
    reply = sealed_read(folder / "reply.json")
    receipt = (folder / "calls" / (REQUEST + ".json")).read_bytes()
    record = json.loads(receipt)
    parent = visible_response(record)
    if (
        hashlib.sha256(receipt).hexdigest() != reply["physical_receipt_sha256"]
        or record["request_hash"] != REQUEST
        or reply["request_hash"] != REQUEST
        or reply["value"] != parent
        or content_hash(parent) != REPLY
    ):
        raise ValueError("The physical reply and accepted seed-102 graph differ.")
    ev = TrainingCompatibilityEvidence.model_validate(inputs["evidence"])
    graph = _named_graph(parent, ev.context.targets)
    prefix = inputs["fitting_contract"]["prefix_policy"]
    supplied = set(prefix["future_inputs"]) | set(prefix["future_auxiliaries"])
    if supplied - set(ev.context.forcing_channels):
        raise ValueError(
            "The archived future-source policy includes a non-forcing channel."
        )
    generation = {c.name for c in graph.components}
    case = {
        "protocol": PROTOCOL,
        "parent": parent,
        "evidence": inputs["evidence"],
        "public_brief": inputs["brief"],
        "target": TARGET,
        "pair": list(PAIR),
        "witness_time": WITNESS_TIME,
        "allowed_sources": sorted((supplied - graph.aliases().keys()) | generation),
        "provenance": {
            "plan": PLAN,
            "intake": plan["intake_sha256"],
            "slot": SLOT,
            "request": REQUEST,
            "receipt": reply["physical_receipt_sha256"],
            "raw_reply": REPLY,
            "training": ev.sha256,
        },
    }
    case["prompt_evidence"] = public_packet(case)
    case["case_sha256"] = content_hash(case)
    validate_case(case)
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returned", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    case = prepare_case(args.returned)
    if args.out.exists():
        if load_case(args.out) != case:
            raise ValueError("Destination already contains a different case.")
    else:
        atomic_json(args.out, case)
    print(json.dumps({"case": str(args.out), "case_sha256": case["case_sha256"]}))


if __name__ == "__main__":
    main()
