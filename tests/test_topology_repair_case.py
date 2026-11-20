"""Check the source boundary and training-pair projection used by both arms."""

import json
from pathlib import Path

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.research import topology_repair_case as case_io
from autoformalism.search.topology_compatibility import (
    CompleteTrajectory,
    EvidenceContract,
    TrainingCompatibilityEvidence,
)


def _row(name, target, flow):
    return CompleteTrajectory(
        trajectory_id=name,
        time=(0, 1, 2, 3),
        targets={"h": target},
        inputs={"flow": flow},
        auxiliaries={},
        covariates={},
    )


def test_witness_packet_uses_only_public_sources_and_ends_at_witness():
    left = _row("a", [3, 3, 2, 9], [0, 1, 0, 1000])
    right = _row("b", [3, 3, 4, 15], [0, 0, 0, 10])
    evidence = TrainingCompatibilityEvidence(
        context=ValidationContext(targets=("h",), external_inputs=("flow",)),
        contract=EvidenceContract(provenance="synthetic source projection"),
        trajectories=(left, right),
        sha256="0" * 64,
    )
    case = {
        "evidence": evidence.model_dump(mode="json"),
        "parent": {},
        "target": "h",
        "allowed_sources": ["h", "flow", "memory"],
        "pair": ["a", "b"],
        "witness_time": 2,
        "public_brief": {
            "scientific_context": "Public basin example",
            "requirements": [],
            "target_dependencies": [],
        },
    }
    packet = case_io.public_packet(case)
    assert packet["paired_values"]["flow"]["first_different_sample"] == {
        "time": 1,
        "values": [1, 0],
    }
    assert packet["paired_values"]["flow"]["maximum_through_witness"] == [1, 0]
    assert packet["paired_values"]["h"]["initial_values"] == [3, 3]
    assert packet["paired_values"]["memory"]["sample_comparison"] == "unavailable"
    case["witness_time"] = 2.5
    with pytest.raises(ValueError, match="same sampled grid"):
        case_io.public_packet(case)


def test_draft_keeps_complete_dependencies_and_no_signed_scientific_claim():
    parent = {
        "hypothesis": "Three-compartment fixture.",
        "targets": {
            "a": {"kind": "dynamic", "sources": ["u", "c"], "role": "measured"},
            "b": {"kind": "algebraic", "sources": ["a"], "role": "measured"},
        },
        "additional_components": [
            {"name": "c", "kind": "dynamic", "sources": ["u"], "role": "storage"}
        ],
    }
    d = case_io.draft_from_parent(parent)
    assert {e.name: e.terms[0].sources for e in d.equations} == {
        "a": ("u", "c"),
        "b": ("a",),
        "c": ("u",),
    }
    assert all(
        e.terms[0].outer_weight_sign.value == "unrestricted" for e in d.equations
    )
    parent["additional_components"][0]["name"] = "a"
    with pytest.raises(ValueError, match="already has a declaration"):
        case_io.draft_from_parent(parent)


def test_partial_or_modified_case_cannot_be_loaded(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps({"parent": {}, "case_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="digest"):
        case_io.load_case(path)


def test_returned_case_and_physical_receipt_when_available():
    archive = Path("output/t2-fresh-20b-results-20261119/returned")
    if not archive.exists():
        pytest.skip("Download the original training inspection for its offline replay")
    case = case_io.prepare_case(archive)
    assert case["provenance"]["request"] == case_io.REQUEST
    assert case["parent"]["targets"]["I"]["sources"] == ["insulin_pmol_per_kg_min", "I"]
    packet = case["prompt_evidence"]["paired_values"]
    assert packet["insulin_pmol_per_kg_min"]["maximum_through_witness"] == [0, 0]
    assert packet["I"]["witness_values"] == pytest.approx([583.1036186, 23.6665505])
    case["allowed_sources"].append("unavailable_future_observation")
    case["case_sha256"] = case_io.content_hash(
        {k: v for k, v in case.items() if k != "case_sha256"}
    )
    with pytest.raises(ValueError, match="permissions"):
        case_io.validate_case(case)
