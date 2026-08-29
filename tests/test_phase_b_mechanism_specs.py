"""Tests for public-only Phase-B mechanism specifications."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.benchmarks import (
    load_suite_spec,
    phase_b_public_spec,
    phase_b_task_mechanism_lines,
    render_phase_b_prompts,
)
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.rebuttal.phase_b_mechanism_specs import (
    phase_b_public_mechanism_spec,
)

DATA_ROOT = Path("data_raw")
SUITE_PATH = Path("configs/benchmarks/phase_b_suite_v1.json")
GENERATED_ROOT = Path("configs/mechanism_eval/phase_b_v1")


def _public_specs():
    suite = load_suite_spec(SUITE_PATH)
    for family in suite.families:
        for task in family.tasks:
            task_argument = task if family.family == "dalla_man" else None
            for condition in family.dynamics_conditions:
                dynamics = "canonical" if condition == "not_applicable" else condition
                for tier in family.tiers:
                    for variant in family.semantic_variants:
                        yield phase_b_public_spec(
                            family.family,
                            tier.name,
                            variant,
                            task=task_argument,
                            dynamics=dynamics,
                            data_root=DATA_ROOT,
                        )


def test_all_40_public_specs_are_prompt_committed_and_channel_bounded() -> None:
    public_specs = tuple(_public_specs())
    assert len(public_specs) == 40

    for public_spec in public_specs:
        mechanism_spec = phase_b_public_mechanism_spec(public_spec)
        proposer, _ = render_phase_b_prompts(public_spec)
        channel_names = {item.public_name for item in public_spec.channels}
        target_names = {
            item.public_name for item in public_spec.channels if item.role == "target"
        }

        assert mechanism_spec.source == "public_prompt"
        assert (
            mechanism_spec.public_prompt_sha256
            == hashlib.sha256(proposer.encode("utf-8")).hexdigest()
        )
        assert len(mechanism_spec.required_mechanisms) == len(
            phase_b_task_mechanism_lines(public_spec)
        )
        for requirement in mechanism_spec.required_mechanisms:
            assert requirement.public_requirement in proposer
            assert set(requirement.required_drivers) <= channel_names
            assert set(requirement.required_targets) <= target_names


def test_obfuscated_specs_contain_no_private_domain_vocabulary() -> None:
    forbidden_by_variant = {
        "obfuscated": (
            "dalla",
            "glucose",
            "insulin",
            "reactor",
            "temperature",
            "concentration",
            "jacket",
        ),
        "opaque": ("alien", "device", "input-driven memory"),
    }
    for public_spec in _public_specs():
        forbidden = forbidden_by_variant.get(public_spec.semantic_variant)
        if forbidden is None:
            continue
        payload = phase_b_public_mechanism_spec(public_spec).model_dump_json()
        lowered = payload.casefold()
        assert all(token not in lowered for token in forbidden)


def test_committed_specs_exactly_match_current_public_contracts() -> None:
    manifest = json.loads(
        (GENERATED_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    paths = sorted((GENERATED_ROOT / "specs").glob("*.json"))
    assert manifest["specification_count"] == len(paths) == 40

    expected = {
        item.benchmark_id: phase_b_public_mechanism_spec(item)
        for item in _public_specs()
    }
    observed = {
        spec.benchmark_id: spec
        for path in paths
        for spec in (
            MechanismEvaluationSpec.model_validate_json(
                path.read_text(encoding="utf-8")
            ),
        )
    }
    assert observed == expected
    for record in manifest["specifications"]:
        payload = (GENERATED_ROOT / record["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["spec_sha256"]


def test_private_gate_metadata_cannot_change_public_mechanism_contract() -> None:
    public_spec = phase_b_public_spec(
        "dalla_man",
        "hard",
        "obfuscated",
        task="T3",
        data_root=DATA_ROOT,
    )
    altered = public_spec.model_copy(
        update={"required_mechanisms": ("private_answer_a", "private_answer_b")}
    )

    assert phase_b_public_mechanism_spec(altered) == (
        phase_b_public_mechanism_spec(public_spec)
    )


def test_public_source_requires_prompt_hash_and_requirement_text() -> None:
    with pytest.raises(ValidationError, match="prompt SHA-256"):
        MechanismEvaluationSpec.model_validate(
            {
                "source": "public_prompt",
                "benchmark_id": "example",
                "tier": "easy",
                "required_mechanisms": [
                    {"id": "input_memory", "public_requirement": "required"}
                ],
            }
        )
    with pytest.raises(ValidationError, match="public source text"):
        MechanismEvaluationSpec.model_validate(
            {
                "source": "public_prompt",
                "benchmark_id": "example",
                "tier": "easy",
                "public_prompt_sha256": "0" * 64,
                "required_mechanisms": [{"id": "input_memory"}],
            }
        )
