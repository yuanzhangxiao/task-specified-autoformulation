"""Tests for deterministic public target-generation contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.benchmarks import (
    load_suite_spec,
    phase_b_public_spec,
    phase_b_public_target_contract,
    render_phase_b_prompts,
)
from autoformalism.schemas import CandidateModel
from autoformalism.targets import PublicTargetContract, evaluate_public_targets

DATA_ROOT = Path("data_raw")
SUITE_PATH = Path("configs/benchmarks/phase_b_suite_v1.json")
GENERATED_ROOT = Path("configs/target_eval/phase_b_v1")
ROLE_AWARE_ROOT = Path("configs/target_eval/phase_b_v2")


def _candidate(total_expression: str) -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": "candidate",
            "parent_candidate_id": None,
            "states": [
                {"name": "Gp", "kind": "observed"},
                {"name": "I", "kind": "observed"},
                {
                    "name": "X",
                    "kind": "latent",
                    "mechanisms": ["delayed_insulin_action"],
                },
            ],
            "processes": [
                {
                    "name": "Uid",
                    "expression": "k_u * X * Gp",
                    "mechanisms": ["insulin_dependent_disposal"],
                },
                {"name": "U", "expression": total_expression},
            ],
            "state_equations": [
                {"state": "Gp", "rhs": "EGP - U - E"},
                {"state": "I", "rhs": "insulin_pmol_per_kg_min - k_i * I"},
                {"state": "X", "rhs": "k_x * (I - X)"},
            ],
            "observation_mappings": [
                {"channel": "Gp", "expression": "Gp"},
                {"channel": "I", "expression": "I"},
                {"channel": "U", "expression": "U"},
            ],
            "parameters": [
                {
                    "name": name,
                    "scope": "global",
                    "bounds": {"lower": 0.0, "upper": 5.0},
                    "initialization_range": {"lower": 0.1, "upper": 1.0},
                }
                for name in ("k_u", "k_i", "k_x")
            ],
            "initial_conditions": [
                {"state": "Gp", "scope": "global", "fixed_value": 1.0},
                {"state": "I", "scope": "global", "fixed_value": 1.0},
                {"state": "X", "scope": "global", "fixed_value": 0.0},
            ],
        }
    )


def test_total_disposal_contract_accepts_recursive_u_ii_path() -> None:
    public = phase_b_public_spec(
        "dalla_man", "easy", "named", task="T2", data_root=DATA_ROOT
    )
    contract = phase_b_public_target_contract(public)

    result = evaluate_public_targets(_candidate("Uii + Uid"), contract)

    assert result.passed
    assert result.mapped_target_fraction == 1.0
    assert result.required_dependency_fraction == 1.0
    assert result.assessed_representation_count == 2
    assert result.representation_consistency_fraction == 1.0


def test_total_disposal_contract_rejects_incomplete_identity_mapping() -> None:
    public = phase_b_public_spec(
        "dalla_man", "easy", "named", task="T2", data_root=DATA_ROOT
    )
    contract = phase_b_public_target_contract(public)

    result = evaluate_public_targets(_candidate("Uid"), contract)

    assert not result.passed
    failed = [item for item in result.predicates if item.status == "failed"]
    assert [item.predicate for item in failed] == [
        "required_dependency:insulin_independent_contribution"
    ]


def test_target_contract_rejects_passthrough_of_supplied_channel() -> None:
    public = phase_b_public_spec(
        "dalla_man", "easy", "named", task="T2", data_root=DATA_ROOT
    )
    contract = phase_b_public_target_contract(public)
    payload = _candidate("Uii + Uid").model_dump(mode="json")
    for mapping in payload["observation_mappings"]:
        if mapping["channel"] == "U":
            mapping["expression"] = "Uii"

    result = evaluate_public_targets(
        CandidateModel.model_validate(payload), contract
    )

    assert not result.passed
    assert any(
        item.target_channel == "U"
        and item.predicate == "generated_model_path"
        and item.status == "failed"
        for item in result.predicates
    )


def test_rate_target_cannot_be_recast_as_an_integrated_state() -> None:
    public = phase_b_public_spec(
        "dalla_man", "easy", "named", task="T2", data_root=DATA_ROOT
    )
    contract = phase_b_public_target_contract(public)
    payload = _candidate("Uii + Uid").model_dump(mode="json")
    payload["processes"] = [
        item for item in payload["processes"] if item["name"] != "U"
    ]
    payload["states"].append({"name": "U", "kind": "observed"})
    payload["state_equations"].append(
        {"state": "U", "rhs": "Uii + k_u * X * Gp"}
    )
    payload["initial_conditions"].append(
        {"state": "U", "scope": "global", "fixed_value": 0.0}
    )

    result = evaluate_public_targets(
        CandidateModel.model_validate(payload), contract
    )

    assert not result.passed
    assert result.representation_consistency_fraction == 0.5
    assert any(
        item.target_channel == "U"
        and item.predicate == "target_representation:instantaneous_process"
        and item.status == "failed"
        for item in result.predicates
    )


def test_legacy_contract_without_public_role_does_not_infer_one() -> None:
    contract = PublicTargetContract.model_validate(
        {
            "schema_version": "public-target-contract-1",
            "benchmark_id": "legacy",
            "tier": "easy",
            "public_prompt_sha256": "0" * 64,
            "targets": [
                {
                    "target_channel": "U",
                    "public_requirement": "generate U",
                }
            ],
        }
    )
    payload = _candidate("Uii + Uid").model_dump(mode="json")
    payload["observation_mappings"] = [
        item for item in payload["observation_mappings"] if item["channel"] == "U"
    ]

    result = evaluate_public_targets(
        CandidateModel.model_validate(payload), contract
    )

    assert result.assessed_representation_count == 0
    assert result.representation_consistency_fraction is None


def test_structured_target_role_requires_public_provenance() -> None:
    with pytest.raises(ValidationError, match="public provenance"):
        PublicTargetContract.model_validate(
            {
                "schema_version": "public-target-contract-2",
                "benchmark_id": "invalid",
                "tier": "easy",
                "public_prompt_sha256": "0" * 64,
                "targets": [
                    {
                        "target_channel": "U",
                        "public_requirement": "generate U",
                        "expected_representation": "instantaneous_process",
                    }
                ],
            }
        )


def test_structured_target_role_requires_v2_contract() -> None:
    with pytest.raises(ValidationError, match="contract-2"):
        PublicTargetContract.model_validate(
            {
                "schema_version": "public-target-contract-1",
                "benchmark_id": "invalid",
                "tier": "easy",
                "public_prompt_sha256": "0" * 64,
                "targets": [
                    {
                        "target_channel": "U",
                        "public_requirement": "generate U",
                        "expected_representation": "instantaneous_process",
                        "representation_requirement": "U is a rate",
                    }
                ],
            }
        )


def test_all_phase_b_target_contracts_are_public_prompt_committed() -> None:
    suite = load_suite_spec(SUITE_PATH)
    contracts = []
    for family in suite.families:
        for task in family.tasks:
            for condition in family.dynamics_conditions:
                for tier in family.tiers:
                    for variant in family.semantic_variants:
                        public = phase_b_public_spec(
                            family.family,
                            tier.name,
                            variant,
                            task=task if family.family == "dalla_man" else None,
                            dynamics=(
                                "canonical"
                                if condition == "not_applicable"
                                else condition
                            ),
                            data_root=DATA_ROOT,
                        )
                        contract = phase_b_public_target_contract(public)
                        proposer, _ = render_phase_b_prompts(public)
                        assert contract.public_prompt_sha256 == hashlib.sha256(
                            proposer.encode("utf-8")
                        ).hexdigest()
                        assert {item.target_channel for item in contract.targets} == {
                            item.public_name
                            for item in public.channels
                            if item.role == "target"
                        }
                        contracts.append(contract)

    assert len(contracts) == 40
    composition_contracts = [
        item
        for item in contracts
        if any(
            target.required_dependencies
            for target in item.targets
        )
    ]
    assert {item.benchmark_id for item in composition_contracts} == {
        "phase_b_dalla_man_t2_canonical_named_easy",
        "phase_b_dalla_man_t2_perturbed_named_easy",
        "phase_b_dalla_man_t3_canonical_named_easy",
        "phase_b_dalla_man_t3_perturbed_named_easy",
    }


def test_committed_target_contracts_match_current_public_prompts() -> None:
    manifest = json.loads(
        (GENERATED_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    paths = sorted((GENERATED_ROOT / "specs").glob("*.json"))

    assert manifest["contract_count"] == len(paths) == 40
    for record in manifest["contracts"]:
        path = GENERATED_ROOT / record["path"]
        payload = path.read_bytes()
        contract = PublicTargetContract.model_validate_json(payload)
        assert contract.benchmark_id == record["benchmark_id"]
        assert hashlib.sha256(payload).hexdigest() == record["contract_sha256"]


def test_role_aware_contract_bundle_matches_public_generator() -> None:
    manifest = json.loads(
        (ROLE_AWARE_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    paths = sorted((ROLE_AWARE_ROOT / "specs").glob("*.json"))
    expected = {
        item.benchmark_id: item
        for item in (
            phase_b_public_target_contract(public)
            for public in _all_public_specs()
        )
    }
    observed = {
        contract.benchmark_id: contract
        for path in paths
        for contract in (
            PublicTargetContract.model_validate_json(path.read_bytes()),
        )
    }

    assert manifest["schema_version"] == "phase-b-public-target-contracts-2"
    assert manifest["contract_schema_version"] == "public-target-contract-2"
    assert manifest["contract_count"] == len(paths) == len(expected) == 40
    assert observed == expected
    assert sum(
        target.expected_representation.value != "unspecified"
        for contract in observed.values()
        for target in contract.targets
    ) > 0


def _all_public_specs():
    suite = load_suite_spec(SUITE_PATH)
    for family in suite.families:
        for task in family.tasks:
            for condition in family.dynamics_conditions:
                for tier in family.tiers:
                    for variant in family.semantic_variants:
                        yield phase_b_public_spec(
                            family.family,
                            tier.name,
                            variant,
                            task=task if family.family == "dalla_man" else None,
                            dynamics=(
                                "canonical"
                                if condition == "not_applicable"
                                else condition
                            ),
                            data_root=DATA_ROOT,
                        )
