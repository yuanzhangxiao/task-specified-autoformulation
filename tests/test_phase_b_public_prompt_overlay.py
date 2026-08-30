"""Tests for the production Phase-B public-prompt overlay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from autoformalism.benchmarks import (
    load_suite_spec,
    phase_b_public_spec,
    render_phase_b_prompts,
)
from autoformalism.targets import PublicTargetContract
from scripts.prepare_phase_b_public_prompt_overlay import prepare_overlay

OVERLAY_CONFIG = Path("configs/phase_b_public_prompt_overlay_v3.json")
TARGET_CONTRACT_ROOT = Path("configs/target_eval/phase_b_v1")
SUITE_PATH = Path("configs/benchmarks/phase_b_suite_v1.json")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_contract_bundle(
    root: Path,
    prompts: dict[str, str],
) -> str:
    specs = root / "specs"
    specs.mkdir(parents=True)
    records = []
    for benchmark_id, prompt in sorted(prompts.items()):
        contract = PublicTargetContract.model_validate(
            {
                "benchmark_id": benchmark_id,
                "tier": "easy",
                "public_prompt_sha256": _sha256(prompt.encode()),
                "targets": [
                    {
                        "target_channel": "target",
                        "public_requirement": "generate target",
                    }
                ],
            }
        )
        payload = (
            json.dumps(contract.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n"
        ).encode()
        relative = Path("specs") / f"{benchmark_id}.json"
        (root / relative).write_bytes(payload)
        records.append(
            {
                "benchmark_id": benchmark_id,
                "contract_sha256": _sha256(payload),
                "path": str(relative),
                "public_prompt_sha256": contract.public_prompt_sha256,
            }
        )
    manifest = {
        "contract_count": len(records),
        "contracts": records,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    (root / "manifest.json").write_bytes(manifest_bytes)
    return _sha256(manifest_bytes)


def _write_public_cell(root: Path, benchmark_id: str, prompt: str) -> None:
    cell = root / "phase_b_v1" / benchmark_id
    cell.mkdir(parents=True)
    (cell / "proposer_prompt.txt").write_text(prompt, encoding="utf-8")
    (cell / "judge_prompt.txt").write_text("unchanged judge", encoding="utf-8")
    (cell / "train.csv").write_text(
        "trajectory_id,t,target\ntrain_000,0,1\n", encoding="utf-8"
    )
    (cell / "validation.csv").write_text(
        "trajectory_id,t,target\nvalidation_000,0,1\n", encoding="utf-8"
    )
    (cell / "test.csv").write_text(
        "trajectory_id,t,target\ntest_000,0,1\n", encoding="utf-8"
    )
    (cell / "manifest.json").write_text(
        json.dumps(
            {
                "benchmark_id": benchmark_id,
                "status": "production_registered",
                "test_sealed": True,
            }
        ),
        encoding="utf-8",
    )


def _write_config(
    path: Path,
    *,
    contract_manifest_sha256: str,
    revised_id: str,
    revised_prompt: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "phase-b-public-prompt-overlay-config-1",
                "suite_version": "phase_b_v1",
                "target_contract_manifest_sha256": contract_manifest_sha256,
                "revisions": [
                    {
                        "benchmark_ids": [revised_id],
                        "expected_source_prompt_sha256": _sha256(
                            b"Public task: old target wording.\n"
                        ),
                        "expected_revised_prompt_sha256": _sha256(
                            revised_prompt.encode()
                        ),
                        "replacements": [
                            {
                                "old": "old target wording",
                                "new": "revised target wording",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, str, str]:
    revised_id = "phase_b_revision_test"
    unchanged_id = "phase_b_unchanged_test"
    source = tmp_path / "source"
    output = tmp_path / "overlay"
    contract_root = tmp_path / "contracts"
    config = tmp_path / "config.json"
    source_prompt = "Public task: old target wording.\n"
    revised_prompt = "Public task: revised target wording.\n"
    unchanged_prompt = "Public task: already final.\n"
    _write_public_cell(source, revised_id, source_prompt)
    _write_public_cell(source, unchanged_id, unchanged_prompt)
    manifest_sha256 = _write_contract_bundle(
        contract_root,
        {revised_id: revised_prompt, unchanged_id: unchanged_prompt},
    )
    _write_config(
        config,
        contract_manifest_sha256=manifest_sha256,
        revised_id=revised_id,
        revised_prompt=revised_prompt,
    )
    return source, output, contract_root, config, revised_id, unchanged_id


def test_overlay_changes_only_reviewed_prompts_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source, output, contract_root, config, revised_id, unchanged_id = _fixture(
        tmp_path
    )
    source_files = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    first = prepare_overlay(
        source_data_root=source,
        output_data_root=output,
        contract_root=contract_root,
        config_path=config,
    )
    second = prepare_overlay(
        source_data_root=source,
        output_data_root=output,
        contract_root=contract_root,
        config_path=config,
    )

    assert first == second
    assert first["changed_prompt_count"] == 1
    assert first["changed_benchmark_ids"] == [revised_id]
    assert first["non_proposer_files_byte_identical"] is True
    assert (
        output / "phase_b_v1" / revised_id / "proposer_prompt.txt"
    ).read_text() == "Public task: revised target wording.\n"
    assert (
        output / "phase_b_v1" / unchanged_id / "proposer_prompt.txt"
    ).read_text() == "Public task: already final.\n"
    assert source_files == {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    for relative, source_bytes in source_files.items():
        if relative.name != "proposer_prompt.txt":
            assert (output / relative).read_bytes() == source_bytes


def test_existing_overlay_rejects_non_prompt_tampering(tmp_path: Path) -> None:
    source, output, contract_root, config, revised_id, _ = _fixture(tmp_path)
    prepare_overlay(
        source_data_root=source,
        output_data_root=output,
        contract_root=contract_root,
        config_path=config,
    )
    (output / "phase_b_v1" / revised_id / "train.csv").write_text(
        "tampered\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="non-proposer file"):
        prepare_overlay(
            source_data_root=source,
            output_data_root=output,
            contract_root=contract_root,
            config_path=config,
        )


def test_overlay_rejects_unreviewed_prompt_mismatch(tmp_path: Path) -> None:
    source, output, contract_root, config, _, unchanged_id = _fixture(tmp_path)
    (source / "phase_b_v1" / unchanged_id / "proposer_prompt.txt").write_text(
        "unexpected prompt\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="outside the reviewed revision"):
        prepare_overlay(
            source_data_root=source,
            output_data_root=output,
            contract_root=contract_root,
            config_path=config,
        )


def test_overlay_rejects_staging_source(tmp_path: Path) -> None:
    source, output, contract_root, config, revised_id, _ = _fixture(tmp_path)
    manifest_path = source / "phase_b_v1" / revised_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "staging_not_registered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not a production release"):
        prepare_overlay(
            source_data_root=source,
            output_data_root=output,
            contract_root=contract_root,
            config_path=config,
        )


def test_frozen_overlay_config_matches_target_contract_bundle() -> None:
    config = json.loads(OVERLAY_CONFIG.read_text(encoding="utf-8"))
    manifest = (TARGET_CONTRACT_ROOT / "manifest.json").read_bytes()
    revision_ids = {
        benchmark_id
        for revision in config["revisions"]
        for benchmark_id in revision["benchmark_ids"]
    }

    assert _sha256(manifest) == config["target_contract_manifest_sha256"]
    assert revision_ids == {
        "phase_b_dalla_man_t1_canonical_named_easy",
        "phase_b_dalla_man_t1_perturbed_named_easy",
        "phase_b_dalla_man_t2_canonical_named_easy",
        "phase_b_dalla_man_t2_perturbed_named_easy",
        "phase_b_dalla_man_t2_canonical_named_hard",
        "phase_b_dalla_man_t2_perturbed_named_hard",
        "phase_b_dalla_man_t3_canonical_named_easy",
        "phase_b_dalla_man_t3_perturbed_named_easy",
        "phase_b_dalla_man_t4_canonical_named_easy",
        "phase_b_dalla_man_t4_perturbed_named_easy",
    }
    for revision in config["revisions"]:
        for benchmark_id in revision["benchmark_ids"]:
            contract = PublicTargetContract.model_validate_json(
                (
                    TARGET_CONTRACT_ROOT / "specs" / f"{benchmark_id}.json"
                ).read_bytes()
            )
            assert (
                contract.public_prompt_sha256
                == revision["expected_revised_prompt_sha256"]
            )


def test_full_overlay_revises_only_ten_reviewed_cells(tmp_path: Path) -> None:
    config = json.loads(OVERLAY_CONFIG.read_text(encoding="utf-8"))
    revisions = {
        benchmark_id: revision["replacements"]
        for revision in config["revisions"]
        for benchmark_id in revision["benchmark_ids"]
    }
    source = tmp_path / "source"
    output = tmp_path / "overlay"
    suite = load_suite_spec(SUITE_PATH)
    expected_prompts: dict[str, str] = {}
    for family in suite.families:
        for task in family.tasks:
            for condition in family.dynamics_conditions:
                dynamics = (
                    "canonical" if condition == "not_applicable" else condition
                )
                for tier in family.tiers:
                    for variant in family.semantic_variants:
                        spec = phase_b_public_spec(
                            family.family,
                            tier.name,
                            variant,
                            task=task if family.family == "dalla_man" else None,
                            dynamics=dynamics,
                            data_root=Path("data_raw"),
                        )
                        prompt = render_phase_b_prompts(spec)[0]
                        expected_prompts[spec.benchmark_id] = prompt
                        source_prompt = prompt
                        for replacement in reversed(
                            revisions.get(spec.benchmark_id, [])
                        ):
                            source_prompt = source_prompt.replace(
                                replacement["new"], replacement["old"]
                            )
                        _write_public_cell(
                            source, spec.benchmark_id, source_prompt
                        )

    manifest = prepare_overlay(
        source_data_root=source,
        output_data_root=output,
        contract_root=TARGET_CONTRACT_ROOT,
        config_path=OVERLAY_CONFIG,
    )

    assert manifest["cell_count"] == 40
    assert manifest["changed_prompt_count"] == 10
    assert set(manifest["changed_benchmark_ids"]) == set(revisions)
    for benchmark_id, prompt in expected_prompts.items():
        assert (
            output / "phase_b_v1" / benchmark_id / "proposer_prompt.txt"
        ).read_text(encoding="utf-8") == prompt
