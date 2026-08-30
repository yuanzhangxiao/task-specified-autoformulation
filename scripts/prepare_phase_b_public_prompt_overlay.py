#!/usr/bin/env python3
"""Prepare and verify the versioned Phase-B public-prompt overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoformalism.targets import PublicTargetContract

_OVERLAY_MANIFEST = "prompt_overlay_manifest.json"


@dataclass(frozen=True)
class PromptRevision:
    """One reviewed exact-text revision shared by named benchmark cells."""

    expected_source_sha256: str | None
    expected_sha256: str
    replacements: tuple[tuple[str, str], ...]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_digest(items: dict[str, str]) -> str:
    payload = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(payload)


def _file_inventory(root: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"public release contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root)
            if any(
                part.lower() == "private" or part.lower().startswith("hidden")
                for part in relative.parts
            ):
                raise ValueError(
                    f"public release contains a forbidden path: {relative}"
                )
            inventory[str(relative)] = _sha256(path.read_bytes())
    return inventory


def _non_prompt_inventory(inventory: dict[str, str]) -> dict[str, str]:
    return {
        path: digest
        for path, digest in inventory.items()
        if Path(path).name != "proposer_prompt.txt"
    }


def _load_revisions(config: dict[str, Any]) -> dict[str, PromptRevision]:
    revisions: dict[str, PromptRevision] = {}
    for record in config.get("revisions", []):
        expected_source = record.get("expected_source_prompt_sha256")
        expected = str(record["expected_revised_prompt_sha256"])
        replacements = tuple(
            (str(item["old"]), str(item["new"]))
            for item in record["replacements"]
        )
        if not replacements:
            raise ValueError("prompt revision must contain at least one replacement")
        for benchmark_id in record["benchmark_ids"]:
            identifier = str(benchmark_id)
            if identifier in revisions:
                raise ValueError(f"duplicate prompt revision: {identifier}")
            revisions[identifier] = PromptRevision(
                None if expected_source is None else str(expected_source),
                expected,
                replacements,
            )
    return revisions


def _load_contracts(
    contract_root: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[dict[str, PublicTargetContract], str]:
    manifest_path = contract_root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = _sha256(manifest_bytes)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "target-contract manifest differs from the frozen overlay config: "
            f"{manifest_sha256}"
        )
    manifest = json.loads(manifest_bytes)
    contracts: dict[str, PublicTargetContract] = {}
    for record in manifest["contracts"]:
        path = contract_root / str(record["path"])
        payload = path.read_bytes()
        if _sha256(payload) != record["contract_sha256"]:
            raise ValueError(f"target contract differs from its manifest: {path}")
        contract = PublicTargetContract.model_validate_json(payload)
        if contract.benchmark_id != record["benchmark_id"]:
            raise ValueError(f"target-contract identifier mismatch: {path}")
        if contract.public_prompt_sha256 != record["public_prompt_sha256"]:
            raise ValueError(f"target-contract prompt hash mismatch: {path}")
        if contract.benchmark_id in contracts:
            raise ValueError(f"duplicate target contract: {contract.benchmark_id}")
        contracts[contract.benchmark_id] = contract
    if len(contracts) != manifest["contract_count"]:
        raise ValueError("target-contract manifest count is inconsistent")
    return contracts, manifest_sha256


def _apply_revision(
    source: bytes,
    *,
    benchmark_id: str,
    expected_sha256: str,
    revision: PromptRevision | None,
) -> tuple[bytes, bool]:
    source_sha256 = _sha256(source)
    if source_sha256 == expected_sha256:
        return source, False
    if revision is None:
        raise ValueError(
            "unexpected proposer-prompt mismatch outside the reviewed revision: "
            f"benchmark={benchmark_id}, source_sha256={source_sha256}, "
            f"expected_sha256={expected_sha256}"
        )
    if revision.expected_sha256 != expected_sha256:
        raise ValueError(
            f"revision and target contract disagree for {benchmark_id}"
        )
    if (
        revision.expected_source_sha256 is not None
        and source_sha256 != revision.expected_source_sha256
    ):
        raise ValueError(
            "source proposer prompt differs from the frozen reviewed predecessor: "
            f"benchmark={benchmark_id}, source_sha256={source_sha256}, "
            f"expected_source_sha256={revision.expected_source_sha256}"
        )
    revised = source.decode("utf-8")
    for old, new in revision.replacements:
        old_count = revised.count(old)
        new_count = revised.count(new)
        if old_count != 1 or new_count != 0:
            raise ValueError(
                "reviewed prompt phrase is not present exactly once in its "
                f"unrevised form: benchmark={benchmark_id}, old={old!r}, "
                f"old_count={old_count}, new_count={new_count}"
            )
        revised = revised.replace(old, new)
    result = revised.encode("utf-8")
    result_sha256 = _sha256(result)
    if result_sha256 != expected_sha256:
        raise ValueError(
            "reviewed replacements did not produce the prompt committed by the "
            f"target contract: benchmark={benchmark_id}, "
            f"actual_sha256={result_sha256}, expected_sha256={expected_sha256}"
        )
    return result, True


def _validate_source_cells(
    suite_root: Path,
    expected_ids: set[str],
) -> None:
    if not suite_root.is_dir():
        raise FileNotFoundError(f"public suite does not exist: {suite_root}")
    actual_ids = {path.name for path in suite_root.iterdir() if path.is_dir()}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"public suite cell inventory differs; missing={missing}, extra={extra}"
        )
    for benchmark_id in sorted(expected_ids):
        cell_root = suite_root / benchmark_id
        manifest_path = cell_root / "manifest.json"
        prompt_path = cell_root / "proposer_prompt.txt"
        if not manifest_path.is_file() or not prompt_path.is_file():
            raise ValueError(f"public cell is incomplete: {benchmark_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("benchmark_id") != benchmark_id:
            raise ValueError(
                f"public cell manifest identifier mismatch: {benchmark_id}"
            )
        if manifest.get("status") != "production_registered":
            raise ValueError(f"public cell is not a production release: {benchmark_id}")
        if not manifest.get("test_sealed"):
            raise ValueError(
                f"public production cell has no sealed test: {benchmark_id}"
            )


def _build_overlay_manifest(
    *,
    config: dict[str, Any],
    source_data_root: Path,
    source_inventory: dict[str, str],
    overlay_inventory: dict[str, str],
    contract_manifest_sha256: str,
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    source_non_prompt = _non_prompt_inventory(source_inventory)
    overlay_non_prompt = _non_prompt_inventory(overlay_inventory)
    if source_non_prompt != overlay_non_prompt:
        changed = sorted(
            set(source_non_prompt) ^ set(overlay_non_prompt)
            | {
                path
                for path in set(source_non_prompt) & set(overlay_non_prompt)
                if source_non_prompt[path] != overlay_non_prompt[path]
            }
        )
        raise ValueError(f"overlay changed non-proposer files: {changed}")
    return {
        "schema_version": "phase-b-public-prompt-overlay-1",
        "status": "ready",
        "suite_version": config["suite_version"],
        "source_data_root": str(source_data_root.resolve()),
        "target_contract_manifest_sha256": contract_manifest_sha256,
        "only_proposer_prompts_may_differ": True,
        "non_proposer_files_byte_identical": True,
        "source_inventory_sha256": _canonical_digest(source_inventory),
        "overlay_inventory_sha256": _canonical_digest(overlay_inventory),
        "non_proposer_inventory_sha256": _canonical_digest(source_non_prompt),
        "cell_count": len(cells),
        "changed_prompt_count": sum(bool(item["changed"]) for item in cells),
        "changed_benchmark_ids": [
            item["benchmark_id"] for item in cells if item["changed"]
        ],
        "cells": cells,
    }


def _verify_existing_overlay(
    *,
    source_suite_root: Path,
    output_data_root: Path,
    expected_manifest: dict[str, Any],
    prompts: dict[str, bytes],
) -> None:
    manifest_path = output_data_root / _OVERLAY_MANIFEST
    if not manifest_path.is_file():
        raise ValueError("existing overlay has no prompt-overlay manifest")
    actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual_manifest != expected_manifest:
        raise ValueError("existing prompt-overlay manifest differs")
    output_suite_root = output_data_root / expected_manifest["suite_version"]
    source_inventory = _file_inventory(source_suite_root)
    output_inventory = _file_inventory(output_suite_root)
    if set(source_inventory) != set(output_inventory):
        raise ValueError("existing overlay file inventory differs from source")
    if _non_prompt_inventory(source_inventory) != _non_prompt_inventory(
        output_inventory
    ):
        raise ValueError("existing overlay changed a non-proposer file")
    for benchmark_id, expected in prompts.items():
        path = output_suite_root / benchmark_id / "proposer_prompt.txt"
        if path.read_bytes() != expected:
            raise ValueError(f"existing overlay prompt differs: {benchmark_id}")


def prepare_overlay(
    *,
    source_data_root: Path,
    output_data_root: Path,
    contract_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Create or verify a full copy-on-write prompt overlay."""
    source_data_root = source_data_root.expanduser().resolve()
    output_data_root = output_data_root.expanduser().resolve()
    contract_root = contract_root.expanduser().resolve()
    if source_data_root == output_data_root:
        raise ValueError("prompt overlay must not overwrite its source release")
    if source_data_root in output_data_root.parents:
        raise ValueError("prompt overlay must not be nested inside its source release")
    if output_data_root in source_data_root.parents:
        raise ValueError("source release must not be nested inside the overlay")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "phase-b-public-prompt-overlay-config-1":
        raise ValueError("unsupported prompt-overlay configuration")
    contracts, contract_manifest_sha256 = _load_contracts(
        contract_root,
        expected_manifest_sha256=str(config["target_contract_manifest_sha256"]),
    )
    revisions = _load_revisions(config)
    unknown_revisions = set(revisions) - set(contracts)
    if unknown_revisions:
        raise ValueError(
            f"prompt revisions have no target contract: {unknown_revisions}"
        )

    suite_version = str(config["suite_version"])
    source_suite_root = source_data_root / suite_version
    _validate_source_cells(source_suite_root, set(contracts))
    source_inventory = _file_inventory(source_suite_root)
    prompts: dict[str, bytes] = {}
    cells: list[dict[str, Any]] = []
    prompt_errors: list[str] = []
    for benchmark_id, contract in sorted(contracts.items()):
        source_path = source_suite_root / benchmark_id / "proposer_prompt.txt"
        source_prompt = source_path.read_bytes()
        try:
            revised_prompt, changed = _apply_revision(
                source_prompt,
                benchmark_id=benchmark_id,
                expected_sha256=contract.public_prompt_sha256,
                revision=revisions.get(benchmark_id),
            )
        except ValueError as exc:
            prompt_errors.append(str(exc))
            continue
        prompts[benchmark_id] = revised_prompt
        cells.append(
            {
                "benchmark_id": benchmark_id,
                "source_prompt_sha256": _sha256(source_prompt),
                "overlay_prompt_sha256": _sha256(revised_prompt),
                "changed": changed,
            }
        )
    if prompt_errors:
        details = "\n- ".join(prompt_errors)
        raise ValueError(f"public prompt overlay preflight failed:\n- {details}")

    overlay_inventory = dict(source_inventory)
    for benchmark_id, prompt in prompts.items():
        relative = f"{benchmark_id}/proposer_prompt.txt"
        overlay_inventory[relative] = _sha256(prompt)
    expected_manifest = _build_overlay_manifest(
        config=config,
        source_data_root=source_data_root,
        source_inventory=source_inventory,
        overlay_inventory=overlay_inventory,
        contract_manifest_sha256=contract_manifest_sha256,
        cells=cells,
    )

    if output_data_root.exists():
        _verify_existing_overlay(
            source_suite_root=source_suite_root,
            output_data_root=output_data_root,
            expected_manifest=expected_manifest,
            prompts=prompts,
        )
        return expected_manifest

    output_data_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_data_root.name}.tmp-",
            dir=output_data_root.parent,
        )
    )
    try:
        temporary_suite_root = temporary / suite_version
        shutil.copytree(source_suite_root, temporary_suite_root)
        for benchmark_id, prompt in prompts.items():
            path = temporary_suite_root / benchmark_id / "proposer_prompt.txt"
            path.write_bytes(prompt)
        actual_overlay_inventory = _file_inventory(temporary_suite_root)
        if actual_overlay_inventory != overlay_inventory:
            raise ValueError("new overlay inventory differs from its frozen plan")
        (temporary / _OVERLAY_MANIFEST).write_text(
            json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_data_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    _verify_existing_overlay(
        source_suite_root=source_suite_root,
        output_data_root=output_data_root,
        expected_manifest=expected_manifest,
        prompts=prompts,
    )
    return expected_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--output-data-root", type=Path, required=True)
    parser.add_argument(
        "--target-contract-root",
        type=Path,
        default=Path("configs/target_eval/phase_b_v1"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase_b_public_prompt_overlay_v3.json"),
    )
    args = parser.parse_args()
    manifest = prepare_overlay(
        source_data_root=args.source_data_root,
        output_data_root=args.output_data_root,
        contract_root=args.target_contract_root,
        config_path=args.config,
    )
    print(
        "verified Phase-B public prompt overlay: "
        f"cells={manifest['cell_count']} "
        f"changed_prompts={manifest['changed_prompt_count']} "
        f"output={args.output_data_root.resolve()}"
    )


if __name__ == "__main__":
    main()
