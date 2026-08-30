#!/usr/bin/env python3
"""Freeze the 30-call GPT-5.6 refresh for prompts changed by public v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

_SCHEMA = "raw-data-agent-fitted-model-prompt-v3-refresh-1"
_SUITE = "phase_b_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_text_once(path: Path, content: str) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite frozen artifact: {path}")
    path.write_text(content, encoding="utf-8")


def freeze_prompt_v3_refresh(
    *,
    config_path: Path,
    output_root: Path,
    public_data_root: Path,
    overlay_config_path: Path,
    source_full_config_path: Path,
) -> dict[str, Any]:
    """Validate public inputs and freeze one task row per refresh call."""
    config_path = config_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    public_data_root = public_data_root.expanduser().resolve()
    overlay_config_path = overlay_config_path.expanduser().resolve()
    source_full_config_path = source_full_config_path.expanduser().resolve()
    config = _read_object(config_path)
    if config.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported GPT-5.6 prompt-refresh configuration")
    if config.get("status") != "frozen_before_refresh_calls":
        raise ValueError("prompt-refresh configuration is not frozen")
    if config.get("provider") != "openai" or config.get("model") != "gpt-5.6-sol":
        raise ValueError("prompt refresh must use the frozen GPT-5.6-sol provider")
    if config.get("output_contract") != "fitted_model":
        raise ValueError("prompt refresh must request a fitted model")
    repetitions = config.get("repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise ValueError("repetitions must be an integer")
    if repetitions != 3:
        raise ValueError("prompt refresh must retain three repetitions")
    expected_source_sha = str(config["source_full_protocol_config_sha256"])
    if _sha256(source_full_config_path) != expected_source_sha:
        raise ValueError("source full-agent protocol differs from the frozen refresh")
    if _sha256(overlay_config_path) != str(
        config["prompt_overlay_config_sha256"]
    ):
        raise ValueError("prompt-overlay configuration differs from the refresh")

    overlay_manifest_path = public_data_root / "prompt_overlay_manifest.json"
    if _sha256(overlay_manifest_path) != str(
        config["prompt_overlay_manifest_sha256"]
    ):
        raise ValueError("prompt-overlay manifest differs from the frozen refresh")
    overlay = _read_object(overlay_manifest_path)
    if overlay.get("status") != "ready":
        raise ValueError("public prompt overlay is not ready")
    if overlay.get("suite_version") != _SUITE:
        raise ValueError("public prompt overlay has the wrong suite version")
    if overlay.get("non_proposer_files_byte_identical") is not True:
        raise ValueError("public prompt overlay changed non-proposer files")
    if overlay.get("target_contract_manifest_sha256") != config.get(
        "target_contract_manifest_sha256"
    ):
        raise ValueError("prompt overlay and refresh target contracts differ")

    benchmarks = config.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("refresh has no benchmark cells")
    benchmark_ids = [str(item["benchmark_id"]) for item in benchmarks]
    if len(benchmark_ids) != len(set(benchmark_ids)):
        raise ValueError("refresh benchmark identifiers are not unique")
    if set(benchmark_ids) != set(overlay.get("changed_benchmark_ids", [])):
        raise ValueError("refresh cells do not exactly match changed v3 prompts")
    overlay_cells = {
        str(item["benchmark_id"]): item for item in overlay.get("cells", [])
    }

    tasks: list[dict[str, Any]] = []
    input_cells: list[dict[str, Any]] = []
    suite_root = public_data_root / _SUITE
    for item in benchmarks:
        benchmark_id = str(item["benchmark_id"])
        tier = str(item["tier"])
        if tier not in {"easy", "hard"}:
            raise ValueError(f"invalid tier for {benchmark_id}: {tier}")
        cell_root = suite_root / benchmark_id
        prompt_path = cell_root / "proposer_prompt.txt"
        train_path = cell_root / "train.csv"
        validation_path = cell_root / "validation.csv"
        for path in (prompt_path, train_path, validation_path):
            if not path.is_file():
                raise ValueError(f"missing public development input: {path}")
        prompt_sha = _sha256(prompt_path)
        if prompt_sha != str(item["public_prompt_sha256"]):
            raise ValueError(f"public prompt differs for {benchmark_id}")
        overlay_cell = overlay_cells.get(benchmark_id)
        if (
            not isinstance(overlay_cell, dict)
            or overlay_cell.get("changed") is not True
        ):
            raise ValueError(f"overlay does not mark prompt changed: {benchmark_id}")
        if overlay_cell.get("overlay_prompt_sha256") != prompt_sha:
            raise ValueError(f"overlay prompt digest differs for {benchmark_id}")
        train_sha = _sha256(train_path)
        validation_sha = _sha256(validation_path)
        input_cells.append(
            {
                "benchmark_id": benchmark_id,
                "tier": tier,
                "public_prompt_sha256": prompt_sha,
                "train_sha256": train_sha,
                "validation_sha256": validation_sha,
            }
        )
        for repetition in range(repetitions):
            tasks.append(
                {
                    "task_id": len(tasks),
                    "benchmark_id": benchmark_id,
                    "tier": tier,
                    "repetition": repetition,
                    "public_prompt_sha256": prompt_sha,
                    "train_sha256": train_sha,
                    "validation_sha256": validation_sha,
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "plan.json"
    task_path = output_root / "task_plan.jsonl"
    manifest_path = output_root / "freeze_manifest.json"
    plan_bytes = config_path.read_bytes()
    task_text = "".join(
        json.dumps(task, sort_keys=True, separators=(",", ":")) + "\n"
        for task in tasks
    )
    manifest = {
        "schema_version": "raw-data-agent-prompt-v3-refresh-freeze-1",
        "status": "frozen_before_refresh_calls",
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "cell_count": len(input_cells),
        "repetitions": repetitions,
        "task_count": len(tasks),
        "cells": input_cells,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "task_plan_sha256": hashlib.sha256(task_text.encode()).hexdigest(),
        "prompt_overlay_config_sha256": _sha256(overlay_config_path),
        "prompt_overlay_manifest_sha256": _sha256(overlay_manifest_path),
        "source_full_protocol_config_sha256": _sha256(source_full_config_path),
        "test_data_opened": False,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _write_text_once(plan_path, plan_bytes.decode("utf-8"))
    _write_text_once(task_path, task_text)
    _write_text_once(manifest_path, manifest_text)
    for path in (plan_path, task_path, manifest_path):
        _write_text_once(
            path.with_name(f"{path.name}.sha256"),
            f"{_sha256(path)}  {path.name}\n",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--overlay-config", type=Path, required=True)
    parser.add_argument("--source-full-config", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_prompt_v3_refresh(
        config_path=args.config,
        output_root=args.output_root,
        public_data_root=args.public_data_root,
        overlay_config_path=args.overlay_config,
        source_full_config_path=args.source_full_config,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
