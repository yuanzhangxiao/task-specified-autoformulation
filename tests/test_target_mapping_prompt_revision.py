"""Tests for the matched target-mapping public-prompt revision."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from autoformalism.data import BenchmarkRegistry
from scripts.prepare_target_mapping_prompt_revision import (
    prepare_overlay,
    revise_prompt,
)

CONFIG = Path("configs/hybrid_judge_target_mapping_prompt_revision_v3.json")
BENCHMARK_ID = "phase_b_dalla_man_t2_canonical_named_easy"


def test_revise_prompt_changes_only_three_frozen_phrases() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = Path(BENCHMARK_ID, "proposer_prompt.txt").read_text(encoding="utf-8")

    revised = revise_prompt(
        source, config["public_prompt_revision"]["replacements"]
    )

    assert "total glucose utilization/disposal rate" in revised
    assert "supplied insulin-independent contribution" in revised
    assert "insulin-dependent contribution to total glucose disposal U(t)" in revised
    assert hashlib.sha256(revised.encode()).hexdigest() == config[
        "public_prompt_revision"
    ]["revised_prompt_sha256"]


def test_revise_prompt_rejects_missing_or_repeated_source_phrase() -> None:
    replacement = [{"old": "old", "new": "new"}]

    with pytest.raises(ValueError, match="count=0"):
        revise_prompt("different", replacement)
    with pytest.raises(ValueError, match="count=2"):
        revise_prompt("old old", replacement)


def test_prepare_overlay_preserves_every_non_prompt_file(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    relative_root = BenchmarkRegistry().get(BENCHMARK_ID).relative_root
    source_root = tmp_path / "source" / relative_root
    source_root.parent.mkdir(parents=True)
    shutil.copytree(Path(BENCHMARK_ID), source_root)
    pair_path = tmp_path / "pairs.jsonl"
    pair_path.write_text(
        json.dumps({"benchmark_id": BENCHMARK_ID}) + "\n", encoding="utf-8"
    )
    pair_hash = hashlib.sha256(pair_path.read_bytes()).hexdigest()
    config["matched_control"]["source_pairs_sha256"] = pair_hash
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_root = tmp_path / "output"
    manifest_path = tmp_path / "prompt_revision_manifest.json"

    first = prepare_overlay(
        source_data_root=tmp_path / "source",
        output_data_root=output_root,
        pairs=pair_path,
        protocol_config=config_path,
        manifest_path=manifest_path,
    )
    second = prepare_overlay(
        source_data_root=tmp_path / "source",
        output_data_root=output_root,
        pairs=pair_path,
        protocol_config=config_path,
        manifest_path=manifest_path,
    )

    assert first == second
    copied_root = output_root / relative_root
    assert first["only_proposer_prompt_changed"] is True
    assert (
        copied_root / "proposer_prompt.txt"
    ).read_text() != (source_root / "proposer_prompt.txt").read_text()
    for source_file in source_root.iterdir():
        if source_file.name != "proposer_prompt.txt":
            copied_bytes = (copied_root / source_file.name).read_bytes()
            assert copied_bytes == source_file.read_bytes()
