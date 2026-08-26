"""Prepare an audited prompt-only overlay for target-mapping calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from autoformalism.data import BenchmarkRegistry


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def revise_prompt(source: str, replacements: list[dict[str, str]]) -> str:
    """Apply exact one-occurrence replacements and reject ambiguous source text."""
    revised = source
    for replacement in replacements:
        old = replacement["old"]
        new = replacement["new"]
        count = revised.count(old)
        if count != 1:
            raise ValueError(
                f"prompt phrase must occur exactly once before replacement: "
                f"{old!r}; count={count}"
            )
        revised = revised.replace(old, new)
    return revised


def _pair_benchmark_ids(path: Path) -> set[str]:
    identifiers = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        benchmark_id = payload.get("benchmark_id")
        if not isinstance(benchmark_id, str):
            raise ValueError(f"pair line {line_number} has no benchmark_id")
        identifiers.add(benchmark_id)
    if not identifiers:
        raise ValueError("pair file is empty")
    return identifiers


def _verify_overlay_files(source: Path, output: Path, revised_prompt: bytes) -> None:
    source_files = {
        path.relative_to(source) for path in source.rglob("*") if path.is_file()
    }
    output_files = {
        path.relative_to(output) for path in output.rglob("*") if path.is_file()
    }
    if source_files != output_files:
        raise ValueError("prompt overlay file inventory differs from source release")
    for relative in source_files:
        actual = (output / relative).read_bytes()
        if relative == Path("proposer_prompt.txt"):
            if actual != revised_prompt:
                raise ValueError("existing overlay has a different proposer prompt")
        elif actual != (source / relative).read_bytes():
            raise ValueError(f"overlay changed a non-prompt file: {relative}")


def prepare_overlay(
    *,
    source_data_root: Path,
    output_data_root: Path,
    pairs: Path,
    protocol_config: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Copy one public cell and change only the frozen prompt phrases."""
    config = json.loads(protocol_config.read_text(encoding="utf-8"))
    revision = config["public_prompt_revision"]
    benchmark_id = str(revision["benchmark_id"])
    pair_bytes = pairs.read_bytes()
    pair_hash = _sha256_bytes(pair_bytes)
    expected_pair_hash = str(config["matched_control"]["source_pairs_sha256"])
    if pair_hash != expected_pair_hash:
        raise ValueError(
            f"pair SHA-256 differs from frozen control: {pair_hash}"
        )
    pair_benchmarks = _pair_benchmark_ids(pairs)
    if pair_benchmarks != {benchmark_id}:
        raise ValueError(
            f"pair benchmarks differ from prompt revision: {sorted(pair_benchmarks)}"
        )

    spec = BenchmarkRegistry().get(benchmark_id)
    source_root = source_data_root.resolve() / spec.relative_root
    output_root = output_data_root.resolve() / spec.relative_root
    if not source_root.is_dir():
        raise FileNotFoundError(f"source benchmark does not exist: {source_root}")
    source_prompt = (source_root / "proposer_prompt.txt").read_bytes()
    source_prompt_hash = _sha256_bytes(source_prompt)
    if source_prompt_hash != revision["source_prompt_sha256"]:
        raise ValueError(
            "source prompt SHA-256 differs from frozen matched control: "
            f"{source_prompt_hash}"
        )
    revised_text = revise_prompt(
        source_prompt.decode("utf-8"), list(revision["replacements"])
    )
    revised_prompt = revised_text.encode("utf-8")
    revised_prompt_hash = _sha256_bytes(revised_prompt)
    if revised_prompt_hash != revision["revised_prompt_sha256"]:
        raise ValueError(
            f"revised prompt SHA-256 differs from frozen value: {revised_prompt_hash}"
        )

    if output_root.exists():
        _verify_overlay_files(source_root, output_root, revised_prompt)
    else:
        output_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_root, output_root)
        (output_root / "proposer_prompt.txt").write_bytes(revised_prompt)
        _verify_overlay_files(source_root, output_root, revised_prompt)

    unchanged_files = {
        str(path.relative_to(source_root)): _sha256_bytes(path.read_bytes())
        for path in sorted(source_root.rglob("*"))
        if path.is_file() and path.name != "proposer_prompt.txt"
    }
    manifest = {
        "schema_version": "target-mapping-prompt-overlay-1",
        "status": "frozen_before_judge_calls",
        "benchmark_id": benchmark_id,
        "source_relative_root": str(spec.relative_root),
        "source_prompt_sha256": source_prompt_hash,
        "revised_prompt_sha256": revised_prompt_hash,
        "source_pairs_sha256": pair_hash,
        "only_proposer_prompt_changed": True,
        "unchanged_file_sha256": unchanged_files,
        "replacements": revision["replacements"],
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("existing prompt-overlay manifest differs")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--output-data-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_overlay(
        source_data_root=args.source_data_root,
        output_data_root=args.output_data_root,
        pairs=args.pairs,
        protocol_config=args.protocol_config,
        manifest_path=args.manifest,
    )
    print(
        "prepared prompt-only overlay for "
        f"{manifest['benchmark_id']}; prompt_sha256="
        f"{manifest['revised_prompt_sha256']}"
    )


if __name__ == "__main__":
    main()
