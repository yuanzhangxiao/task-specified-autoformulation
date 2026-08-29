#!/usr/bin/env python3
"""Build prompt-committed public mechanism specifications for Phase B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autoformalism.benchmarks import (
    load_suite_spec,
    phase_b_public_spec,
)
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.rebuttal.phase_b_mechanism_specs import (
    phase_b_public_mechanism_spec,
)


def build_specs(
    suite_path: Path, data_root: Path
) -> tuple[MechanismEvaluationSpec, ...]:
    """Derive every full-factorial cell from public benchmark contracts."""
    suite = load_suite_spec(suite_path)
    result: list[MechanismEvaluationSpec] = []
    for family in suite.families:
        for task in family.tasks:
            task_argument = task if family.family == "dalla_man" else None
            for condition in family.dynamics_conditions:
                dynamics = "canonical" if condition == "not_applicable" else condition
                for tier in family.tiers:
                    for variant in family.semantic_variants:
                        public_spec = phase_b_public_spec(
                            family.family,
                            tier.name,
                            variant,
                            task=task_argument,
                            dynamics=dynamics,
                            data_root=data_root,
                        )
                        result.append(phase_b_public_mechanism_spec(public_spec))
    if len(result) != suite.number_of_cells:
        raise RuntimeError(
            f"expected {suite.number_of_cells} specifications, got {len(result)}"
        )
    identifiers = [item.benchmark_id for item in result]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("generated benchmark identifiers are not unique")
    return tuple(result)


def write_bundle(
    specs: tuple[MechanismEvaluationSpec, ...],
    *,
    suite_path: Path,
    output_root: Path,
) -> None:
    """Write stable specifications and a provenance manifest."""
    specs_root = output_root / "specs"
    specs_root.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{item.benchmark_id}.json" for item in specs}
    stale = {path.name for path in specs_root.glob("*.json")} - expected_names
    if stale:
        raise ValueError(
            f"refusing to ignore stale specification files: {sorted(stale)}"
        )

    records: list[dict[str, object]] = []
    for spec in specs:
        payload = (
            json.dumps(spec.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        path = specs_root / f"{spec.benchmark_id}.json"
        path.write_text(payload, encoding="utf-8")
        records.append(
            {
                "benchmark_id": spec.benchmark_id,
                "tier": spec.tier,
                "requirement_count": len(spec.required_mechanisms),
                "public_prompt_sha256": spec.public_prompt_sha256,
                "spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "path": str(path.relative_to(output_root)),
            }
        )
    manifest = {
        "schema_version": "phase-b-public-mechanism-specs-1",
        "status": "frozen_public_prompt_derived",
        "suite_path": str(suite_path),
        "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "specification_count": len(specs),
        "specifications": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("configs/benchmarks/phase_b_suite_v1.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("configs/mechanism_eval/phase_b_v1"),
    )
    args = parser.parse_args()
    specs = build_specs(args.suite, args.data_root)
    write_bundle(specs, suite_path=args.suite, output_root=args.output_root)
    print(
        f"wrote {len(specs)} public mechanism specifications to "
        f"{(args.output_root / 'specs').resolve()}"
    )


if __name__ == "__main__":
    main()
