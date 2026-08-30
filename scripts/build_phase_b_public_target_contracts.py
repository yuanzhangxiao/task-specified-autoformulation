#!/usr/bin/env python3
"""Build prompt-committed public target contracts for Phase B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autoformalism.benchmarks import (
    load_suite_spec,
    phase_b_public_spec,
    phase_b_public_target_contract,
)
from autoformalism.targets import PublicTargetContract


def build_contracts(
    suite_path: Path,
    data_root: Path,
) -> tuple[PublicTargetContract, ...]:
    """Derive every full-factorial cell from public benchmark contracts."""
    suite = load_suite_spec(suite_path)
    result: list[PublicTargetContract] = []
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
                        result.append(phase_b_public_target_contract(public_spec))
    if len(result) != suite.number_of_cells:
        raise RuntimeError(
            f"expected {suite.number_of_cells} contracts, got {len(result)}"
        )
    identifiers = [item.benchmark_id for item in result]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("generated benchmark identifiers are not unique")
    return tuple(result)


def write_bundle(
    contracts: tuple[PublicTargetContract, ...],
    *,
    suite_path: Path,
    output_root: Path,
) -> None:
    """Write stable contracts and their provenance manifest."""
    specs_root = output_root / "specs"
    specs_root.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{item.benchmark_id}.json" for item in contracts}
    stale = {path.name for path in specs_root.glob("*.json")} - expected_names
    if stale:
        raise ValueError(f"refusing to ignore stale contracts: {sorted(stale)}")

    records: list[dict[str, object]] = []
    for contract in contracts:
        payload = (
            json.dumps(contract.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n"
        )
        path = specs_root / f"{contract.benchmark_id}.json"
        path.write_text(payload, encoding="utf-8")
        records.append(
            {
                "benchmark_id": contract.benchmark_id,
                "tier": contract.tier,
                "target_count": len(contract.targets),
                "dependency_count": sum(
                    len(item.required_dependencies) for item in contract.targets
                ),
                "public_prompt_sha256": contract.public_prompt_sha256,
                "contract_sha256": hashlib.sha256(
                    payload.encode("utf-8")
                ).hexdigest(),
                "path": str(path.relative_to(output_root)),
            }
        )
    manifest = {
        "schema_version": "phase-b-public-target-contracts-1",
        "status": "frozen_public_prompt_derived",
        "suite_path": str(suite_path),
        "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "contract_count": len(contracts),
        "contracts": records,
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
        default=Path("configs/target_eval/phase_b_v1"),
    )
    args = parser.parse_args()
    contracts = build_contracts(args.suite, args.data_root)
    write_bundle(contracts, suite_path=args.suite, output_root=args.output_root)
    print(
        f"wrote {len(contracts)} public target contracts to "
        f"{(args.output_root / 'specs').resolve()}"
    )


if __name__ == "__main__":
    main()
