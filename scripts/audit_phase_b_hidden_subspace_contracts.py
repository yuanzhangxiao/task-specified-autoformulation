#!/usr/bin/env python3
"""Audit private Phase-B response-subspace ranks before method evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry, FrozenTestAccess
from autoformalism.data.scaling import TrainingScaler
from autoformalism.rebuttal.phase_b_hidden_subspace import (
    PhaseBHiddenSubspaceContract,
    phase_b_hidden_subspace_contract,
    phase_b_reference_directions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", action="append", required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--private-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark_ids = tuple(dict.fromkeys(args.benchmark_id))
    selection_hash = hashlib.sha256("\n".join(benchmark_ids).encode()).hexdigest()
    registry = BenchmarkRegistry()
    loader = BenchmarkLoader(registry)
    rows = []
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for benchmark_id in benchmark_ids:
        tier = benchmark_id.rsplit("_", maxsplit=1)[-1]
        if tier not in {"easy", "hard"}:
            rows.append(
                _failure_row(
                    benchmark_id,
                    ValueError(
                        f"cannot infer tier from benchmark ID: {benchmark_id}"
                    ),
                )
            )
            continue
        contract = None
        try:
            config = DataConfig(
                root=args.public_data_root.expanduser().resolve(),
                benchmark_id=benchmark_id,
                tier=tier,
            )
            development = loader.load_development(config)
            test = loader.load_test(
                config,
                access=FrozenTestAccess(
                    benchmark_id=benchmark_id,
                    tier=tier,
                    selection_hash=selection_hash,
                ),
            )
            contract = phase_b_hidden_subspace_contract(
                benchmark_id,
                data_root=args.private_data_root.expanduser().resolve(),
            )
            if contract.mode == "not_applicable":
                rows.append(
                    {
                        **_contract_identity(contract),
                        "audit_status": "not_applicable",
                        "claimed_dimension": 0,
                        "train_rank_at_1e3": None,
                        "test_rank_at_1e3": None,
                        "claimed_rank_pass": True,
                        "train_condition_number": None,
                        "train_matrix_sha256": None,
                        "test_matrix_sha256": None,
                        "error_type": None,
                        "error": None,
                    }
                )
                continue
            fitted = TrainingScaler().fit(development.train).scales
            scales = {
                target: float(fitted[f"target:{target}"].standard_deviation)
                for target in contract.target_sources
            }
            train, heldout = phase_b_reference_directions(
                training_split=development.train,
                test_split=test,
                contract=contract,
                normalization_scales=scales,
                private_data_root=args.private_data_root,
            )
            train_singular = np.linalg.svd(train, compute_uv=False)
            test_singular = np.linalg.svd(heldout, compute_uv=False)
            train_rank = _relative_rank(train_singular)
            test_rank = _relative_rank(test_singular)
            claimed_index = contract.claimed_dimension - 1
            condition = float(
                train_singular[0]
                / max(float(train_singular[claimed_index]), 1e-15)
            )
            rows.append(
                {
                    **_contract_identity(contract),
                    "audit_status": "pass",
                    "claimed_dimension": contract.claimed_dimension,
                    "train_rank_at_1e3": train_rank,
                    "test_rank_at_1e3": test_rank,
                    "claimed_rank_pass": train_rank >= contract.claimed_dimension,
                    "train_condition_number": condition,
                    "train_matrix_sha256": _array_sha256(train),
                    "test_matrix_sha256": _array_sha256(heldout),
                    "error_type": None,
                    "error": None,
                }
            )
            matrices[benchmark_id] = (train, heldout)
            print(
                f"{benchmark_id}: dimension={contract.claimed_dimension} "
                f"train_rank={train_rank} test_rank={test_rank} "
                f"condition={condition:.3g}",
                flush=True,
            )
        except Exception as error:
            rows.append(_failure_row(benchmark_id, error, contract=contract))
            print(
                f"{benchmark_id}: FAILED {type(error).__name__}: {error}",
                flush=True,
            )
    semantic_checks = _semantic_pair_checks(rows, matrices)
    payload = {
        "schema_version": "phase-b-hidden-subspace-contract-audit-1",
        "status": "pass"
        if all(item["claimed_rank_pass"] for item in rows)
        and all(item["identical"] for item in semantic_checks)
        else "fail",
        "benchmark_ids": benchmark_ids,
        "failed_benchmark_count": sum(
            item["audit_status"] == "failed" for item in rows
        ),
        "claimed_rank_pass": all(item["claimed_rank_pass"] for item in rows),
        "semantic_pair_identity_pass": all(
            item["identical"] for item in semantic_checks
        ),
        "rows": rows,
        "semantic_pair_checks": semantic_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(
        args.output,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    print(f"overall hidden contract audit: {payload['status']}")
    if payload["status"] != "pass":
        raise SystemExit(1)


def _contract_identity(
    contract: PhaseBHiddenSubspaceContract,
) -> dict[str, object]:
    return {
        "benchmark_id": contract.benchmark_id,
        "family": contract.family,
        "task": contract.task,
        "tier": contract.tier,
        "dynamics": contract.dynamics,
        "mode": contract.mode,
    }


def _failure_row(
    benchmark_id: str,
    error: Exception,
    *,
    contract: PhaseBHiddenSubspaceContract | None = None,
) -> dict[str, object]:
    identity = (
        _contract_identity(contract)
        if contract is not None
        else {
            "benchmark_id": benchmark_id,
            "family": None,
            "task": None,
            "tier": None,
            "dynamics": None,
            "mode": None,
        }
    )
    return {
        **identity,
        "audit_status": "failed",
        "claimed_dimension": None,
        "train_rank_at_1e3": None,
        "test_rank_at_1e3": None,
        "claimed_rank_pass": False,
        "train_condition_number": None,
        "train_matrix_sha256": None,
        "test_matrix_sha256": None,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _relative_rank(singular_values: np.ndarray) -> int:
    leading = max(float(singular_values[0]), 1e-15)
    return int(np.count_nonzero(singular_values / leading >= 1e-3))


def _array_sha256(values: np.ndarray) -> str:
    normalized = np.asarray(values, dtype="<f8")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _semantic_pair_checks(
    rows: list[dict[str, object]],
    matrices: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[str]] = {}
    for row in rows:
        benchmark_id = str(row["benchmark_id"])
        if row["family"] is None:
            continue
        key = (
            str(row["family"]),
            str(row["task"]),
            str(row["tier"]),
            str(row["dynamics"]),
        )
        grouped.setdefault(key, []).append(benchmark_id)
    checks = []
    for identifiers in grouped.values():
        if len(identifiers) != 2:
            continue
        first, second = sorted(identifiers)
        if first not in matrices or second not in matrices:
            checks.append(
                {
                    "first": first,
                    "second": second,
                    "identical": False,
                    "error": "one or both semantic-pair matrices are unavailable",
                }
            )
            continue
        left_train, left_test = matrices[first]
        right_train, right_test = matrices[second]
        checks.append(
            {
                "first": first,
                "second": second,
                "identical": bool(
                    np.array_equal(left_train, right_train)
                    and np.array_equal(left_test, right_test)
                ),
            }
        )
    return checks


def _write_text_atomic(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


if __name__ == "__main__":
    main()
