#!/usr/bin/env python3
"""Evaluate one frozen classical baseline on the sealed public test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from autoformalism.baselines.core import baseline_validation_context, target_scales
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry, FrozenTestAccess
from autoformalism.rebuttal.baseline_postfreeze import (
    BaselinePredictiveTestResult,
    FrozenBaselineModel,
    evaluate_frozen_baseline_predictively,
)


def main() -> None:
    """Validate the full freeze before opening one task's test trajectories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-model-root", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    result = evaluate_task(
        args.final_model_root,
        args.public_data_root,
        args.output_root,
        task_index=args.task_index,
    )
    print(result.model_dump_json(indent=2))


def evaluate_task(
    final_model_root: Path,
    public_data_root: Path,
    output_root: Path,
    *,
    task_index: int,
) -> BaselinePredictiveTestResult:
    """Open test through a grant bound to the immutable final-model manifest."""
    frozen = final_model_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    manifest_path = frozen / "final_model_freeze.json"
    manifest_sha = _sha256(manifest_path)
    _validate_digest_file(frozen / "final_model_freeze.json.sha256", manifest_path)
    manifest = _read_object(manifest_path)
    if (
        manifest.get("status") != "frozen_before_test_or_private_evaluation"
        or manifest.get("model_count") != 360
        or manifest.get("test_data_opened") is not False
        or manifest.get("private_reference_opened") is not False
        or manifest.get("frozen_models_sha256")
        != _sha256(frozen / "frozen_baseline_models.jsonl")
    ):
        raise ValueError("final-model freeze is not eligible for sealed evaluation")
    models = _read_models(frozen / "frozen_baseline_models.jsonl")
    if len(models) != 360 or {item.task_index for item in models} != set(range(360)):
        raise ValueError("frozen final-model matrix is incomplete")
    if not 0 <= task_index < len(models):
        raise ValueError("task index lies outside the frozen model matrix")
    model = next(item for item in models if item.task_index == task_index)
    model_path = frozen / "models" / f"task_{task_index:03d}.json"
    model_sha = _sha256(model_path)
    if FrozenBaselineModel.model_validate_json(
        model_path.read_text(encoding="utf-8")
    ) != model:
        raise ValueError("task model differs from the frozen model ledger")

    result_path = output / "tasks" / f"task_{task_index:03d}.json"
    if result_path.is_file():
        existing = BaselinePredictiveTestResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        _validate_result_identity(existing, model, model_sha)
        return existing

    public = public_data_root.expanduser().resolve()
    registry = BenchmarkRegistry()
    loader = BenchmarkLoader(registry)
    data_config = DataConfig(
        root=public,
        benchmark_id=model.benchmark_id,
        tier=model.tier,
    )
    development = loader.load_development(data_config)
    context = baseline_validation_context(
        development,
        registry.get(model.benchmark_id),
    )
    if (
        development.train.fingerprint != model.train_fingerprint
        or development.validation.fingerprint != model.validation_fingerprint
        or target_scales(development.train, context.targets)
        != model.normalization_scales
    ):
        raise ValueError("public development data differs from the final-model freeze")
    _write_or_validate_receipt(
        output / "predictive_test_freeze_receipt.json",
        final_model_manifest_sha256=manifest_sha,
        frozen_models_sha256=str(manifest["frozen_models_sha256"]),
        model_count=len(models),
    )
    test = loader.load_test(
        data_config,
        access=FrozenTestAccess(
            benchmark_id=model.benchmark_id,
            tier=model.tier,
            selection_hash=manifest_sha,
        ),
    )
    result = evaluate_frozen_baseline_predictively(
        model,
        test,
        context,
        frozen_model_sha256=model_sha,
    )
    _validate_result_identity(result, model, model_sha)
    _write_once(result_path, result.model_dump_json(indent=2) + "\n")
    return result


def _read_models(path: Path) -> tuple[FrozenBaselineModel, ...]:
    return tuple(
        FrozenBaselineModel.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _validate_result_identity(
    result: BaselinePredictiveTestResult,
    model: FrozenBaselineModel,
    model_sha256: str,
) -> None:
    if (
        result.task_index,
        result.method,
        result.benchmark_id,
        result.tier,
        result.seed,
        result.frozen_model_sha256,
    ) != (
        model.task_index,
        model.method,
        model.benchmark_id,
        model.tier,
        model.seed,
        model_sha256,
    ):
        raise ValueError("predictive result identity differs from the frozen model")


def _write_or_validate_receipt(
    path: Path,
    *,
    final_model_manifest_sha256: str,
    frozen_models_sha256: str,
    model_count: int,
) -> None:
    payload = {
        "schema_version": "phase-b-baseline-predictive-test-freeze-receipt-1",
        "final_model_manifest_sha256": final_model_manifest_sha256,
        "frozen_models_sha256": frozen_models_sha256,
        "model_count": model_count,
        "all_models_validated_before_test_access": True,
        "test_data_opened_after_freeze": True,
        "private_reference_opened": False,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("predictive test freeze receipt differs")
        return
    _write_once(path, text)


def _validate_digest_file(digest_path: Path, target: Path) -> None:
    fields = digest_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != _sha256(target):
        raise ValueError("final-model digest file differs")


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_once(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"post-freeze artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required frozen artifact is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
