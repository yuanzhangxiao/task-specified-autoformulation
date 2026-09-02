#!/usr/bin/env python3
"""Freeze final classical baseline models before any sealed test access."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from autoformalism.baselines.core import baseline_validation_context
from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry, DevelopmentDataset
from autoformalism.rebuttal.baseline_pilot import BaselinePilotTask
from autoformalism.rebuttal.baseline_postfreeze import (
    FrozenBaselineModel,
    freeze_baseline_model,
)
from autoformalism.rebuttal.final_evaluation_adapters import SourceAdapterRequest


def main() -> None:
    """Validate the readiness freeze and apply public-only final-fit rules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-freeze-root", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = finalize_models(
        args.development_freeze_root,
        args.public_data_root,
        args.output_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def finalize_models(
    development_freeze_root: Path,
    public_data_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Create content-addressed final models from frozen development selections."""
    source = development_freeze_root.expanduser().resolve()
    public = public_data_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    source_manifest_path = source / "development_result_freeze.json"
    source_manifest_sha = _sha256(source_manifest_path)
    _validate_digest_file(
        source / "development_result_freeze.json.sha256",
        source_manifest_path,
    )
    source_manifest = _read_object(source_manifest_path)
    if (
        source_manifest.get("status")
        != "frozen_before_test_or_oracle_evaluation"
        or source_manifest.get("test_data_opened") is not False
        or source_manifest.get("private_reference_opened") is not False
        or source_manifest.get("oracle_derivatives_used") is not False
        or source_manifest.get("oracle_latent_states_used") is not False
        or source_manifest.get("task_count") != 360
        or source_manifest.get("selected_result_count") != 360
        or set(source_manifest.get("methods", []))
        != {"persistence", "sindy", "pysr"}
    ):
        raise ValueError("development readiness freeze is not the 360-task matrix")
    _validate_artifact_ledger(source)
    tasks = _read_tasks(source / "inputs" / "task_plan.jsonl")
    if len(tasks) != 360 or {item.task_index for item in tasks} != set(range(360)):
        raise ValueError("readiness task plan is not contiguous and complete")
    plan = _read_object(source / "inputs" / "plan.json")
    cells = {
        (str(item["benchmark_id"]), str(item["tier"])): str(
            item["public_prompt_sha256"]
        )
        for item in plan.get("cells", [])
        if isinstance(item, dict)
    }
    if len(cells) != 40:
        raise ValueError("readiness plan does not contain 40 public cells")

    registry = BenchmarkRegistry()
    loader = BenchmarkLoader(registry)
    development_cache: dict[tuple[str, str], DevelopmentDataset] = {}
    models: list[FrozenBaselineModel] = []
    for task in tasks:
        key = (task.benchmark_id, task.tier)
        if key not in development_cache:
            prompt = public / "phase_b_v1" / task.benchmark_id / "proposer_prompt.txt"
            if _sha256(prompt) != cells.get(key):
                raise ValueError(f"public prompt differs: {task.benchmark_id}")
            development_cache[key] = loader.load_development(
                DataConfig(
                    root=public,
                    benchmark_id=task.benchmark_id,
                    tier=task.tier,
                )
            )
        development = development_cache[key]
        context = baseline_validation_context(
            development,
            registry.get(task.benchmark_id),
        )
        source_result = source / "tasks" / f"task_{task.task_index:03d}.json"
        result = BaselineDevelopmentResult.model_validate_json(
            source_result.read_text(encoding="utf-8")
        )
        if (
            result.method,
            result.benchmark_id,
            result.tier,
            result.seed,
        ) != (
            task.method,
            task.benchmark_id,
            task.tier,
            task.repetition,
        ):
            raise ValueError(f"task identity differs: {task.task_index}")
        model = freeze_baseline_model(
            task_index=task.task_index,
            result=result,
            development=development,
            context=context,
            source_development_result_sha256=_sha256(source_result),
            source_development_freeze_sha256=source_manifest_sha,
        )
        model_path = output / "models" / f"task_{task.task_index:03d}.json"
        _write_once(model_path, model.model_dump_json(indent=2) + "\n")
        models.append(model)

    models_path = output / "frozen_baseline_models.jsonl"
    _write_once(
        models_path,
        "".join(item.model_dump_json() + "\n" for item in models),
    )
    requests = tuple(
        SourceAdapterRequest(
            request_id=f"classical-{item.task_index:03d}",
            source_kind=item.method,
            source_path=output / "models" / f"task_{item.task_index:03d}.json",
            expected_benchmark_id=item.benchmark_id,
            expected_tier=item.tier,
            expected_repetition=item.seed,
        )
        for item in models
        if item.method in {"sindy", "pysr"}
    )
    requests_path = output / "source_adapter_requests.jsonl"
    _write_once(
        requests_path,
        "".join(item.model_dump_json() + "\n" for item in requests),
    )
    manifest = {
        "schema_version": "phase-b-public-baseline-final-model-freeze-1",
        "status": "frozen_before_test_or_private_evaluation",
        "model_count": len(models),
        "symbolic_subject_request_count": len(requests),
        "method_counts": {
            method: sum(item.method == method for item in models)
            for method in ("persistence", "sindy", "pysr")
        },
        "development_refit_policy": {
            "persistence": "not_applicable",
            "sindy": "refit_selected_threshold_on_train_plus_validation",
            "pysr": "preserve_selected_equations_without_new_search",
        },
        "development_freeze_sha256": source_manifest_sha,
        "frozen_models_sha256": _sha256(models_path),
        "source_adapter_requests_sha256": _sha256(requests_path),
        "test_data_opened": False,
        "private_reference_opened": False,
        "oracle_derivatives_used": False,
        "oracle_latent_states_used": False,
        "weighted_overall_score_defined": False,
    }
    manifest_path = output / "final_model_freeze.json"
    _write_once(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_once(
        output / "final_model_freeze.json.sha256",
        f"{_sha256(manifest_path)}  final_model_freeze.json\n",
    )
    return manifest


def _read_tasks(path: Path) -> tuple[BaselinePilotTask, ...]:
    return tuple(
        BaselinePilotTask.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _validate_artifact_ledger(root: Path) -> None:
    ledger = root / "artifact_ledger.jsonl"
    manifest = _read_object(root / "development_result_freeze.json")
    if _sha256(ledger) != manifest.get("artifact_ledger_sha256"):
        raise ValueError("development artifact ledger hash differs")
    for line_number, line in enumerate(
        ledger.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        item = json.loads(line)
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or _sha256(path) != item.get("sha256"):
            raise ValueError(
                f"development artifact differs at ledger line {line_number}"
            )
        if path.stat().st_size != item.get("size_bytes"):
            raise ValueError(f"development artifact size differs at line {line_number}")


def _validate_digest_file(digest_path: Path, target: Path) -> None:
    fields = digest_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != _sha256(target):
        raise ValueError("development freeze digest file differs")


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_once(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"frozen artifact differs: {path}")
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
