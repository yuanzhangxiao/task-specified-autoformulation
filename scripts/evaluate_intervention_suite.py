#!/usr/bin/env python3
"""Evaluate frozen model artifacts on a generated private intervention suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.data.scaling import TrainingScaler
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.intervention_evaluation import (
    evaluate_frozen_model,
    load_frozen_model,
)
from autoformalism.rebuttal.interventions import (
    load_intervention_suite,
    load_system_spec,
    simulate_reference,
)


def _context_and_scale(
    data_root: Path, benchmark_id: str, tier: str
) -> tuple[ValidationContext, float]:
    registry = BenchmarkRegistry()
    spec = registry.get(benchmark_id)
    development = BenchmarkLoader(registry).load_development(
        DataConfig(
            root=data_root,
            benchmark_id=benchmark_id,
            tier=tier,
            use_clean_observations=(benchmark_id == "benchmark5"),
        )
    )
    context = ValidationContext(
        targets=development.roles.targets,
        auxiliaries=development.roles.auxiliaries,
        external_inputs=spec.external_inputs,
        fixed_covariates=spec.fixed_covariates,
        lagged_targets=(
            development.roles.targets if spec.one_step_target_history else ()
        ),
    )
    target = context.targets[0]
    scale = (
        TrainingScaler()
        .fit(development.train)
        .scales[f"target:{target}"]
        .standard_deviation
    )
    return context, scale


def _system_spec(data_root: Path, benchmark_id: str) -> dict[str, Any]:
    if benchmark_id not in {"benchmark5", "benchmark6"}:
        return {}
    _, spec = load_system_spec(data_root, benchmark_id)
    if benchmark_id == "benchmark6":
        mapping_path = data_root / "benchmark6_alien_device/private/secret_mapping.json"
        spec["secret_mapping"] = json.loads(mapping_path.read_text(encoding="utf-8"))
    return spec


def evaluate_suite(
    *,
    suite_path: Path,
    data_root: Path,
    tier: str,
    model_specs: list[str],
    reset_observed_states: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate label/path specifications and return JSON-serializable rows."""

    suite = load_intervention_suite(suite_path)
    parsed_models: list[tuple[str, Path]] = []
    for item in model_specs:
        try:
            label, encoded_path = item.split("=", 1)
        except ValueError as exc:
            raise ValueError("--model must use LABEL=PATH") from exc
        parsed_models.append((label, Path(encoded_path)))

    cached: dict[str, tuple[ValidationContext, float, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for case in suite.cases:
        benchmark = case.benchmark_id
        if benchmark not in cached:
            context, scale = _context_and_scale(data_root, benchmark, tier)
            cached[benchmark] = (context, scale, _system_spec(data_root, benchmark))
        context, scale, system_spec = cached[benchmark]
        reference = simulate_reference(case, system_spec=system_spec)
        for label, path in parsed_models:
            if f"/{benchmark}/" not in f"/{label}/" and not label.startswith(
                f"{benchmark}:"
            ):
                continue
            target = context.targets[0]
            try:
                model = load_frozen_model(path, target=target)
                result = evaluate_frozen_model(
                    model,
                    case=case,
                    reference=reference,
                    context=context,
                    tier=tier,
                    system_spec=system_spec,
                    fallback_target_scale=scale,
                    reset_observed_states=reset_observed_states,
                )
                row = result.model_dump()
                row["model_label"] = label
                row["evaluation_protocol"] = (
                    "one_step_reset" if reset_observed_states else "free_rollout"
                )
            except Exception as exc:
                row = {
                    "model_label": label,
                    "source": str(path),
                    "case_id": case.case_id,
                    "benchmark_id": benchmark,
                    "success": False,
                    "target_mse": None,
                    "target_nmse": None,
                    "in_distribution_nmse": None,
                    "nmse_degradation_ratio": None,
                    "message": f"artifact adapter failed: {exc}",
                }
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tier", choices=("easy", "medium", "hard"), default="hard")
    parser.add_argument(
        "--free-rollout",
        action="store_true",
        help="Propagate observed states without resetting them from measurements.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="label must begin with its registered benchmark ID and a colon",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = evaluate_suite(
        suite_path=args.suite,
        data_root=args.data_root,
        tier=args.tier,
        model_specs=args.model,
        reset_observed_states=not args.free_rollout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    successes = sum(bool(row["success"]) for row in rows)
    print(
        f"evaluations={len(rows)} successful={successes} failed={len(rows) - successes}"
    )


if __name__ == "__main__":
    main()
