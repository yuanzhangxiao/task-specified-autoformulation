"""Refit frozen failed candidates using development data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

import pandas as pd

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.execution import ExecutionArguments, _context
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import FitConfig, fit_candidate
from autoformalism.schemas import CandidateModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark", default="benchmark5")
    parser.add_argument("--tier", default="hard")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--screen-max-nfev", type=int, default=1)
    parser.add_argument("--screen-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--refine-top-k", type=int, default=3)
    parser.add_argument("--refine-max-nfev", type=int, default=10)
    parser.add_argument("--refine-timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(
        args.artifact_root, args.benchmark, args.tier, args.seed
    )
    loader = BenchmarkLoader(BenchmarkRegistry())
    data_config = DataConfig(
        root=args.data_root.resolve(),
        benchmark_id=args.benchmark,
        tier=args.tier,
    )
    development = loader.load_development(data_config)
    arguments = ExecutionArguments(
        data_root=data_config.root,
        benchmark_id=args.benchmark,
        tier=args.tier,
        seed=args.seed,
        proposer_model=None,
        judge_model=None,
        iteration_budget=1,
        beam_size=1,
        output_root=args.output_root,
        resume=False,
        dry_run=False,
        mock_llm=True,
        use_clean_observations=False,
    )
    context = _context(arguments, development)
    screen_path = args.output_root / "screening_results.jsonl"
    completed = _completed_keys(screen_path)
    for item in candidates:
        for profile in ("declared_midpoint", "scale_aware_near_zero"):
            key = f"{item['structural_hash']}:{profile}"
            if key in completed:
                continue
            candidate = CandidateModel.model_validate(item["candidate"])
            row = _fit_one(
                candidate,
                item,
                profile,
                development,
                context,
                args.seed,
                args.screen_max_nfev,
                args.screen_timeout_seconds,
            )
            _append_jsonl(screen_path, row)
            completed.add(key)
            print(
                f"screen {len(completed)}/{2 * len(candidates)} "
                f"{candidate.candidate_id} {profile} "
                f"success={row['success']} val={row['validation_nmse']}",
                flush=True,
            )

    screening = _read_jsonl(screen_path)
    successful = [row for row in screening if row["success"]]
    successful.sort(key=lambda row: (row["validation_nmse"], row["key"]))
    finalists = successful[: args.refine_top_k]
    refinement_path = args.output_root / "refinement_results.jsonl"
    refined_keys = _completed_keys(refinement_path)
    by_hash = {item["structural_hash"]: item for item in candidates}
    for finalist in finalists:
        key = finalist["key"]
        if key in refined_keys:
            continue
        item = by_hash[finalist["structural_hash"]]
        candidate = CandidateModel.model_validate(item["candidate"])
        row = _fit_one(
            candidate,
            item,
            finalist["initialization_profile"],
            development,
            context,
            args.seed,
            args.refine_max_nfev,
            args.refine_timeout_seconds,
            preferred_parameters=finalist["global_parameters"],
        )
        _append_jsonl(refinement_path, row)
        refined_keys.add(key)
        print(
            f"refine {len(refined_keys)}/{len(finalists)} "
            f"{candidate.candidate_id} success={row['success']} "
            f"val={row['validation_nmse']}",
            flush=True,
        )

    refinement = _read_jsonl(refinement_path)
    eligible = [row for row in refinement if row["success"]]
    if not eligible:
        eligible = successful
    if not eligible:
        raise RuntimeError("no frozen candidate passed development refitting")
    selected = min(eligible, key=lambda row: (row["validation_nmse"], row["key"]))
    selected_item = by_hash[selected["structural_hash"]]
    manifest = {
        "schema_version": "1",
        "stage": "development_selection_frozen",
        "benchmark_id": args.benchmark,
        "tier": args.tier,
        "seed": args.seed,
        "uses_test_data": False,
        "selection_metric": "validation_normalized_mse",
        "candidate_id": selected["candidate_id"],
        "structural_hash": selected["structural_hash"],
        "source_checkpoint": selected["source_checkpoint"],
        "candidate": selected_item["candidate"],
        "initialization_profile": selected["initialization_profile"],
        "global_parameters": selected["global_parameters"],
        "training_normalized_mse": selected["training_nmse"],
        "validation_normalized_mse": selected["validation_nmse"],
        "screen_max_nfev": args.screen_max_nfev,
        "refine_max_nfev": args.refine_max_nfev,
    }
    (args.output_root / "frozen_development_selection.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary(args.output_root, screening, refinement, manifest)


def collect_candidates(
    root: Path, benchmark: str, tier: str, seed: int
) -> list[dict]:
    name = f"{benchmark}_{tier}_seed{seed}"
    records: dict[str, dict] = {}
    for path in sorted(root.rglob(f"{name}/checkpoints/round_*.json")):
        if any(part.startswith("noj-") for part in path.parts):
            continue
        payload = _read_json(path)
        structural_hash = payload.get("structural_hash")
        candidate = payload.get("candidate")
        if not structural_hash or not candidate:
            continue
        records.setdefault(
            structural_hash,
            {
                "structural_hash": structural_hash,
                "candidate": candidate,
                "source_checkpoint": str(path),
            },
        )
    if not records:
        raise ValueError(f"no failed candidates found for {name}")
    return [records[key] for key in sorted(records)]


def initial_parameters(candidate: CandidateModel, profile: str) -> dict[str, float]:
    values = {}
    for parameter in candidate.parameters:
        if parameter.initialization_range is None:
            values[parameter.name] = 1.0
            continue
        lower = parameter.initialization_range.lower
        upper = parameter.initialization_range.upper
        if profile == "declared_midpoint":
            value = (lower + upper) / 2.0
        elif profile == "scale_aware_near_zero":
            if lower <= 0.0 <= upper:
                value = 0.0
            elif lower > 0.0:
                value = lower
            else:
                value = upper
        else:
            raise ValueError(f"unknown initialization profile: {profile}")
        values[parameter.name] = value
    return values


def _fit_one(
    candidate,
    item,
    profile,
    development,
    context,
    seed,
    max_nfev,
    timeout,
    *,
    preferred_parameters=None,
):
    model = compile_candidate(candidate, context)
    settings = FitConfig(
        number_of_starts=1,
        random_seed=seed,
        integration_backend="fixed_rk4",
        fixed_step_substeps=1,
        maximum_function_evaluations=max_nfev,
        maximum_wall_time_seconds=timeout,
    )
    preferred = preferred_parameters or initial_parameters(candidate, profile)
    started = monotonic()
    fitted = fit_candidate(
        model,
        development.train,
        development.validation,
        settings,
        initial_global_parameters=preferred,
    )
    return {
        "key": f"{item['structural_hash']}:{profile}",
        "structural_hash": item["structural_hash"],
        "candidate_id": candidate.candidate_id,
        "source_checkpoint": item["source_checkpoint"],
        "initialization_profile": profile,
        "success": fitted.success,
        "elapsed_seconds": monotonic() - started,
        "training_nmse": fitted.training_metrics.normalized_mse,
        "validation_nmse": fitted.validation_metrics.normalized_mse,
        "global_parameters": dict(fitted.global_parameters),
        "function_evaluations": sum(
            item.function_evaluations for item in fitted.diagnostics
        ),
        "integration_failures": sum(
            item.integration_failures for item in fitted.diagnostics
        ),
        "message": fitted.message,
    }


def _write_summary(root, screening, refinement, manifest):
    pd.DataFrame(screening).to_csv(root / "screening_results.csv", index=False)
    pd.DataFrame(refinement).to_csv(root / "refinement_results.csv", index=False)
    lines = [
        "# Frozen failed-candidate refit",
        "",
        "No test data were loaded. Structures and numerical settings were selected "
        "using validation NMSE only.",
        "",
        f"- screened structures: {len({row['structural_hash'] for row in screening})}",
        f"- successful screening fits: {sum(row['success'] for row in screening)}",
        f"- successful refinement fits: {sum(row['success'] for row in refinement)}",
        f"- selected candidate: `{manifest['candidate_id']}`",
        f"- initialization: `{manifest['initialization_profile']}`",
        f"- training NMSE: {manifest['training_normalized_mse']:.6g}",
        f"- validation NMSE: {manifest['validation_normalized_mse']:.6g}",
        "",
    ]
    (root / "refit_report.md").write_text("\n".join(lines), encoding="utf-8")


def _completed_keys(path: Path) -> set[str]:
    return {str(row["key"]) for row in _read_jsonl(path)}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


if __name__ == "__main__":
    main()
