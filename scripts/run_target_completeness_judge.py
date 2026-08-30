"""Run the candidate-specific absolute target-completeness judge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.execution import (
    ExecutionArguments,
    _context,
    _prediction_protocol_prompt,
    _symbol_contract,
)
from autoformalism.expressions import ValidationContext, repair_protected_declarations
from autoformalism.judging import extract_public_requirements, structural_facts
from autoformalism.judging.prompts import (
    TARGET_COMPLETENESS_JUDGE_PROMPT,
    TARGET_COMPLETENESS_JUDGE_PROTOCOL_VERSION,
)
from autoformalism.llm import (
    LLMConfig,
    LLMProvider,
    VLLMReasoningEffort,
    create_llm_client,
)
from autoformalism.llm.exceptions import LLMError
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel

FIELDS = (
    "pair_id",
    "benchmark_id",
    "tier",
    "mutation_type",
    "judge_model",
    "repetition",
    "candidate_role",
    "candidate_id",
    "requested_target_ids",
    "target_assessments",
    "overall_verdict",
    "candidate_repairs",
    "response_transport",
    "provider_attempts",
    "successful_attempt_seed",
    "request_hash",
)
FAILURE_SCHEMA_VERSION = "target-completeness-judge-failure-1"
RUN_MANIFEST_SCHEMA_VERSION = "target-completeness-judge-run-1"


def _task_context(
    data_root: Path,
    pair: AdversarialPair,
) -> tuple[str, ValidationContext]:
    """Load the public prompt and development-only symbol context."""
    registry = BenchmarkRegistry()
    development = BenchmarkLoader(registry).load_development(
        DataConfig(root=data_root, benchmark_id=pair.benchmark_id, tier=pair.tier)
    )
    arguments = ExecutionArguments(
        data_root=data_root,
        benchmark_id=pair.benchmark_id,
        tier=pair.tier,
        seed=0,
        proposer_model=None,
        judge_model=None,
        iteration_budget=1,
        beam_size=1,
        output_root=Path("artifacts/rebuttal/target-completeness"),
        resume=False,
        dry_run=False,
        mock_llm=True,
        use_clean_observations=False,
    )
    context = _context(arguments, development)
    spec = registry.get(pair.benchmark_id)
    root = data_root / spec.relative_root
    if spec.data_layout == "legacy_split_files":
        root /= spec.tier_directory_template.format(tier=pair.tier)
    return (root / "proposer_prompt.txt").read_text(encoding="utf-8"), context


def target_completeness_system_prompt(
    public_prompt: str,
    context: ValidationContext,
    model: str,
) -> str:
    """Build the absolute, one-candidate target-completeness prompt."""
    return (
        f"Configured judge model: {model}\n\n"
        f"Public scientific task:\n{public_prompt}\n\n"
        f"{_prediction_protocol_prompt(context)}\n\n"
        f"{_symbol_contract(context)}\n\n"
        "Candidate-specific target-completeness protocol:\n"
        f"{TARGET_COMPLETENESS_JUDGE_PROMPT}"
    )


def _blinded_candidate_payload(candidate: CandidateModel) -> dict[str, object]:
    """Remove lineage and experiment labels from the provider-visible payload."""
    payload = candidate.model_dump(mode="json")
    payload["candidate_id"] = "candidate"
    payload["parent_candidate_id"] = None
    payload["change_summary"] = "unspecified"
    return payload


def target_completeness_request(
    candidate: CandidateModel,
    *,
    public_prompt: str,
    context: ValidationContext,
) -> dict[str, object]:
    """Return the provider-visible request for exactly one candidate."""
    target_ids = tuple(context.targets)
    return {
        "schema_version": "target-completeness-request-1",
        "public_requirement_registry": extract_public_requirements(
            public_prompt
        ).model_dump(mode="json"),
        "requested_target_ids": list(target_ids),
        "candidate": _blinded_candidate_payload(candidate),
        "deterministic_structural_facts": structural_facts(
            candidate,
            task_inputs=tuple(context.external_inputs),
            include_model_semantics=True,
            causal_observation_resets=bool(context.lagged_targets),
        ),
    }


def _select_shard(
    pairs: tuple[AdversarialPair, ...],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[AdversarialPair, ...]:
    """Select one deterministic contiguous pair shard."""
    start = len(pairs) * shard_index // shard_count
    stop = len(pairs) * (shard_index + 1) // shard_count
    return pairs[start:stop]


def _planned_keys(
    pairs: tuple[AdversarialPair, ...],
    *,
    judge_models: list[str],
    repetitions: int,
) -> set[tuple[str, str, int, str]]:
    """Return every candidate-specific logical call owned by this shard."""
    return {
        (pair.pair_id, model, repetition, candidate_role)
        for pair in pairs
        for model in judge_models
        for repetition in range(repetitions)
        for candidate_role in ("baseline", "mutated")
    }


def _completed(path: Path) -> set[tuple[str, str, int, str]]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (
                row["pair_id"],
                row["judge_model"],
                int(row["repetition"]),
                row["candidate_role"],
            )
            for row in csv.DictReader(handle)
        }


def _failed(path: Path) -> set[tuple[str, str, int, str]]:
    if not path.is_file():
        return set()
    keys: set[tuple[str, str, int, str]] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != FAILURE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported failure schema on line {line_number}: "
                f"{row.get('schema_version')!r}"
            )
        key = (
            str(row["pair_id"]),
            str(row["judge_model"]),
            int(row["repetition"]),
            str(row["candidate_role"]),
        )
        if key in keys:
            raise ValueError(f"duplicate persistent-failure key: {key}")
        keys.add(key)
    return keys


def _ensure_outcome_files(score_path: Path, failure_path: Path) -> None:
    """Create merge-safe empty outputs before any provider call."""
    if not score_path.exists():
        with score_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
    failure_path.touch(exist_ok=True)


def _append_score(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)


def _append_failure(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _ensure_run_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Create one immutable manifest or reject incompatible resume state."""
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(
                "target-completeness resume configuration differs from manifest"
            )
        return
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True)


def main() -> None:
    if os.environ.get("AF_TARGET_COMPLETENESS_PROTOCOL") == "paired_v8":
        if __package__:
            from scripts.run_paired_target_completeness_judge import (
                main as paired_main,
            )
        else:
            from run_paired_target_completeness_judge import main as paired_main

        paired_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--judge-models", nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-output-tokens", type=int, default=3072)
    parser.add_argument("--max-attempts", type=int, default=10, choices=range(1, 12))
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--vllm-reasoning-effort",
        choices=tuple(item.value for item in VLLMReasoningEffort),
        default=VLLMReasoningEffort.LOW.value,
    )
    parser.add_argument("--vllm-temperature", type=float, default=0.2)
    parser.add_argument("--vllm-seed-base", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard index must be in [0, shard count)")

    args.output_root.mkdir(parents=True, exist_ok=True)
    source_bytes = args.pairs.read_bytes()
    source_pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    )
    pairs = _select_shard(
        source_pairs,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    planned = _planned_keys(
        pairs,
        judge_models=args.judge_models,
        repetitions=args.repetitions,
    )
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "protocol_version": TARGET_COMPLETENESS_JUDGE_PROTOCOL_VERSION,
        "pairs_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "selected_pair_ids": [pair.pair_id for pair in pairs],
        "judge_models": args.judge_models,
        "repetitions": args.repetitions,
        "max_attempts": args.max_attempts,
        "vllm_reasoning_effort": args.vllm_reasoning_effort,
        "vllm_temperature": args.vllm_temperature,
        "vllm_seed_base": args.vllm_seed_base,
        "candidate_policy": "one_candidate_per_call",
        "target_policy": "all_public_targets",
        "atomic_stage_enabled": False,
        "comparative_questions_enabled": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    _ensure_run_manifest(
        args.output_root / "target_completeness_run_manifest.json", manifest
    )
    score_path = args.output_root / "target_completeness_scores.csv"
    failure_path = args.output_root / "target_completeness_failures.jsonl"
    _ensure_outcome_files(score_path, failure_path)
    successful = _completed(score_path)
    failed = _failed(failure_path)
    if successful & failed:
        raise SystemExit("keys occur in both success and failure outputs")
    completed = successful | failed
    if completed - planned:
        raise SystemExit("saved outcomes do not belong to the selected shard")

    contexts: dict[tuple[str, str], tuple[str, ValidationContext]] = {}
    for pair in pairs:
        key = (pair.benchmark_id, pair.tier)
        if key not in contexts:
            contexts[key] = _task_context(args.data_root.resolve(), pair)
    if args.dry_run:
        targets = {
            f"{benchmark_id}/{tier}": list(context.targets)
            for (benchmark_id, tier), (_prompt, context) in contexts.items()
        }
        print(
            f"pairs={len(pairs)} expected_calls={len(planned)} "
            f"successful={len(successful)} failed={len(failed)} "
            f"remaining={len(planned - completed)} targets={targets}"
        )
        return

    expected = len(planned)
    for model_spec in args.judge_models:
        provider_name, model = model_spec.split(":", 1)
        provider = LLMProvider(provider_name)
        storage_name = model.replace("/", "__")
        clients = tuple(
            create_llm_client(
                LLMConfig(
                    provider=provider,
                    model=model,
                    cache_directory=(
                        args.output_root
                        / "cache"
                        / provider.value
                        / storage_name
                        / f"repetition_{repetition}"
                    ),
                    log_path=(
                        args.output_root
                        / f"{provider.value}_{storage_name}_events.jsonl"
                    ),
                    max_attempts=args.max_attempts,
                    vllm_base_url=args.vllm_base_url,
                    vllm_reasoning_effort=VLLMReasoningEffort(
                        args.vllm_reasoning_effort
                    ),
                    vllm_temperature=args.vllm_temperature,
                    vllm_seed=args.vllm_seed_base + repetition,
                    timeout_seconds=args.timeout_seconds,
                    max_output_tokens=args.max_output_tokens,
                )
            )
            for repetition in range(args.repetitions)
        )
        for pair in pairs:
            public_prompt, context = contexts[(pair.benchmark_id, pair.tier)]
            system_prompt = target_completeness_system_prompt(
                public_prompt, context, model_spec
            )
            candidates = (
                ("baseline", pair.valid_candidate),
                ("mutated", pair.adversarial_candidate),
            )
            for candidate_role, source_candidate in candidates:
                candidate, repairs = repair_protected_declarations(
                    source_candidate, context
                )
                request = target_completeness_request(
                    candidate,
                    public_prompt=public_prompt,
                    context=context,
                )
                target_ids = set(context.targets)
                for repetition in range(args.repetitions):
                    key = (pair.pair_id, model_spec, repetition, candidate_role)
                    if key in completed:
                        continue
                    try:
                        result = clients[repetition].assess_target_completeness(
                            system_prompt=system_prompt,
                            user_prompt=json.dumps(request, sort_keys=True),
                            expected_target_ids=target_ids,
                        )
                    except LLMError as exc:
                        category = getattr(exc, "category", None)
                        _append_failure(
                            failure_path,
                            {
                                "schema_version": FAILURE_SCHEMA_VERSION,
                                "pair_id": pair.pair_id,
                                "benchmark_id": pair.benchmark_id,
                                "tier": pair.tier,
                                "mutation_type": pair.mutation_type,
                                "judge_model": model_spec,
                                "repetition": repetition,
                                "candidate_role": candidate_role,
                                "candidate_id": source_candidate.candidate_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:4000],
                                "failure_category": getattr(category, "value", None),
                                "provider_attempt_limit": args.max_attempts,
                            },
                        )
                        failed.add(key)
                        completed.add(key)
                        print(
                            f"failed {len(completed)}/{expected}: {key}: "
                            f"{type(exc).__name__}",
                            flush=True,
                        )
                        continue
                    retry = result.raw_response.get("_autoformalism_retry")
                    if not isinstance(retry, dict):
                        retry = {}
                    _append_score(
                        score_path,
                        {
                            "pair_id": pair.pair_id,
                            "benchmark_id": pair.benchmark_id,
                            "tier": pair.tier,
                            "mutation_type": pair.mutation_type,
                            "judge_model": model_spec,
                            "repetition": repetition,
                            "candidate_role": candidate_role,
                            "candidate_id": source_candidate.candidate_id,
                            "requested_target_ids": _json(sorted(target_ids)),
                            "target_assessments": _json(
                                [
                                    item.model_dump(mode="json")
                                    for item in result.parsed.target_assessments
                                ]
                            ),
                            "overall_verdict": result.parsed.overall_verdict.value,
                            "candidate_repairs": _json(repairs),
                            "response_transport": retry.get("format_mode"),
                            "provider_attempts": result.provider_attempts,
                            "successful_attempt_seed": retry.get("sampling_seed"),
                            "request_hash": result.request_hash,
                        },
                    )
                    successful.add(key)
                    completed.add(key)
                    print(f"completed {len(completed)}/{expected}: {key}", flush=True)


if __name__ == "__main__":
    main()
