"""Run the V8 paired target-only judge with symmetry-preserving retries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.execution import (
    ExecutionArguments,
    _context,
    _prediction_protocol_prompt,
    _symbol_contract,
)
from autoformalism.expressions import ValidationContext, repair_protected_declarations
from autoformalism.judging import (
    extract_public_requirements,
    paired_target_question_consensus,
    structural_facts,
)
from autoformalism.judging.prompts import (
    PAIRED_TARGET_COMPLETENESS_JUDGE_PROMPT,
    PAIRED_TARGET_COMPLETENESS_JUDGE_PROTOCOL_VERSION,
)
from autoformalism.llm import (
    LLMConfig,
    LLMProvider,
    VLLMReasoningEffort,
    create_llm_client,
)
from autoformalism.llm.exceptions import LLMError
from autoformalism.llm.models import LLMCallResult
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import (
    CandidateModel,
    PairedTargetCompletenessJudgeResult,
)

FAILURE_SCHEMA_VERSION = "paired-target-completeness-failure-1"
RUN_MANIFEST_SCHEMA_VERSION = "paired-target-completeness-run-1"
SCORE_SCHEMA_VERSION = "paired-target-completeness-score-1"


def _task_context(
    data_root: Path,
    pair: AdversarialPair,
) -> tuple[str, ValidationContext]:
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
        output_root=Path("artifacts/rebuttal/paired-target-completeness"),
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


def paired_target_completeness_system_prompt(
    public_prompt: str,
    context: ValidationContext,
    model: str,
) -> str:
    """Build the target-only paired system prompt."""
    return (
        f"Configured judge model: {model}\n\n"
        f"Public scientific task:\n{public_prompt}\n\n"
        f"{_prediction_protocol_prompt(context)}\n\n"
        f"{_symbol_contract(context)}\n\n"
        "Paired target-only protocol:\n"
        f"{PAIRED_TARGET_COMPLETENESS_JUDGE_PROMPT}"
    )


def _blinded_candidate_payload(
    candidate: CandidateModel,
    runtime_id: str,
) -> dict[str, object]:
    payload = candidate.model_dump(mode="json")
    payload["candidate_id"] = runtime_id
    payload["parent_candidate_id"] = None
    payload["change_summary"] = "unspecified"
    return payload


def paired_target_completeness_request(
    candidate_a: CandidateModel,
    candidate_b: CandidateModel,
    *,
    public_prompt: str,
    context: ValidationContext,
) -> dict[str, object]:
    """Build one blinded request with two independently assessed candidates."""
    return {
        "schema_version": "paired-target-completeness-request-1",
        "public_requirement_registry": extract_public_requirements(
            public_prompt
        ).model_dump(mode="json"),
        "requested_target_ids": list(context.targets),
        "candidate_a": _blinded_candidate_payload(candidate_a, "candidate_a"),
        "candidate_b": _blinded_candidate_payload(candidate_b, "candidate_b"),
        "deterministic_structural_facts": {
            "candidate_a": structural_facts(
                candidate_a,
                task_inputs=tuple(context.external_inputs),
                include_model_semantics=True,
                causal_observation_resets=bool(context.lagged_targets),
            ),
            "candidate_b": structural_facts(
                candidate_b,
                task_inputs=tuple(context.external_inputs),
                include_model_semantics=True,
                causal_observation_resets=bool(context.lagged_targets),
            ),
        },
    }


def _select_shard(
    pairs: tuple[AdversarialPair, ...],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[AdversarialPair, ...]:
    start = len(pairs) * shard_index // shard_count
    stop = len(pairs) * (shard_index + 1) // shard_count
    return pairs[start:stop]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _ensure_manifest(path: Path, manifest: dict[str, object]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("paired target-only resume configuration changed")
        return
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _seed_for_attempt(
    base: int,
    repetition: int,
    seed_attempt: int,
    *,
    seed_attempts: int,
    provider_attempts: int,
) -> int:
    """Allocate nonoverlapping seed blocks across repeats and pair retries."""
    return (
        base
        + repetition * seed_attempts * provider_attempts
        + seed_attempt * provider_attempts
    )


def _orientation_metadata(
    result: LLMCallResult[PairedTargetCompletenessJudgeResult],
) -> dict[str, object]:
    raw_response = result.raw_response
    retry = raw_response.get("_autoformalism_retry")
    if not isinstance(retry, dict):
        retry = {}
    return {
        "request_hash": result.request_hash,
        "provider_attempts": result.provider_attempts,
        "response_transport": retry.get("format_mode"),
        "successful_provider_seed": retry.get("sampling_seed"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--judge-models", nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-output-tokens", type=int, default=3072)
    parser.add_argument("--max-attempts", type=int, default=10, choices=range(1, 12))
    parser.add_argument("--paired-seed-attempts", type=int, default=2)
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
    if args.repetitions < 1 or args.paired_seed_attempts < 1:
        raise SystemExit("repetitions and paired seed attempts must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard index must be in [0, shard count)")

    args.output_root.mkdir(parents=True, exist_ok=True)
    source_bytes = args.pairs.read_bytes()
    all_pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    )
    pairs = _select_shard(
        all_pairs,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    planned = {
        (pair.pair_id, model, repetition)
        for pair in pairs
        for model in args.judge_models
        for repetition in range(args.repetitions)
    }
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "protocol_version": PAIRED_TARGET_COMPLETENESS_JUDGE_PROTOCOL_VERSION,
        "pairs_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "selected_pair_ids": [pair.pair_id for pair in pairs],
        "judge_models": args.judge_models,
        "repetitions": args.repetitions,
        "max_provider_attempts": args.max_attempts,
        "max_paired_seed_attempts": args.paired_seed_attempts,
        "seed_block_policy": "nonoverlapping_provider_attempt_blocks_v1",
        "vllm_reasoning_effort": args.vllm_reasoning_effort,
        "vllm_temperature": args.vllm_temperature,
        "vllm_seed_base": args.vllm_seed_base,
        "candidate_order_policy": "both_orientations_same_seed",
        "consensus": "fail_dominant_per_candidate_target",
        "atomic_stage_enabled": False,
        "comparative_questions_enabled": False,
        "numeric_score_defined": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    _ensure_manifest(
        args.output_root / "paired_target_completeness_run_manifest.json",
        manifest,
    )
    score_path = args.output_root / "paired_target_completeness_scores.jsonl"
    failure_path = args.output_root / "paired_target_completeness_failures.jsonl"
    score_path.touch(exist_ok=True)
    failure_path.touch(exist_ok=True)
    successes = _read_jsonl(score_path)
    failures = _read_jsonl(failure_path)
    completed = {
        (str(row["pair_id"]), str(row["judge_model"]), int(row["repetition"]))
        for row in (*successes, *failures)
    }
    if len(completed) != len(successes) + len(failures):
        raise ValueError("duplicate paired target-only outcome key")
    if completed - planned:
        raise ValueError("saved outcome does not belong to this run")

    contexts: dict[tuple[str, str], tuple[str, ValidationContext]] = {}
    for pair in pairs:
        key = (pair.benchmark_id, pair.tier)
        if key not in contexts:
            contexts[key] = _task_context(args.data_root.resolve(), pair)
    if args.dry_run:
        print(
            f"pairs={len(pairs)} planned_trials={len(planned)} "
            f"initial_orientation_calls={2 * len(planned)} "
            f"maximum_orientation_calls={2 * args.paired_seed_attempts * len(planned)} "
            f"remaining_trials={len(planned - completed)}"
        )
        return

    expected = len(planned)
    for model_spec in args.judge_models:
        provider_name, model = model_spec.split(":", 1)
        provider = LLMProvider(provider_name)
        storage_name = model.replace("/", "__")
        for pair in pairs:
            public_prompt, context = contexts[(pair.benchmark_id, pair.tier)]
            system_prompt = paired_target_completeness_system_prompt(
                public_prompt,
                context,
                model_spec,
            )
            baseline, baseline_repairs = repair_protected_declarations(
                pair.valid_candidate,
                context,
            )
            mutated, mutated_repairs = repair_protected_declarations(
                pair.adversarial_candidate,
                context,
            )
            target_ids = set(context.targets)
            requests = (
                paired_target_completeness_request(
                    baseline,
                    mutated,
                    public_prompt=public_prompt,
                    context=context,
                ),
                paired_target_completeness_request(
                    mutated,
                    baseline,
                    public_prompt=public_prompt,
                    context=context,
                ),
            )
            for repetition in range(args.repetitions):
                key = (pair.pair_id, model_spec, repetition)
                if key in completed:
                    continue
                seed_failures: list[dict[str, object]] = []
                final_results: tuple[
                    LLMCallResult[PairedTargetCompletenessJudgeResult],
                    LLMCallResult[PairedTargetCompletenessJudgeResult],
                ] | None = None
                selected_seed: int | None = None
                selected_attempt: int | None = None
                for seed_attempt in range(args.paired_seed_attempts):
                    seed = _seed_for_attempt(
                        args.vllm_seed_base,
                        repetition,
                        seed_attempt,
                        seed_attempts=args.paired_seed_attempts,
                        provider_attempts=args.max_attempts,
                    )
                    client = create_llm_client(
                        LLMConfig(
                            provider=provider,
                            model=model,
                            cache_directory=(
                                args.output_root
                                / "cache"
                                / provider.value
                                / storage_name
                                / f"seed_{seed}"
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
                            vllm_seed=seed,
                            timeout_seconds=args.timeout_seconds,
                            max_output_tokens=args.max_output_tokens,
                        )
                    )
                    orientation_results: list[
                        LLMCallResult[PairedTargetCompletenessJudgeResult] | None
                    ] = []
                    orientation_errors: list[dict[str, object]] = []
                    for orientation, request in zip(
                        ("baseline_a", "baseline_b"), requests, strict=True
                    ):
                        try:
                            result = client.assess_paired_target_completeness(
                                system_prompt=system_prompt,
                                user_prompt=json.dumps(request, sort_keys=True),
                                expected_target_ids=target_ids,
                            )
                            orientation_results.append(result)
                        except LLMError as exc:
                            orientation_results.append(None)
                            category = getattr(exc, "category", None)
                            orientation_errors.append(
                                {
                                    "orientation": orientation,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc)[:4000],
                                    "failure_category": getattr(
                                        category, "value", None
                                    ),
                                }
                            )
                    if not orientation_errors:
                        assert orientation_results[0] is not None
                        assert orientation_results[1] is not None
                        final_results = (
                            orientation_results[0],
                            orientation_results[1],
                        )
                        selected_seed = seed
                        selected_attempt = seed_attempt + 1
                        break
                    seed_failures.append(
                        {
                            "seed_attempt": seed_attempt + 1,
                            "seed": seed,
                            "orientation_errors": orientation_errors,
                            "discarded_successful_orientation_count": sum(
                                item is not None for item in orientation_results
                            ),
                        }
                    )

                if final_results is None:
                    _append_jsonl(
                        failure_path,
                        {
                            "schema_version": FAILURE_SCHEMA_VERSION,
                            "pair_id": pair.pair_id,
                            "benchmark_id": pair.benchmark_id,
                            "tier": pair.tier,
                            "mutation_type": pair.mutation_type,
                            "judge_model": model_spec,
                            "repetition": repetition,
                            "paired_seed_attempt_limit": args.paired_seed_attempts,
                            "seed_failures": seed_failures,
                        },
                    )
                    completed.add(key)
                    print(
                        f"failed {len(completed)}/{expected}: {key}",
                        flush=True,
                    )
                    continue

                forward_call, reverse_call = final_results
                forward = forward_call.parsed
                reverse = reverse_call.parsed
                baseline_consensus, mutated_consensus, disagreements = (
                    paired_target_question_consensus(forward, reverse)
                )
                _append_jsonl(
                    score_path,
                    {
                        "schema_version": SCORE_SCHEMA_VERSION,
                        "pair_id": pair.pair_id,
                        "benchmark_id": pair.benchmark_id,
                        "tier": pair.tier,
                        "mutation_type": pair.mutation_type,
                        "judge_model": model_spec,
                        "repetition": repetition,
                        "requested_target_ids": sorted(target_ids),
                        "selected_seed_attempt": selected_attempt,
                        "selected_seed": selected_seed,
                        "prior_seed_failures": seed_failures,
                        "baseline_repairs": baseline_repairs,
                        "mutated_repairs": mutated_repairs,
                        "forward": {
                            "candidate_a_role": "baseline",
                            "candidate_b_role": "mutated",
                            "result": forward.model_dump(mode="json"),
                            **_orientation_metadata(forward_call),
                        },
                        "reverse": {
                            "candidate_a_role": "mutated",
                            "candidate_b_role": "baseline",
                            "result": reverse.model_dump(mode="json"),
                            **_orientation_metadata(reverse_call),
                        },
                        "consensus": {
                            "baseline": baseline_consensus.model_dump(mode="json"),
                            "mutated": mutated_consensus.model_dump(mode="json"),
                            "baseline_overall_verdict": (
                                baseline_consensus.overall_verdict.value
                            ),
                            "mutated_overall_verdict": (
                                mutated_consensus.overall_verdict.value
                            ),
                            "orientation_disagreements": list(disagreements),
                        },
                    },
                )
                completed.add(key)
                print(f"completed {len(completed)}/{expected}: {key}", flush=True)


if __name__ == "__main__":
    main()
