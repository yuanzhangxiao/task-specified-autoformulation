"""Score frozen valid/adversarial pairs without exposing fit or pair labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.execution import (
    _JUDGE_CONTROLLER_PROMPT,
    ExecutionArguments,
    _context,
    _prediction_protocol_prompt,
    _symbol_contract,
)
from autoformalism.llm import (
    LLMConfig,
    LLMProvider,
    OllamaThinking,
    create_llm_client,
)
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.search.controller import _DETERMINISTIC_CERTIFICATIONS

FIELDS = (
    "pair_id",
    "benchmark_id",
    "tier",
    "mutation_type",
    "known_label",
    "judge_model",
    "repetition",
    "aggregate_score",
    "category_scores",
    "category_justifications",
    "hard_red_flags",
    "missing_requirements",
    "actionable_edits",
    "request_hash",
)


def _judge_prompt(data_root: Path, pair: AdversarialPair, model: str) -> str:
    registry = BenchmarkRegistry()
    development = BenchmarkLoader(registry).load_development(
        DataConfig(
            root=data_root,
            benchmark_id=pair.benchmark_id,
            tier=pair.tier,
        )
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
        output_root=Path("artifacts/rebuttal/adversarial"),
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
    proposer = (root / "proposer_prompt.txt").read_text(encoding="utf-8")
    judge = (root / "judge_prompt.txt").read_text(encoding="utf-8")
    return (
        f"Configured judge model: {model}\n\n"
        f"Proposer task:\n{proposer}\n\n"
        f"{_prediction_protocol_prompt(context)}\n\n"
        f"{_symbol_contract(context)}\n\n"
        f"Judge task:\n{judge}\n\n"
        f"Runtime judge amendment:\n{_JUDGE_CONTROLLER_PROMPT}"
    )


def _completed(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (
                row["pair_id"],
                row["known_label"],
                row["judge_model"],
                int(row["repetition"]),
            )
            for row in csv.DictReader(handle)
        }


def _append(path: Path, row: dict[str, object]) -> None:
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _select_shard(
    pairs: tuple[AdversarialPair, ...],
    *,
    shard_index: int,
    shard_count: int,
    strategy: str,
) -> tuple[AdversarialPair, ...]:
    """Select a deterministic round-robin or contiguous pair shard."""
    if strategy == "contiguous":
        start = len(pairs) * shard_index // shard_count
        stop = len(pairs) * (shard_index + 1) // shard_count
        return pairs[start:stop]
    return tuple(
        pair
        for index, pair in enumerate(pairs)
        if index % shard_count == shard_index
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--judge-models", nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--baseline-label",
        choices=("valid", "baseline"),
        default="valid",
        help="label unmutated candidates as historical valid or prospective baseline",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--ollama-thinking",
        choices=tuple(item.value for item in OllamaThinking),
        default=OllamaThinking.AUTO.value,
    )
    parser.add_argument("--ollama-temperature", type=float, default=0.0)
    parser.add_argument("--ollama-seed-base", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--shard-strategy",
        choices=("round_robin", "contiguous"),
        default="round_robin",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if not 0.0 <= args.ollama_temperature <= 2.0:
        raise SystemExit("--ollama-temperature must be in [0, 2]")
    if args.ollama_seed_base is not None and args.ollama_seed_base < 0:
        raise SystemExit("--ollama-seed-base must be nonnegative")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard index must be in [0, shard count)")
    pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    pairs = _select_shard(
        pairs,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        strategy=args.shard_strategy,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    score_path = args.output_root / "adversarial_judge_scores.csv"
    completed = _completed(score_path)
    expected = len(pairs) * 2 * args.repetitions * len(args.judge_models)
    if args.dry_run:
        prompts = {
            (pair.benchmark_id, pair.tier): _judge_prompt(
                args.data_root.resolve(), pair, args.judge_models[0]
            )
            for pair in pairs
        }
        print(
            f"pairs={len(pairs)} prompts={len(prompts)} expected_calls={expected} "
            f"completed={len(completed)} remaining={expected - len(completed)}"
        )
        return
    for model_spec in args.judge_models:
        provider_name, model = model_spec.split(":", 1)
        provider = LLMProvider(provider_name)
        clients = tuple(
            create_llm_client(
                LLMConfig(
                    provider=provider,
                    model=model,
                    cache_directory=(
                        args.output_root
                        / "cache"
                        / provider.value
                        / model
                        / f"repetition_{repetition}"
                    ),
                    log_path=(
                        args.output_root
                        / f"{provider.value}_{model}_events.jsonl"
                    ),
                    ollama_base_url=args.ollama_base_url,
                    ollama_thinking=OllamaThinking(args.ollama_thinking),
                    ollama_temperature=args.ollama_temperature,
                    ollama_seed=(
                        None
                        if args.ollama_seed_base is None
                        else args.ollama_seed_base + repetition
                    ),
                    timeout_seconds=args.timeout_seconds,
                    max_output_tokens=args.max_output_tokens,
                )
            )
            for repetition in range(args.repetitions)
        )
        for pair in pairs:
            system_prompt = _judge_prompt(
                args.data_root.resolve(), pair, model_spec
            )
            for label, candidate in (
                (args.baseline_label, pair.valid_candidate),
                (
                    "adversarial" if args.baseline_label == "valid" else "mutated",
                    pair.adversarial_candidate,
                ),
            ):
                for repetition in range(args.repetitions):
                    key = (pair.pair_id, label, model_spec, repetition)
                    if key in completed:
                        continue
                    request = {
                        "deterministic_certifications": list(
                            _DETERMINISTIC_CERTIFICATIONS
                        ),
                        "candidate": candidate.model_dump(mode="json"),
                    }
                    result = clients[repetition].judge(
                        system_prompt=system_prompt,
                        user_prompt=json.dumps(request, sort_keys=True),
                    )
                    _append(
                        score_path,
                        {
                            "pair_id": pair.pair_id,
                            "benchmark_id": pair.benchmark_id,
                            "tier": pair.tier,
                            "mutation_type": pair.mutation_type,
                            "known_label": label,
                            "judge_model": model_spec,
                            "repetition": repetition,
                            "aggregate_score": result.parsed.aggregate_score,
                            "category_scores": json.dumps(
                                result.parsed.numeric_category_scores,
                                sort_keys=True,
                            ),
                            "category_justifications": json.dumps(
                                {
                                    name: item.justification
                                    for name, item in (
                                        result.parsed.category_scores.__dict__.items()
                                    )
                                },
                                sort_keys=True,
                            ),
                            "hard_red_flags": json.dumps(
                                [
                                    item.model_dump(mode="json")
                                    for item in result.parsed.hard_red_flags
                                ],
                                sort_keys=True,
                            ),
                            "missing_requirements": json.dumps(
                                list(result.parsed.missing_requirements)
                            ),
                            "actionable_edits": json.dumps(
                                [
                                    item.model_dump(mode="json")
                                    for item in result.parsed.actionable_edits
                                ],
                                sort_keys=True,
                            ),
                            "request_hash": result.request_hash,
                        },
                    )
                    completed.add(key)
                    print(f"completed {len(completed)}/{expected}: {key}", flush=True)


if __name__ == "__main__":
    main()
