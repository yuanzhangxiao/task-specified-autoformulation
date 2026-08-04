"""Score frozen valid/adversarial pairs without exposing fit or pair labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from autoformalism.llm import LLMConfig, LLMProvider, create_llm_client
from autoformalism.rebuttal.adversarial import AdversarialPair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--judge-system-prompt", type=Path, required=True)
    parser.add_argument("--judge-models", nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    system_prompt = args.judge_system_prompt.read_text(encoding="utf-8")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_spec in args.judge_models:
        provider_name, model = model_spec.split(":", 1)
        provider = LLMProvider(provider_name)
        client = create_llm_client(
            LLMConfig(
                provider=provider,
                model=model,
                cache_directory=args.output_root / "cache" / provider.value / model,
                log_path=args.output_root / f"{provider.value}_{model}_events.jsonl",
                timeout_seconds=args.timeout_seconds,
                max_output_tokens=args.max_output_tokens,
            )
        )
        for pair in pairs:
            for label, candidate in (
                ("valid", pair.valid_candidate),
                ("adversarial", pair.adversarial_candidate),
            ):
                for repetition in range(args.repetitions):
                    # The nonce creates independent cached calls. The known label
                    # is retained only in the local output row, never in prompts.
                    request = {
                        "evaluation_replicate": repetition,
                        "candidate": candidate.model_dump(mode="json"),
                    }
                    result = client.judge(
                        system_prompt=system_prompt,
                        user_prompt=json.dumps(request, sort_keys=True),
                    )
                    rows.append(
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
                            "request_hash": result.request_hash,
                        }
                    )
    if rows:
        with (args.output_root / "adversarial_judge_scores.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
