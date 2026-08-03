#!/usr/bin/env python3
"""Make one manual proposer call; never calls a provider without --live."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.llm import LLMConfig, LLMProvider, create_llm_client


def build_parser() -> argparse.ArgumentParser:
    """Build the live-call command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly authorize one real proposer request",
    )
    parser.add_argument(
        "--provider", choices=("openai", "gemini", "ollama"), default="openai"
    )
    parser.add_argument("--model", help="provider model identifier")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/llm_cache"))
    parser.add_argument("--log", type=Path, default=Path("artifacts/llm_events.jsonl"))
    return parser


def main() -> None:
    """Run exactly one proposer call after an explicit live gate."""
    args = build_parser().parse_args()
    if not args.live:
        print("No call made. Pass --live to authorize one proposer request.")
        return
    if not args.model:
        raise SystemExit("--model is required with --live")

    user_prompt = (
        args.prompt_file.read_text(encoding="utf-8")
        if args.prompt_file is not None
        else (
            "Propose a minimal continuous-time model with one observed target state "
            "and one latent state driven by input_u. Include all fields required by "
            "the response schema."
        )
    )
    client = create_llm_client(
        LLMConfig(
            provider=LLMProvider(args.provider),
            model=args.model,
            cache_directory=args.cache_dir,
            log_path=args.log,
            max_attempts=1,
            proposal_target_channels=("target",),
        )
    )
    result = client.propose(
        system_prompt=(
            "Return one proposer candidate. Follow the supplied structured-output "
            "schema exactly. Use explicit analytic expression strings."
        ),
        user_prompt=user_prompt,
    )
    print(json.dumps(result.parsed.model_dump(mode="json"), indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "request_hash": result.request_hash,
                "cache_hit": result.cache_hit,
                "latency_ms": result.latency_ms,
                "usage": (
                    None
                    if result.usage is None
                    else {
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": result.usage.total_tokens,
                    }
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
