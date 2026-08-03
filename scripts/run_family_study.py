#!/usr/bin/env python3
"""Run the preregistered GPT/Gemini proposer-judge family study."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PAIRINGS = (("gpt", "gemini"), ("gemini", "gpt"), ("gemini", "gemini"))
CONDITIONS = (("original_b1", "hard"), ("benchmark6", "hard"))


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed-design study CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpt-model", default="openai:gpt-5.6-terra")
    parser.add_argument("--gemini-model", default="gemini:gemini-3.6-flash")
    parser.add_argument("--seeds", type=int, nargs="+", default=range(5))
    parser.add_argument("--iteration-budget", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    """Run incremental cells, resuming partial checkpoints."""
    args = build_parser().parse_args()
    models = {"gpt": args.gpt_model, "gemini": args.gemini_model}
    records: list[dict[str, object]] = []
    for proposer_family, judge_family in PAIRINGS:
        cell = f"{proposer_family}_proposer__{judge_family}_judge"
        cell_root = args.output_root / cell
        for benchmark, tier in CONDITIONS:
            for seed in args.seeds:
                name = f"{benchmark}_{tier}_seed{seed}"
                checkpoint_root = cell_root / name / "checkpoints"
                if (checkpoint_root / "final.json").exists():
                    records.append({"cell": cell, "run": name, "status": "skip"})
                    continue
                resume = (checkpoint_root / "run.json").exists()
                entrypoint = (
                    "scripts/resume_experiment.py"
                    if resume
                    else "scripts/run_experiment.py"
                )
                command = [
                    sys.executable,
                    entrypoint,
                    "--data-root", str(args.data_root),
                    "--benchmark-id", benchmark,
                    "--tier", tier,
                    "--seed", str(seed),
                    "--proposer-model", models[proposer_family],
                    "--judge-model", models[judge_family],
                    "--iteration-budget", str(args.iteration_budget),
                    "--beam-size", str(args.beam_size),
                    "--llm-timeout-seconds", "900",
                    "--llm-max-output-tokens", "4096",
                    "--fit-starts", "1",
                    "--fit-max-nfev", "50",
                    "--fit-timeout-seconds", "300",
                    "--final-fit-max-nfev", "150",
                    "--final-fit-timeout-seconds", "300",
                    "--output-root", str(cell_root),
                ]
                if args.dry_run:
                    print(" ".join(command))
                    continue
                print(f"START {cell} {benchmark} {tier} seed{seed}", flush=True)
                result = subprocess.run(command, check=False)
                status = "complete" if result.returncode == 0 else "failed"
                records.append({"cell": cell, "run": name, "status": status})
                args.output_root.mkdir(parents=True, exist_ok=True)
                (args.output_root / "study_status.json").write_text(
                    json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
                )
    if not args.dry_run:
        print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
