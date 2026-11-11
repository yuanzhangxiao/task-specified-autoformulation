"""Fit and score the literature models on the nine-case Phase B comparison.

    python scripts/phase_b_literature_models.py run --inputs STAGED --output DIR
    python scripts/phase_b_literature_models.py report --output DIR

`run` freezes DIR/plan.json from the verified staged inputs on first use and
refuses a later run whose plan differs; every start and every unfitted score
is a job sealed in DIR/results, so a rerun does only what is missing. See
docs/PHASE_B_LITERATURE_MODELS.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _commit() -> str:
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return head + ("-dirty" if dirty else "")


def _job(arguments: tuple[str, str, str, str, float]) -> tuple[str, str, str]:
    from autoformalism.rebuttal.phase_b_literature import run_job

    inputs, output, key, name, wall_seconds = arguments
    return key, name, run_job(Path(inputs), Path(output), key, name, wall_seconds)


def main() -> None:
    """Literature-model fits run on CPU and make no provider call."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--wall-seconds", type=float, default=3 * 3600.0)
    summary = sub.add_parser("report")
    summary.add_argument("--output", type=Path, required=True)
    extra = sub.add_parser("cstr")
    extra.add_argument("--pair", required=True)
    extra.add_argument("--source", type=Path, required=True)
    extra.add_argument("--output", type=Path, required=True)
    extra.add_argument("--wall-seconds", type=float, default=3 * 3600.0)
    args = parser.parse_args()
    for name in THREADS:
        os.environ[name] = "1"

    from autoformalism.rebuttal import phase_b_literature as lit

    if args.command == "report":
        print(json.dumps(lit.report(args.output), indent=2, allow_nan=True))
        return
    commit = _commit()
    if commit.endswith("-dirty"):
        sys.exit("commit or stash tracked changes first; the plan records the commit")
    if args.command == "cstr":
        from autoformalism.rebuttal import cstr_literature

        prepared, identity = cstr_literature.prepare(
            args.pair, args.source.expanduser().resolve()
        )
        result = cstr_literature.run(
            prepared,
            identity,
            args.output.expanduser().resolve(),
            source_commit=commit,
            wall_seconds=args.wall_seconds,
        )
        print(json.dumps(result, indent=2, allow_nan=True))
        return
    inputs = args.inputs.expanduser().resolve()
    output = args.output.expanduser().resolve()
    frozen = lit.plan(inputs, source_commit=commit, wall_seconds=args.wall_seconds)
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != json.loads(json.dumps(frozen)):
            sys.exit(f"{plan_path} differs from this run's plan; use a new output")
    else:
        output.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(frozen, indent=2, sort_keys=True))
    pending = lit.jobs(output)
    print(f"{len(pending)} jobs pending", flush=True)
    work = [
        (str(inputs), str(output), key, name, args.wall_seconds)
        for key, name in pending
    ]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(_job, item) for item in work]
        for future in as_completed(futures):
            print(*future.result(), flush=True)
    print(json.dumps(lit.report(output), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
