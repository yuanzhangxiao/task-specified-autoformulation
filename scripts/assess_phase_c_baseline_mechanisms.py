#!/usr/bin/env python3
"""Public mechanism tests of the Phase C external baselines' sealed picks.

``prepare`` freezes every pick of the named runs against the qualified public
release (development data only), with the reviewed LLM-SR transcriptions;
``run`` assesses frozen rows, one or all, in worker processes; ``report``
writes summary.json and SUMMARY.md. Nothing is fitted, selected or called,
and no test file is opened. See docs/PHASE_C_BASELINE_MECHANISMS.md.
"""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from autoformalism.rebuttal import phase_c_baseline_mechanisms as mechanisms
from autoformalism.rebuttal.prefit_replay import sealed_read


def _assess(root: Path, index: int) -> tuple[int, str, dict]:
    result = mechanisms.assess(root, index)
    return index, result["row_id"], result["counts"]


def main() -> None:
    """Prepare, run or report one assessment root."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="freeze the picks")
    prepare.add_argument("--release", type=Path, required=True)
    prepare.add_argument("--construction-config", type=Path, required=True)
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--classical", type=Path,
                         help="the SINDy/PySR run (both methods)")
    prepare.add_argument("--d3", type=Path)
    prepare.add_argument("--llm-ode", type=Path)
    prepare.add_argument("--llm-ode-published-prompt", type=Path)
    prepare.add_argument("--llm-sr", type=Path)
    prepare.add_argument("--llm-sr-transcriptions", type=Path)
    run = commands.add_parser("run", help="assess frozen rows")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--index", type=int, action="append",
                     help="a row to assess; repeat for more (default: all)")
    run.add_argument("--workers", type=int, default=1)
    report = commands.add_parser("report", help="summarize")
    report.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        sources = {
            method: path
            for method, path in (
                ("sindy", args.classical),
                ("pysr", args.classical),
                ("d3", args.d3),
                ("llm_ode", args.llm_ode),
                ("llm_ode_published_prompt", args.llm_ode_published_prompt),
                ("llm_sr", args.llm_sr),
            )
            if path is not None
        }
        plan = mechanisms.prepare(
            args.release,
            args.construction_config,
            sources,
            args.root,
            args.llm_sr_transcriptions,
        )
        print(f"froze {len(plan['rows'])} rows; plan {plan['artifact_sha256']}")
    elif args.command == "run":
        rows = len(sealed_read(args.root / "plan.json")["rows"])
        indices = args.index or list(range(rows))
        if args.workers <= 1:
            for index in indices:
                print(*_assess(args.root, index), flush=True)
            return
        with ProcessPoolExecutor(
            args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(_assess, args.root, i) for i in indices]
            for future in as_completed(futures):
                print(*future.result(), flush=True)
    else:
        value = mechanisms.report(args.root)
        print((args.root / "SUMMARY.md").read_text())
        print(f"rows: {len(value['rows'])}")


if __name__ == "__main__":
    main()
