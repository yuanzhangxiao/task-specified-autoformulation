"""CSTR literature models beside the approved Phase B campaign.

The approved sheet fits the textbook temperature balance with C and Tj read
from data (``cstr_balance`` in ``phase_b_literature``). With C read from data
that balance is open-loop unstable at these operating points: the heat it
generates grows with T while nothing consumes the reactant, so a free rollout
amplifies small errors, such as the linear interpolation of C between
samples, into a runaway even at the generator's documented parameters. The
textbook CSTR with its reactant balance closes that loop.

Agreed on 2026-11-10: report both models, beside the approved row, on the
historical Phase B cell and on its corrected Phase C counterpart, which has
the same roles (target T; C and Tj supplied; Cf, Tf and Tjf as inputs):

- ``cstr_closed_loop``: the closed loop on the Phase B cell;
- ``cstr_balance_phase_c``: the temperature balance on the Phase C cell;
- ``cstr_closed_loop_phase_c``: the closed loop on the Phase C cell.

Same protocol as ``phase_b_literature``: generic starts, three starts, least
squares on the training rollout, selection on validation among starts whose
replay matched, shared scoring. Each pair is also scored at the generator's
documented parameters, as a diagnostic of the data, never as a start. Only
public training and validation tables are read.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from autoformalism.rebuttal import phase_b_literature as lit
from autoformalism.rebuttal import phase_b_nine_case as nine
from autoformalism.rebuttal import phase_c_baselines

PROTOCOL = "cstr-literature-2"
PHASE_C_CELL = "phase_c_cstr_controlled_reactor_mechanism_canonical_named_easy_reset_v1"
PAIRS = (
    lit.CSTR_CLOSED,
    lit.Pair(
        "cstr_balance_phase_c",
        PHASE_C_CELL,
        "cstr",
        "textbook temperature balance on the corrected Phase C data",
        False,
    ),
    lit.Pair(
        "cstr_closed_loop_phase_c",
        PHASE_C_CELL,
        "cstr_closed",
        "textbook CSTR with its reactant balance on the corrected Phase C data",
        False,
    ),
)
PAIR_BY_KEY = {pair.key: pair for pair in PAIRS}
#: The generator's documented CSTR parameters, written about 350 K:
#: r_ref = beta * k0 * exp(-ER / 350) with k0 7.2e10 and beta 80.
DOCUMENTED = {
    "q": 1.0,
    "h": 2.5,
    "ER": 8750.0,
    "r_ref": 80.0 * 7.2e10 * math.exp(-8750.0 / 350.0),
    "beta": 80.0,
}


def prepare(key: str, source: Path) -> tuple[lit.Prepared, dict[str, str]]:
    """Load a pair's public splits, from the staged Phase B cells or the
    Phase C release, and build its candidate."""
    pair = PAIR_BY_KEY[key]
    if pair.benchmark_id == PHASE_C_CELL:
        cell = phase_c_baselines.load_cell(source, PHASE_C_CELL)
    else:
        cell = nine.load_cell(source, pair.benchmark_id, context="baseline")
    prepared = lit.assemble(
        pair, cell.dataset.train, cell.dataset.validation, cell.context
    )
    return prepared, cell.identity


def documented_scores(prepared: lit.Prepared) -> dict[str, Any]:
    """The model at the documented parameters: a diagnostic, not a start."""
    parameters = {
        name: value
        for name, value in DOCUMENTED.items()
        if name in prepared.structure.starts
    }
    return {
        "parameters": parameters,
        "train": lit.score(prepared, prepared.train, parameters),
        "validation": lit.score(prepared, prepared.validation, parameters),
    }


def run(
    prepared: lit.Prepared,
    identity: dict[str, str],
    output: Path,
    *,
    source_commit: str,
    wall_seconds: float,
) -> dict[str, Any]:
    """Fit every start not yet sealed, then select and summarize."""
    plan = {
        "protocol": PROTOCOL,
        "source_commit": source_commit,
        "public_identity": identity,
        "pair": prepared.description,
        "starts": list(lit.STARTS),
        "wall_seconds_per_start": wall_seconds,
        "fast_rollout": lit.FAST_ROLLOUT,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
            raise ValueError(f"{plan_path} differs from this run's plan")
    else:
        lit._write(plan_path, plan)
    starts = []
    for start in lit.STARTS:
        path = output / "results" / f"start_{start}.json"
        if not path.exists():
            lit._write(path, lit.fit(prepared, start, wall_seconds=wall_seconds))
        starts.append(json.loads(path.read_text()))
    eligible = [item for item in starts if item["replay"]["matches"]]
    selected = min(
        eligible,
        key=lambda item: item["validation"]["normalized_mse"],
        default=None,
    )
    summary = {
        "protocol": PROTOCOL,
        "key": prepared.pair.key,
        "benchmark_id": prepared.pair.benchmark_id,
        "model": prepared.pair.model,
        "label": prepared.pair.label,
        "selection": "lowest validation NMSE among starts whose replay matches",
        "starts": {
            item["start"]: {
                "train_nmse": item["train"]["normalized_mse"],
                "validation_nmse": item["validation"]["normalized_mse"],
                "parameters": item["parameters"],
                "replay_matches": item["replay"]["matches"],
            }
            for item in starts
        },
        "selected_start": None if selected is None else selected["start"],
        "validation_nmse": (
            None if selected is None else selected["validation"]["normalized_mse"]
        ),
        "documented_parameters_diagnostic": documented_scores(prepared),
        "test_data_opened": False,
    }
    lit._write(output / "summary.json", summary)
    return summary
