"""The textbook CSTR temperature balance on the corrected Phase C CSTR cell.

The historical Phase B CSTR export carries known generation defects, which
Phase C corrected. This fits the same textbook balance, by the same protocol
(``phase_b_literature``: generic starts, three starts, least squares on the
training rollout, selection on validation, shared scoring with a replay
check), to the Phase C cell with the same roles: target T, supplied C and Tj,
inputs Cf, Tf and Tjf. Only the cell's public training and validation tables
are read.

As a diagnostic, never a start, the balance is also scored at the generator's
documented parameters, which says whether the data follow the structure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from autoformalism.rebuttal import phase_b_literature as lit
from autoformalism.rebuttal import phase_c_baselines

PROTOCOL = "phase-c-literature-cstr-1"
CELL = "phase_c_cstr_controlled_reactor_mechanism_canonical_named_easy_reset_v1"
PAIR = lit.Pair(
    "cstr_balance_phase_c",
    CELL,
    "cstr",
    "textbook structure on the corrected Phase C data",
    False,
)
#: The generator's documented CSTR parameters, written about 350 K:
#: r_ref = beta_r * k0 * exp(-ER / 350) with k0 7.2e10 and beta_r 80.
DOCUMENTED = {
    "q": 1.0,
    "h": 2.5,
    "ER": 8750.0,
    "r_ref": 80.0 * 7.2e10 * math.exp(-8750.0 / 350.0),
}


def prepare(release: Path) -> tuple[lit.Prepared, dict[str, str]]:
    """Load the Phase C cell's public splits and build the candidate."""
    cell = phase_c_baselines.load_cell(release, CELL)
    prepared = lit.assemble(
        PAIR, cell.dataset.train, cell.dataset.validation, cell.context
    )
    return prepared, cell.identity


def documented_scores(prepared: lit.Prepared) -> dict[str, Any]:
    """The balance at the documented parameters: a diagnostic, not a start."""
    parameters = dict(DOCUMENTED)
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
        "benchmark_id": CELL,
        "label": PAIR.label,
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
