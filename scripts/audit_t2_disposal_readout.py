"""Quantify the observable disposal component under an explicit additive readout."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research.trajectory_function_fitting import _rows
from scripts.audit_hidden_shape_t2 import FROZEN
from scripts.audit_pooled_hidden_shape_t2 import _pool
from scripts.matched_equation_experiment import DEFAULT, _registered

OUTPUT = DEFAULT.parent / "hidden-form-forecast-20261126/disposal-readout"


def inspect_readout(rows: list[dict]) -> dict:
    """Compare effective disposal at identical measured initial insulin values."""
    by_insulin = defaultdict(list)
    samples = 0
    smallest, largest = float("inf"), float("-inf")
    for row in _rows(rows, "train"):
        total, independent = row.targets["U"], row.auxiliaries["Uii"]
        effective = [u - uii for u, uii in zip(total, independent, strict=True)]
        samples += len(effective)
        smallest = min(smallest, min(effective))
        largest = max(largest, max(effective))
        value = Fraction(total[0]) - Fraction(independent[0])
        by_insulin[row.targets["I"][0]].append((value, row.trajectory_id))
    # At a common argument, the best common prediction is the midpoint of the
    # extreme outputs. The ordering is deterministic when several events tie.
    best = None
    for insulin, group in sorted(by_insulin.items()):
        if len(group) < 2:
            continue
        low, high = min(group), max(group)
        radius = (high[0] - low[0]) / 2
        if best is None or radius > best[0]:
            best = radius, insulin, low, high
    witness = None
    if best is not None:
        radius, insulin, low, high = best
        witness = {
            "I_at_start": insulin,
            "events": [low[1], high[1]],
            "disposal_at_start": list(map(float, (low[0], high[0]))),
            "minimum_uniform_error_at_start": float(radius),
            "exact_error_in_recorded_values": str(radius),
        }
    return {
        "readout_assumption": "U = Uii + D",
        "events": len(rows),
        "samples": samples,
        "sample_range": [smallest, largest],
        "memoryless_hypothesis": "D(0) = h(I(0)) with one shared h",
        "largest_repeated_input_witness": witness,
    }


def run(source: Path, output: Path) -> dict:
    """Read only the frozen training packets and verify the receipt on resume."""
    plan = _registered(source / "plan.json", FROZEN)
    panels = sorted(
        (p for p in plan["panels"] if p["panel_id"].startswith("graph02-fold")),
        key=lambda p: p["fold"],
    )
    _, rows = _pool(source, panels)
    with _lock(output):
        return sealed_write(
            output / "summary.json",
            {
                "protocol": "measured-disposal-readout-1",
                "source_plan": FROZEN,
                "source_packets": [p["packet"] for p in panels],
                **inspect_readout(rows),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))
