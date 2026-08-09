#!/usr/bin/env python3
"""Run private Phase B0 local observability diagnostics for Dalla Man T1--T4."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from autoformalism.rebuttal.dalla_man import (
    STATE_INDEX,
    DallaManParameters,
    compute_dalla_man_basal,
)
from autoformalism.rebuttal.observability import (
    empirical_dalla_observability,
    empirical_dalla_parameter_sensitivity,
)

PROTOCOLS = {
    "single_90g": ((0.0, 90.0),),
    "multi_timing_mass": ((0.0, 60.0), (75.0, 35.0), (170.0, 55.0)),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    basal = compute_dalla_man_basal(DallaManParameters())
    shifted_initial = basal.initial_state.copy()
    shifted_initial[STATE_INDEX["Gp"]] *= 1.10
    shifted_initial[STATE_INDEX["Gt"]] *= 0.90
    shifted_initial[STATE_INDEX["Ip"]] *= 1.15
    shifted_initial[STATE_INDEX["Il"]] *= 1.10
    initial_conditions = {
        "basal": tuple(basal.initial_state),
        "shifted_glucose_insulin": tuple(shifted_initial),
    }
    for protocol, meals in PROTOCOLS.items():
        duration = 360.0 if protocol == "multi_timing_mass" else 300.0
        for task in ("T1", "T2", "T3", "T4"):
            for initial_label, initial_state in initial_conditions.items():
                result = empirical_dalla_observability(
                    task,
                    meals=meals,
                    duration=duration,
                    protocol=protocol,
                    initial_state=initial_state,
                )
                row = asdict(result)
                row["initial_condition"] = initial_label
                row["outputs"] = json.dumps(row["outputs"])
                row["hidden_states"] = json.dumps(row["hidden_states"])
                row["singular_values"] = json.dumps(row["singular_values"])
                rows.append(row)
            for quantity_kind in ("outputs", "fluxes"):
                result = empirical_dalla_parameter_sensitivity(
                    task,
                    meals=meals,
                    duration=duration,
                    protocol=protocol,
                    quantity_kind=quantity_kind,
                )
                row = asdict(result)
                for field in ("quantities", "parameters", "singular_values"):
                    row[field] = json.dumps(row[field])
                parameter_rows.append(row)
    _write_csv(output_root / "observability.csv", rows)
    _write_csv(output_root / "parameter_sensitivity.csv", parameter_rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
