#!/usr/bin/env python3
"""Emit the \\AFSetResult lines Table 1 reads, from a results package.

The statistic is the one docs/EXPERIMENTS_DRAFT_NOTES.md specifies, not the
row-level median the roster report carries:

    case_median[m,b] = median over repetitions within a case
    macro_median[m]  = median over the nine cases
    macro_MAD[m]     = median |case_median - macro_median|, unscaled

The nine cases are equally weighted. Confirmed failed predictions rank as
+infinity, before either median. Missing evaluations with unknown outcomes
remain unavailable. Graph/complexity statistics use recorded evidence only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

#: The nine evaluation cases, in the order the draft introduces them.
NINE_CASES = (
    "phase_b_dalla_man_t1_canonical_named_easy",
    "phase_b_dalla_man_t1_canonical_named_hard",
    "phase_b_dalla_man_t2_canonical_named_easy",
    "phase_b_dalla_man_t2_canonical_named_hard",
    "phase_b_anonymous_system_t1_canonical_obfuscated_easy",
    "phase_b_dalla_man_t1_perturbed_named_easy",
    "phase_b_anonymous_system_t1_perturbed_obfuscated_easy",
    "phase_b_cstr_controlled_reactor_mechanism_canonical_named_easy",
    "phase_b_alien_device_unknown_device_mechanism_canonical_functional_easy",
)

#: Result-key stem per method identifier in the evaluation records.
METHOD_KEYS = {
    "sindy": "sindy",
    "pysr": "pysr",
    "d3_native_no_tools": "d3",
    "raw_data_agent:gpt-5.6-sol": "agent",
}


def case_median(values: list[float]) -> float | None:
    """Median over a case's repetitions, including worst-ranked failures."""
    return median(values) if values else None


def macro(case_values: list[float | None]) -> tuple[float | None, float | None]:
    """Macro median and unscaled MAD over cases, or None if any case is absent.

    Infinity is a known failed outcome, not a missing case. Its MAD is
    undefined if the centre itself is infinite.
    """
    if not case_values or any(value is None for value in case_values):
        return None, None
    present = [float(value) for value in case_values]
    centre = median(present)
    return centre, (
        median([abs(value - centre) for value in present])
        if math.isfinite(centre) else None
    )


def render(value: float | None, spread: float | None, digits: int = 3) -> str:
    """`median [MAD]`, or the draft's own placeholder when unavailable."""
    if value is None:
        return r"\textbf{TBD}"
    if value == math.inf:
        return r"\text{Failed}"
    if spread == math.inf:
        return f"{value:.{digits}f} [$\\infty$]"
    if spread is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{spread:.{digits}f}]"


def collect(rows: list[dict], field: str) -> dict[str, list[float | None]]:
    """Per method, one value per case, in the fixed case order."""
    per_method: dict[str, list[float | None]] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        values: list[float | None] = []
        for case in NINE_CASES:
            subset = [r for r in rows if str(r["method"]) == method
                      and str(r["benchmark_id"]) == case]
            samples = []
            unknown = field == "target_nmse" and len(subset) != 3
            for row in subset:
                value = row.get(field)
                if value is not None:
                    if (type(value) not in (int, float)
                            or not math.isfinite(value) or value < 0):
                        raise ValueError(f"invalid numeric endpoint: {field}")
                    samples.append(float(value))
                elif field == "target_nmse":
                    if (row.get("target_status") == "failed"
                            or row.get("terminal_status") in {"failed", "timed_out"}):
                        samples.append(math.inf)
                    else:
                        unknown = True
            values.append(None if unknown else case_median(samples))
        per_method[method] = values
    return per_method


def coverage(rows: list[dict], method: str) -> tuple[int, int]:
    """Evaluated and planned repetitions over the nine cases only."""
    subset = [
        row
        for row in rows
        if str(row["method"]) == method and str(row["benchmark_id"]) in NINE_CASES
    ]
    scored = sum(1 for row in subset if row.get("target_nmse") is not None)
    return scored, len(subset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.models.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors = collect(rows, "target_nmse")
    terms = collect(rows, "complexity_terms")
    compliance = collect(rows, "mechanism_compliance")

    lines = [
        "% Generated from a results package by "
        "scripts/build_experiments_table_results.py",
        "% Statistic: macro median [unscaled MAD] over the nine cases,",
        "% case medians taken over repetitions first "
        "(EXPERIMENTS_DRAFT_NOTES.md).",
        "% Failed predictions rank +infinity before either median; "
        "unknown outcomes are unavailable.",
        "",
    ]
    notes: list[str] = []
    for method, key in sorted(METHOD_KEYS.items()):
        if method not in errors:
            notes.append(f"{method}: absent from this package")
            continue
        scored, planned = coverage(rows, method)
        absent = [
            case
            for case, value in zip(NINE_CASES, errors[method], strict=True)
            if value is None
        ]
        lines += [
            f"\\AFSetResult{{base.{key}.error}}{{{render(*macro(errors[method]))}}}",
            f"\\AFSetResult{{base.{key}.terms}}"
            f"{{{render(*macro(terms[method]), digits=1)}}}",
            f"\\AFSetResult{{base.{key}.coverage}}{{{scored}/{planned}}}",
        ]
        if absent:
            notes.append(
                f"{method}: unknown NMSE outcomes for {len(absent)} of nine cases "
                f"({', '.join(item.replace('phase_b_', '') for item in absent)})"
                " -- headline aggregate unavailable"
            )
        centre, spread = macro(compliance[method])
        if centre is not None:
            notes.append(
                f"{method}: graph-only compliance {render(centre, spread)} "
                "(fraction; Table 1 wants pass/unresolved percentages, which "
                "need the per-obligation counts)"
            )
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    if notes:
        print("% Notes:")
        for note in notes:
            print(f"%   {note}")


if __name__ == "__main__":
    main()
