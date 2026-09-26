#!/usr/bin/env python3
"""Independently aggregate an exported baseline package without model replay.

The existing table builder supplies only the roster, parsed as literal data.
This auditor reads JSONL and CSV independently, checks their agreement, and
computes endpoint-specific case medians followed by macro medians and raw MADs.
It never imports the table builder or evaluates candidate equation strings.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import tarfile
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

METHODS = {
    "sindy": "SINDy",
    "pysr": "PySR",
    "d3_native_no_tools": "D3",
    "raw_data_agent:gpt-5.6-sol": "GPT-5.6 Sol agent",
}
FIELDS = (
    "target_nmse",
    "mechanism_compliance",
    "complexity_terms",
    "complexity_states",
    "complexity_latent_states",
    "complexity_parameters",
)


def read_roster(script: Path, config: Path) -> tuple[str, ...]:
    """Read the declared tuple without executing another implementation."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    roster = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "NINE_CASES"
            for target in node.targets
        )
    )
    campaign = json.loads(config.read_text(encoding="utf-8"))["public_cells"]
    if len(roster) != 9 or len(set(roster)) != 9 or set(roster) != set(campaign):
        raise ValueError("table roster and nine-case campaign disagree")
    return tuple(roster)


def finite(value: Any) -> bool:
    """Boolean flags and nonfinite values are not numerical observations."""
    return type(value) in (int, float) and math.isfinite(value)


def statistics(values: list[float]) -> dict[str, Any]:
    """Return a median and unscaled MAD, retaining the observed denominator."""
    if not values:
        return {"n": 0, "median": None, "mad": None}
    centre = median(values)
    return {
        "n": len(values),
        "median": centre,
        "mad": median([abs(value - centre) for value in values]),
    }


def load_package(archive: Path) -> tuple[list[dict], dict, dict]:
    """Read three allowlisted files in memory and verify every CSV field."""
    with tarfile.open(archive, "r:gz") as bundle:
        payloads = {}
        for name in ("models.jsonl", "models_and_metrics.csv", "manifest.json"):
            matches = [
                member
                for member in bundle.getmembers()
                if Path(member.name).name == name and member.isfile()
            ]
            if len(matches) != 1 or matches[0].size > 100_000_000:
                raise ValueError(f"missing, duplicated, or oversized member: {name}")
            stream = bundle.extractfile(matches[0])
            assert stream is not None
            payloads[name] = stream.read()
    rows = [
        json.loads(line)
        for line in payloads["models.jsonl"].splitlines()
        if line.strip()
    ]
    flat = list(
        csv.DictReader(io.StringIO(payloads["models_and_metrics.csv"].decode("utf-8")))
    )
    original = {row["request_id"]: row for row in rows}
    flattened = {row["request_id"]: row for row in flat}
    identities = {(r["method"], r["benchmark_id"], r["repetition"]) for r in rows}
    if len(original) != len(rows) or len(identities) != len(rows):
        raise ValueError("duplicate JSONL request or method/case/repetition")
    if len(flattened) != len(flat) or original.keys() != flattened.keys():
        raise ValueError("CSV and JSONL request identities differ")
    comparisons = 0
    for request, row in original.items():
        for key, actual in flattened[request].items():
            if key not in row:
                raise ValueError(f"CSV-only field: {key}")
            value = row[key]
            expected = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else ""
                if value is None
                else str(value)
            )
            if actual != expected:
                raise ValueError(f"CSV/JSONL field differs: {request}/{key}")
            comparisons += 1
    manifest = json.loads(payloads["manifest.json"])
    counts = {
        "planned_total": len(rows),
        "with_model": sum(bool(row["equation_count"]) for row in rows),
        "with_score": sum(finite(row.get("target_nmse")) for row in rows),
    }
    if any(manifest.get(key) != value for key, value in counts.items()):
        raise ValueError("manifest counts disagree with exported records")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    source_hashes = sum(len(e.get("files", {})) for e in manifest["evaluations"])
    audit = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "computed_payload_sha256": hashes,
        "csv_jsonl_field_comparisons": comparisons,
        "csv_jsonl_mismatches": 0,
        "manifest_counts": counts,
        "upstream_hashes_listed": source_hashes,
        "upstream_hashes_verified": 0,
        "hash_scope": (
            "Manifest hashes identify upstream files absent from this archive."
        ),
        "execution_semantics_in_export": any("execution_semantics" in r for r in rows),
        "token_fields_in_export": any("token" in k for r in rows for k in r),
        "replay_performed": False,
    }
    return rows, manifest, audit


def summarize(rows: list[dict], roster: tuple[str, ...]) -> list[dict]:
    """Apply missing-case guards separately to each recorded endpoint."""
    result = []
    for method, label in METHODS.items():
        selected = [
            row
            for row in rows
            if row["method"] == method and row["benchmark_id"] in roster
        ]
        if {(r["benchmark_id"], r["repetition"]) for r in selected} != {
            (case, repetition) for case in roster for repetition in range(3)
        } or len(selected) != 3 * len(roster):
            raise ValueError(
                f"incomplete/duplicate planned three-repetition roster: {method}"
            )
        cases = []
        for case in roster:
            subset = sorted(
                (r for r in selected if r["benchmark_id"] == case),
                key=lambda row: row["repetition"],
            )
            metrics = {}
            for field in FIELDS:
                samples = []
                for row in subset:
                    value = row.get(field)
                    if value is None:
                        continue
                    if not finite(value) or value < 0:
                        raise ValueError(
                            f"invalid numeric endpoint: {row['request_id']}/{field}"
                        )
                    status_key = {
                        "target_nmse": "target_status",
                        "mechanism_compliance": "mechanism_status",
                    }.get(field)
                    if status_key and row.get(status_key) != "available":
                        raise ValueError(f"score on unavailable endpoint: {status_key}")
                    if field == "mechanism_compliance" and value > 1:
                        raise ValueError("graph compliance is not in [0, 1]")
                    samples.append(value)
                metrics[field] = statistics(samples)
            cases.append(
                {
                    "benchmark_id": case,
                    "metrics": metrics,
                    "rows": [
                        {
                            k: r.get(k)
                            for k in (
                                "request_id",
                                "repetition",
                                "target_status",
                                "target_nmse",
                                "mechanism_status",
                                "mechanism_compliance",
                                "mechanism_coverage",
                                "runtime_valid",
                                "terminal_status",
                                "reason",
                                "complexity_terms",
                            )
                        }
                        for r in subset
                    ],
                }
            )
        macros = {}
        for field in FIELDS:
            medians = [case["metrics"][field]["median"] for case in cases]
            missing = [
                case["benchmark_id"]
                for case in cases
                if case["metrics"][field]["median"] is None
            ]
            macros[field] = {
                **(statistics(medians) if not missing else statistics([])),
                "available_cases": len(roster) - len(missing),
                "missing_cases": missing,
                "record_coverage": sum(case["metrics"][field]["n"] for case in cases),
            }
        result.append(
            {
                "method": method,
                "label": label,
                "planned": len(selected),
                "macros": macros,
                "cases": cases,
                "target_statuses": dict(
                    Counter(str(r.get("target_status")) for r in selected)
                ),
                "graph_statuses": dict(
                    Counter(str(r.get("mechanism_status")) for r in selected)
                ),
            }
        )
    return result


def display(stat: dict, *, scale: float = 1, digits: int = 3) -> str:
    """Format only at presentation time, keeping full precision in JSON."""
    if stat["median"] is None:
        return "Unavailable"
    centre, spread = stat["median"] * scale, stat["mad"] * scale
    if abs(centre) >= 1e4 or 0 < abs(centre) < 1e-3:
        return f"{centre:.3g} [{spread:.3g}]"
    return f"{centre:.{digits}f} [{spread:.{digits}f}]"


def write_report(output: Path, report: dict) -> None:
    """Write the headline table, case-level evidence, and a LaTeX table fragment."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# External-baseline calculation cross-check",
        "",
        "Case median across repetitions, then median [unscaled MAD] "
        "across the same nine cases.",
        "Statistics are conditional on recorded endpoints; "
        "coverage is endpoint-specific.",
        "",
        "| Method | Test NMSE | Graph compliance (%) | Additive terms "
        "| NMSE coverage | Graph coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    tex = [
        "% Independently aggregated from the supplied package; no model replay.",
        "% Median [unscaled MAD] of nine case medians; endpoint-wise coverage.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{External baselines on the nine-case roster. Values are "
        r"median [unscaled MAD] of case medians. Graph compliance is the "
        r"recorded equation-graph endpoint, not annotation coverage or a "
        r"general scientific-validity score. Statistics use available "
        r"endpoints, with coverage shown separately. An aggregate is "
        r"unavailable if an entire case has no scored repetition. Token "
        r"usage and unresolved-obligation counts are absent from this export.}",
        r"\label{tab:external_baseline_crosscheck}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Test NMSE & Graph (\%) & Terms & NMSE $n/N$ & Graph $n/N$ \\",
        r"\midrule",
    ]
    for method in report["methods"]:
        stats = method["macros"]
        cells = [
            method["label"],
            display(stats["target_nmse"]),
            display(stats["mechanism_compliance"], scale=100, digits=1),
            display(stats["complexity_terms"], digits=1),
            f"{stats['target_nmse']['record_coverage']}/{method['planned']}",
            f"{stats['mechanism_compliance']['record_coverage']}/{method['planned']}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    lines.append("")
    for method in report["methods"]:
        for case in method["macros"]["target_nmse"]["missing_cases"]:
            lines.append(f"{method['label']}: no scored repetition for `{case}`.")
    lines += [
        "A 100% macro median does not mean every benchmark passed. "
        "Read graph coverage separately from prediction coverage.",
        "The graph endpoint is not the separate fitted-equation/activity assessment.",
        "",
        "## Per-case NMSE: median [MAD]",
        "",
        "| Case | SINDy | PySR | D3 | GPT-5.6 Sol agent |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for index, case in enumerate(report["roster"]):
        cells = [case.removeprefix("phase_b_")]
        for method in report["methods"]:
            stat = method["cases"][index]["metrics"]["target_nmse"]
            cells.append(display(stat) + (f" ({stat['n']}/3)" if stat["n"] < 3 else ""))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Incomplete cells show (scored/3). A zero MAD from one observation "
        "does not establish repeatability.",
        "",
        "## Audit scope",
        "",
    ]
    lines += [f"- {item}" for item in report["limitations"]]
    lines += [
        "",
        "## Integrity",
        "",
        "```json",
        json.dumps(report["integrity"], indent=2),
        "```",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "BASELINE_TABLE.tex").write_text("\n".join(tex), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    roster = read_roster(
        repo / "scripts/build_experiments_table_results.py",
        repo / "configs/final_component_campaign_v1.json",
    )
    rows, _, integrity = load_package(args.package)
    report = {
        "protocol": "independent-baseline-package-aggregate-audit-1",
        "roster": roster,
        "integrity": integrity,
        "methods": summarize(rows, roster),
        "limitations": [
            "Aggregates were independently recomputed; individual rollout NMSEs "
            "were not replayed.",
            "JSONL and CSV agree field by field. Manifest hashes name upstream "
            "files not included here; their contents cannot be verified from "
            "this archive.",
            "The supplied v1 archive lacks execution_semantics, fitted parameter "
            "values, and replay trajectories. D3 uses the recorded d3 evaluation "
            "scores, never an ODE reinterpretation; per-subject execution "
            "receipts are not independently verifiable here.",
            "The manifest's test_data_opened=false describes packaging; the "
            "package script reads already-produced final-evaluation endpoints.",
            "Graph scores use mechanism_compliance, not annotation-based "
            "mechanism_coverage. Unresolved-obligation counts and token usage "
            "are absent from the supplied v1 export.",
            "Fitted parameters are not substituted by the graph evaluation "
            "entry point. This graph endpoint differs from the separately "
            "implemented fitted-equation assessment.",
        ],
    }
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "selected_rows": sum(m["planned"] for m in report["methods"]),
                "csv_jsonl_mismatches": integrity["csv_jsonl_mismatches"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
